#!/usr/bin/env python3
"""
HEARS-LM command-line scorer.

Input: a CSV with columns `gene` and `protein_change` (p.Arg143Trp, Arg143Trp or R143W; positions on the gene's
reference protein = the RefSeq transcript used by ClinVar, normally MANE Select), optionally `chrom`, `pos`,
`ref`, `alt` (GRCh38, single-nucleotide substitution).

  * rows WITH a genomic coordinate are scored by the full model (3-seed ensemble); its conservation inputs are
    fetched from dbNSFP through MyVariant.info by that coordinate (network access required)
  * rows WITHOUT a coordinate are scored by the sequence-only model, which needs nothing but the sequence
  * with --no-finetune, rows with a coordinate are scored instead by the configuration without any fine-tuning
    (frozen full-protein language-model values + descriptors); it runs quickly on a CPU

Output: one row per input variant with the probability, the model used, the calibrated ACMG/AMP evidence level
(PP3/BP4 Supporting or Moderate; thresholds from the Tavtigian-Pejaver calibration of each model on the pooled test
sets), the number of training variants of the gene, and the predicted folding free-energy change (DDGun-like,
kcal/mol, positive = destabilising). No evidence level is given for genes that contributed no training variants to
the evaluated models (the ten genes withheld for evaluation: OTOF, CDH23, TECTA, WFS1, TMC1, LOXHD1, STRC, CLDN14,
TBC1D24, ACTG1, and genes without labelled variants).

  python code/hearslm_score.py variants.csv -o scores.csv

The evidence level is computational evidence for use within ACMG/AMP-ClinGen classification; it is not a
classification. It is intended for genes with curated variants in the training data.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/model'))
AA3 = {'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H',
       'Ile': 'I', 'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S', 'Thr': 'T', 'Trp': 'W',
       'Tyr': 'Y', 'Val': 'V'}
MODELS = {'full': (['runs/full/seed0', 'runs/full/seed1', 'runs/full/seed2'], 'results/calibration_thresholds.csv'),
          'sequence-only': (['runs/seq_plm_ddg/seed0', 'runs/seq_plm_ddg/seed1', 'runs/seq_plm_ddg/seed2'],
                            'results/calibration_thresholds_seq_plm_ddg.csv'),
          'no-finetune': (['runs/feat_plm_only/seed0', 'runs/feat_plm_only/seed1', 'runs/feat_plm_only/seed2'],
                          'results/calibration_thresholds_feat_plm_only.csv')}


def parse(pc):
    s = str(pc).strip().replace('p.', '').strip('()')
    m = re.match(r'^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$', s)
    if m and m.group(1) in AA3 and m.group(3) in AA3:
        return AA3[m.group(1)], int(m.group(2)), AA3[m.group(3)]
    m = re.match(r'^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$', s)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None, None, None


def thresholds(path):
    t = pd.read_csv(ROOT / path)
    if 'method' in t:
        t = t[t.method == 'HEARS-LM']
    t = t[t.evidence != 'ANY_DISCORDANT'].dropna(subset=['threshold'])
    return dict(zip(t.evidence, t.threshold))


NOT_ASSIGNED = 'not assigned (gene absent from the training data of the evaluated model)'


def evidence(p, thr):
    """Evidence level up to Moderate, the strength reported and evaluated in the paper."""
    lvl = 'indeterminate'
    for s in ['supporting', 'moderate']:
        if f'PP3_{s}' in thr and p >= thr[f'PP3_{s}']:
            lvl = f'PP3_{s}'
        if f'BP4_{s}' in thr and p <= thr[f'BP4_{s}']:
            lvl = f'BP4_{s}'
    return lvl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--no-finetune', action='store_true',
                    help='score rows with a coordinate by the configuration without fine-tuning')
    a = ap.parse_args()
    from hearslm import load_reference_sequences
    from predict import prepare, score
    seqs = load_reference_sequences()
    df = pd.read_csv(a.input)
    p = df.protein_change.map(parse)
    df['ref1'], df['ref_pos'], df['alt1'] = [x[0] for x in p], [x[1] for x in p], [x[2] for x in p]
    bad = df.ref1.isna() | ~df.gene.isin(seqs)
    ok = df[~bad].copy()
    ok['ref_pos'] = ok.ref_pos.astype(int)
    match = [seqs[g][q - 1] == r if q <= len(seqs[g]) else False for g, q, r in zip(ok.gene, ok.ref_pos, ok.ref1)]
    ok = ok[match]
    has_coord = ok[['chrom', 'pos', 'ref', 'alt']].notna().all(axis=1) if {'chrom', 'pos', 'ref', 'alt'} <= set(ok) \
        else pd.Series(False, index=ok.index)
    outs = []
    n_train = pd.read_csv(ROOT / 'results/training_variants_per_gene.csv').set_index('gene').n_training_variants
    coord_model = 'no-finetune' if a.no_finetune else 'full'
    ok['ann_ok'] = False
    feats = None
    if has_coord.any():
        # annotate once; a coordinate that dbNSFP does not resolve to this amino-acid change is not used
        import importlib.util
        import os
        import subprocess
        import tempfile
        rows = ok[has_coord].copy()
        rows['chrom'] = (rows.chrom.astype(str).str.replace(r'\.0$', '', regex=True)
                         .str.replace('^chr', '', regex=True, case=False))
        rows['pos'] = pd.to_numeric(rows.pos).astype(int)
        v = prepare(rows[['gene', 'ref_pos', 'ref1', 'alt1']], seqs)
        tmp = Path(tempfile.mkdtemp())
        tab = v.merge(rows.assign(variant_id=rows.gene + ':' + rows.ref1 + rows.ref_pos.astype(str) + rows.alt1)
                      [['variant_id', 'chrom', 'pos', 'ref', 'alt']], on='variant_id')
        tab = tab.rename(columns={'ref': 'ref_nt', 'alt': 'alt_nt'}).assign(partition='user', y='')
        tab.to_csv(tmp / 'variants.csv', index=False)
        env = dict(os.environ, ANNOT_IN=str(tmp / 'variants.csv'), ANNOT_OUT=str(tmp / 'dbnsfp.csv'),
                   ANNOT_VERSIONS_KEY='user_query')
        subprocess.run([sys.executable, str(ROOT / 'code/data/05_annotate_dbnsfp.py')], env=env, check=True,
                       stdout=subprocess.DEVNULL)
        ann = pd.read_csv(tmp / 'dbnsfp.csv', low_memory=False)
        good = set(ann.variant_id[ann.found.astype(bool) & ann.dbnsfp_aa_match.fillna(False).astype(bool)])
        vid = ok.gene + ':' + ok.ref1 + ok.ref_pos.astype(str) + ok.alt1
        ok['ann_ok'] = (has_coord & vid.isin(good)).to_numpy()
        n_bad = int((has_coord & ~ok.ann_ok).sum())
        if n_bad:
            print(f'[note] {n_bad} variant(s): the coordinate did not resolve to the stated protein change in dbNSFP; '
                  f'scored by the sequence-only model instead')
        spec = importlib.util.spec_from_file_location('f9', ROOT / 'code/data/09_build_features.py')
        f9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f9)
        keep = tab[tab.variant_id.isin(good)]
        if len(keep):
            ft, _, _ = f9.assemble(keep, ann[ann.variant_id.isin(good)])
            ft.to_csv(tmp / 'features.csv', index=False)
            feats = tmp / 'features.csv'
    for name, rows in [(coord_model, ok[ok.ann_ok]), ('sequence-only', ok[~ok.ann_ok])]:
        if rows.empty:
            continue
        runs, thr_file = MODELS[name]
        v = prepare(rows[['gene', 'ref_pos', 'ref1', 'alt1']], seqs)
        res = score(v, [ROOT / r for r in runs], features_csv=feats if name != 'sequence-only' else None, seqs=seqs)
        thr = thresholds(thr_file)
        res['model'] = name
        res['n_training_variants_gene'] = res.gene.map(n_train).fillna(0).astype(int)
        res['evidence'] = [evidence(x, thr) if n > 0 else NOT_ASSIGNED
                           for x, n in zip(res.prob, res.n_training_variants_gene)]
        outs.append(res[['variant_id', 'gene', 'ref_pos', 'ref1', 'alt1', 'model', 'prob', 'evidence',
                         'n_training_variants_gene']
                        + (['ddg_pred'] if 'ddg_pred' in res else [])])
    out = pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()
    n_skip = len(df) - len(out)
    out.to_csv(a.output, index=False)
    if len(out) and (out.n_training_variants_gene == 0).any():
        g = sorted(out.gene[out.n_training_variants_gene == 0].unique())
        print(f'[note] no evidence level for {", ".join(g)}: these genes contributed no training variants to the '
              f'evaluated models (ten genes were withheld for evaluation); the score is given for information, and '
              f'a general-purpose predictor is recommended for them')
    print(f'scored {len(out)} variants -> {a.output}' + (f'; {n_skip} not scored (unknown gene, unparsable change '
                                                         f'or reference residue mismatch)' if n_skip else ''))


if __name__ == '__main__':
    main()
