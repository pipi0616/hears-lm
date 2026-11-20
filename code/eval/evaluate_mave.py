#!/usr/bin/env python3
"""
Orthogonal validation against experimental multiplexed assays (MaveDB), independent of clinical labels.

For every primary score set (KCNQ4 homozygous current, KCNE1 function, 13 'domainome' abundance sets that report
cellular abundance, a proxy of folding stability) the experimental score is oriented so that higher = more
damaging (all primary sets: higher score = more functional / more stable) and compared by Spearman correlation with
  HEARS-LM pathogenicity probability and HEARS-LM ddG head of the SEQUENCE-ONLY configuration (leave-gene-family-out
      models: the gene was never seen in training). The sequence-only configuration is used because most assayed
      substitutions are not single-nucleotide changes and have no dbNSFP conservation record, so the conservation
      inputs of the selected configuration would have to be imputed,
  DDGun ddG (Tollefson et al. 2023; available for variants observed in the Deafness Variation Database),
  AlphaMissense (canonical-isoform table; equal to dbNSFP AlphaMissense up to the multi-transcript rule,
      Spearman 0.995 on the labelled variants),
  ESM-2 150M zero-shot (full context, wild-type marginal log-likelihood ratio).
Correlations are computed on the variants scored by all methods compared (common set) with bootstrap 95% CIs,
and DDGun separately on its own (smaller) coverage.

Output: results/mave_correlations.csv, results/mave_variant_scores.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS  # noqa: E402

# AlphaMissense scores of every substitution of the panel proteins (canonical isoform), needed because the assays cover
# substitutions that are not single-nucleotide changes and therefore absent from dbNSFP
AM_TABLE = DATA / 'annotations/alphamissense_all_substitutions.csv.gz'
AA = 'ACDEFGHIKLMNPQRSTVWY'


def am_lookup(df):
    am = pd.read_csv(AM_TABLE, usecols=['gene', 'protein_short', 'am_score'])
    x = am.protein_short.str.extract(r'^([A-Z])(\d+)([A-Z])$')
    am['ref1'], am['uniprot_pos'], am['alt1'] = x[0], pd.to_numeric(x[1]), x[2]
    res = pd.read_csv(DATA / 'annotations/residue_structure_table.csv', usecols=['gene', 'ref_pos', 'uniprot_pos'])
    d = df[['variant_id', 'gene', 'ref_pos', 'ref1', 'alt1']].merge(res, on=['gene', 'ref_pos'], how='left')
    d = d.merge(am[['gene', 'uniprot_pos', 'ref1', 'alt1', 'am_score']], on=['gene', 'uniprot_pos', 'ref1', 'alt1'],
                how='left')
    return d.set_index('variant_id').am_score


def esm_llr(df):
    w = pd.read_csv(DATA / 'annotations/esm2_wt_marginals.csv.gz')
    d = df[['variant_id', 'gene', 'ref_pos', 'ref1', 'alt1']].merge(w, on=['gene', 'ref_pos'], how='left')
    rec = d.to_dict('records')
    return pd.Series([r[f"lp_{r['alt1']}"] - r[f"lp_{r['ref1']}"] for r in rec], index=d.variant_id)


def spearman_ci(x, y, nboot=1000, seed=0):
    rng = np.random.default_rng(seed)
    est = spearmanr(x, y).statistic
    vals = [spearmanr(x[i], y[i]).statistic for i in (rng.integers(0, len(x), len(x)) for _ in range(nboot))]
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return est, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cv-arm', default='cv_seq_plm_ddg')
    ap.add_argument('--nboot', type=int, default=1000)
    a = ap.parse_args()
    mv = pd.read_csv(DATA / 'annotations/mave_v8.csv', low_memory=False)
    ss = pd.read_csv(DATA / 'annotations/mave_scoresets_v8.csv')
    prim = ss[ss.primary].set_index('urn')
    mv = mv[mv.urn.isin(prim.index)].copy()
    sign = np.where(prim.higher_score_means.reindex(mv.urn).str.startswith('more').to_numpy(), -1.0, 1.0)
    mv['damage'] = sign * mv.score
    pred = pd.read_csv(RESULTS / f'external_predictions_{a.cv_arm}.csv').set_index('variant_id')
    ddg = pd.read_csv(DATA / 'annotations/ddgun_v8.csv', usecols=['variant_id', 'ddg']).set_index('variant_id').ddg
    uniq = mv.drop_duplicates('variant_id')
    am = am_lookup(uniq)
    esm = esm_llr(uniq)
    mv['HEARS-LM (pathogenicity)'] = pred.prob.reindex(mv.variant_id).to_numpy()
    if 'ddg_pred' in pred:
        mv['HEARS-LM (ddG head)'] = pred.ddg_pred.reindex(mv.variant_id).to_numpy()
    mv['AlphaMissense'] = am.reindex(mv.variant_id).to_numpy()
    mv['ESM-2 150M zero-shot'] = -esm.reindex(mv.variant_id).to_numpy()
    mv['DDGun ddG'] = ddg.reindex(mv.variant_id).to_numpy()
    methods = [c for c in ['HEARS-LM (pathogenicity)', 'HEARS-LM (ddG head)', 'AlphaMissense', 'ESM-2 150M zero-shot']
               if c in mv]
    mv.to_csv(RESULTS / 'mave_variant_scores.csv', index=False)
    rows = []
    for urn, d in mv.groupby('urn'):
        info = prim.loc[urn]
        common = d[methods + ['damage']].notna().all(axis=1).to_numpy()
        for m in methods + ['DDGun ddG']:
            k = common & d[m].notna().to_numpy() if m != 'DDGun ddG' else d[[m, 'damage']].notna().all(axis=1).to_numpy()
            if k.sum() < 20:
                continue
            est, lo, hi = spearman_ci(d[m].to_numpy()[k], d.damage.to_numpy()[k], a.nboot)
            rows.append({'urn': urn, 'gene': info.gene, 'assay_type': info.assay_type, 'method': m,
                         'subset': 'common' if m != 'DDGun ddG' else 'DDGun coverage', 'n': int(k.sum()),
                         'spearman': est, 'lo': lo, 'hi': hi})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / 'mave_correlations.csv', index=False)
    pd.set_option('display.width', 220)
    print(out.pivot_table(index=['gene', 'assay_type'], columns='method', values='spearman').round(3).to_string())
    dom = out[out.assay_type == 'domainome abundance']
    print('\nmean over domainome sets:', dom.groupby('method').spearman.mean().round(3).to_dict())


if __name__ == '__main__':
    main()
