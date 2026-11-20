#!/usr/bin/env python3
"""
Score arbitrary missense variants with one or several trained HEARS-LM runs (ensemble = mean over runs).

Input variants: CSV with columns gene, ref_pos, ref1, alt1 (reference-protein numbering, 1-letter codes),
or --saturation GENE [GENE ...] to enumerate all 19 substitutions at every residue of those genes.
Output: variant_id, gene, ref_pos, ref1, alt1, prob_<run>, ..., prob (mean), ddg_pred (mean, if available).

Descriptor inputs (models trained with feature groups): read from --features-csv (built with
code/data/09_build_features.py --variants ... --dbnsfp ...); without it they are assembled on the fly from the
residue/gene tables, in which case genomic-position groups (cons, af) are missing and a warning is printed.

  python predict.py --runs runs/seq_ddg/seed0 runs/seq_ddg/seed1 --variants my.csv --out scores.csv
  python predict.py --runs runs/cv_seq_ddg/fold2 --saturation KCNQ4 --out results/saturation_KCNQ4.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hearslm import (AA, ROOT, VariantSet, feature_columns, load_feature_table, load_reference_sequences,  # noqa: E402
                     load_trained, predict)


def saturation_table(genes, seqs):
    rows = []
    for g in genes:
        s = seqs[g]
        for i, ref in enumerate(s, start=1):
            if ref not in AA:
                continue
            for alt in AA:
                if alt != ref:
                    rows.append((g, i, ref, alt))
    return pd.DataFrame(rows, columns=['gene', 'ref_pos', 'ref1', 'alt1'])


def prepare(df, seqs):
    """Keep variants whose reference residue matches the reference protein; add variant_id."""
    df = df[['gene', 'ref_pos', 'ref1', 'alt1']].copy()
    df['ref_pos'] = df.ref_pos.astype(int)
    ok = [g in seqs and 1 <= p <= len(seqs[g]) and seqs[g][p - 1] == r
          for g, p, r in zip(df.gene, df.ref_pos, df.ref1)]
    n_bad = len(df) - int(np.sum(ok))
    if n_bad:
        print(f'[warn] {n_bad} variants do not match the reference protein and are dropped')
    df = df[ok].reset_index(drop=True)
    df['variant_id'] = df.gene + ':' + df.ref1 + df.ref_pos.astype(str) + df.alt1
    return df.drop_duplicates('variant_id').reset_index(drop=True)


def score(df, run_dirs, features_csv=None, batch_size=64, seqs=None, device=None):
    """Score prepared variants with each run; returns per-run and mean probability (and ddG if available)."""
    from transformers import AutoTokenizer
    seqs = seqs or load_reference_sequences()
    out = df[['variant_id', 'gene', 'ref_pos', 'ref1', 'alt1']].copy()
    probs, ddgs = [], []
    for rd in run_dirs:
        model, cfg, feat_cols, stats = load_trained(rd, device)
        tok = AutoTokenizer.from_pretrained(cfg['backbone'])
        if not cfg['feature_groups']:
            ft, cols = None, []
        elif features_csv:
            ft, cols = load_feature_table(cfg['feature_groups'], features_csv)
        else:
            missing = [g for g in cfg['feature_groups'] if g in ('cons', 'af')]
            if missing:
                print(f'[warn] groups {missing} need genomic annotation; they are missing (imputed as training mean)')
            import importlib.util
            spec = importlib.util.spec_from_file_location('feat09', ROOT / 'code/data/09_build_features.py')
            feat09 = importlib.util.module_from_spec(spec); spec.loader.exec_module(feat09)
            ft, _, _ = feat09.assemble(df, None, want=cfg['feature_groups'])
            ft = ft.set_index('variant_id')
            cols = feature_columns(cfg['feature_groups'])
        assert cols == feat_cols, 'feature definition changed since training'
        vs = VariantSet(df, seqs, tok, cfg['window'], ft, feat_cols, None)
        vs.standardise(stats)
        pred = predict(model, vs, cfg['device'], batch_size)
        tag = '_'.join(Path(rd).parts[-2:])
        out[f'prob_{tag}'] = pred.prob.to_numpy()
        probs.append(pred.prob.to_numpy())
        if 'ddg_pred_z' in pred:
            d = pred.ddg_pred_z.to_numpy() * stats['ddg_std'] + stats['ddg_mean']
            out[f'ddg_{tag}'] = d
            ddgs.append(d)
        del model
        print(f'scored {len(df)} variants with {rd}', flush=True)
    out['prob'] = np.mean(probs, axis=0)
    if ddgs:
        out['ddg_pred'] = np.mean(ddgs, axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--variants')
    ap.add_argument('--saturation', nargs='*')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--features-csv')
    a = ap.parse_args()
    seqs = load_reference_sequences()
    df = saturation_table(a.saturation, seqs) if a.saturation else pd.read_csv(a.variants)
    df = prepare(df, seqs)
    out = score(df, a.runs, a.features_csv, a.batch_size, seqs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f'wrote {a.out} ({len(out)} rows)')


if __name__ == '__main__':
    main()
