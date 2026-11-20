#!/usr/bin/env python3
"""
Score variants outside the labelled set with the leave-gene-family-out models, so that every gene is scored by a
model that never saw that gene or any homologous panel gene during training.

Variant sets
  * all MaveDB variants of data/annotations/mave_v8.csv (functional / abundance validation)
  * all 19 substitutions at every residue of the genes given with --saturation (mutational landscapes)
Each gene goes to the model of its cross-validation fold (data/gene_families_v8.csv).

  python predict_external.py --cv-arm cv_seq_plm_ddg --saturation KCNQ4 OTOF GJB2
  -> results/external_predictions_<cv-arm>.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
from hearslm import load_reference_sequences  # noqa: E402
from predict import prepare, saturation_table, score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cv-arm', required=True)
    ap.add_argument('--saturation', nargs='*', default=[])
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--genes', nargs='*', help='restrict to these genes (testing)')
    ap.add_argument('--device')
    ap.add_argument('--runs-root', default=str(ROOT / 'runs'))
    ap.add_argument('--out')
    a = ap.parse_args()
    seqs = load_reference_sequences()
    mave = pd.read_csv(ROOT / 'data/annotations/mave_v8.csv', usecols=['gene', 'ref_pos', 'ref1', 'alt1'])
    sat = saturation_table(a.saturation, seqs) if a.saturation else mave.iloc[:0]
    df = prepare(pd.concat([mave, sat], ignore_index=True), seqs)
    df['in_mave'] = df.variant_id.isin(prepare(mave, seqs).variant_id)
    df['in_saturation'] = df.gene.isin(a.saturation)
    fam = pd.read_csv(ROOT / 'data/gene_families_v8.csv')
    fold_of = dict(zip(fam.gene, fam.cv_fold))
    df['fold'] = df.gene.map(fold_of)
    if a.genes:
        df = df[df.gene.isin(a.genes)].copy()
    assert df.fold.notna().all()
    outs = []
    for k, d in df.groupby('fold'):
        run = Path(a.runs_root) / a.cv_arm / f'fold{int(k)}'
        assert (run / 'model_trainable.pt').exists(), f'missing {run}'
        o = score(d.reset_index(drop=True), [run], batch_size=a.batch_size, seqs=seqs, device=a.device)
        outs.append(o[['variant_id', 'prob'] + (['ddg_pred'] if 'ddg_pred' in o else [])].merge(
            d[['variant_id', 'gene', 'ref_pos', 'ref1', 'alt1', 'fold', 'in_mave', 'in_saturation']], on='variant_id'))
    out = pd.concat(outs, ignore_index=True)
    dest = Path(a.out) if a.out else ROOT / 'results' / f'external_predictions_{a.cv_arm}.csv'
    out.to_csv(dest, index=False)
    print(f'wrote {dest}: {len(out)} variants; by fold {out.fold.value_counts().to_dict()}')


if __name__ == '__main__':
    main()
