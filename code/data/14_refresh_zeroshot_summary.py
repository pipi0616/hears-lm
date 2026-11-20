#!/usr/bin/env python3
"""
Recompute the per-partition zero-shot AUROCs of esm2_zeroshot_v8_summary.json and esm2_wt_marginals_summary.json
against the current labelled table (data/variants_v8.csv), without re-running the language model.

The per-variant zero-shot scores do not depend on the labels or partitions, so only the summary statistics need
refreshing when the labelled table is rebuilt. The modification time and size of the table that was used are recorded.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/eval'))
import metrics as M  # noqa: E402

V = ROOT / 'data/variants_v8.csv'
ANN = ROOT / 'data/annotations'


def stamp():
    st = os.stat(V)
    return {'path': str(V.relative_to(ROOT)), 'rows': int(len(pd.read_csv(V, usecols=['variant_id']))),
            'modified': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}


def per_partition(df, cols):
    out = {}
    for part, d in df.groupby('partition'):
        d = d[d.y.notna()]
        if d.y.nunique() < 2:
            continue
        out[part] = {'n': int(len(d)), 'n_P': int(d.y.sum())}
        for name, col, sign in cols:
            k = d[col].notna()
            out[part][name] = float(M.auroc(d.y[k].to_numpy().astype(int), sign * d[col][k].to_numpy()))
            out[part][f'n_{name}'] = int(k.sum())
    return out


def main():
    v = pd.read_csv(V, usecols=['variant_id', 'partition', 'y'])
    z = pd.read_csv(ANN / 'esm2_zeroshot_v8.csv', usecols=['variant_id', 'esm2_150m_llr', 'esm2_150m_llr_w32'])
    d = v.merge(z, on='variant_id', how='left')
    s = json.load(open(ANN / 'esm2_zeroshot_v8_summary.json'))
    s['auroc_by_partition'] = per_partition(d, [('auroc_neg_llr_full', 'esm2_150m_llr', -1),
                                                ('auroc_neg_llr_w32', 'esm2_150m_llr_w32', -1)])
    s['variants_v8_used'] = stamp()
    json.dump(s, open(ANN / 'esm2_zeroshot_v8_summary.json', 'w'), indent=2)
    w = pd.read_csv(ANN / 'esm2_wt_marginals.csv.gz')
    k = pd.read_csv(ANN / 'esm2_zeroshot_v8.csv', usecols=['variant_id', 'gene', 'ref_pos', 'ref1', 'alt1'])
    k = k.merge(w, on=['gene', 'ref_pos'], how='left')
    k['wt_llr'] = [r[f'lp_{a}'] - r[f'lp_{b}'] if isinstance(a, str) and f'lp_{a}' in r else np.nan
                   for (_, r), a, b in zip(k.iterrows(), k.alt1, k.ref1)]
    d = v.merge(k[['variant_id', 'wt_llr']], on='variant_id', how='left')
    s2 = json.load(open(ANN / 'esm2_wt_marginals_summary.json'))
    pp = per_partition(d, [('wt_marginal', 'wt_llr', -1)])
    for part, r in pp.items():
        s2['auroc'][part] = {**s2['auroc'].get(part, {}), 'wt_marginal': r['wt_marginal'], 'n': r['n']}
    s2['variants_v8_used'] = stamp()
    json.dump(s2, open(ANN / 'esm2_wt_marginals_summary.json', 'w'), indent=2)
    dev = s['auroc_by_partition']['dev']
    print(f"dev n={dev['n']} ({dev['n_P']} P): full {dev['auroc_neg_llr_full']:.3f}, window {dev['auroc_neg_llr_w32']:.3f}, "
          f"wt-marginal {s2['auroc']['dev']['wt_marginal']:.3f} ")


if __name__ == '__main__':
    main()
