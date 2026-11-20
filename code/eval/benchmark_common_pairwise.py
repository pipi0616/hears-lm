#!/usr/bin/env python3
"""
Paired comparison of the final model with every method of the common comparison set, on exactly the variants that
all of these methods score (the set of benchmark_common.csv), per test partition.

  results/benchmark_common_pairwise.csv  partition x method: AUROC of both on the common set, paired bootstrap
                                         difference (HEARS-LM minus method) with 95% CI, DeLong p, BH q over all
                                         partition x method comparisons; within-gene AUROC of both

Run after evaluate_fixed.py --stage test (uses the same final model, baselines and common-set definition).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from common import RESULTS, TESTS, gene_prior, load_baselines, load_preds, load_variants, panel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nboot', type=int, default=2000)
    a = ap.parse_args()
    final = json.load(open(RESULTS / 'final_model.json'))['final_arm']
    CORE = panel(final)
    v = load_variants()
    base, af = load_baselines()
    vi = v.set_index('variant_id')
    gp = gene_prior(v)
    gp.index = v.variant_id
    rows = []
    for part in TESTS:
        t = load_preds(final, part)[['variant_id', 'y', 'prob']].set_index('variant_id')
        ext = base.reindex(t.index).copy()
        ext['Gene prior (training label fraction)'] = gp.reindex(t.index)
        ext['Allele frequency only (-log10 AF)'] = -np.log10(af['af_any'].reindex(t.index).fillna(0) + 1e-7)
        core = [c for c in CORE if c in ext.columns]
        k = ext[core].notna().all(axis=1).to_numpy()
        y = t.y.to_numpy()[k].astype(int); s = t.prob.to_numpy()[k]
        genes = vi.gene.reindex(t.index).to_numpy()[k]
        for name in core + ['Gene prior (training label fraction)', 'Allele frequency only (-log10 AF)']:
            b = ext[name].to_numpy()[k]
            pb = M.paired_bootstrap_diff(y, s, b, M.auroc, a.nboot)
            dl = M.delong_test(y, s, b)
            rows.append({'partition': part, 'method': name, 'n': int(k.sum()), 'n_P': int(y.sum()),
                         'auroc_hearslm': dl['auc1'], 'auroc_method': dl['auc2'],
                         'diff_hearslm_minus_method': pb['diff'], 'diff_lo': pb['lo'], 'diff_hi': pb['hi'],
                         'p_delong': dl['p'], 'within_gene_hearslm': M.within_group_auroc(y, s, genes),
                         'within_gene_method': M.within_group_auroc(y, b, genes)})
    d = pd.DataFrame(rows)
    d['q_delong'] = M.bh_fdr(d.p_delong.to_numpy())
    d.to_csv(RESULTS / 'benchmark_common_pairwise.csv', index=False)
    print(d.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
