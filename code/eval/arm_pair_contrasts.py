#!/usr/bin/env python3
"""
Contrasts between pairs of configurations that isolate one component each (Supplementary Table S5c):
  full-protein values (seq -> seq_plm), auxiliary objective (seq -> seq_ddg; seq_plm -> seq_plm_ddg),
  each descriptor group added to seq_plm_ddg, all three groups (seq_plm_ddg -> full), backbone size (seq_t12 -> seq),
  fine-tuning (feat_plm_only -> full).

For every pair and partition: pooled AUROC of the seed ensembles on identical variants with paired bootstrap
difference and DeLong test; within-gene AUROC of both with paired bootstrap difference; Welch test across seeds for
the pooled and the within-gene AUROC.

  results/arm_pair_contrasts.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from common import PARTS, RESULTS, load_preds, load_variants  # noqa: E402

PAIRS = [('seq', 'seq_plm', 'full-protein language-model values'),
         ('seq', 'seq_ddg', 'auxiliary objective (window encoder)'),
         ('seq_plm', 'seq_plm_ddg', 'auxiliary objective'),
         ('seq_plm_ddg', 'seq_plm_ddg_cons', 'conservation'),
         ('seq_plm_ddg', 'seq_plm_ddg_struct', 'structure annotation'),
         ('seq_plm_ddg', 'seq_plm_ddg_hear', 'gene-level descriptors'),
         ('seq_plm_ddg', 'full', 'all three descriptor groups'),
         ('seq_t12', 'seq', 'backbone 35M -> 150M (window encoder)'),
         ('feat_plm_only', 'full', 'adapted window encoder (fine-tuning)')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nboot', type=int, default=2000)
    a = ap.parse_args()
    gene = load_variants().set_index('variant_id').gene
    ps = pd.read_csv(RESULTS / 'arms_per_seed.csv')
    rows = []
    for base, other, what in PAIRS:
        for part in PARTS:
            pa, pb = load_preds(base, part), load_preds(other, part)
            if pa is None or pb is None:
                continue
            m = pa[['variant_id', 'y', 'prob']].merge(pb[['variant_id', 'prob']], on='variant_id', suffixes=('_a', '_b'))
            y = m.y.to_numpy().astype(int); sa = m.prob_a.to_numpy(); sb = m.prob_b.to_numpy()
            g = gene.reindex(m.variant_id).to_numpy()
            d = M.paired_bootstrap_diff(y, sb, sa, M.auroc, a.nboot)
            dl = M.delong_test(y, sb, sa)
            wg = lambda yy, ss, gg=None: np.nan  # noqa: E731
            rng = np.random.default_rng(0)
            wa, wb = M.within_group_auroc(y, sa, g), M.within_group_auroc(y, sb, g)
            diffs = []
            for _ in range(min(a.nboot, 1000)):                  # stratified by class, as for the pooled statistic
                i = np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum()) for c in (0, 1)])
                diffs.append(M.within_group_auroc(y[i], sb[i], g[i]) - M.within_group_auroc(y[i], sa[i], g[i]))
            lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
            xa = ps[(ps.arm == base) & (ps.partition == part)]; xb = ps[(ps.arm == other) & (ps.partition == part)]
            two = len(xa) > 1 and len(xb) > 1
            rows.append({'component': what, 'without': base, 'with': other, 'partition': part, 'n': len(m),
                         'auroc_without': dl['auc2'], 'auroc_with': dl['auc1'], 'diff': d['diff'], 'diff_lo': d['lo'],
                         'diff_hi': d['hi'], 'p_delong': dl['p'],
                         'within_gene_without': wa, 'within_gene_with': wb, 'within_gene_diff': wb - wa,
                         'within_gene_diff_lo': lo, 'within_gene_diff_hi': hi,
                         'n_seeds_without': len(xa), 'n_seeds_with': len(xb),
                         'p_welch_seeds': float(ttest_ind(xb.auroc, xa.auroc, equal_var=False).pvalue) if two else np.nan,
                         'p_welch_seeds_within_gene': float(ttest_ind(xb.auroc_within_gene, xa.auroc_within_gene,
                                                                      equal_var=False).pvalue) if two else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / 'arm_pair_contrasts.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 100)
    print(out[['component', 'partition', 'n', 'diff', 'diff_lo', 'diff_hi', 'p_delong', 'within_gene_diff',
               'within_gene_diff_lo', 'within_gene_diff_hi', 'p_welch_seeds', 'p_welch_seeds_within_gene']].round(4).to_string())


if __name__ == '__main__':
    main()
