#!/usr/bin/env python3
"""
Table 3 - what each component of the model contributes.

Every row is a separately trained configuration (not a leave-one-out ablation of a fixed model), reported as the
mean over random seeds on the development set and on the three test sets, plus the leave-gene-family-out
cross-validation where it was run. The allele-frequency row is a diagnostic: it quantifies how much of the task
is solvable from a feature that the ACMG/AMP criteria BA1, BS1 and PM2 already score, and is not part of the
released model.

Output: results/table_components.csv, results/table_components.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from common import RESULTS, RUNS, load_baselines, load_variants  # noqa: E402

ROWS = [
    ('feat_only', 'Descriptors only (no language model)'),
    ('frozen', 'Frozen language-model window embeddings (no adaptation)'),
    ('feat_plm_only', 'Frozen full-protein language-model values + descriptors (no adaptation)'),
    ('seq', 'Adapted language model, 65-residue window'),
    ('seq_plm', '  + full-protein language-model values'),
    ('seq_plm_ddg', '  + thermodynamic objective'),
    ('seq_plm_ddg_cons', '    + conservation'),
    ('seq_plm_ddg_struct', '    + structure annotation'),
    ('seq_plm_ddg_hear', '    + gene-level descriptors'),
    ('full', '    + all three (released model)'),
    ('full_af', 'Released model + allele frequency (diagnostic only)'),
    ('seq_t12', 'Adapted 35M language model, window only'),
]


def cv_auroc(arm, v, gene):
    fs = sorted((RUNS / f'cv_{arm}').glob('fold*/preds_cvtest.csv'))
    if len(fs) < 5:
        return None
    o = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return M.auroc(o.y, o.prob), M.within_group_auroc(o.y, o.prob, gene.reindex(o.variant_id).to_numpy())


def main():
    s = pd.read_csv(RESULTS / 'arms_summary.csv')
    v = load_variants()
    gene = v.set_index('variant_id').gene
    rows = []
    for arm, label in ROWS:
        d = s[s.arm == arm]
        if d.empty:
            continue
        r = {'configuration': label, 'arm': arm, 'seeds': int(d.n_seeds.max())}
        for p in ['dev', 'test_time', 'test_gene', 'test_db']:
            q = d[d.partition == p]
            if len(q):
                m, sd = q.auroc_mean.iloc[0], q.auroc_sd.iloc[0]
                r[p] = f'{m:.3f}' + ('' if np.isnan(sd) else f' ± {sd:.3f}')
                r[p + '_value'] = round(float(m), 4)
        c = cv_auroc(arm, v, gene)
        if c:
            r['cross_validation'] = f'{c[0]:.3f}'
            r['cross_validation_within_gene'] = f'{c[1]:.3f}'
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / 'table_components.csv', index=False)
    show = ['configuration', 'seeds', 'dev', 'test_time', 'test_gene', 'test_db', 'cross_validation']
    show = [c for c in show if c in t]
    pd.set_option('display.width', 230); pd.set_option('display.max_colwidth', 72)
    print(t[show].to_string(index=False))
    (RESULTS / 'table_components.md').write_text(t[show].to_markdown(index=False))


if __name__ == '__main__':
    main()
