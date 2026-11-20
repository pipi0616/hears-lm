#!/usr/bin/env python3
"""
Check that the command-line scorer reproduces the evaluated models.

Takes a few temporal-test variants, scores them through code/hearslm_score.py exactly as a user would (gene, protein
change, GRCh38 coordinate; mixed with rows without a coordinate), and compares the probabilities with the stored
test-set predictions of the selected configuration (runs/full) and of the configuration without fine-tuning
(runs/feat_plm_only). Needs network access (dbNSFP via MyVariant.info).

  python code/tests/check_cli_reproduces_evaluation.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    v = pd.read_csv(ROOT / 'data/variants_v8.csv', low_memory=False)
    t = v[(v.partition == 'test_time') & v.chrom.notna()].sample(6, random_state=7)
    a = pd.DataFrame({'gene': t.gene, 'protein_change': t.ref1 + t.ref_pos.astype(str) + t.alt1,
                      'chrom': t.chrom.astype(str), 'pos': t.pos.astype(int), 'ref': t.ref_nt, 'alt': t.alt_nt})
    b = pd.DataFrame({'gene': ['GJB2', 'KCNQ4'], 'protein_change': ['p.Leu90Pro', 'G285S']})   # no coordinate
    tmp = Path(tempfile.mkdtemp())
    pd.concat([a, b]).to_csv(tmp / 'in.csv', index=False)
    ok = True
    for flag, arm, label in [([], 'full', 'full'), (['--no-finetune'], 'feat_plm_only', 'no-finetune')]:
        subprocess.run([sys.executable, str(ROOT / 'code/hearslm_score.py'), str(tmp / 'in.csv'), '-o',
                        str(tmp / 'out.csv')] + flag, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        o = pd.read_csv(tmp / 'out.csv').set_index('variant_id')
        ev = pd.concat([pd.read_csv(ROOT / f'runs/{arm}/seed{s}/preds_test_time.csv') for s in range(3)]) \
            .groupby('variant_id').prob.mean()
        c = o[o.model == label]
        d = np.abs(c.prob - ev.reindex(c.index)).max()
        n_seq = int((o.model == 'sequence-only').sum())
        good = len(c) == len(a) and d < 1e-6 and n_seq == len(b)
        ok &= good
        print(f'{label:12s}: {len(c)} coordinate variants, max |difference| {d:.2e}; {n_seq} sequence-only rows '
              f'-> {"PASS" if good else "FAIL"}')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
