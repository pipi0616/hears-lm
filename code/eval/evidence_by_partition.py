#!/usr/bin/env python3
"""
Evidence yield and discordance of the calibrated PP3/BP4 thresholds, per test partition.

The thresholds are those of results/calibration_thresholds.csv (one set per method, derived on the pooled test
variants); here they are evaluated separately in each setting:
  P_with_PP3  fraction of pathogenic variants that receive PP3 at any strength (useful evidence)
  B_with_BP4  fraction of benign variants that receive BP4 at any strength (useful evidence)
  P_with_BP4  fraction of pathogenic variants that receive BP4 (discordant - evidence in the wrong direction)
  B_with_PP3  fraction of benign variants that receive PP3 (discordant)
Output: results/evidence_by_partition.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from common import DATA, RESULTS  # noqa: E402


def main():
    a = pd.read_csv(RESULTS / 'evidence_assignment.csv')
    v = pd.read_csv(DATA / 'variants_v8.csv', usecols=['variant_id', 'partition'])
    a = a.merge(v, on='variant_id')
    a['PP3'] = a.evidence.str.startswith('PP3')
    a['BP4'] = a.evidence.str.startswith('BP4')
    rows = []
    for (m, p), d in a.groupby(['method', 'partition']):
        P, B = d[d.y == 1], d[d.y == 0]
        rows.append({'method': m, 'partition': p, 'n_P': len(P), 'n_B': len(B),
                     'P_with_PP3': P.PP3.mean(), 'B_with_BP4': B.BP4.mean(),
                     'P_with_BP4': P.BP4.mean(), 'B_with_PP3': B.PP3.mean(),
                     'P_with_PP3_moderate_or_stronger': P.evidence.isin(['PP3_moderate', 'PP3_strong',
                                                                         'PP3_very_strong']).mean()})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / 'evidence_by_partition.csv', index=False)
    pd.set_option('display.width', 220)
    print(out.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
