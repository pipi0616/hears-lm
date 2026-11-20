#!/usr/bin/env python3
"""
Evidence-strength thresholds (Tavtigian-Pejaver local posterior, prior 0.0441) for another released configuration,
derived exactly as for the selected model in evaluate_clinical.py: same calibration variants (the pooled test
variants scored by HEARS-LM and every comparison predictor), same number of bootstrap resamples, and out-of-sample
yields by the same five-fold cross-fitting.

  python calibrate_arm.py seq_plm_ddg
      -> results/calibration_thresholds_seq_plm_ddg.csv, results/evidence_crossfit_seq_plm_ddg.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from common import RESULTS, TESTS, load_baselines, load_preds  # noqa: E402
from evaluate_clinical import COMPARATORS, NBOOT, _yields, assign_evidence, final_arm, test_predictions  # noqa: E402


def main():
    arm = sys.argv[1]
    base, _ = load_baselines()
    ref = test_predictions(final_arm())                       # defines the calibration variants
    ext = base.reindex(ref.variant_id)
    common = np.all([~np.isnan(ext[c].to_numpy()) for c in COMPARATORS if c in ext] + [ref.prob.notna().to_numpy()],
                    axis=0)
    d = pd.concat([load_preds(arm, p)[['variant_id', 'y', 'prob']].assign(partition=p) for p in TESTS]) \
        .set_index('variant_id').reindex(ref.variant_id[common]).reset_index()
    y = d.y.to_numpy().astype(int); sc = d.prob.to_numpy()
    r = M.evidence_thresholds(sc, y, n_boot=NBOOT, seed=0)
    out = pd.DataFrame([{'arm': arm, 'evidence': k, 'threshold': v, 'required_posterior': r['required_posterior'][k],
                         'n_calibration': len(d)} for k, v in r['thresholds'].items()])
    out.to_csv(RESULTS / f'calibration_thresholds_{arm}.csv', index=False)
    print(out.to_string(index=False))
    rng = np.random.default_rng(0)
    fold = np.empty(len(d), int)
    for _, idx in d.groupby(['partition', 'y']).indices.items():
        idx = rng.permutation(idx)
        fold[idx] = np.arange(len(idx)) % 5
    ev = np.empty(len(d), dtype=object)
    for f in range(5):
        tr = fold != f
        ev[~tr] = assign_evidence(sc[~tr], M.evidence_thresholds(sc[tr], y[tr], n_boot=NBOOT // 2, seed=0)['thresholds'])
    rows = [{'design': 'kfold', 'method': f'HEARS-LM ({arm})', 'partition': p,
             **_yields(ev[(d.partition == p).to_numpy()], y[(d.partition == p).to_numpy()])} for p in TESTS]
    cf = pd.DataFrame(rows)
    cf.to_csv(RESULTS / f'evidence_crossfit_{arm}.csv', index=False)
    print(cf.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
