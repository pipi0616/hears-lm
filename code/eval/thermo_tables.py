#!/usr/bin/env python3
"""
Tables behind Figure 5a-b (the auxiliary folding-energy head).

  results/ddg_by_plddt.csv
      Spearman correlation between the auxiliary head of the selected model (three-seed mean) and DDGun on the
      withheld-gene test variants that have a DDGun value, by AlphaFold2 pLDDT of the residue.
  results/mave_head_vs_ddgun_same_variants.csv
      Per Human Domainome abundance assay, on the variants that have both a DDGun value and a head estimate:
      Spearman correlation with the experimental abundance score (oriented so that higher = more damaging) of the head
      (sequence-only cross-validation model, the model used for all assay comparisons) and of DDGun.
      Assays with fewer than 20 such variants are omitted.

Run after evaluate_fixed.py --stage test and evaluate_mave.py.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, RUNS, list_seeds  # noqa: E402


def main():
    arm = json.load(open(RESULTS / 'final_model.json'))['final_arm']
    p = pd.concat([pd.read_csv(RUNS / arm / s / 'preds_test_gene.csv') for s in list_seeds(arm)])
    p = p.groupby('variant_id')[['ddg_pred', 'ddg_target']].mean().dropna()
    st = pd.read_csv(DATA / 'annotations/structure_v8.csv', usecols=['variant_id', 'plddt', 'plddt_missing']) \
        .set_index('variant_id').reindex(p.index)
    lab = pd.Series('no model', index=p.index)
    ok = st.plddt_missing.fillna(1) == 0
    lab[ok & (st.plddt >= 90)] = 'pLDDT >= 90'
    lab[ok & (st.plddt >= 70) & (st.plddt < 90)] = '70-90'
    lab[ok & (st.plddt >= 50) & (st.plddt < 70)] = '50-70'
    lab[ok & (st.plddt < 50)] = '< 50'
    rows = [{'plddt': g, 'n': int((lab == g).sum()),
             'spearman': round(spearmanr(p.ddg_pred[lab == g], p.ddg_target[lab == g]).statistic, 3)}
            for g in ['pLDDT >= 90', '70-90', '50-70', '< 50', 'no model'] if (lab == g).sum() >= 20]
    a = pd.DataFrame(rows)
    a.to_csv(RESULTS / 'ddg_by_plddt.csv', index=False)
    print(a.to_string(index=False), f'\nall: n = {len(p)}, Spearman {spearmanr(p.ddg_pred, p.ddg_target).statistic:.3f}')

    mv = pd.read_csv(RESULTS / 'mave_variant_scores.csv', low_memory=False)
    ss = pd.read_csv(DATA / 'annotations/mave_scoresets_v8.csv').set_index('urn')
    rows = []
    for urn, d in mv.groupby('urn'):
        if ss.loc[urn, 'assay_type'] != 'domainome abundance':
            continue
        d = d.dropna(subset=['HEARS-LM (ddG head)', 'DDGun ddG', 'damage'])
        if len(d) < 20:
            continue
        rows.append({'urn': urn, 'gene': ss.loc[urn, 'gene'], 'n': len(d),
                     'head': spearmanr(d['HEARS-LM (ddG head)'], d.damage).statistic,
                     'ddgun': spearmanr(d['DDGun ddG'], d.damage).statistic})
    b = pd.DataFrame(rows)
    b.to_csv(RESULTS / 'mave_head_vs_ddgun_same_variants.csv', index=False)
    print(b.round(3).to_string(index=False))
    print(f'mean over {len(b)} assays (n = {int(b.n.sum())}): head {b["head"].mean():.3f}, DDGun {b.ddgun.mean():.3f}')


if __name__ == '__main__':
    main()
