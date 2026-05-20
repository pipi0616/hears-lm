#!/usr/bin/env python3
"""
Clean HL-VCEP comparison (NO train leakage).
Use only HL-VCEP variants NOT in our training set.
Compare us vs AM vs REVEL on this held-out subset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

ROOT = Path('/Users/pipi/Projects/QAFI_Paper/hearing_v2')
EVAL = ROOT/'eval_results'

# 1. Load HL-VCEP curated
hl = pd.read_csv(ROOT/'data/external/hl_vcep_curated.csv')
hl = hl.dropna(subset=['protein_change']).drop_duplicates(subset=['gene','protein_change'])
def map_class_ordinal(c):
    if c is None or pd.isna(c): return None
    s = str(c).lower()
    if 'benign - stand alone' in s or s.strip()=='benign': return 0
    if 'likely benign' in s: return 1
    if 'likely pathogenic' in s: return 3
    if 'pathogenic' in s: return 4
    if 'uncertain' in s: return 2
    return None
hl['ordinal'] = hl.final_classification.apply(map_class_ordinal)
hl = hl.dropna(subset=['ordinal'])
print(f'HL-VCEP variants: {len(hl)}, classes:')
print(hl.ordinal.value_counts().sort_index().to_dict())

# 2. Identify train variants — leakage
train = pd.read_csv(ROOT/'data/splits/train_with_features.csv', low_memory=False)
train_keys = set(zip(train.gene, train.protein_change))
hl['in_train'] = hl.apply(lambda r: (r.gene, r.protein_change) in train_keys, axis=1)
hl_clean = hl[~hl.in_train].copy()
print(f'\nLeakage: {hl.in_train.sum()}/{len(hl)} HL-VCEP variants in our train set')
print(f'Clean HL-VCEP (NOT in train): {len(hl_clean)}')
print(f'Clean class distribution:')
print(hl_clean.ordinal.value_counts().sort_index().to_dict())

# 3. Aggregate predictions from val_time / val_gene / val_db
all_preds = []
for s in ['val_time','val_gene','val_db','val_func_mave']:
    df = pd.read_csv(ROOT/f'data/splits/{s}_with_features.csv', low_memory=False)
    cols = ['gene','protein_change']
    for c in ['am_score_best','baseline_alphamissense_score','revel','baseline_revel_score',
              'cadd_phred','baseline_cadd_phred','phylop100','gerp_rs','phastcons100',
              'abs_ddg_fold','plddt','gnomad_mis_z','af_popmax','baseline_metarnn_score',
              'baseline_bayesdel_add_af_score','baseline_clinpred_score','baseline_vest4_score',
              'baseline_eve_score','baseline_provean_score','baseline_sift_score',
              'baseline_polyphen2_hvar_score','baseline_mvp_score','baseline_mutpred_score',
              'baseline_primateai_score']:
        if c in df.columns: cols.append(c)
    # Add our model predictions
    if s in ('val_db','val_func_mave'):
        ours = pd.read_csv(EVAL/f'{s}_predictions_v2_ep2.csv')
    else:
        ours = pd.read_csv(EVAL/f'{s}_predictions_v2_ep2.csv')
    df_with = df[cols].merge(ours[['gene','protein_change','pred_prob']].rename(columns={'pred_prob':'Ours_v2'}),
                              on=['gene','protein_change'], how='left')
    df_with['source'] = s
    all_preds.append(df_with)
all_preds = pd.concat(all_preds, ignore_index=True).drop_duplicates(subset=['gene','protein_change'])
print(f'\nPredictors collected: {len(all_preds)} unique variants')

# Coalesce dual-source columns
def coal(df, *cols, out=None):
    out = out or cols[0]
    s = pd.Series([np.nan]*len(df), index=df.index)
    for c in cols:
        if c in df.columns:
            s = s.combine_first(df[c])
    df[out] = s
coal(all_preds, 'am_score_best', 'baseline_alphamissense_score', out='AM')
coal(all_preds, 'revel', 'baseline_revel_score', out='REVEL')
coal(all_preds, 'cadd_phred', 'baseline_cadd_phred', out='CADD')

# 4. Merge HL-VCEP_clean ↔ predictions
m = hl_clean.merge(all_preds, on=['gene','protein_change'], how='inner')
print(f'\nClean HL-VCEP with at least one prediction: {len(m)}')

# Per-predictor n
predictors = ['Ours_v2', 'AM', 'REVEL', 'CADD', 'phylop100', 'abs_ddg_fold',
              'baseline_metarnn_score','baseline_bayesdel_add_af_score','baseline_clinpred_score',
              'baseline_vest4_score','baseline_eve_score','baseline_mvp_score',
              'baseline_mutpred_score','baseline_primateai_score']
print('\nCoverage per predictor (on clean HL-VCEP set):')
for p in predictors:
    if p in m.columns:
        print(f'  {p:35} {m[p].notna().sum():3d}/{len(m)}')

# 5. Strict fair comparison: only variants where ALL key predictors have score
key = ['Ours_v2','AM','REVEL','CADD']
key = [c for c in key if c in m.columns]
mask_strict = m[key].notna().all(axis=1)
m_strict = m[mask_strict].copy()
print(f'\n=== Strict intersection (all of {key} have score): n={len(m_strict)} ===')
print(f'Class distribution: {m_strict.ordinal.value_counts().sort_index().to_dict()}')

# 6. Evaluate predictors on strict intersection
def evaluate(scores, ordinal, name):
    valid = scores.notna() & ordinal.notna()
    if valid.sum() < 10:
        return {'name': name, 'n': int(valid.sum())}
    s = scores[valid].values; o = ordinal[valid].astype(int).values
    rho = spearmanr(s, o).statistic
    if rho < 0:
        s = -s; rho = -rho
    auc_path = roc_auc_score((o >= 3).astype(int), s) if len(set(o >= 3)) > 1 else np.nan
    auc_ben  = roc_auc_score((o <= 1).astype(int), -s) if len(set(o <= 1)) > 1 else np.nan
    return {'name': name, 'n': int(valid.sum()),
            'spearman_ordinal': round(rho, 3),
            'AUC_PLP_vs_rest': round(auc_path, 3),
            'AUC_BLB_vs_rest': round(auc_ben, 3)}

print('\nResults on strict intersection:')
rows = []
for p in predictors:
    if p in m_strict.columns:
        rows.append(evaluate(m_strict[p], m_strict.ordinal, p))
strict_df = pd.DataFrame(rows)
if 'spearman_ordinal' in strict_df.columns:
    strict_df = strict_df.sort_values('spearman_ordinal', ascending=False)
    print(strict_df.to_string(index=False))
else:
    print(f'  (insufficient data: strict n={len(m_strict)} — skipping)')

# 7. Loose comparison: each predictor on its own coverage (apples to oranges but informative)
print('\n=== Loose (per-predictor own coverage) ===')
rows_loose = []
for p in predictors:
    if p in m.columns:
        rows_loose.append(evaluate(m[p], m.ordinal, p))
loose_df = pd.DataFrame(rows_loose).sort_values('spearman_ordinal', ascending=False)
print(loose_df.to_string(index=False))

# Save
OUT = EVAL/'hl_vcep_clean'
OUT.mkdir(exist_ok=True)
strict_df.to_csv(OUT/'strict_intersection_compare.csv', index=False)
loose_df.to_csv(OUT/'loose_per_predictor.csv', index=False)
m_strict.to_csv(OUT/'strict_variants.csv', index=False)
m.to_csv(OUT/'all_clean_variants.csv', index=False)
print(f'\n✅ Saved → {OUT}')
