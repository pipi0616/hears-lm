#!/usr/bin/env python3
"""
Comprehensive leaderboard v2: include AlphaMissense (full, 90% val_db) + dbNSFP baselines.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, brier_score_loss, roc_curve)
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

ROOT = Path('/Users/pipi/Projects/QAFI_Paper')
DATA = ROOT / 'hearing_v2/data'
EVAL = ROOT / 'hearing_v2/eval_results'

# -------- METRICS --------
def ece(y_true, y_prob, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0; n = len(y_true)
    for i in range(n_bins):
        m = (y_prob >= edges[i]) & (y_prob < edges[i+1])
        if i == n_bins-1: m = m | (y_prob == 1.0)
        if m.sum() == 0: continue
        e += abs(y_true[m].mean() - y_prob[m].mean()) * m.sum() / n
    return e

def sens_at_spec(y, s, t=0.90):
    fpr, tpr, _ = roc_curve(y, s)
    spec = 1 - fpr
    valid = spec >= t
    return tpr[valid].max() if valid.any() else float('nan')

def spec_at_sens(y, s, t=0.95):
    fpr, tpr, _ = roc_curve(y, s)
    valid = tpr >= t
    return (1-fpr)[valid].max() if valid.any() else float('nan')

def binary_metrics(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    m = ~np.isnan(y_score)
    if m.sum() < 20 or len(np.unique(y_true[m])) < 2:
        return {'n': int(m.sum())}
    y_true = y_true[m]; y_score = y_score[m]
    auc = roc_auc_score(y_true, y_score)
    flipped = False
    if auc < 0.5:
        y_score = -y_score; auc = roc_auc_score(y_true, y_score); flipped = True
    ap = average_precision_score(y_true, y_score)
    lo, hi = y_score.min(), y_score.max()
    prob = (y_score - lo) / (hi - lo) if hi > lo else np.zeros_like(y_score)
    y_pred = (prob >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, prob)
    e = ece(y_true, prob)
    s90 = sens_at_spec(y_true, y_score, 0.90)
    sp95 = spec_at_sens(y_true, y_score, 0.95)
    ppv = (y_true[y_pred==1].mean() if (y_pred==1).sum() > 0 else float('nan'))
    npv = (1 - y_true[y_pred==0].mean() if (y_pred==0).sum() > 0 else float('nan'))
    return {'AUC':auc,'AP':ap,'F1':f1,'MCC':mcc,'Brier':brier,'ECE':e,
            'Sens@90Spec':s90,'Spec@95Sens':sp95,'PPV':ppv,'NPV':npv,
            'n':int(m.sum()),'flip':int(flipped)}


# -------- LOAD DATA --------
print('Loading data...')

def load_split(s):
    df = pd.read_csv(DATA/f'splits/{s}_with_features.csv', low_memory=False)
    # Add our model predictions
    for tag in ['', 'v2', 'v3']:
        suffix = '_ep2' if tag == '' else f'_{tag}_ep2'
        # train_local_test wrote val_db_predictions_ep2.csv etc
        # for val_time/val_gene we wrote {s}_predictions_{vtag}_ep2.csv
        if s in ('val_db','val_func_mave'):
            pname = f'{s}_predictions{suffix}.csv'
        else:
            vtag = 'ep2' if tag == '' else f'{tag}_ep2'
            pname = f'{s}_predictions_{vtag}.csv'
        p = EVAL/pname
        col = f'Ours_v{1 if tag=="" else tag[-1]}'
        if p.exists():
            d = pd.read_csv(p)
            df = df.merge(d[['gene','protein_change','pred_prob']].rename(columns={'pred_prob':col}),
                          on=['gene','protein_change'], how='left')
    # Add zero-shot
    zs = EVAL/f'esm_zeroshot_{s}.csv'
    if zs.exists():
        d = pd.read_csv(zs)
        df = df.merge(d[['gene','protein_change','esm_llr']].rename(columns={'esm_llr':'ESM-2 zero-shot'}),
                      on=['gene','protein_change'], how='left')
    elif 'esm_llr' in df.columns:
        df['ESM-2 zero-shot'] = df.esm_llr
    return df

val_db = load_split('val_db')
val_time = load_split('val_time')
val_gene = load_split('val_gene')
val_mv  = load_split('val_func_mave')

print(f'val_db {val_db.shape}, val_time {val_time.shape}, val_gene {val_gene.shape}, val_func_mave {val_mv.shape}')

# -------- DEFINE METHODS --------
def get_methods(df, label_col):
    methods = {}
    # Our models
    for tag in ['Ours_v1','Ours_v2','Ours_v3']:
        if tag in df.columns: methods[tag] = tag
    # Zero-shot
    if 'ESM-2 zero-shot' in df.columns: methods['ESM-2 zero-shot'] = 'ESM-2 zero-shot'
    # In-split predictors
    if 'revel' in df.columns: methods['REVEL_split'] = 'revel'
    if 'cadd_phred' in df.columns: methods['CADD_split'] = 'cadd_phred'
    # AlphaMissense full (direct)
    if 'am_score_full' in df.columns: methods['AlphaMissense'] = 'am_score_full'
    # dbNSFP baselines
    for c in df.columns:
        if c.startswith('baseline_') and c.endswith('_score'):
            name = c.replace('baseline_','').replace('_score','')
            methods[name.upper()] = c
        elif c == 'baseline_cadd_phred' and 'CADD_dbNSFP' not in methods:
            methods['CADD_dbNSFP'] = c
        elif c == 'baseline_polyphen2_hvar_score':
            methods['PolyPhen2_HVAR'] = c
    return methods


def bench(df, label_col, name):
    methods = get_methods(df, label_col)
    print(f'\n=== {name} ({len(methods)} methods, label={label_col}) ===')
    rows = []
    for n, col in methods.items():
        if col not in df.columns: continue
        m = binary_metrics(df[label_col].values, df[col].values)
        m['Method'] = n; rows.append(m)
    return pd.DataFrame(rows).set_index('Method').sort_values('AUC', ascending=False)


# val_db: binary y
df_db = bench(val_db, 'y', 'val_db')
print(df_db[['AUC','AP','F1','MCC','Brier','ECE','Sens@90Spec','Spec@95Sens','n']].round(3).to_string())

# val_time: binary y
df_vt = bench(val_time, 'y', 'val_time')
print(df_vt[['AUC','AP','F1','MCC','Brier','ECE','Sens@90Spec','Spec@95Sens','n']].round(3).to_string())

# val_gene: binary y
df_vg = bench(val_gene, 'y', 'val_gene')
print(df_vg[['AUC','AP','F1','MCC','Brier','ECE','Sens@90Spec','Spec@95Sens','n']].round(3).to_string())

# val_func_mave: binary func_label + continuous mean_score_z
df_mv = bench(val_mv, 'func_label', 'val_func_mave binary')
print(df_mv[['AUC','AP','F1','MCC','Brier','ECE','Sens@90Spec','Spec@95Sens','n']].round(3).to_string())

# Continuous MAVE z
print('\n=== val_func_mave continuous (mean_score_z) ===')
rows = []
methods_mv = get_methods(val_mv, 'func_label')
for name, col in methods_mv.items():
    if col not in val_mv.columns: continue
    y = val_mv.mean_score_z.values
    s = val_mv[col].values
    m = ~np.isnan(y) & ~np.isnan(s)
    if m.sum() < 50: continue
    rho = spearmanr(s[m], y[m]).statistic
    r = pearsonr(s[m], y[m]).statistic
    flipped = False
    if rho > 0: rho = -rho; r = -r; flipped = True
    rows.append({'Method':name,'spearman':rho,'pearson':r,'R2':r*r,'n':int(m.sum()),'flip':int(flipped)})
df_cont = pd.DataFrame(rows).set_index('Method').sort_values('spearman')
print(df_cont.round(3).to_string())

# Per-gene MAVE Spearman
print('\n=== val_func_mave: per-gene median Spearman ===')
rows = []
for name, col in methods_mv.items():
    if col not in val_mv.columns: continue
    per_g = []
    for g in val_mv.gene.unique():
        sub = val_mv[(val_mv.gene==g) & val_mv.mean_score_z.notna() & val_mv[col].notna()]
        if len(sub) > 20:
            rho = spearmanr(sub[col], sub.mean_score_z).statistic
            per_g.append(rho)
    if not per_g: continue
    arr = np.array(per_g)
    if np.median(arr) > 0: arr = -arr
    rows.append({'Method':name,'n_genes':len(per_g),
                 'median_rho':np.median(arr),'mean_rho':np.mean(arr),
                 'best':arr.min(),'worst':arr.max()})
df_pg = pd.DataFrame(rows).set_index('Method').sort_values('median_rho')
print(df_pg.round(3).to_string())

# Within-gene-centered
print('\n=== val_func_mave: within-gene-centered Spearman (per-variant signal) ===')
rows = []
for name, col in methods_mv.items():
    if col not in val_mv.columns: continue
    tmp = val_mv[['gene','mean_score_z',col]].copy()
    tmp[f'{col}_c'] = tmp[col] - tmp.groupby('gene')[col].transform('mean')
    tmp['z_c'] = tmp.mean_score_z - tmp.groupby('gene').mean_score_z.transform('mean')
    m = tmp[f'{col}_c'].notna() & tmp['z_c'].notna()
    if m.sum() < 100: continue
    rho = spearmanr(tmp.loc[m, f'{col}_c'], tmp.loc[m, 'z_c']).statistic
    if rho > 0: rho = -rho
    rows.append({'Method':name,'within_gene_rho':rho,'n':int(m.sum())})
print(pd.DataFrame(rows).set_index('Method').sort_values('within_gene_rho').round(3).to_string())

# Save
OUT = EVAL / 'comprehensive_benchmark_v2'
OUT.mkdir(exist_ok=True)
df_db.to_csv(OUT/'val_db_binary.csv')
df_vt.to_csv(OUT/'val_time_binary.csv')
df_vg.to_csv(OUT/'val_gene_binary.csv')
df_mv.to_csv(OUT/'val_func_mave_binary.csv')
df_cont.to_csv(OUT/'val_func_mave_continuous.csv')
df_pg.to_csv(OUT/'val_func_mave_per_gene.csv')
print(f'\n✅ Saved → {OUT}')
