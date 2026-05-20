#!/usr/bin/env python3
"""
Rebuild train + val splits per the v7 design:

Train (v7):
  ClinVar pre-2024 missense (current train.csv)
  - 50 (修 MAVE 泄漏)
  - V_gene 10 个 hold-out 基因的所有变异
  + V3_dvd_full B 子集 (排除 V_func MAVE 13 基因防泄漏)

Val (v7), 4 个干净维度:
  V_time:   V2_temporal  (不变)
  V_gene:   10 hold-out 基因 (取代 V1_otof)
  V_db:     V3_dvd_strong (不变)
  V_func:   MAVE (扣 50 个 train 泄漏)

V3_dvd_full 命运: B 全吃进 train, P 4,580 留作 supp val
"""
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd

ROOT = Path('/Users/pipi/Projects/QAFI_Paper/hearing')
SPLITS = ROOT / 'data/splits'
EXT = ROOT / 'data/external'
OUT = ROOT / 'feasibility/v7_train_expansion/splits_v7'
OUT.mkdir(parents=True, exist_ok=True)

def gp_set(df):
    return set(zip(df.gene.fillna(''), df.protein_change.fillna('')))

# ============== Load ==============
train = pd.read_csv(SPLITS / 'train.csv')
v1   = pd.read_csv(SPLITS / 'val_holdgene_otof.csv')
v2   = pd.read_csv(SPLITS / 'val_temporal_extended.csv')
v3s  = pd.read_csv(SPLITS / 'val_dvd_strong.csv')
v3f  = pd.read_csv(SPLITS / 'val_dvd_full.csv')
mave = pd.read_csv(EXT / 'mavedb_functional_cohort.csv').rename(columns={'protein_change_3letter':'protein_change'})
mave_genes_13 = set(mave.gene.unique())
print(f'MAVE covers {len(mave_genes_13)} genes: {sorted(mave_genes_13)}')

# ============== Pick 10 hold-out genes for V_gene ==============
# 准则: 多样化(不同 family / mechanism), 中等-中低频(每基因 30-200 变异),
# 不在 MAVE 13(避免跟 V_func 冲突), 不全是 dominant 模式
# 数据驱动: 看 train 里每个基因变异数
gene_counts = train.gene.value_counts()
print(f'\nTrain has variants in {len(gene_counts)} genes')

# 候选 hold-out:
HOLDOUT_GENES = [
    'OTOF',      # synapse,AR,profound
    'CLDN14',    # tight junction,AR,DFNB29
    'STRC',      # stereocilia coat,AR,DFNB16
    'TECTA',     # tectorial membrane,AR/AD
    'WFS1',      # Wolfram,AD,LFSNHL
    'TMC1',      # MET channel,AR/AD
    'CDH23',     # cadherin/Usher,AR
    'LOXHD1',    # large recessive,AR DFNB77
    'TBC1D24',   # multi-syndrome,AR/AD
    'ACTG1',     # cytoskeleton,AD
]
# 检查这 10 基因没有跟 MAVE 13 重叠
overlap_mave = set(HOLDOUT_GENES) & mave_genes_13
assert len(overlap_mave) == 0, f'Holdout in MAVE: {overlap_mave}'
print(f'\nHold-out 10 genes (none in MAVE 13):')
for g in HOLDOUT_GENES:
    n = gene_counts.get(g, 0)
    print(f'  {g:10} {n:>4} variants in current train')

# ============== Build V_gene split ==============
# Take ALL ClinVar P/B variants for these 10 genes, including current train + V1 + V2
print(f'\n========== Building V_gene ==========')
all_clinvar = pd.read_csv(ROOT / 'data/clinvar_hhl_all.csv')
all_clinvar['year'] = pd.to_numeric(all_clinvar['year'], errors='coerce')

def is_pb(s):
    s = str(s).lower()
    if 'pathogenic' in s and 'uncertain' not in s and 'conflict' not in s:
        return 'P'
    if 'benign' in s and 'uncertain' not in s and 'conflict' not in s and 'pathogenic' not in s:
        return 'B'
    return None

all_clinvar['_label'] = all_clinvar['ClinicalSignificance'].apply(is_pb)
all_clinvar['y'] = all_clinvar['_label'].map({'P': 1.0, 'B': 0.0})
v_gene_pool = all_clinvar[
    all_clinvar.gene.isin(HOLDOUT_GENES)
    & all_clinvar._label.notna()
    & all_clinvar.protein_change.notna()
    & (all_clinvar.protein_change.astype(str).str.len() > 4)
].copy()
print(f'  Raw pool: {len(v_gene_pool)} variants')

# Dedup
v_gene_pool = v_gene_pool.drop_duplicates(subset=['gene','protein_change'], keep='first')
print(f'  Dedup:    {len(v_gene_pool)}')

# Schema match train.csv
v_gene = v_gene_pool.rename(columns={
    'ClinicalSignificance':'clinical_significance',
    'ReviewStatus':'review_status',
    'Chromosome':'chromosome',
})[['gene','protein_change','aa_pos','y','clinical_significance','review_status',
    'VariationID','variation_id','year','chromosome']].copy()
print(f'  V_gene final: {len(v_gene)}  (P={int((v_gene.y==1).sum())}, B={int((v_gene.y==0).sum())})')

# ============== Build train_v7 ==============
print(f'\n========== Building train_v7 ==========')

# Step 1: 从 current train 删 hold-out 基因
train_clean = train[~train.gene.isin(HOLDOUT_GENES)].copy()
print(f'  After removing hold-out genes: {len(train_clean)} (was {len(train)})')

# Step 2: 删 MAVE 50 个泄漏
mave_keys = gp_set(mave)
train_clean['_key'] = list(zip(train_clean.gene, train_clean.protein_change))
train_clean = train_clean[~train_clean._key.isin(mave_keys)].copy()
print(f'  After removing MAVE leak:      {len(train_clean)}')

# Step 3: 吃 V3_dvd_full B 子集进 train
v3f_b = v3f[v3f.y == 0].copy()
print(f'  V3_dvd_full B pool:            {len(v3f_b)}')

# 排除已在 train 的
v3f_b['_key'] = list(zip(v3f_b.gene, v3f_b.protein_change))
v3f_b_new = v3f_b[~v3f_b._key.isin(set(train_clean._key))].copy()
print(f'    not already in train_clean:  {len(v3f_b_new)}')

# 排除在 hold-out genes (那些归 V_gene)
v3f_b_new = v3f_b_new[~v3f_b_new.gene.isin(HOLDOUT_GENES)].copy()
print(f'    not in hold-out genes:       {len(v3f_b_new)}')

# 排除在 MAVE 13 genes (防 V_func 泄漏)
v3f_b_new = v3f_b_new[~v3f_b_new.gene.isin(mave_genes_13)].copy()
print(f'    not in MAVE 13 genes:        {len(v3f_b_new)}')

# 排除在 V2_temporal (防 V_time 泄漏)
v2_keys = gp_set(v2)
v3f_b_new = v3f_b_new[~v3f_b_new._key.isin(v2_keys)].copy()
print(f'    not in V2_temporal:          {len(v3f_b_new)}')

# 排除在 V3_dvd_strong (防 V_db 泄漏)
v3s_keys = gp_set(v3s)
v3f_b_new = v3f_b_new[~v3f_b_new._key.isin(v3s_keys)].copy()
print(f'    not in V3_dvd_strong:        {len(v3f_b_new)}')

# Format to train schema
v3f_b_add = v3f_b_new.rename(columns={'pathogenicity':'clinical_significance'}).copy()
v3f_b_add['review_status'] = 'DVD curated (n_pmid=' + v3f_b_add['n_pmid'].astype(str) + ')'
v3f_b_add['VariationID'] = v3f_b_add['dvd_id']
v3f_b_add['variation_id'] = v3f_b_add['dvd_id']
v3f_b_add['year'] = 2023.0
v3f_b_add['chromosome'] = ''
v3f_b_add = v3f_b_add[['gene','protein_change','aa_pos','y','clinical_significance','review_status',
                       'VariationID','variation_id','year','chromosome']].copy()

# Combine
train_clean = train_clean.drop(columns=['_key'])
train_v7 = pd.concat([train_clean, v3f_b_add], ignore_index=True)
train_v7 = train_v7.drop_duplicates(subset=['gene','protein_change'], keep='first')
print(f'\n  train_v7 final: {len(train_v7)}  (P={int((train_v7.y==1).sum())}, B={int((train_v7.y==0).sum())})')

# ============== V_func MAVE clean ==============
# Drop the 50 train leakage (using ORIGINAL train.csv keys)
orig_train_keys = gp_set(train)
mave_keys_before = list(zip(mave.gene, mave.protein_change))
keep_mask = [k not in orig_train_keys for k in mave_keys_before]
mave_clean = mave[keep_mask].copy()
print(f'\n  V_func MAVE: {len(mave)} → {len(mave_clean)} after removing 50 original-train leakage')

# ============== Save splits ==============
print(f'\n========== Save ==========')
train_v7.to_csv(OUT / 'train_v7.csv', index=False)
v_gene.to_csv(OUT / 'val_gene_v7.csv', index=False)
mave_clean.to_csv(OUT / 'val_func_mave_v7.csv', index=False)
v2.to_csv(OUT / 'val_time_v7.csv', index=False)
v3s.to_csv(OUT / 'val_db_v7.csv', index=False)
# Supp
v3f_p_supp = v3f[v3f.y == 1].copy()
v3f_p_supp.to_csv(OUT / 'val_dvd_full_P_supp.csv', index=False)

print(f'  train_v7.csv               {len(train_v7)}')
print(f'  val_gene_v7.csv            {len(v_gene)}  (was V1_otof 205)')
print(f'  val_time_v7.csv            {len(v2)}')
print(f'  val_db_v7.csv              {len(v3s)}')
print(f'  val_func_mave_v7.csv       {len(mave_clean)}')
print(f'  val_dvd_full_P_supp.csv    {len(v3f_p_supp)} (supp only)')

# ============== Final leakage audit ==============
print(f'\n========== FINAL LEAKAGE AUDIT ==========')
train_keys = gp_set(train_v7)
all_clean = True
for name, df in [('val_gene_v7',v_gene),('val_time_v7',v2),('val_db_v7',v3s),
                  ('val_func_mave_v7',mave_clean)]:
    overlap = train_keys & gp_set(df)
    status = '✅' if len(overlap) == 0 else f'❌ {len(overlap)} OVERLAP'
    if len(overlap): all_clean = False
    print(f'  train ∩ {name:25} = {len(overlap):>4}  {status}')

print()
print('  pairwise val ∩ val (no overlap allowed):')
val_dfs = {'val_gene_v7':v_gene, 'val_time_v7':v2, 'val_db_v7':v3s, 'val_func_mave_v7':mave_clean}
val_names = list(val_dfs.keys())
for i in range(len(val_names)):
    for j in range(i+1, len(val_names)):
        a, b = val_names[i], val_names[j]
        ov = gp_set(val_dfs[a]) & gp_set(val_dfs[b])
        status = '✅' if len(ov)==0 else f'⚠️ {len(ov)}'
        print(f'    {a} ∩ {b} = {len(ov):>3}  {status}')

print()
if all_clean:
    print('🎉 TRAIN ↔ ALL VAL CLEAN')
print(f'\nTrain class balance: P/B = {(train_v7.y==1).sum()}/{(train_v7.y==0).sum()} = {(train_v7.y==1).sum()/(train_v7.y==0).sum():.2f}:1')
