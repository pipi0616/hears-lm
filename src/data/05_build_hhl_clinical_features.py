#!/usr/bin/env python3
"""
Build HHL-specific per-gene clinical features (PLM cannot learn from sequence).

Sources:
  - HPO (Human Phenotype Ontology) — phenotype categories per gene
  - CGD (Clinical Genomic Database) — inheritance + age + intervention
  - gnomAD constraint — LoF intolerance (LOEUF)

Output: hhl_clinical_features.csv (per gene)
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path

ROOT = Path('/Users/pipi/Projects/QAFI_Paper/hearing')
OUT = Path('/Users/pipi/Projects/QAFI_Paper/hearing_v2/data/features')
PANEL = pd.read_csv(ROOT/'data/gene_panel_hhl_green.csv').gene.dropna().unique()

# ==================== A. HPO features ====================
print('[1/3] Build HPO clinical features...')
hpo = pd.read_csv(ROOT/'data/external/phenotype/hpo_genes_to_phenotype.txt', sep='\t')
hpo_hhl = hpo[hpo.gene_symbol.isin(PANEL)]

# Define HHL-specific phenotype categories
HPO_CATEGORIES = {
    'hpo_AR':                'autosomal recessive',
    'hpo_AD':                'autosomal dominant',
    'hpo_X_linked':          'x-linked',
    'hpo_mitochondrial':     'mitochondrial inheritance',
    'hpo_SNHL':              'sensorineural hearing|sensorineural',
    'hpo_progressive_HL':    'progressive sensorineural|progressive hearing|progressive.*hear',
    'hpo_congenital_onset':  'congenital onset|prelingual',
    'hpo_childhood_onset':   'childhood onset|infantile onset|juvenile onset',
    'hpo_adult_onset':       'adult onset',
    'hpo_conductive_HL':     'conductive hearing',
    'hpo_mixed_HL':          'mixed.*hearing',
    'hpo_vestibular':        'vestibular|balance|vertigo|ataxia',
    'hpo_visual_USH':        'retinal|retinitis|blind|visual loss|visual impairment|optic atrophy',
    'hpo_renal':             'renal|kidney|nephro',
    'hpo_thyroid_pendred':   'thyroid|goiter|hypothyroidism',
    'hpo_enlarged_VA':       'enlarged vestibular',
    'hpo_cardiac':           'long qt|cardiac arrhythmia|cardiomyopathy',
}

feat = pd.DataFrame(index=PANEL); feat.index.name = 'gene'
for col, kw in HPO_CATEGORIES.items():
    matched_genes = set(hpo_hhl[hpo_hhl.hpo_name.str.contains(kw, case=False, na=False, regex=True)].gene_symbol)
    feat[col] = feat.index.isin(matched_genes).astype(int)

# Severity proxy: total HPO terms per gene
feat['hpo_n_total_phenotypes'] = feat.index.map(hpo_hhl.groupby('gene_symbol').size()).fillna(0).astype(int)
# HHL-specific phenotype count
hpo_hhl_only = hpo_hhl[hpo_hhl.hpo_name.str.contains('hearing|sensorineural|conductive|deaf', case=False, na=False, regex=True)]
feat['hpo_n_HL_phenotypes'] = feat.index.map(hpo_hhl_only.groupby('gene_symbol').size()).fillna(0).astype(int)
# Syndromic flag: has visual OR renal OR thyroid OR cardiac
feat['hpo_is_syndromic'] = ((feat.hpo_visual_USH | feat.hpo_renal | feat.hpo_thyroid_pendred | feat.hpo_cardiac) > 0).astype(int)

print(f'  HPO features built: {len([c for c in feat.columns if c.startswith("hpo_")])} cols')


# ==================== B. CGD features ====================
print('\n[2/3] Build CGD features...')
cgd = pd.read_csv(ROOT/'data/external/CGD.txt', sep='\t')
cgd = cgd.rename(columns={'#GENE':'gene'})
cgd_hhl = cgd[cgd.gene.isin(PANEL)]
print(f'  CGD covers {cgd_hhl.gene.nunique()}/{len(PANEL)} HHL genes')

# Inheritance from CGD
def inh_to_flags(inh_str):
    if pd.isna(inh_str): return {'cgd_AD': 0, 'cgd_AR': 0, 'cgd_XL': 0, 'cgd_MT': 0}
    return {
        'cgd_AD': int('AD' in inh_str),
        'cgd_AR': int('AR' in inh_str),
        'cgd_XL': int('XL' in inh_str),
        'cgd_MT': int('Mt' in inh_str or 'MT' in inh_str),
    }

cgd_per_gene = cgd_hhl.groupby('gene').agg({
    'INHERITANCE': lambda x: ';'.join(set(x.dropna())),
    'AGE GROUP':   lambda x: ';'.join(set(x.dropna())),
    'CONDITION':   lambda x: ';'.join(set(x.dropna())),
    'MANIFESTATION CATEGORIES': lambda x: ';'.join(set(x.dropna())),
})

for g in PANEL:
    if g not in cgd_per_gene.index:
        for c in ['cgd_AD','cgd_AR','cgd_XL','cgd_MT','cgd_pediatric','cgd_HHL_condition']:
            feat.loc[g, c] = 0
    else:
        row = cgd_per_gene.loc[g]
        flags = inh_to_flags(row['INHERITANCE'])
        for k, v in flags.items(): feat.loc[g, k] = v
        feat.loc[g, 'cgd_pediatric'] = int('Pediatric' in str(row['AGE GROUP']))
        feat.loc[g, 'cgd_HHL_condition'] = int(re.search(r'deaf|hearing', str(row['CONDITION']), re.I) is not None)
print(f'  CGD features added: cgd_AD, cgd_AR, cgd_XL, cgd_MT, cgd_pediatric, cgd_HHL_condition')


# ==================== C. gnomAD constraint (LOEUF / pLI) ====================
print('\n[3/3] Build gnomAD constraint features (LOEUF, pLI)...')
gnom = pd.read_csv(ROOT/'data/external/gnomad_v4_constraint.tsv', sep='\t', low_memory=False)
print(f'  gnomAD shape: {gnom.shape}, sample cols: {list(gnom.columns)[:10]}')
# Check what's available
for col in ['lof.oe_ci.upper','lof.pLI','mis.oe_ci.upper','syn.oe_ci.upper','lof_hc_lc.pLI','lof_hc_lc.oe_ci.upper']:
    if col in gnom.columns: print(f'    has {col}')
# Find right column
loeuf_col = None
for cand in ['lof.oe_ci.upper','lof_hc_lc.oe_ci.upper','oe_lof_upper','lof_upper']:
    if cand in gnom.columns: loeuf_col = cand; break
pli_col = None
for cand in ['lof.pLI','lof_hc_lc.pLI','pLI']:
    if cand in gnom.columns: pli_col = cand; break

print(f'  Using LOEUF={loeuf_col}, pLI={pli_col}')

# Take canonical transcript per gene (first one)
gnom_panel = gnom[gnom.gene.isin(PANEL)].drop_duplicates('gene', keep='first')
if loeuf_col:
    feat['loeuf'] = feat.index.map(gnom_panel.set_index('gene')[loeuf_col])
    feat['loeuf_constrained'] = (feat.loeuf < 0.35).astype(int)  # standard threshold
if pli_col:
    feat['pLI'] = feat.index.map(gnom_panel.set_index('gene')[pli_col])

# Final cleanup
for c in feat.columns:
    if feat[c].dtype == 'float64':
        feat[c] = feat[c].fillna(0)
    else:
        feat[c] = feat[c].fillna(0).astype(int)

# Save
feat = feat.reset_index()
feat.to_csv(OUT/'hhl_clinical_features.csv', index=False)
print(f'\n✅ Saved hhl_clinical_features.csv: {feat.shape}')

# Audit
print('\n=== Per-feature sanity ===')
for c in feat.columns[1:]:
    n_pos = (feat[c] > 0).sum()
    print(f'  {c:30}  positive/non-zero: {n_pos}/{len(feat)} ({100*n_pos/len(feat):.0f}%)')
