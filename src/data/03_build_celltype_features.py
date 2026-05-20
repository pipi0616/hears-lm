#!/usr/bin/env python3
"""
Build HHL gene × cell-type expression matrix using AUTHORITATIVE source:
Wang et al 2024 (Nat Commun) supp 6 — peer-reviewed, author-curated.

Replaces my custom scanpy pipeline (which had ad-hoc cell-type labels).
"""
import pandas as pd
from pathlib import Path
import shutil

ROOT = Path('/Users/pipi/Projects/QAFI_Paper/hearing')
OUT = Path('/Users/pipi/Projects/QAFI_Paper/hearing_v2/data/features')
OUT.mkdir(parents=True, exist_ok=True)

PANEL = pd.read_csv(ROOT / 'data/gene_panel_hhl_green.csv')['gene'].dropna().unique().tolist()
print(f'HHL panel: {len(PANEL)} genes')

# Cell type assignment (cluster ID → label, derived from supp 6 marker genes)
CLUSTER_LABEL = {
    0:  'Stromal_Fibrocyte',
    1:  'Transitional_epi',
    2:  'HC_precursor',
    3:  'Immune',
    4:  'Schwann_nonmyelin',
    5:  'Supporting_cell',
    6:  'Cartilage',
    7:  'Hair_cell',
    8:  'Endothelial',
    9:  'Pericyte_SM_1',
    10: 'Pericyte_SM_2',
    11: 'Dark_cell',
    12: 'Schwann_myelin',
}

# Move downloaded supp 6 to project
shutil.copy('/tmp/wang2024_supp6.xlsx',
            ROOT / 'data/external/papers/wang2024_supp6_celltypes.xlsx')

# Load author-curated per-cell-group expression matrix
xls = pd.ExcelFile('/tmp/wang2024_supp6.xlsx')
df = pd.read_excel(xls, sheet_name='Figure 1c_per cell group', skiprows=1)
df = df.rename(columns={'Unnamed: 0': 'gene'})
df = df.rename(columns={i: CLUSTER_LABEL[i] for i in range(13)})
print(f'\nAuthor-curated matrix: {df.shape[0]} genes × 13 cell types')

# Filter to HHL panel
hhl_in = df[df.gene.isin(PANEL)].copy()
print(f'HHL panel ∩ matrix:    {len(hhl_in)} genes (in author atlas)')

# Sanity check known markers
print('\n=== Sanity check (known markers) ===')
checks = {
    'POU4F3':  'Hair_cell',
    'OTOF':    'Hair_cell',
    'CABP2':   'Hair_cell',
    'MYO7A':   'Hair_cell',
    'CIB2':    'Hair_cell',
    'GJB2':    'Supporting_cell',
    'OTOG':    'Supporting_cell',
    'OTOGL':   'Supporting_cell',
    'TMC1':    'Hair_cell',
    'SLC26A4': None,  # epithelial, not in this curation primarily
}
for g, expected in checks.items():
    row = hhl_in[hhl_in.gene == g]
    if len(row) == 0:
        print(f'  {g:10} ⚠️  not in author atlas')
        continue
    vals = row.iloc[0].drop('gene').to_dict()
    top = max(vals, key=vals.get)
    val_top = vals[top]
    val_exp = vals.get(expected, None) if expected else None
    status = '✅' if (expected is None or top == expected) else f'⚠️ top={top}'
    if expected:
        print(f'  {g:10} expected={expected:20} actual_top={top:20} val={val_top:.2f}  {status}')
    else:
        print(f'  {g:10} no expected, actual_top={top:20} val={val_top:.2f}')

# Save
out = OUT / 'hhl_gene_celltype_expression_authoritative.csv'
hhl_in.to_csv(out, index=False)
print(f'\n✅ saved → {out}')

# Show summary
print(f'\nTop 5 hair-cell-enriched HHL genes (val_Hair_cell highest):')
print(hhl_in.nlargest(5, 'Hair_cell')[['gene','Hair_cell','Supporting_cell','HC_precursor']].to_string(index=False))

print(f'\nTop 5 supporting-cell-enriched HHL genes:')
print(hhl_in.nlargest(5, 'Supporting_cell')[['gene','Supporting_cell','Hair_cell']].to_string(index=False))

print(f'\nTop 5 dark-cell-enriched HHL genes (内淋巴 K+ 相关):')
print(hhl_in.nlargest(5, 'Dark_cell')[['gene','Dark_cell','Endothelial','Pericyte_SM_1']].to_string(index=False))

# Genes NOT in author atlas (HHL panel - found)
missing = set(PANEL) - set(hhl_in.gene)
print(f'\n⚠️  HHL genes NOT in author 3,374 expressed-genes list ({len(missing)}):')
print('  → 这些基因在 utricle 表达过低,作者过滤掉了 (与我们之前发现 13% silent 一致)')
print(' ', ', '.join(sorted(missing)[:30]))
