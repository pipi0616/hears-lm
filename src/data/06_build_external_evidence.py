#!/usr/bin/env python3
"""
Final v5 — collect 5 standard, working API sources + build unified summary.

Sources (all standard, citable, no manual curation):
  Complex / PPI / Pathway:
    1. EBI Complex Portal       (already have) — 131 genes
    2. STRING-DB ≥700           (already have) — 110 genes
    3. Reactome (fixed)         — runs here
    4. UniProt SUBUNIT (text)   (already have) — 126 genes
  Cell type expression:
    5. Wang 2024 supp 6         (already have) — 66 genes utricle 13 cell types

Output: hhl_external_evidence_unified.csv — per-gene cross-source summary
"""
import warnings; warnings.filterwarnings('ignore')
import json, time
from pathlib import Path
import requests
import pandas as pd

ROOT = Path('/Users/pipi/Projects/QAFI_Paper/hearing')
PANEL = pd.read_csv(ROOT / 'data/gene_panel_hhl_green.csv')['gene'].dropna().unique().tolist()
OUT = Path('/Users/pipi/Projects/QAFI_Paper/hearing_v2/data/features')
CACHE = OUT / '_api_cache_v5'
CACHE.mkdir(parents=True, exist_ok=True)
HEADERS = {'User-Agent': 'hearing-research/0.1', 'Accept': 'application/json'}


def query_reactome(gene):
    cache_f = CACHE / f'rx_{gene}.json'
    if cache_f.exists():
        return json.load(open(cache_f))
    url = f'https://reactome.org/ContentService/search/query'
    params = {'query': gene, 'species': 'Homo sapiens', 'types': 'Pathway'}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            json.dump([], open(cache_f, 'w'))
            return []
        data = r.json()
        results = []
        for grp in data.get('results', []):
            for item in grp.get('entries', [])[:20]:
                results.append({
                    'pathway_id': item.get('stId', '') or str(item.get('dbId', '')),
                    'pathway_name': item.get('name', ''),
                })
        json.dump(results, open(cache_f, 'w'))
        return results
    except Exception:
        return []


# ============== STEP 1: Reactome (only new query needed) ==============
print('[1/2] Querying Reactome for 165 HHL genes...')
rx_records = []
for i, g in enumerate(PANEL):
    if (i+1) % 25 == 0: print(f'  {i+1}/{len(PANEL)}…', flush=True)
    rx = query_reactome(g)
    for p in rx:
        rx_records.append({'gene': g, 'source': 'Reactome', **p})
    time.sleep(0.12)
rx_df = pd.DataFrame(rx_records)
n_rx_genes = rx_df.gene.nunique() if len(rx_df) else 0
print(f'  → {len(rx_df)} pathway entries across {n_rx_genes} genes')
rx_df.to_csv(OUT / '_reactome_v5.csv', index=False)


# ============== STEP 2: Build unified summary ==============
print('\n[2/2] Building unified per-gene summary...')

# Load all 5 sources
cp = pd.read_csv(OUT / '_cp_complex_portal.csv') if (OUT / '_cp_complex_portal.csv').exists() else pd.DataFrame()
string_df = pd.read_csv(OUT / '_string_ppi_high.csv') if (OUT / '_string_ppi_high.csv').exists() else pd.DataFrame()
wang = pd.read_csv(OUT / 'hhl_gene_celltype_expression_authoritative.csv') if (OUT / 'hhl_gene_celltype_expression_authoritative.csv').exists() else pd.DataFrame()

# Load UniProt subunit (from earlier run — long format)
unip = pd.read_csv(OUT / 'hhl_complex_membership.csv') if (OUT / 'hhl_complex_membership.csv').exists() else pd.DataFrame()

print(f'  Loaded:')
print(f'    Complex Portal:          {cp.gene.nunique() if len(cp) else 0} genes, {len(cp)} memberships')
print(f'    STRING-DB ≥700:          {string_df.gene.nunique() if len(string_df) else 0} genes, {len(string_df)} edges')
print(f'    Reactome (new):          {n_rx_genes} genes, {len(rx_df)} pathway entries')
print(f'    UniProt SUBUNIT:         {(unip.subunit_text.fillna("").str.len().gt(10)).sum() if "subunit_text" in unip.columns else 0} genes')
print(f'    Wang 2024 cell-type:     {len(wang)} genes')

# Per-gene unified summary
summary_rows = []
for g in PANEL:
    n_cp_complexes = (cp.gene == g).sum() if len(cp) else 0
    n_string_partners = (string_df.gene == g).sum() if len(string_df) else 0
    n_reactome_pathways = (rx_df.gene == g).sum() if len(rx_df) else 0
    has_uniprot_subunit = 0
    if 'subunit_text' in unip.columns:
        row = unip[unip.gene == g]
        if len(row):
            txt = row.iloc[0].subunit_text
            has_uniprot_subunit = int(isinstance(txt, str) and len(txt) > 10)
    in_wang_atlas = int(g in set(wang.gene)) if len(wang) else 0
    n_sources = (n_cp_complexes > 0) + (n_string_partners > 0) + (n_reactome_pathways > 0) + has_uniprot_subunit + in_wang_atlas
    summary_rows.append({
        'gene': g,
        'n_complex_portal': n_cp_complexes,
        'n_string_partners': n_string_partners,
        'n_reactome_pathways': n_reactome_pathways,
        'has_uniprot_subunit': has_uniprot_subunit,
        'in_wang_atlas': in_wang_atlas,
        'n_sources_supported': n_sources,
    })
sum_df = pd.DataFrame(summary_rows)
sum_df.to_csv(OUT / 'hhl_external_evidence_unified.csv', index=False)
print(f'\n✅ Unified summary saved → hhl_external_evidence_unified.csv')

# Coverage report
print('\n========== COVERAGE REPORT ==========')
print(f'Total HHL panel genes:                 {len(PANEL)}')
print(f'\nPer-source coverage:')
print(f'  Complex Portal      : {(sum_df.n_complex_portal > 0).sum():>3}/{len(PANEL)} ({100*(sum_df.n_complex_portal>0).sum()/len(PANEL):.0f}%)')
print(f'  STRING ≥700         : {(sum_df.n_string_partners > 0).sum():>3}/{len(PANEL)} ({100*(sum_df.n_string_partners>0).sum()/len(PANEL):.0f}%)')
print(f'  Reactome            : {(sum_df.n_reactome_pathways > 0).sum():>3}/{len(PANEL)} ({100*(sum_df.n_reactome_pathways>0).sum()/len(PANEL):.0f}%)')
print(f'  UniProt SUBUNIT     : {(sum_df.has_uniprot_subunit > 0).sum():>3}/{len(PANEL)} ({100*(sum_df.has_uniprot_subunit>0).sum()/len(PANEL):.0f}%)')
print(f'  Wang 2024 atlas     : {(sum_df.in_wang_atlas > 0).sum():>3}/{len(PANEL)} ({100*(sum_df.in_wang_atlas>0).sum()/len(PANEL):.0f}%)')

print(f'\nGenes by # sources covered:')
for n in range(6):
    c = (sum_df.n_sources_supported == n).sum()
    print(f'  {n} sources: {c:>3} genes ({100*c/len(PANEL):.0f}%)')

print(f'\nUnion (≥1 source):  {(sum_df.n_sources_supported > 0).sum()}/{len(PANEL)} ({100*(sum_df.n_sources_supported>0).sum()/len(PANEL):.0f}%)')
print(f'Strong (≥3 sources): {(sum_df.n_sources_supported >= 3).sum()}/{len(PANEL)} ({100*(sum_df.n_sources_supported>=3).sum()/len(PANEL):.0f}%)')
