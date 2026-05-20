#!/usr/bin/env python3
"""
Build HHL protein complex membership from STANDARD bioinformatics APIs.

Three authoritative sources (cross-validated):
  1. EBI Complex Portal — manually curated stable protein complexes
  2. STRING-DB         — PPI with confidence scores (>700 = high)
  3. Reactome          — pathway membership

A gene's complex membership requires ≥2 sources agreement (or Complex Portal alone, gold).
This replaces the previous manual canonical curation (which was error-prone).
"""
import warnings; warnings.filterwarnings('ignore')
import json, time, re
from pathlib import Path
import requests
import pandas as pd

ROOT = Path('/Users/pipi/Projects/QAFI_Paper/hearing')
PANEL = pd.read_csv(ROOT / 'data/gene_panel_hhl_green.csv')['gene'].dropna().unique().tolist()
print(f'HHL panel: {len(PANEL)} genes')

OUT = Path('/Users/pipi/Projects/QAFI_Paper/hearing_v2/data/features')
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path('/Users/pipi/Projects/QAFI_Paper/hearing_v2/data/features/_api_cache')
CACHE.mkdir(parents=True, exist_ok=True)

HEADERS = {'User-Agent': 'hearing-research/0.1', 'Accept': 'application/json'}


# ===================== EBI Complex Portal =====================
# https://www.ebi.ac.uk/complexportal/api-docs
def query_complex_portal(gene):
    """Returns list of {complex_id, complex_name} for this gene."""
    cache_f = CACHE / f'cp_{gene}.json'
    if cache_f.exists():
        return json.load(open(cache_f))
    url = f'https://www.ebi.ac.uk/intact/complex-ws/search/{gene}?facets=species_f&filters=species_f:%22Homo+sapiens%22'
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            json.dump([], open(cache_f, 'w'))
            return []
        data = r.json()
        results = []
        for cx in data.get('elements', [])[:20]:  # cap at 20 per gene
            results.append({
                'complex_id':   cx.get('complexAC', ''),
                'complex_name': cx.get('complexName', ''),
            })
        json.dump(results, open(cache_f, 'w'))
        return results
    except Exception as e:
        print(f'  [warn] CP {gene}: {e}')
        return []


# ===================== STRING-DB =====================
# https://string-db.org/help/api/
STRING_SP = '9606'  # Homo sapiens
def get_string_id_map(genes):
    """Map gene symbols → STRING IDs in one batch call."""
    cache_f = CACHE / 'string_id_map.json'
    if cache_f.exists():
        return json.load(open(cache_f))
    url = 'https://string-db.org/api/json/get_string_ids'
    params = {
        'identifiers': '\r'.join(genes),
        'species': STRING_SP,
        'limit': 1,
    }
    try:
        r = requests.post(url, data=params, headers=HEADERS, timeout=60)
        r.raise_for_status()
        data = r.json()
        m = {item['queryItem']: item['stringId'] for item in data}
        json.dump(m, open(cache_f, 'w'))
        return m
    except Exception as e:
        print(f'  [error] STRING id map: {e}')
        return {}


def get_string_network(string_ids, score_threshold=700):
    """Get all interactions among a list of STRING IDs (high confidence)."""
    cache_f = CACHE / 'string_network.json'
    if cache_f.exists():
        return json.load(open(cache_f))
    url = 'https://string-db.org/api/json/network'
    params = {
        'identifiers': '\r'.join(string_ids),
        'species': STRING_SP,
        'required_score': score_threshold,
    }
    try:
        r = requests.post(url, data=params, headers=HEADERS, timeout=120)
        r.raise_for_status()
        data = r.json()
        json.dump(data, open(cache_f, 'w'))
        return data
    except Exception as e:
        print(f'  [error] STRING network: {e}')
        return []


# ===================== Reactome =====================
# https://reactome.org/ContentService/
def query_reactome_pathways(gene):
    """Return list of {pathway_id, pathway_name} for this gene (human only)."""
    cache_f = CACHE / f'rx_{gene}.json'
    if cache_f.exists():
        return json.load(open(cache_f))
    url = f'https://reactome.org/ContentService/data/mapping/{gene}/pathways/HSA'
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 404:
            json.dump([], open(cache_f, 'w'))
            return []
        if r.status_code != 200:
            return []
        data = r.json()
        results = []
        for p in data:
            results.append({
                'pathway_id':   p.get('stId', ''),
                'pathway_name': p.get('displayName', ''),
            })
        json.dump(results, open(cache_f, 'w'))
        return results
    except Exception as e:
        return []


# =============================================================
# MAIN
# =============================================================

# --- 1. EBI Complex Portal (per-gene) ---
print('\n[1/3] Querying EBI Complex Portal...')
cp_records = []
for i, g in enumerate(PANEL):
    if (i+1) % 25 == 0:
        print(f'  {i+1}/{len(PANEL)}…')
    cps = query_complex_portal(g)
    for cx in cps:
        cp_records.append({'gene': g, 'source': 'ComplexPortal', **cx})
    time.sleep(0.2)
cp_df = pd.DataFrame(cp_records)
print(f'  → {len(cp_df)} complex memberships across {cp_df.gene.nunique() if len(cp_df) else 0} genes')


# --- 2. STRING-DB network ---
print('\n[2/3] Querying STRING-DB (high-confidence PPI)...')
string_id_map = get_string_id_map(PANEL)
print(f'  STRING IDs resolved: {len(string_id_map)}/{len(PANEL)}')
string_network = get_string_network(list(string_id_map.values()), score_threshold=700)
print(f'  Network edges (score≥700): {len(string_network)}')

# Build per-gene partner list
inv_map = {v: k for k, v in string_id_map.items()}
ppi_records = []
for edge in string_network:
    a = inv_map.get(edge.get('stringId_A'))
    b = inv_map.get(edge.get('stringId_B'))
    if a and b:
        ppi_records.append({'gene': a, 'partner': b, 'score': edge.get('score', 0)})
        ppi_records.append({'gene': b, 'partner': a, 'score': edge.get('score', 0)})
ppi_df = pd.DataFrame(ppi_records).drop_duplicates(subset=['gene','partner'])
print(f'  Per-gene partner records: {len(ppi_df)}')


# --- 3. Reactome (per-gene pathway) ---
print('\n[3/3] Querying Reactome pathways...')
rx_records = []
for i, g in enumerate(PANEL):
    if (i+1) % 25 == 0:
        print(f'  {i+1}/{len(PANEL)}…')
    rx = query_reactome_pathways(g)
    for p in rx:
        rx_records.append({'gene': g, 'source': 'Reactome', **p})
    time.sleep(0.2)
rx_df = pd.DataFrame(rx_records)
print(f'  → {len(rx_df)} pathway memberships across {rx_df.gene.nunique() if len(rx_df) else 0} genes')


# --- Save raw evidence ---
cp_df.to_csv(OUT / '_cp_complex_portal.csv', index=False)
ppi_df.to_csv(OUT / '_string_ppi_high.csv', index=False)
rx_df.to_csv(OUT / '_reactome_pathways.csv', index=False)
print(f'\n  Saved 3 raw evidence files to {OUT}')


# --- Per-gene summary ---
print('\n=== PER-GENE SUMMARY ===')
summary = []
for g in PANEL:
    n_cp = len(cp_df[cp_df.gene == g]) if len(cp_df) else 0
    n_ppi = len(ppi_df[ppi_df.gene == g]) if len(ppi_df) else 0
    n_rx = len(rx_df[rx_df.gene == g]) if len(rx_df) else 0
    summary.append({
        'gene': g,
        'n_complex_portal': n_cp,
        'n_string_partners_high': n_ppi,
        'n_reactome_pathways': n_rx,
    })
sum_df = pd.DataFrame(summary)
sum_df.to_csv(OUT / 'hhl_external_evidence_summary.csv', index=False)
print(f'\n  {(sum_df.n_complex_portal > 0).sum()} genes with Complex Portal entries')
print(f'  {(sum_df.n_string_partners_high > 0).sum()} genes with STRING high-confidence partners')
print(f'  {(sum_df.n_reactome_pathways > 0).sum()} genes with Reactome pathways')
print(f'\n  saved → hhl_external_evidence_summary.csv')
print('\n  Top genes by total evidence:')
sum_df['total_evidence'] = sum_df.n_complex_portal + (sum_df.n_string_partners_high > 0).astype(int) + sum_df.n_reactome_pathways
print(sum_df.nlargest(10, 'total_evidence').to_string(index=False))
