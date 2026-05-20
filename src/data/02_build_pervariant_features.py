#!/usr/bin/env python3
"""
Build per-variant features for HHL training/validation sets:
  1. pLDDT at variant position (from AlphaFold structures, local parse)
  2. UniProt domain at variant position (UniProt API)
  3. PhyloP / GERP conservation (myvariant.info API)

Output: per-variant feature CSV that joins onto train/val on (gene, aa_pos)
"""
import pandas as pd
import numpy as np
import json
import time
import requests
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

ROOT_HEARING = Path('/Users/pipi/Projects/QAFI_Paper/hearing')
ROOT_V2 = Path('/Users/pipi/Projects/QAFI_Paper/hearing_v2')
STRUCT_DIR = ROOT_HEARING / 'data/structures'
OUT = ROOT_V2 / 'data/features'
CACHE = OUT / '_pervariant_cache'
CACHE.mkdir(parents=True, exist_ok=True)

UNIPROT_MAP = pd.read_csv(ROOT_HEARING / 'data/gene_uniprot_mapping.csv').set_index('gene').uniprot_id.to_dict()
print(f'gene→UniProt map: {len(UNIPROT_MAP)} genes')


# =====================================================================
# Step 1: pLDDT per residue from AF structures
# =====================================================================
def parse_pdb_plddt(pdb_path):
    """Parse a PDB file (AlphaFold v2/v4 monomer) → dict {residue_idx (1-based): mean B-factor=pLDDT}.
    AF stores pLDDT in the B-factor column."""
    plddt_per_res = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('ATOM'): continue
            res_id = int(line[22:26].strip())
            atom_name = line[12:16].strip()
            if atom_name != 'CA': continue   # only Cα has pLDDT (same for all atoms in a residue)
            bf = float(line[60:66].strip())
            plddt_per_res[res_id] = bf
    return plddt_per_res


def build_plddt_lookup(force_rebuild=False):
    """For each HHL gene, parse ALL its AF PDBs → {(gene, aa_pos): max_pLDDT}.
    Multiple isoform PDBs per gene → take max pLDDT at each residue (best confidence)."""
    cache_f = CACHE / 'plddt_lookup.json'
    if cache_f.exists() and not force_rebuild:
        with open(cache_f) as f:
            data = json.load(f)
        return {tuple(json.loads(k)): v for k, v in data.items()}
    lookup = {}
    n_genes_done = 0
    for gene_dir in STRUCT_DIR.iterdir():
        if not gene_dir.is_dir(): continue
        gene = gene_dir.name
        pdbs = list(gene_dir.glob('*.pdb'))
        if not pdbs: continue
        # Parse ALL pdbs (different isoforms / fragments), take max pLDDT per residue
        gene_pl = {}
        for pdb in pdbs:
            try:
                pl = parse_pdb_plddt(pdb)
                for res, bf in pl.items():
                    if res not in gene_pl or bf > gene_pl[res]:
                        gene_pl[res] = bf
            except Exception as e:
                print(f'  pLDDT parse fail {gene}/{pdb.name}: {e}')
        for res, bf in gene_pl.items():
            lookup[(gene, res)] = bf
        if gene_pl: n_genes_done += 1
    print(f'  pLDDT loaded for {n_genes_done} genes, {len(lookup)} (gene, pos) entries')
    serial = {json.dumps(list(k)): v for k, v in lookup.items()}
    with open(cache_f, 'w') as f:
        json.dump(serial, f)
    return lookup


# =====================================================================
# Step 2: UniProt domain annotations per gene → domain at each position
# =====================================================================
def fetch_uniprot_domains(uniprot_id):
    """Get list of {start, end, name, type} for protein from UniProt REST API."""
    cache_f = CACHE / f'uniprot_dom_{uniprot_id}.json'
    if cache_f.exists():
        return json.load(open(cache_f))
    url = f'https://rest.uniprot.org/uniprotkb/{uniprot_id}.json'
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            json.dump([], open(cache_f, 'w'))
            return []
        data = r.json()
        domains = []
        # UniProt features array contains domains, regions, motifs, etc.
        for feat in data.get('features', []):
            ft_type = feat.get('type', '')
            if ft_type not in ('Domain','Region','Motif','Repeat','Active site','Binding site','Transmembrane','Topological domain','Signal','Disulfide bond'):
                continue
            loc = feat.get('location', {})
            start = loc.get('start', {}).get('value')
            end   = loc.get('end',   {}).get('value')
            if start is None or end is None: continue
            domains.append({
                'type': ft_type,
                'name': feat.get('description', ft_type),
                'start': int(start),
                'end':   int(end),
            })
        json.dump(domains, open(cache_f, 'w'))
        return domains
    except Exception as e:
        print(f'  UniProt fail {uniprot_id}: {e}')
        return []


def build_domain_lookup(genes):
    """For each gene, get domain list. Return per-gene list."""
    lookup = {}
    for i, g in enumerate(genes):
        if g not in UNIPROT_MAP: continue
        uid = UNIPROT_MAP[g]
        if (i+1) % 25 == 0: print(f'    UniProt fetched {i+1}/{len(genes)}…')
        doms = fetch_uniprot_domains(uid)
        lookup[g] = doms
        time.sleep(0.1)
    return lookup


def position_domain_features(gene, aa_pos, domain_lookup):
    """For (gene, aa_pos), return:
       - in_domain: 1 if any 'Domain'/'Region'/'Repeat'/'Topological domain' covers it
       - in_active_site: 1 if 'Active site'/'Binding site' at this position
       - in_transmembrane: 1 if 'Transmembrane'
       - in_signal_peptide: 1 if 'Signal'
       - dist_to_domain_boundary: min distance to nearest domain start/end (NaN if not in domain)
    """
    doms = domain_lookup.get(gene, [])
    in_domain = 0; in_active = 0; in_tm = 0; in_signal = 0; in_disulfide_loop = 0
    dist_to_bnd = np.nan
    for d in doms:
        s, e = d['start'], d['end']
        ttype = d['type']
        if s <= aa_pos <= e:
            if ttype in ('Domain','Region','Repeat','Topological domain'):
                in_domain = 1
                bnd = min(aa_pos - s, e - aa_pos)
                if pd.isna(dist_to_bnd) or bnd < dist_to_bnd:
                    dist_to_bnd = bnd
            elif ttype in ('Active site','Binding site'):
                in_active = 1
            elif ttype == 'Transmembrane':
                in_tm = 1
            elif ttype == 'Signal':
                in_signal = 1
            elif ttype == 'Disulfide bond':
                in_disulfide_loop = 1
    return {
        'in_domain': in_domain,
        'in_active_site': in_active,
        'in_transmembrane': in_tm,
        'in_signal_peptide': in_signal,
        'in_disulfide_bond': in_disulfide_loop,
        'dist_to_domain_boundary': dist_to_bnd,
    }


# =====================================================================
# Step 3: PhyloP / GERP via myvariant.info batch
# =====================================================================
def fetch_phylop_batch(hgvs_list, batch_size=200):
    """Use myvariant.info /v1/variant POST endpoint. Field: dbnsfp.phylop_100way_vertebrate, dbnsfp.gerp_pp_rs."""
    results = {}
    cache_f = CACHE / 'phylop_lookup.json'
    if cache_f.exists():
        results = json.load(open(cache_f))
    queried_set = set(results.keys())
    todo = [h for h in hgvs_list if h and h not in queried_set]
    print(f'  PhyloP: cached {len(queried_set)}, to query {len(todo)}')
    n_done = 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i+batch_size]
        try:
            r = requests.post('https://myvariant.info/v1/variant',
                              json={'ids': batch,
                                    'fields': 'dbnsfp.phylop_100way_vertebrate,dbnsfp.gerp_pp_rs,dbnsfp.phylop_30way_mammalian'},
                              timeout=60)
            if r.status_code == 200:
                for d in r.json():
                    qid = d.get('query', d.get('_id'))
                    if d.get('notfound'):
                        results[qid] = {'phylop': None, 'gerp': None}
                    else:
                        dn = d.get('dbnsfp', {}) or {}
                        results[qid] = {
                            'phylop': dn.get('phylop_100way_vertebrate'),
                            'gerp':   dn.get('gerp_pp_rs'),
                        }
            n_done += len(batch)
            if n_done % 1000 == 0 or n_done == len(todo):
                print(f'    {n_done}/{len(todo)} queried…')
                json.dump(results, open(cache_f,'w'))
            time.sleep(0.3)
        except Exception as e:
            print(f'    batch fail at {i}: {e}')
            time.sleep(2)
    json.dump(results, open(cache_f,'w'))
    return results


# =====================================================================
# Main
# =====================================================================
def hgvs_from_row(row):
    """Build hgvs from chromosome + ... if available; otherwise return None."""
    if 'hgvs' in row and pd.notna(row.get('hgvs')):
        return str(row['hgvs'])
    # Build chr-based hgvs from VCF-like fields
    if all(c in row.index for c in ['chromosome']):
        chrom = row['chromosome']
        if pd.notna(chrom):
            # We don't have ref/alt/pos in the train.csv so cannot construct
            return None
    return None


def main():
    print('\n[1/4] Build pLDDT lookup from AF structures...')
    plddt_lookup = build_plddt_lookup()

    print('\n[2/4] Build UniProt domain lookup...')
    panel = sorted(UNIPROT_MAP.keys())
    domain_lookup = build_domain_lookup(panel)
    print(f'  Got domains for {sum(1 for v in domain_lookup.values() if len(v)>0)}/{len(panel)} genes')

    print('\n[3/4] Build per-variant feature table for train + val sets...')
    SPLITS = ROOT_V2 / 'data/splits'

    feat_records = []
    for sname in ['train','val_time','val_gene','val_db','val_func_mave']:
        df = pd.read_csv(SPLITS / f'{sname}_with_celltype.csv', low_memory=False)
        if 'aa_pos' not in df.columns:
            # val_func_mave doesn't have aa_pos — extract from protein_change
            if 'protein_change' in df.columns:
                df['aa_pos'] = df.protein_change.str.extract(r'(\d+)').astype(float)
            else:
                print(f'  {sname}: no aa_pos, skipping')
                continue
        for _, row in df.iterrows():
            g = row.gene
            aa = row.get('aa_pos')
            if pd.isna(aa): continue
            aa = int(aa)
            plddt = plddt_lookup.get((g, aa), np.nan)
            dom_feats = position_domain_features(g, aa, domain_lookup)
            rec = {'gene': g, 'aa_pos': aa, 'protein_change': row.get('protein_change'),
                   'split': sname, 'plddt': plddt, **dom_feats}
            feat_records.append(rec)
    feat_df = pd.DataFrame(feat_records).drop_duplicates(subset=['gene','protein_change','split'])
    print(f'  Per-variant features: {feat_df.shape}')
    print(f'  pLDDT non-NaN: {feat_df.plddt.notna().sum()}/{len(feat_df)} ({100*feat_df.plddt.notna().mean():.0f}%)')
    print(f'  in_domain == 1: {feat_df.in_domain.sum()}')

    feat_df.to_csv(OUT / 'pervariant_features_struct_domain.csv', index=False)
    print(f'\n✅ Saved → pervariant_features_struct_domain.csv  ({feat_df.shape})')

    # Audit
    print('\n=== Per-feature summary ===')
    for col in ['plddt','in_domain','in_active_site','in_transmembrane','in_signal_peptide','in_disulfide_bond','dist_to_domain_boundary']:
        sub = feat_df[col].dropna()
        n_uniq = sub.nunique()
        if pd.api.types.is_numeric_dtype(sub):
            print(f'  {col:30}  nonNaN={len(sub):>5}  unique={n_uniq}  range=[{sub.min():.2f},{sub.max():.2f}]  mean={sub.mean():.2f}')
        else:
            print(f'  {col:30}  nonNaN={len(sub):>5}  unique={n_uniq}')

if __name__ == '__main__':
    main()
