#!/usr/bin/env python3
"""
Step 01 - Extract raw variant records for the hearing-loss gene panel from the primary sources.

Inputs (read-only, raw):
  - ClinVar variant_summary snapshots (2025-01, 2025-06, 2026-01, 2026-04), GRCh38 rows
  - Deafness Variation Database (DVD) full export (dvd_all_curated.csv)
  - PanelApp "green" hearing-loss gene panel (gene_panel_hhl_green.csv)

Outputs (data/sources/):
  - clinvar_<snapshot>.csv : one row per ClinVar SNV with a parseable protein change in a panel gene
  - dvd_missense.csv       : one row per DVD missense variant with its evidence basis
  - extract_summary.json   : row counts for the Methods / reconciliation table

No filtering on label quality is done here; that happens in step 04.
"""
import json
import re
from pathlib import Path

import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data'))
from paths import legacy_root  # noqa: E402

REV_ROOT = Path(__file__).resolve().parents[2]                   # the release folder
RAW = legacy_root() / 'data'
OUT = REV_ROOT / 'data/sources'
OUT.mkdir(parents=True, exist_ok=True)

SNAPSHOTS = ['2025-01', '2025-06', '2026-01', '2026-04']
AA3 = {'Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile', 'Leu', 'Lys', 'Met',
       'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val'}

P_TERMS = {'pathogenic', 'likely pathogenic', 'pathogenic/likely pathogenic'}
B_TERMS = {'benign', 'likely benign', 'benign/likely benign'}
STARS = {
    'practice guideline': 4,
    'reviewed by expert panel': 3,
    'criteria provided, multiple submitters, no conflicts': 2,
    'criteria provided, single submitter': 1,
    'criteria provided, conflicting classifications': 1,
    'criteria provided, conflicting interpretations': 1,
}

NAME_RE = re.compile(r'^(N[MR]_\d+\.\d+)\(([^)]+)\):(c\.[^ ]+)(?: \(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3}|=)\))?')


def clinsig_status(cs):
    """Map a ClinVar germline classification string to P / B / VUS / CONFLICT / OTHER."""
    s = str(cs).strip().lower()
    head = s.split(';')[0].strip()          # drop modifiers such as "; other", "; risk factor"
    if 'conflicting' in s:
        return 'CONFLICT'
    if 'low penetrance' in s:
        return 'OTHER'
    if head in P_TERMS:
        return 'P'
    if head in B_TERMS:
        return 'B'
    if head.startswith('uncertain significance'):
        return 'VUS'
    return 'OTHER'


def vclass(ref, alt):
    if ref is None or alt is None:
        return 'unparsed'
    if alt == '=' or ref == alt:
        return 'synonymous'
    if alt == 'Ter':
        return 'nonsense'
    if ref == 'Ter':
        return 'stoploss'
    if ref in AA3 and alt in AA3:
        return 'missense'
    return 'other'


def extract_clinvar(snapshot, hgnc2gene):
    path = RAW / f'external/clinvar_snapshots/variant_summary_{snapshot}.txt.gz'
    usecols = ['#AlleleID', 'Type', 'Name', 'GeneSymbol', 'HGNC_ID', 'ClinicalSignificance', 'LastEvaluated',
               'Assembly', 'Chromosome', 'ReviewStatus', 'NumberSubmitters', 'VariationID',
               'PositionVCF', 'ReferenceAlleleVCF', 'AlternateAlleleVCF']
    keep = []
    for chunk in pd.read_csv(path, sep='\t', usecols=usecols, dtype=str, chunksize=400_000, low_memory=False):
        m = (chunk['Assembly'] == 'GRCh38') & (chunk['Type'] == 'single nucleotide variant') \
            & chunk['HGNC_ID'].isin(hgnc2gene)
        if m.any():
            keep.append(chunk[m])
    df = pd.concat(keep, ignore_index=True)
    df['gene'] = df['HGNC_ID'].map(hgnc2gene)
    parsed = df['Name'].str.extract(NAME_RE)
    parsed.columns = ['nm', 'name_gene', 'c_change', 'p_ref', 'p_pos', 'p_alt']
    df = pd.concat([df, parsed], axis=1)
    df['vclass'] = [vclass(r if isinstance(r, str) else None, a if isinstance(a, str) else None)
                    for r, a in zip(df.p_ref, df.p_alt)]
    df['status'] = df['ClinicalSignificance'].map(clinsig_status)
    df['stars'] = df['ReviewStatus'].str.strip().str.lower().map(STARS).fillna(0).astype(int)
    df['snapshot'] = snapshot
    df = df.rename(columns={'#AlleleID': 'AlleleID'})
    cols = ['snapshot', 'gene', 'GeneSymbol', 'VariationID', 'AlleleID', 'Name', 'nm', 'c_change',
            'p_ref', 'p_pos', 'p_alt', 'vclass', 'ClinicalSignificance', 'status', 'ReviewStatus', 'stars',
            'LastEvaluated', 'NumberSubmitters', 'Chromosome', 'PositionVCF', 'ReferenceAlleleVCF',
            'AlternateAlleleVCF']
    return df[cols]


def dvd_basis(comment, pmid):
    c = str(comment)
    if 'prediction data only' in c:
        return 'prediction_only'
    if 'MAF' in c and 'cutoff' in c:
        return 'maf_cutoff'
    has_clinvar = 'ClinVar' in c
    has_lit = ('literature' in c) or ('PubMed' in c)
    if has_clinvar and has_lit:
        return 'clinvar+literature'
    if has_clinvar:
        return 'clinvar_only'
    if has_lit:
        return 'literature'
    if 'curated' in c.lower() or 'MORL' in c:
        return 'manual_curation'
    if pd.notna(pmid):
        return 'unspecified_with_pmid'
    return 'unspecified'


def extract_dvd(panel_genes):
    d = pd.read_csv(RAW / 'external/dvd_all_curated.csv', low_memory=False, dtype=str)
    n_all = len(d)
    d = d[d['consequence'].astype(str).str.contains('missense_variant')].copy()
    hp = d['hgvs_p'].str.extract(r'^(NP_\d+\.\d+):p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})$')
    hp.columns = ['np', 'p_ref', 'p_pos', 'p_alt']
    d = pd.concat([d, hp], axis=1)
    d['nm'] = d['hgvs_c'].str.split(':').str[0]
    d['vclass'] = [vclass(r if isinstance(r, str) else None, a if isinstance(a, str) else None)
                   for r, a in zip(d.p_ref, d.p_alt)]
    d['basis'] = [dvd_basis(c, p) for c, p in zip(d['comments'], d['pmid'])]
    d['n_pmid'] = d['pmid'].fillna('').map(lambda s: len([x for x in s.split(',') if x.strip()]))
    lab = d['pathogenicity'].str.strip().str.lower()
    d['status'] = lab.map(lambda s: 'P' if s in ('pathogenic', 'likely pathogenic')
                          else ('B' if s in ('benign', 'likely benign') else 'OTHER'))
    g38 = d['variation_grch38'].str.extract(r'^([0-9XYMT]+):(\d+):([ACGT]+)>([ACGT]+)$')
    g38.columns = ['Chromosome', 'PositionVCF', 'ReferenceAlleleVCF', 'AlternateAlleleVCF']
    d = pd.concat([d, g38], axis=1)
    d['in_panel'] = d['gene'].isin(panel_genes)
    cols = ['gene', 'dvd_id', 'np', 'nm', 'hgvs_c', 'p_ref', 'p_pos', 'p_alt', 'vclass', 'pathogenicity', 'status',
            'basis', 'n_pmid', 'pmid', 'overall_max_maf', 'maf_source', 'disease', 'Chromosome', 'PositionVCF',
            'ReferenceAlleleVCF', 'AlternateAlleleVCF', 'in_panel', 'comments']
    return d[cols], n_all


def main():
    panel = pd.read_csv(RAW / 'gene_panel_hhl_green.csv')
    seqs = pd.read_csv(RAW / 'features/gene_sequences.csv')
    panel = panel[panel.gene.isin(seqs.gene) & ~panel.gene.str.startswith('MT-')].copy()
    hgnc2gene = dict(zip(panel.hgnc_id.astype(str), panel.gene))
    hgnc2gene = {(k if k.startswith('HGNC:') else f'HGNC:{k}'): v for k, v in hgnc2gene.items()}
    print(f'panel genes with protein sequence: {len(panel)}')

    summary = {'panel_genes': int(len(panel))}
    for snap in SNAPSHOTS:
        df = extract_clinvar(snap, hgnc2gene)
        df.to_csv(OUT / f'clinvar_{snap}.csv', index=False)
        mis = df[df.vclass == 'missense']
        summary[f'clinvar_{snap}'] = {
            'snv_rows_panel': int(len(df)),
            'genes': int(df.gene.nunique()),
            'by_vclass': df.vclass.value_counts().to_dict(),
            'missense_by_status': mis.status.value_counts().to_dict(),
            'missense_definitive_ge1star': int(((mis.status.isin(['P', 'B'])) & (mis.stars >= 1)).sum()),
            'missense_P_ge1star': int(((mis.status == 'P') & (mis.stars >= 1)).sum()),
            'missense_B_ge1star': int(((mis.status == 'B') & (mis.stars >= 1)).sum()),
            'missense_VUS': int((mis.status == 'VUS').sum()),
            'missense_conflicting': int((mis.status == 'CONFLICT').sum()),
        }
        print(snap, json.dumps(summary[f'clinvar_{snap}']))

    dvd, n_all = extract_dvd(set(panel.gene))
    dvd.to_csv(OUT / 'dvd_missense.csv', index=False)
    summary['dvd'] = {
        'all_rows': int(n_all),
        'missense_rows': int(len(dvd)),
        'missense_rows_in_panel': int(dvd.in_panel.sum()),
        'genes': int(dvd.gene.nunique()),
        'unparsed_protein_change': int((dvd.vclass != 'missense').sum()),
        'basis_by_status': pd.crosstab(dvd.basis, dvd.status).to_dict(),
    }
    print('dvd', json.dumps(summary['dvd']))
    json.dump(summary, open(OUT / 'extract_summary.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
