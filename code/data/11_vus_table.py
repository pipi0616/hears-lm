#!/usr/bin/env python3
"""
Step 11 - ClinVar 2026-04 missense variants of the panel genes WITHOUT a definitive classification
(uncertain / conflicting / other), on reference-protein numbering, with residue-context flags.

Purpose: the set of unresolved protein variants that the final model is applied to (VUS prioritisation).
None of them is part of the labelled data set, so none was seen in training or evaluation.

Inputs (read-only)
  - data/sources/clinvar_2026-04_canon.csv : ClinVar missense SNV records mapped to the reference protein
                                             (step 03); only rows with canon_status == 'ok' are used
  - data/variants_v8.csv                   : labelled variants (step 04); used to exclude labelled variants
                                             and to compute the residue-context flags (label = y_final)
  - data/sources/reference_proteins.csv    : only for a consistency check of the reference residue

Rules
  1. Records are aggregated to the protein level (variant_id = GENE:<ref><ref_pos><alt>).
  2. A variant_id is dropped if any of its records is definitive (status P or B with >= 1 star) or if it is
     present in data/variants_v8.csv (any partition; this also removes variants labelled by DVD only and
     variants that were definitive in the 2025-01 training snapshot).
  3. cv_state = 'CONFLICT' if any record has status CONFLICT, else 'VUS' if any record is VUS, else 'OTHER'
     (zero-star P / B assertions, "not provided", risk alleles ...; see `cv_statuses`).
  4. The representative record (name, GRCh38 coordinates) is the first record after sorting by: status
     matching cv_state priority (CONFLICT, VUS, other), stars descending, VariationID ascending. A protein
     change can be produced by more than one nucleotide change (n_records > 1): the coordinates are those
     of the representative record only; all VariationIDs are listed in `variation_ids`.
  5. Residue context, from labelled variants with a final label (y_final not missing; all partitions):
       same_residue_PLP  1 if a labelled P/LP missense variant exists at the same gene + ref_pos
       same_residue_BLB  1 if a labelled B/LB missense variant exists at the same gene + ref_pos
       dist_nearest_PLP  min |ref_pos difference| to any labelled P/LP residue of the gene (0 = same
                         residue); missing if the gene has no labelled P/LP missense variant

Outputs (data/)
  - vus_2026-04.csv : one row per unresolved protein variant
        variant_id, gene, ref_pos, ref1, alt1, cv_state, n_records, variation_ids (pipe-separated),
        name, chrom, pos, ref_nt, alt_nt, max_submitters, same_residue_PLP, same_residue_BLB,
        dist_nearest_PLP, then audit columns: uniprot_pos, cv_statuses (pipe-separated record statuses),
        max_stars
  - vus_2026-04_summary.json : validation counts

Deterministic; no imputation; no network.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[2]                        # the release folder
SRC = REV / 'data/sources'
SNAPSHOT = '2026-04'
OUT = REV / f'data/vus_{SNAPSHOT}.csv'
SUMMARY = REV / f'data/vus_{SNAPSHOT}_summary.json'
STATE_RANK = {'CONFLICT': 0, 'VUS': 1}


def load_labelled(cols):
    """Columns `cols` of data/variants_v8.csv plus a provenance record.

    Other steps rewrite this file in place (04b overwrites cv_fold), so the row count is checked against
    data/split_summary.json to detect a partially written file.
    """
    path = REV / 'data/variants_v8.csv'
    v = pd.read_csv(path, usecols=cols, low_memory=False)
    with open(REV / 'data/split_summary.json') as fh:
        expect = sum(p['n'] for p in json.load(fh)['partitions'].values())
    assert len(v) == expect and v.variant_id.is_unique, \
        f'{path}: {len(v)} rows, expected {expect} unique variants (file being rewritten?)'
    info = {'rows': int(len(v)),
            'modified': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(path.stat().st_mtime))}
    return v, info


def nearest_distance(query_pos, sorted_pos):
    """min |q - p| over p in sorted_pos for every q (both integer arrays; sorted_pos non-empty, ascending)."""
    i = np.searchsorted(sorted_pos, query_pos)
    left = np.abs(query_pos - sorted_pos[np.clip(i - 1, 0, len(sorted_pos) - 1)])
    right = np.abs(sorted_pos[np.clip(i, 0, len(sorted_pos) - 1)] - query_pos)
    return np.minimum(left, right)


def main():
    c = pd.read_csv(SRC / f'clinvar_{SNAPSHOT}_canon.csv', dtype=str)
    n_records_all = len(c)
    c = c[c.canon_status == 'ok'].copy()
    assert c.VariationID.is_unique and c.variant_id.notna().all()
    c['stars'] = c.stars.astype(int)
    c['vid_num'] = c.VariationID.astype(int)
    c['n_sub'] = c.NumberSubmitters.astype(int)
    c['definitive'] = c.status.isin(['P', 'B']) & (c.stars >= 1)
    n_protein_all = c.variant_id.nunique()

    lab, lab_info = load_labelled(['variant_id', 'gene', 'ref_pos', 'y_final'])
    definitive_ids = set(c.variant_id[c.definitive])
    labelled_ids = set(lab.variant_id)
    drop_def = c.variant_id.isin(definitive_ids)
    drop_lab = c.variant_id.isin(labelled_ids)
    n_drop = {'protein_variants_with_definitive_record': int(c.variant_id[drop_def].nunique()),
              'protein_variants_in_variants_v8': int(c.variant_id[drop_lab].nunique()),
              'in_variants_v8_without_definitive_record': int(c.variant_id[drop_lab & ~drop_def].nunique()),
              'definitive_but_not_in_variants_v8': int(c.variant_id[drop_def & ~drop_lab].nunique())}
    u = c[~drop_def & ~drop_lab].copy()

    # ---- protein-level aggregation
    u['is_conflict'] = u.status == 'CONFLICT'
    u['is_vus'] = u.status == 'VUS'
    u['rank'] = u.status.map(STATE_RANK).fillna(2).astype(int)
    u = u.sort_values(['variant_id', 'vid_num'])
    g = u.groupby('variant_id', sort=True)
    agg = g.agg(n_records=('VariationID', 'size'), any_conflict=('is_conflict', 'max'), any_vus=('is_vus', 'max'),
                variation_ids=('VariationID', '|'.join), max_submitters=('n_sub', 'max'), max_stars=('stars', 'max'),
                cv_statuses=('status', lambda s: '|'.join(sorted(set(s)))))
    agg['cv_state'] = np.where(agg.any_conflict, 'CONFLICT', np.where(agg.any_vus, 'VUS', 'OTHER'))
    rep = u.sort_values(['variant_id', 'rank', 'stars', 'vid_num'], ascending=[True, True, False, True]) \
           .drop_duplicates('variant_id', keep='first').set_index('variant_id')
    assert (rep['rank'] == agg.cv_state.map(STATE_RANK).fillna(2).astype(int)).all()
    out = pd.DataFrame({
        'gene': rep.gene, 'ref_pos': rep.ref_pos.astype(int), 'ref1': rep.ref1, 'alt1': rep.alt1,
        'cv_state': agg.cv_state, 'n_records': agg.n_records, 'variation_ids': agg.variation_ids,
        'name': rep.Name, 'chrom': rep.Chromosome, 'pos': rep.PositionVCF.astype(int),
        'ref_nt': rep.ReferenceAlleleVCF, 'alt_nt': rep.AlternateAlleleVCF, 'max_submitters': agg.max_submitters,
    }).rename_axis('variant_id').reset_index()

    # ---- residue context against the labelled set
    fin = lab[lab.y_final.notna()]
    p_pos = fin[fin.y_final == 1].groupby('gene').ref_pos.apply(lambda s: np.unique(s.to_numpy(dtype=int)))
    b_keys = set(zip(fin.gene[fin.y_final == 0], fin.ref_pos[fin.y_final == 0].astype(int)))
    p_keys = set(zip(fin.gene[fin.y_final == 1], fin.ref_pos[fin.y_final == 1].astype(int)))
    keys = list(zip(out.gene, out.ref_pos))
    out['same_residue_PLP'] = [int(k in p_keys) for k in keys]
    out['same_residue_BLB'] = [int(k in b_keys) for k in keys]
    dist = pd.Series(pd.NA, index=out.index, dtype='Int64')
    for gene, idx in out.groupby('gene').groups.items():
        if gene in p_pos.index:
            dist.loc[idx] = nearest_distance(out.loc[idx, 'ref_pos'].to_numpy(dtype=int), p_pos[gene])
    out['dist_nearest_PLP'] = dist
    out['uniprot_pos'] = out.variant_id.map(rep.uniprot_pos).astype('Int64')
    out['cv_statuses'] = out.variant_id.map(agg.cv_statuses)
    out['max_stars'] = out.variant_id.map(agg.max_stars)
    out = out.sort_values(['gene', 'ref_pos', 'alt1']).reset_index(drop=True)

    # ---- invariants
    refseq = dict(pd.read_csv(SRC / 'reference_proteins.csv', usecols=['gene', 'sequence']).values)
    assert out.variant_id.is_unique
    assert (out.variant_id == out.gene + ':' + out.ref1 + out.ref_pos.astype(str) + out.alt1).all()
    assert all(refseq[g][p - 1] == a for g, p, a in zip(out.gene, out.ref_pos, out.ref1))
    assert not set(out.variant_id) & labelled_ids and not set(out.variant_id) & definitive_ids
    assert ((out.same_residue_PLP == 1) == (out.dist_nearest_PLP == 0).fillna(False)).all()
    assert int(out.n_records.sum()) == len(u)
    out.to_csv(OUT, index=False)

    has_p_gene = out.dist_nearest_PLP.notna()
    by_gene = out.gene.value_counts().sort_index().sort_values(ascending=False, kind='mergesort')   # ties: A-Z
    summary = {
        'input': {'snapshot': SNAPSHOT, 'records': int(n_records_all), 'records_canon_ok': int(len(c)),
                  'protein_variants_canon_ok': int(n_protein_all)},
        'variants_v8_used': {**lab_info, 'with_final_label': int(len(fin)),
                             'labelled_PLP_residues': int(len(p_keys)), 'labelled_BLB_residues': int(len(b_keys)),
                             'genes_with_labelled_PLP': int(len(p_pos))},
        'dropped': n_drop,
        'output': {'file': str(OUT), 'rows': int(len(out)), 'genes': int(out.gene.nunique()),
                   'records_represented': int(out.n_records.sum()),
                   'variants_with_more_than_one_record': int((out.n_records > 1).sum()),
                   'without_uniprot_pos': int(out.uniprot_pos.isna().sum())},
        'by_cv_state': {k: int(n) for k, n in out.cv_state.value_counts().items()},
        'cv_statuses_within_OTHER': {k: int(n) for k, n in out.cv_statuses[out.cv_state == 'OTHER'].value_counts().items()},
        'max_stars_by_cv_state': {s: {int(k): int(n) for k, n in g.max_stars.value_counts().sort_index().items()}
                                  for s, g in out.groupby('cv_state')},
        'by_gene_top15': {k: int(n) for k, n in by_gene.head(15).items()},
        'by_gene_all': {k: int(n) for k, n in by_gene.items()},
        'same_residue_PLP': {'n': int(out.same_residue_PLP.sum()),
                             'fraction': round(float(out.same_residue_PLP.mean()), 4),
                             'fraction_by_cv_state': {s: round(float(g.same_residue_PLP.mean()), 4)
                                                      for s, g in out.groupby('cv_state')}},
        'same_residue_BLB': {'n': int(out.same_residue_BLB.sum()),
                             'fraction': round(float(out.same_residue_BLB.mean()), 4)},
        'same_residue_PLP_and_BLB': int(((out.same_residue_PLP == 1) & (out.same_residue_BLB == 1)).sum()),
        'dist_nearest_PLP': {'missing_gene_without_labelled_PLP': int((~has_p_gene).sum()),
                             'genes_without_labelled_PLP': sorted(out.gene[~has_p_gene].unique()),
                             'median': float(out.dist_nearest_PLP.median()),
                             'fraction_within_5_residues': round(float((out.dist_nearest_PLP[has_p_gene] <= 5).mean()), 4),
                             'fraction_within_5_residues_of_all':
                                 round(float((out.dist_nearest_PLP <= 5).fillna(False).mean()), 4)},
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + '\n')

    s = summary
    print(f"ClinVar {SNAPSHOT}: {s['input']['records_canon_ok']} mapped missense records = "
          f"{s['input']['protein_variants_canon_ok']} protein variants")
    print(f"dropped: {s['dropped']}")
    print(f"variants_v8.csv used: {s['variants_v8_used']}")
    print(f"OUTPUT {s['output']['rows']} unresolved protein variants in {s['output']['genes']} genes "
          f"({s['output']['variants_with_more_than_one_record']} with >1 ClinVar record)")
    print(f"by cv_state: {s['by_cv_state']}")
    print(f"record statuses within OTHER: {s['cv_statuses_within_OTHER']}")
    print(f"top 15 genes: {s['by_gene_top15']}")
    print(f"same_residue_PLP: {s['same_residue_PLP']}; same_residue_BLB: {s['same_residue_BLB']}; both: "
          f"{s['same_residue_PLP_and_BLB']}")
    print(f"dist_nearest_PLP: { {k: v for k, v in s['dist_nearest_PLP'].items() if k != 'genes_without_labelled_PLP'} }")
    print(f'wrote {OUT} and {SUMMARY}')


if __name__ == '__main__':
    main()
