#!/usr/bin/env python3
"""
Step 04 - Build the labelled missense data set and all data partitions (revision v8).

Label sources (evidence-based only)
  ClinVar : germline classification P / LP / P-LP (=1) or B / LB / B-LB (=0) with review status >= 1 star
            ("criteria provided ..." / expert panel / practice guideline). Records are aggregated to the
            protein level (gene + reference position + substitution); a protein change with both a
            pathogenic and a benign >=1-star record is dropped.
  DVD     : classifications supported by literature, minor-allele-frequency cut-off or manual curation.
            DVD classifications "based on prediction data only" (n = 33,832 Likely benign) are NOT labels.
            DVD classifications based only on ClinVar submissions are not used as an independent source
            (ClinVar is read directly, with star filter and snapshot control).
            A DVD label is also discarded when ClinVar holds an expert-panel or multi-submitter consensus
            classification of "uncertain significance" (>= 2 stars) for the same protein change in either
            snapshot, because that is a positive statement contradicting the DVD assertion; single-submitter
            VUS records and "conflicting classifications" do not override DVD.

Partitions (mutually exclusive at the protein-variant level; priority gene > db > time > train)
  TEST_gene : every labelled variant in 10 genes withheld from training (same 10 genes as the original study)
  TEST_db   : DVD variants with strong literature evidence (P/LP >= 2 PMIDs; B/LB >= 1 PMID, not MAF-only)
              that had NO definitive ClinVar classification in the training snapshot (2025-01)
  TEST_time : variants with a definitive ClinVar classification in the 2026-04 snapshot that had none in
              the 2025-01 snapshot (absent, VUS, conflicting or zero-star at that time)
  TRAIN pool: ClinVar 2025-01 definitive labels + remaining DVD evidence-based labels; split 90/10 into
              FIT / DEV by residue position (all variants of one residue stay together; assignment by a
              hash of the residue key, so it does not change when other variants are added or removed).
  CV folds  : 5 gene-grouped folds over ALL labelled variants (final labels) for leave-gene-out analyses.

Outputs (data/):
  variants_v8.csv      one row per labelled protein variant with partition, labels and provenance
  split_summary.json   counts for the reconciliation table
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]                       # the release folder
SRC = ROOT / 'data/sources'
OUT = ROOT / 'data'
TRAIN_SNAPSHOT, EVAL_SNAPSHOT = '2025-01', '2026-04'
HOLDOUT_GENES = ['OTOF', 'CLDN14', 'STRC', 'TECTA', 'WFS1', 'TMC1', 'CDH23', 'LOXHD1', 'TBC1D24', 'ACTG1']
DVD_INDEPENDENT = {'literature', 'clinvar+literature', 'maf_cutoff', 'manual_curation', 'unspecified_with_pmid'}
SEED = 20261006


def clinvar_protein_level(snapshot):
    c = pd.read_csv(SRC / f'clinvar_{snapshot}_canon.csv', dtype=str)
    c = c[c.canon_status == 'ok'].copy()
    c['stars'] = c.stars.astype(int)
    c['definitive'] = c.status.isin(['P', 'B']) & (c.stars >= 1)
    # representative record: definitive first, then stars, then most recent VariationID
    c['_vid'] = c.VariationID.astype(int)
    c = c.sort_values(['variant_id', 'definitive', 'stars', '_vid'], ascending=[True, False, False, True])
    rows = []
    for vid, g in c.groupby('variant_id', sort=False):
        d = g[g.definitive]
        has_p, has_b = (d.status == 'P').any(), (d.status == 'B').any()
        if has_p and has_b:
            state = 'PB_CONFLICT'
        elif has_p:
            state = 'P'
        elif has_b:
            state = 'B'
        elif (g.status == 'CONFLICT').any():
            state = 'CONFLICT'
        elif g.status.isin(['P', 'B']).any():
            state = 'LOWSTAR'
        elif (g.status == 'VUS').any():
            state = 'VUS'
        else:
            state = 'OTHER'
        r = g.iloc[0]
        rows.append({'variant_id': vid, 'gene': r.gene, 'ref_pos': int(r.ref_pos), 'ref1': r.ref1, 'alt1': r.alt1,
                     'uniprot_pos': r.uniprot_pos, 'cv_state': state,
                     'cv_stars': int(d.stars.max()) if len(d) else int(g.stars.max()),
                     'cv_n_records': len(g), 'cv_variation_id': r.VariationID, 'cv_name': r.Name,
                     'cv_clinsig': r.ClinicalSignificance, 'cv_review': r.ReviewStatus,
                     'cv_last_evaluated': r.LastEvaluated, 'cv_p_notation': f'{r.p_ref}{r.p_pos}{r.p_alt}',
                     'chrom': r.Chromosome, 'pos': r.PositionVCF, 'ref_nt': r.ReferenceAlleleVCF,
                     'alt_nt': r.AlternateAlleleVCF})
    return pd.DataFrame(rows)


def dvd_protein_level():
    d = pd.read_csv(SRC / 'dvd_missense_canon.csv', dtype=str)
    d = d[d.canon_status == 'ok'].copy()
    d['n_pmid'] = d.n_pmid.astype(int)
    stats = {'dvd_missense_mapped': int(len(d)), 'dvd_prediction_only': int((d.basis == 'prediction_only').sum()),
             'dvd_clinvar_only': int((d.basis == 'clinvar_only').sum())}
    e = d[d.basis.isin(DVD_INDEPENDENT) & d.status.isin(['P', 'B'])].copy()
    stats['dvd_independent_records'] = int(len(e))
    e = e.sort_values(['variant_id', 'n_pmid'], ascending=[True, False])
    rows = []
    for vid, g in e.groupby('variant_id', sort=False):
        has_p, has_b = (g.status == 'P').any(), (g.status == 'B').any()
        r = g.iloc[0]
        state = 'PB_CONFLICT' if (has_p and has_b) else ('P' if has_p else 'B')
        strong = (state == 'P' and r.n_pmid >= 2) or (state == 'B' and r.n_pmid >= 1 and r.basis != 'maf_cutoff')
        rows.append({'variant_id': vid, 'gene': r.gene, 'ref_pos': int(r.ref_pos), 'ref1': r.ref1, 'alt1': r.alt1,
                     'uniprot_pos': r.uniprot_pos, 'dvd_state': state, 'dvd_basis': r.basis,
                     'dvd_n_pmid': int(r.n_pmid), 'dvd_strong': bool(strong), 'dvd_id': r.dvd_id,
                     'dvd_pathogenicity': r.pathogenicity, 'dvd_max_maf': r.overall_max_maf,
                     'dvd_p_notation': f'{r.p_ref}{r.p_pos}{r.p_alt}', 'dvd_np': r.np,
                     'dvd_chrom': r.Chromosome, 'dvd_pos': r.PositionVCF, 'dvd_ref_nt': r.ReferenceAlleleVCF,
                     'dvd_alt_nt': r.AlternateAlleleVCF})
    return pd.DataFrame(rows), stats


def main():
    cv25 = clinvar_protein_level(TRAIN_SNAPSHOT)
    cv26 = clinvar_protein_level(EVAL_SNAPSHOT)
    dvd, stats = dvd_protein_level()
    print(f'ClinVar {TRAIN_SNAPSHOT}: {len(cv25)} protein-level missense; states {cv25.cv_state.value_counts().to_dict()}')
    print(f'ClinVar {EVAL_SNAPSHOT}: {len(cv26)} protein-level missense; states {cv26.cv_state.value_counts().to_dict()}')
    print(f'DVD independent-evidence protein-level: {len(dvd)}; states {dvd.dvd_state.value_counts().to_dict()}')

    base_cols = ['variant_id', 'gene', 'ref_pos', 'ref1', 'alt1', 'uniprot_pos']
    allv = pd.concat([cv26[base_cols], cv25[base_cols], dvd[base_cols]]).drop_duplicates('variant_id')
    v = allv.merge(cv26.drop(columns=base_cols[1:]), on='variant_id', how='left')
    v = v.merge(cv25[['variant_id', 'cv_state', 'cv_stars', 'cv_variation_id', 'chrom', 'pos', 'ref_nt', 'alt_nt']]
                .rename(columns={'cv_state': 'cv25_state', 'cv_stars': 'cv25_stars', 'cv_variation_id': 'cv25_variation_id',
                                 'chrom': 'chrom25', 'pos': 'pos25', 'ref_nt': 'ref_nt25', 'alt_nt': 'alt_nt25'}),
                on='variant_id', how='left')
    v = v.merge(dvd.drop(columns=base_cols[1:]), on='variant_id', how='left')
    v['cv_state'] = v.cv_state.fillna('ABSENT')
    v['cv25_state'] = v.cv25_state.fillna('ABSENT')
    v['dvd_state'] = v.dvd_state.fillna('ABSENT')
    v['dvd_strong'] = v.dvd_strong.fillna(False).astype(bool)

    # representative genomic coordinates (GRCh38): ClinVar 2026-04, else ClinVar 2025-01, else DVD
    for a, b, c in [('chrom', 'chrom25', 'dvd_chrom'), ('pos', 'pos25', 'dvd_pos'),
                    ('ref_nt', 'ref_nt25', 'dvd_ref_nt'), ('alt_nt', 'alt_nt25', 'dvd_alt_nt')]:
        v[a] = v[a].fillna(v[b]).fillna(v[c])
    v = v.drop(columns=['chrom25', 'pos25', 'ref_nt25', 'alt_nt25'])

    lab = {'P': 1, 'B': 0}
    v['y_cv26'] = v.cv_state.map(lab)
    v['y_cv25'] = v.cv25_state.map(lab)
    v['y_dvd'] = v.dvd_state.map(lab)
    # DVD label overridden by an expert-panel / multi-submitter consensus VUS in ClinVar (either snapshot)
    cons_vus = ((v.cv_state == 'VUS') & (v.cv_stars.fillna(0) >= 2)) | ((v.cv25_state == 'VUS') & (v.cv25_stars.fillna(0) >= 2))
    v['dvd_overridden_by_consensus_vus'] = cons_vus & v.y_dvd.notna()
    stats['dvd_labels_overridden_by_consensus_vus'] = int(v.dvd_overridden_by_consensus_vus.sum())
    stats['dvd_labels_overridden_by_expert_panel_vus'] = int(
        (v.y_dvd.notna() & (((v.cv_state == 'VUS') & (v.cv_stars.fillna(0) >= 3))
                            | ((v.cv25_state == 'VUS') & (v.cv25_stars.fillna(0) >= 3)))).sum())
    v.loc[v.dvd_overridden_by_consensus_vus, 'y_dvd'] = np.nan

    # ---- final label (evaluation label): ClinVar 2026-04 and/or DVD; drop cross-source conflicts
    has_any = v.y_cv26.notna() | v.y_dvd.notna()
    conflict_final = v.y_cv26.notna() & v.y_dvd.notna() & (v.y_cv26 != v.y_dvd)
    internal_conflict = (v.cv_state == 'PB_CONFLICT') | (v.dvd_state == 'PB_CONFLICT')
    v['y_final'] = v.y_cv26.where(v.y_cv26.notna(), v.y_dvd)
    v.loc[conflict_final | internal_conflict, 'y_final'] = np.nan
    # ---- training-time label: ClinVar 2025-01 and/or DVD; drop conflicts
    conflict_train = v.y_cv25.notna() & v.y_dvd.notna() & (v.y_cv25 != v.y_dvd)
    v['y_train'] = v.y_cv25.where(v.y_cv25.notna(), v.y_dvd)
    v.loc[conflict_train | (v.cv25_state == 'PB_CONFLICT') | (v.dvd_state == 'PB_CONFLICT'), 'y_train'] = np.nan

    stats.update({
        'protein_variants_any_source': int(len(v)),
        'labelled_any': int(has_any.sum()),
        'cross_source_conflict_final': int(conflict_final.sum()),
        'cross_source_conflict_train': int(conflict_train.sum()),
        'clinvar_internal_PB_conflict_2026': int((v.cv_state == 'PB_CONFLICT').sum()),
        'dvd_internal_PB_conflict': int((v.dvd_state == 'PB_CONFLICT').sum()),
    })

    # ---- partitions
    v['partition'] = 'none'
    in_hold = v.gene.isin(HOLDOUT_GENES)
    definitive25 = v.y_cv25.notna()
    # TEST_gene
    m_gene = in_hold & v.y_final.notna()
    v.loc[m_gene, 'partition'] = 'test_gene'
    # TEST_db: DVD strong evidence, no definitive ClinVar label at training time, no opposite ClinVar label now
    m_db = (~in_hold) & v.dvd_strong & v.y_dvd.notna() & (~definitive25) & v.y_final.notna() \
        & (v.cv25_state != 'PB_CONFLICT')
    v.loc[m_db, 'partition'] = 'test_db'
    # TEST_time: definitive in ClinVar 2026-04, not definitive in 2025-01
    m_time = (~in_hold) & (~m_db) & v.y_cv26.notna() & (~definitive25) & v.y_final.notna() \
        & (v.cv25_state != 'PB_CONFLICT')
    v.loc[m_time, 'partition'] = 'test_time'
    # TRAIN pool
    m_train = (~in_hold) & (~m_db) & (~m_time) & v.y_train.notna()
    v.loc[m_train, 'partition'] = 'train_pool'

    # evaluation label per row
    v['y'] = np.where(v.partition == 'train_pool', v.y_train, v.y_final)

    # FIT / DEV split of the train pool by residue position (10% of positions -> dev)
    key = v.gene + ':' + v.ref_pos.astype(str)
    in_dev = key.map(lambda k: int(hashlib.md5(f'{SEED}:{k}'.encode()).hexdigest(), 16) % 10 == 0)
    v.loc[(v.partition == 'train_pool') & in_dev, 'partition'] = 'dev'
    v.loc[v.partition == 'train_pool', 'partition'] = 'fit'

    # ---- gene-grouped 5-fold CV over all variants with a final label
    cvm = v.y_final.notna()
    gsize = v[cvm].groupby('gene').size().sort_values(ascending=False)
    fold_load = np.zeros(5)
    fold_of = {}
    genes = list(gsize.index)
    # greedy balanced assignment (largest genes first), deterministic tie-break by shuffled order
    order = np.array(genes)
    for g in order:
        f = int(np.argmin(fold_load))
        fold_of[g] = f
        fold_load[f] += gsize[g]
    v['cv_fold'] = np.where(cvm, v.gene.map(fold_of), -1).astype(int)

    # ---- leakage-relevant flags
    fit = v[v.partition == 'fit']
    fit_pos = set(fit.gene + ':' + fit.ref_pos.astype(str))
    fit_gene_n = fit.groupby('gene').size()
    v['pos_in_fit'] = key.isin(fit_pos)
    v['n_fit_gene'] = v.gene.map(fit_gene_n).fillna(0).astype(int)
    v['source'] = np.select([v.y_cv26.notna() & v.y_dvd.notna(), v.y_cv26.notna(), v.y_dvd.notna(),
                             v.y_cv25.notna()], ['clinvar+dvd', 'clinvar', 'dvd', 'clinvar2025_only'], 'none')
    v['prev_state_2025'] = v.cv25_state

    keep = v[v.partition != 'none'].copy()
    # variant-level disjointness is guaranteed by construction; assert it anyway
    assert keep.variant_id.is_unique
    keep = keep.sort_values(['partition', 'gene', 'ref_pos', 'alt1']).reset_index(drop=True)
    keep.to_csv(OUT / 'variants_v8.csv', index=False)

    def desc(df):
        return {'n': int(len(df)), 'P': int((df.y == 1).sum()), 'B': int((df.y == 0).sum()),
                'genes': int(df.gene.nunique()), 'positions': int((df.gene + ':' + df.ref_pos.astype(str)).nunique()),
                'pos_in_fit': int(df.pos_in_fit.sum()), 'source': df.source.value_counts().to_dict()}
    stats['partitions'] = {p: desc(keep[keep.partition == p]) for p in ['fit', 'dev', 'test_db', 'test_time', 'test_gene']}
    stats['test_time_prev_state'] = keep[keep.partition == 'test_time'].prev_state_2025.value_counts().to_dict()
    stats['test_db_clinvar_state_2026'] = keep[keep.partition == 'test_db'].cv_state.value_counts().to_dict()
    stats['test_db_prev_state_2025'] = keep[keep.partition == 'test_db'].prev_state_2025.value_counts().to_dict()
    stats['train_label_drift'] = {
        'fit_dev_n': int(keep.partition.isin(['fit', 'dev']).sum()),
        'label_changed_by_2026': int((keep.partition.isin(['fit', 'dev']) & keep.y_final.notna() & (keep.y != keep.y_final)).sum()),
        'no_longer_definitive_2026': int((keep.partition.isin(['fit', 'dev']) & keep.y_final.isna()).sum())}
    cvd = keep[keep.cv_fold >= 0]
    stats['cv'] = {'n': int(len(cvd)), 'P': int((cvd.y_final == 1).sum()), 'B': int((cvd.y_final == 0).sum()),
                   'genes': int(cvd.gene.nunique()),
                   'folds': {int(f): {'n': int(len(g)), 'P': int((g.y_final == 1).sum()), 'genes': int(g.gene.nunique())}
                             for f, g in cvd.groupby('cv_fold')}}
    stats['genes_total'] = int(keep.gene.nunique())
    stats['train_snapshot'], stats['eval_snapshot'], stats['holdout_genes'] = TRAIN_SNAPSHOT, EVAL_SNAPSHOT, HOLDOUT_GENES
    json.dump(stats, open(OUT / 'split_summary.json', 'w'), indent=2)
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
