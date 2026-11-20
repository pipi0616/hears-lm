#!/usr/bin/env python3
"""Supplementary Tables S1-S10 as one Excel workbook (manuscript/supplementary_tables.xlsx), from result files."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R, D = ROOT / 'results', ROOT / 'data'


def s1():
    ref = pd.read_csv(D / 'sources/reference_proteins.csv').drop(columns=['sequence'])
    fam = pd.read_csv(D / 'gene_families_v8.csv')[['gene', 'family_id', 'family_genes', 'cv_fold']]
    inh = pd.read_csv(R / 'table_per_gene.csv')[['gene', 'mode_of_inheritance']]
    ntr = pd.read_csv(R / 'training_variants_per_gene.csv')
    out = ref.merge(fam, on='gene', how='left').merge(inh, on='gene', how='left').merge(ntr, on='gene', how='left')
    out['n_training_variants'] = out.n_training_variants.fillna(0).astype(int)
    return out


def s2():
    t = pd.read_csv(R / 'table_per_gene.csv')
    return t.rename(columns={'P_total': 'P_cross_validation_pool', 'B_total': 'B_cross_validation_pool'})


def s2b():
    """Mapping of source records onto the reference proteins, per source (Methods 2.2)."""
    c = json.load(open(D / 'sources/canonicalize_summary.json'))
    rows = []
    for src in ['clinvar_2025-01', 'clinvar_2026-04', 'dvd']:
        r = c[src]
        n = r['ok'] + r['not_in_reference_isoform'] + r['ref_mismatch_reference']
        rows.append({'source': src, 'records': n, 'mapped': r['ok'],
                     'failed_position_not_in_reference_isoform': r['not_in_reference_isoform'],
                     'failed_reference_residue_mismatch': r['ref_mismatch_reference'],
                     'failed_fraction': round((n - r['ok']) / n, 4),
                     'mapped_with_renumbering': r['renumbered'],
                     'mapped_without_uniprot_position': r['ok_without_uniprot_pos']})
    return pd.DataFrame(rows)


def s3():
    groups = json.load(open(D / 'features/feature_groups.json'))
    src = {'plm': 'frozen ESM-2 150M, full-protein wild-type marginals (this study)',
           'cons': 'dbNSFP v4.8a via MyVariant.info, GRCh38 coordinate', 'af': 'gnomAD v2.1.1 exomes / v3.1.1 genomes (diagnostic arm only)',
           'struct': 'AlphaFold DB (pLDDT); UniProt features', 'hear_celltype': 'inner-ear single-cell expression (Kolla et al. 2020 supplementary tables; GEO GSE135913; GEO GSE207817)',
           'hear_hpo': 'Human Phenotype Ontology annotations', 'hear_cgd': 'Clinical Genomic Database',
           'hear_impc': 'International Mouse Phenotyping Consortium', 'hear_network': 'STRING, Reactome, Complex Portal',
           'hear_constraint': 'gnomAD v4 gene constraint (MANE Select)'}
    rows = [{'group': g, 'feature': c, 'level': 'gene' if g.startswith('hear') else 'variant/residue', 'source': src.get(g, '')}
            for g, cols in groups.items() if g != 'hear' for c in cols]
    return pd.DataFrame(rows)


def s4():
    a = pd.read_csv(R / 'arms_per_seed.csv')
    return a.pivot_table(index=['arm', 'seed'], columns='partition', values=['auroc', 'auprc', 'auroc_within_gene'])\
        .round(4).reset_index()


def s5():
    """Predictors: properties, coverage per test set and membership of the common comparison panel (coverage rule)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import MIN_PANEL_COVERAGE, panel
    p = pd.read_csv(ROOT / 'manuscript/supp_table_S5_predictors.csv')
    cov = pd.read_csv(R / 'benchmark_pairwise.csv')
    c = cov.pivot_table(index='method', columns='partition', values='coverage').round(3)
    c.columns = [f'coverage_{x}' for x in c.columns]
    out = p.merge(c, left_on='benchmark_name', right_index=True, how='left')
    final = json.load(open(R / 'final_model.json'))['final_arm']
    members = set(panel(final)) | {'Gene prior (training label fraction)', 'Allele frequency only (-log10 AF)'}
    out['in_common_panel'] = out.benchmark_name.isin(members)
    out['panel_rule'] = f'dbNSFP predictors with coverage >= {MIN_PANEL_COVERAGE:.0%} in every test set'
    return out.drop(columns='benchmark_name')


def s6():
    """MaveDB score sets analysed (primary sets only), with the orientation used."""
    m = pd.read_csv(D / 'annotations/mave_scoresets_v8.csv')
    m = m[m.primary]
    keep = [c for c in ['urn', 'gene', 'assay_type', 'title', 'higher_score_means', 'n_missense', 'ref_start', 'ref_end',
                        'published_date', 'publication', 'license'] if c in m.columns]
    return m[keep] if keep else m


def s7():
    v = json.load(open(D / 'annotations/annotation_versions.json'))
    rows = [
        ('ClinVar variant_summary', 'monthly archive releases 2025-01 and 2026-04', 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/', 'downloaded April 2026'),
        ('Deafness Variation Database', 'web interface export', 'https://deafnessvariationdatabase.org', '21 April 2026'),
        ('PanelApp Hearing loss (panel 126)', 'green genes', 'https://panelapp.genomicsengland.co.uk', 'April 2026'),
        ('RefSeq protein sequences', 'NCBI E-utilities', 'https://eutils.ncbi.nlm.nih.gov', 'October 2026'),
        ('dbNSFP', v['myvariant_dbnsfp_gnomad']['dbnsfp']['version'], 'via MyVariant.info build ' +
         v['myvariant_dbnsfp_gnomad']['service']['build_version'], v['myvariant_dbnsfp_gnomad']['access_date']),
        ('gnomAD allele frequencies', 'v2.1.1 exomes (GRCh38 liftover), v3.1.1 genomes', 'via MyVariant.info', v['myvariant_dbnsfp_gnomad']['access_date']),
        ('gnomAD gene constraint', 'v4', 'https://gnomad.broadinstitute.org', 'April 2026'),
        ('AlphaFold Protein Structure Database', 'model v4 (canonical) and v6 (isoforms)', 'https://alphafold.ebi.ac.uk', 'April-May 2026'),
        ('UniProt', 'REST entries', 'https://rest.uniprot.org', 'April 2026'),
        ('DDGun folding free energies', 'Tollefson et al. 2023, Table S4', 'Hum Genet 142:819-834', 'April 2026'),
        ('MaveDB', '30 score sets (15 primary)', 'https://api.mavedb.org', 'October 2026'),
        ('ClinGen Evidence Repository', 'export of hearing-loss genes', 'https://erepo.clinicalgenome.org', 'May 2026'),
        ('ESM-2', 'facebook/esm2_t30_150M_UR50D, commit a695f604', 'https://huggingface.co', 'October 2026'),
        ('Mouse cochlear epithelium single-cell expression', 'Kolla et al. 2020, supplementary tables (hair and supporting cells)', 'published supplementary data', 'May 2026'),
        ('Mouse adult and human fetal inner-ear single-cell expression', 'GEO GSE135913 (processed matrix)', 'https://www.ncbi.nlm.nih.gov/geo/', 'May 2026'),
        ('Human utricle single-cell expression', 'GEO GSE207817 (organ-donor and vestibular-schwannoma utricles)', 'https://www.ncbi.nlm.nih.gov/geo/', 'April 2026'),
        ('Human Phenotype Ontology', 'genes_to_phenotype.txt (release current in April 2026)', 'https://hpo.jax.org', 'April 2026'),
        ('Clinical Genomic Database', 'CGD.txt (release current in May 2026)', 'https://research.nhgri.nih.gov/CGD/', 'May 2026'),
        ('International Mouse Phenotyping Consortium', 'genotype-phenotype associations, hearing and vestibular parameters (release current in April 2026)', 'https://www.mousephenotype.org', 'April 2026'),
        ('STRING, Reactome, Complex Portal', 'interaction partners, pathways and complexes per gene (releases current in April 2026)', 'https://string-db.org; https://reactome.org; https://www.ebi.ac.uk/complexportal', 'April 2026'),
    ]
    return pd.DataFrame(rows, columns=['resource', 'version', 'location', 'accessed'])


def s4b():
    return pd.read_csv(R / 'arm_contrasts.csv')


def s8b():
    return pd.read_csv(R / 'cv_per_residue_cv_full.csv')


def s9e():
    """HEARS-LM thresholds by calibration population, and in each cross-fitting fold (stability of the thresholds)."""
    pop = pd.read_csv(R / 'calibration_thresholds_by_population.csv')
    ft = pd.read_csv(R / 'evidence_crossfit_fold_thresholds.csv')
    ft = ft[ft.method == 'HEARS-LM'].copy()
    ft['calibration_population'] = 'pooled test sets without cross-fitting fold ' + ft.fold.astype(str)
    ft = ft.rename(columns={'n_calibration': 'n'})
    cols = ['calibration_population', 'n'] + [c for c in pop.columns if c.startswith(('PP3', 'BP4'))]
    return pd.concat([pop, ft[cols]], ignore_index=True)


def s9c():
    return pd.read_csv(R / 'evidence_crossfit.csv')


def s9d():
    return pd.read_csv(R / 'reclassification.csv')


def s8():
    return pd.read_csv(R / 'cv_per_gene_cv_full.csv')


def s9():
    t = pd.read_csv(R / 'calibration_thresholds.csv')
    q = pd.read_csv(R / 'evidence_by_partition.csv')
    return t, q


def s10():
    return pd.read_csv(R / 'vus_catalogue.csv')


def main():
    out = ROOT / 'manuscript/supplementary_tables.xlsx'
    t9, q9 = s9()
    cal_other = pd.concat([pd.read_csv(R / f'calibration_thresholds_{a}.csv') for a in ['seq_plm_ddg', 'feat_plm_only']])
    cf = pd.concat([s9c()] + [pd.read_csv(R / f'evidence_crossfit_{a}.csv') for a in ['seq_plm_ddg', 'feat_plm_only']])
    sheets = [('S1_genes', s1()), ('S2_counts_per_gene', s2()), ('S2b_mapping', s2b()), ('S3_resources', s7()),
              ('S4_descriptors', s3()),
              ('S5_all_runs', s4()), ('S5b_vs_selected_config', s4b()),
              ('S5c_component_contrasts', pd.read_csv(R / 'arm_pair_contrasts.csv')),
              ('S6_predictors', s5()),
              ('S6b_common_set_metrics', pd.read_csv(R / 'benchmark_common.csv')),
              ('S6c_common_set_paired', pd.read_csv(R / 'benchmark_common_pairwise.csv')),
              ('S6d_strata_identical_variants', pd.read_csv(R / 'benchmark_strata.csv')),
              ('S6e_pairwise_own_coverage', pd.read_csv(R / 'benchmark_pairwise.csv')),
              ('S7_MAVE_score_sets', s6()),
              ('S8_per_gene_CV', s8()), ('S8b_residue_level_CV', s8b()),
              ('S8c_CV_pooled', pd.read_csv(R / 'cv_overall_cv_full.csv')),
              ('S9_calibration', t9), ('S9b_thresholds_other_configs', cal_other),
              ('S9c_evidence_by_set', q9), ('S9d_evidence_crossfit', cf),
              ('S9e_thresholds_by_population', s9e()),
              ('S9f_expert_panel', pd.read_csv(R / 'expert_panel.csv')), ('S9g_reclassification', s9d()),
              ('S10_VUS_catalogue', s10())]
    with pd.ExcelWriter(out) as xw:
        for name, df in sheets:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(str(x) for x in c if x) for c in df.columns]
            df.to_excel(xw, sheet_name=name, index=False)
    print('wrote', out, [(n, len(d)) for n, d in sheets])


if __name__ == '__main__':
    main()
