#!/usr/bin/env python3
"""
Data reconciliation tables (one source of truth for every count quoted in the manuscript).

Outputs (results/)
  table_partitions.csv      per partition: definition, n, P, B, genes, residue positions, label source,
                            fraction of variants at a residue that also carries a training (fit) variant
  table_per_gene.csv        per gene: labelled variants per partition, CV fold, gene family, inheritance
  data_flow.json            counts for the data-flow figure (sources -> filters -> partitions)
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, load_variants  # noqa: E402

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data'))
from paths import legacy_root  # noqa: E402

PANEL = Path(__file__).resolve().parents[2] / 'data/sources/gene_panel_hhl_green.csv'   # PanelApp panel 126, green genes
DEFS = {
    'fit': 'training: ClinVar 2025-01 (>=1 star) and DVD evidence-based labels, 90% of residue positions',
    'dev': 'development: remaining 10% of residue positions of the training pool (model selection only)',
    'test_time': 'definitive ClinVar classification first present in 2026-04 (absent, VUS, conflicting or 0-star in 2025-01)',
    'test_db': 'DVD literature-curated (P/LP >= 2 PMIDs; B/LB >= 1 PMID, not MAF-based) without a definitive ClinVar '
               'classification in 2025-01',
    'test_gene': 'all labelled variants of 10 genes withheld from training',
}


def main():
    v = load_variants()
    ext = json.load(open(DATA / 'sources/extract_summary.json'))
    can = json.load(open(DATA / 'sources/canonicalize_summary.json'))
    spl = json.load(open(DATA / 'split_summary.json'))
    cvf = json.load(open(DATA / 'cv_folds_summary.json'))
    rows = []
    for p in ['fit', 'dev', 'test_time', 'test_db', 'test_gene']:
        d = v[v.partition == p]
        rows.append({'partition': p, 'definition': DEFS[p], 'n': len(d), 'P_LP': int((d.y == 1).sum()),
                     'B_LB': int((d.y == 0).sum()), 'frac_P': round(float(d.y.mean()), 3), 'genes': d.gene.nunique(),
                     'residues': d.pos_key.nunique(),
                     'label_source_clinvar_only': int((d.source == 'clinvar').sum()),
                     'label_source_dvd_only': int((d.source == 'dvd').sum()),
                     'label_source_both': int((d.source == 'clinvar+dvd').sum()),
                     'label_source_clinvar2025_only': int((d.source == 'clinvar2025_only').sum()),
                     'frac_at_training_residue': round(float(d.pos_in_fit.mean()), 3) if p != 'fit' else None})
    cv = v[v.cv_fold >= 0]
    rows.append({'partition': 'cross-validation pool', 'definition': 'all variants with a final (2026-04) label; '
                 '5 folds of whole gene families', 'n': len(cv), 'P_LP': int((cv.y_final == 1).sum()),
                 'B_LB': int((cv.y_final == 0).sum()), 'frac_P': round(float(cv.y_final.mean()), 3),
                 'genes': cv.gene.nunique(), 'residues': cv.pos_key.nunique()})
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / 'table_partitions.csv', index=False)
    print(t.drop(columns=['definition']).to_string())

    fam = pd.read_csv(DATA / 'gene_families_v8.csv')
    panel = pd.read_csv(PANEL)[['gene', 'mode_of_inheritance']]
    g = pd.crosstab(v.gene, v.partition)
    g['P_total'] = v[v.y_final == 1].groupby('gene').size()
    g['B_total'] = v[v.y_final == 0].groupby('gene').size()
    g = g.fillna(0).astype(int).reset_index().merge(fam[['gene', 'family_id', 'family_genes', 'cv_fold']], on='gene',
                                                    how='left').merge(panel, on='gene', how='left')
    g.to_csv(RESULTS / 'table_per_gene.csv', index=False)

    flow = {'panel_genes_with_protein': ext['panel_genes'],
            'clinvar_2026_04_missense_records': ext['clinvar_2026-04']['by_vclass']['missense'],
            'clinvar_2025_01_missense_records': ext['clinvar_2025-01']['by_vclass']['missense'],
            'clinvar_2026_04_missense_vus_records': ext['clinvar_2026-04']['missense_VUS'],
            'clinvar_2026_04_missense_conflicting_records': ext['clinvar_2026-04']['missense_conflicting'],
            'dvd_missense_records': ext['dvd']['missense_rows'],
            'dvd_prediction_only_benign_excluded': spl.get('dvd_prediction_only'),
            'dvd_clinvar_only_not_independent': spl.get('dvd_clinvar_only'),
            'canonicalisation': can,
            'dvd_labels_overridden_by_consensus_vus': spl['dvd_labels_overridden_by_consensus_vus'],
            'cross_source_conflicts_removed_final': spl['cross_source_conflict_final'],
            'cross_source_conflicts_removed_train': spl['cross_source_conflict_train'],
            'labelled_protein_variants': int(len(v)), 'genes_with_labels': int(v.gene.nunique()),
            'partitions': {r['partition']: {'n': r['n'], 'P': r['P_LP'], 'B': r['B_LB'], 'genes': r['genes']}
                           for r in rows},
            'gene_families': {'n_families': cvf['n_families'], 'n_multi_gene_families': cvf['n_multi_gene_families']},
            'cv_folds': cvf['folds'],
            'holdout_genes_with_homologue_in_training': cvf['holdout_genes_with_family_member_in_fit']}
    json.dump(flow, open(RESULTS / 'data_flow.json', 'w'), indent=1, default=str)
    print(json.dumps({k: flow[k] for k in flow if k not in ('canonicalisation', 'partitions', 'cv_folds')}, indent=1,
                     default=str))


if __name__ == '__main__':
    main()
