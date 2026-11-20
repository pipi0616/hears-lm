#!/usr/bin/env python3
"""Main tables of the manuscript, generated from the results files (manuscript/sections/06_tables.md)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'results'


def md(df):
    cols = list(df.columns)
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows():
        out.append('| ' + ' | '.join('' if pd.isna(x) else str(x) for x in r) + ' |')
    return '\n'.join(out)


def frac_at_training_residue():
    """Exact fraction per partition from the variant table (the stored value is rounded to three decimals)."""
    v = pd.read_csv(ROOT / 'data/variants_v8.csv', usecols=['partition', 'pos_in_fit'], low_memory=False)
    return v.groupby('partition').pos_in_fit.mean()


def table1():
    t = pd.read_csv(R / 'table_partitions.csv')
    exact = frac_at_training_residue()
    t['frac_at_training_residue'] = [exact.get(p, x) if pd.notna(x) else x
                                     for p, x in zip(t.partition, t.frac_at_training_residue)]
    names = {'fit': 'Training (fit)', 'dev': 'Development', 'test_time': 'Temporal test (test_time)',
             'test_db': 'Literature-evidence test (test_db)', 'test_gene': 'Withheld-gene test (test_gene)',
             'cross-validation pool': 'Leave-gene-family-out cross-validation'}
    d = pd.DataFrame({
        'Set': t.partition.map(names),
        'Definition': t.definition,
        'Variants': t.n.map('{:,}'.format),
        'P/LP': t.P_LP.map('{:,}'.format),
        'B/LB': t.B_LB.map('{:,}'.format),
        'Genes': t.genes,
        'At a training residue': t.frac_at_training_residue.map(lambda x: '' if pd.isna(x) else f'{x:.0%}'),
    })
    return ('**Table 1. Labelled data and partitions.** Partitions are disjoint at the protein-variant level. '
            '"At a training residue": fraction of variants located at a residue position that also carries a '
            'training variant. P/LP, pathogenic or likely pathogenic; B/LB, benign or likely benign.\n\n' + md(d))


def table2():
    cm = pd.read_csv(R / 'benchmark_common.csv')
    n_methods = cm[cm.partition == 'test_time'].method.nunique()
    n_db = n_methods - 4                           # minus HEARS-LM, ESM-2 zero-shot, gene prior, allele frequency
    mv = pd.read_csv(R / 'mave_correlations.csv')
    dom = mv[(mv.subset == 'common') & (mv.assay_type == 'domainome abundance')]
    ncl = pd.read_csv(R / 'evidence_by_partition.csv').method.nunique() - 1
    rows = [
        ('Development', 'Model selection only', 'P/LP vs B/LB', 'HEARS-LM configurations', 'Development partition'),
        ('Temporal test', 'New variants in panel genes', 'ClinVar 2026-04, first definitive after 2025-01',
         f'HEARS-LM, {n_db} dbNSFP predictors, zero-shot ESM-2, gene prior, allele frequency ({n_methods} methods)',
         'Scored by all methods of the panel (coverage >= 90% in every test set); other predictors pairwise'),
        ('Withheld genes', 'Genes absent from training', 'ClinVar 2026-04 and DVD', 'As above', 'As above'),
        ('Literature evidence', 'Curated labels without ClinVar classification', 'DVD (95% P/LP; AUPRC)', 'As above',
         'As above'),
        ('Gene-family CV', 'Genes and homologues unseen', 'Final labels', 'HEARS-LM configurations, dbNSFP predictors, '
         'zero-shot ESM-2', 'Out-of-fold; per gene >= 10 P/LP and >= 10 B/LB'),
        ('Thermodynamic head', 'Fidelity of the distilled estimate', 'DDGun (Tollefson et al.)', 'Auxiliary head',
         'Withheld-gene variants with DDGun'),
        ('Experimental assays', 'Agreement free of clinical labels',
         f'Domainome abundance ({dom.urn.nunique()} assays, {dom.gene.nunique()} proteins), KCNQ4 current, KCNE1 function',
         'Sequence-only HEARS-LM (CV models) and its head, AlphaMissense, zero-shot ESM-2; DDGun',
         'Scored by the first three; DDGun on its coverage'),
        ('Evidence calibration', 'ACMG/AMP evidence strength', 'Local posterior, prior 0.0441',
         f'HEARS-LM and {ncl} predictors', 'Pooled test variants scored by all of these; five-fold cross-fitting'),
        ('Expert panel', 'Agreement with ClinGen HL-VCEP', 'HL-VCEP P/LP vs B/LB', 'As above',
         'Out-of-fold scores only; scored by all methods'),
        ('Reclassification', 'Prospective-style', 'VUS/conflicting 2025-01 -> definitive 2026-04; VUS -> P/LP vs still VUS',
         'As above', 'Scored by all methods; ROC and precision-recall'),
    ]
    d = pd.DataFrame(rows, columns=['Benchmark', 'Question', 'Reference labels', 'Methods compared',
                                    'Variant set and missing scores'])
    return ('**Table 2. Benchmark specifications.** Main comparisons use the variants scored by every method of the '
            'common panel; coverage of each method is given in Supplementary Table S6.\n\n' + md(d))


def table3():
    t = pd.read_csv(R / 'table_components.csv')
    keep = ['configuration', 'seeds', 'dev', 'test_time', 'test_gene', 'test_db', 'cross_validation']
    d = t[[c for c in keep if c in t]].rename(columns={
        'configuration': 'Configuration', 'seeds': 'Seeds', 'dev': 'Development', 'test_time': 'Temporal test',
        'test_gene': 'Withheld genes', 'test_db': 'Literature evidence',
        'cross_validation': 'Leave-family-out CV'})
    d['Configuration'] = d['Configuration'].str.replace('^ +', lambda m: ' ' * (len(m.group(0)) // 2), regex=True)
    after = t.arm.isin(['feat_only', 'frozen', 'feat_plm_only', 'seq_t12']).to_numpy()
    d.loc[after, 'Configuration'] = d.loc[after, 'Configuration'] + ' *'
    cap = ('**Table 3. Contribution of each component.** Each row is a separately trained configuration. Values are '
           'AUROC on all variants of each partition, mean ± standard deviation across random seeds (Seeds: number of '
           'seeds); cross-validation: one model per fold, pooled out-of-fold predictions. The comparisons with other '
           'methods (Figures 3 and 4) use the seed ensemble on the variants scored by every method, so their values '
           'differ. * Pre-specified control, trained after the selection had been recorded and not eligible '
           'for selection; the cross-validation of the configuration without fine-tuning was added after the test '
           'evaluation (Supplementary Note S1). The '
           'allele-frequency row is a diagnostic and is not part of the selected model.')
    return cap + '\n\n' + md(d)


def main():
    out = '## Tables\n\n' + table1() + '\n\n' + table2() + '\n\n' + table3() + '\n'
    (ROOT / 'manuscript/sections').mkdir(parents=True, exist_ok=True)
    (ROOT / 'manuscript/sections/06_tables.md').write_text(out)
    print(out[:1500])


if __name__ == '__main__':
    main()
