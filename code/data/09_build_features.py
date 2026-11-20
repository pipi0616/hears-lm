#!/usr/bin/env python3
"""
Step 09 - Assemble the tabular descriptor table used by the feature arms of the model.

Groups (data/features/feature_groups.json)
  plm    (variant level)  full-context log-probabilities of the pretrained, frozen ESM-2 150M at the residue
                          (wild-type marginals, step 13): log-likelihood ratio of the substitution, log p(wt),
                          log p(mut), entropy. Sequence-derived: needs no external annotation.
  cons   (variant level)  conservation at the genomic position, from dbNSFP by GRCh38 coordinate
  struct (residue level)  AlphaFold2 pLDDT and UniProt sequence annotations at the residue (step 06)
  hear   (gene level)     inner-ear / hearing-loss biology of the gene: single-cell expression in cochlear and
                          vestibular cell types, HPO and CGD phenotype/inheritance annotations, IMPC mouse
                          auditory phenotypes, interaction/pathway counts, gnomAD gene constraint
                          (sub-groups hear_celltype, hear_hpo, hear_cgd, hear_impc, hear_network, hear_constraint)
  af     (variant level)  gnomAD allele frequency - DIAGNOSTIC ARM ONLY, never part of the released model

Nothing that the model predicts (pathogenicity label, ddG) and no other predictor's score is a feature.
Missing values stay NaN here (the training code imputes the training mean after standardisation); explicit
*_missing indicator columns are provided per group.

The gene-level constraint metrics are recomputed from the gnomAD v4 constraint table (MANE Select transcript);
that table has no X-chromosome genes, so the six X-linked panel genes have missing constraint (gnomad_in_db = 0).
The remaining gene-level descriptors are carried over unchanged from the original study's gene tables.

Usage
  python 09_build_features.py                                  # labelled variants -> data/features/features_v8.csv
  python 09_build_features.py --variants T.csv --dbnsfp D.csv --out F.csv   # any other variant table
     (T needs variant_id, gene, ref_pos, ref1, alt1; D is the matching step-05 annotation, optional:
      without it the cons and af groups are missing)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data'))
from paths import legacy_root  # noqa: E402

REV = Path(__file__).resolve().parents[2]          # the release folder (works wherever it is unpacked)
ROOT = REV.parents[1]
ANN = REV / 'data/annotations'
OUT = REV / 'data/features'
OUT.mkdir(parents=True, exist_ok=True)

# current HGNC symbols for panel genes that are stored under older symbols
ALIAS = {'DFNA5': 'GSDME', 'DFNB59': 'PJVK', 'KARS': 'KARS1', 'KIAA1024L': 'MINAR2', 'SPATA5': 'AFG2A',
         'SPATA5L1': 'AFG2B'}
STRUCT_COLS = ['plddt', 'plddt_missing', 'in_domain', 'in_transmembrane', 'in_signal_or_propeptide', 'in_disulfide',
               'in_binding_or_active_site', 'annot_missing', 'rel_pos']
PLM_COLS = ['plm_llr', 'plm_lp_wt', 'plm_lp_mut', 'plm_entropy']
CONS_COLS = ['cons_phylop100', 'cons_phylop_mammal', 'cons_phylop_primate', 'cons_phastcons100', 'cons_gerp_rs']


def constraint_table(genes):
    c = pd.read_csv(legacy_root() / 'data/external/gnomad_v4_constraint.tsv', sep='\t', low_memory=False)
    c = c[c.transcript.astype(str).str.startswith('ENST')]
    mane = c[c.mane_select.astype(str).str.lower() == 'true'].drop_duplicates('gene').set_index('gene')
    canon = c[c.canonical.astype(str).str.lower() == 'true'].drop_duplicates('gene').set_index('gene')
    rows = []
    for g in genes:
        r = None
        for sym in (ALIAS.get(g, g), g):          # current symbol first, then the symbol used in the panel
            if sym in mane.index:
                r = mane.loc[sym]; break
            if sym in canon.index:
                r = canon.loc[sym]; break
        if r is None:
            rows.append({'gene': g, 'gnomad_in_db': 0})
            continue
        rows.append({'gene': g, 'gnomad_in_db': 1,
                     'gnomad_lof_oe': r['lof.oe'], 'gnomad_lof_oe_upper': r['lof.oe_ci.upper'],
                     'gnomad_lof_pLI': r['lof.pLI'], 'gnomad_mis_oe': r['mis.oe'],
                     'gnomad_mis_oeuf': r['mis.oe_ci.upper'], 'gnomad_mis_z': r['mis.z_score'],
                     'gnomad_mis_pphen_oe': r['mis_pphen.oe'], 'gnomad_syn_z': r['syn.z_score']})
    t = pd.DataFrame(rows)
    for col in t.columns:
        if col != 'gene':
            t[col] = pd.to_numeric(t[col], errors='coerce')
    t['gnomad_loeuf_constrained'] = (t.gnomad_lof_oe_upper < 0.6).astype(float).where(t.gnomad_lof_oe_upper.notna())
    t['gnomad_pLI_high'] = (t.gnomad_lof_pLI > 0.9).astype(float).where(t.gnomad_lof_pLI.notna())
    t['gnomad_mis_constrained'] = (t.gnomad_mis_z > 3.09).astype(float).where(t.gnomad_mis_z.notna())
    return t


def plm_features(v):
    w = pd.read_csv(ANN / 'esm2_wt_marginals.csv.gz')
    t = v[['variant_id', 'gene', 'ref_pos', 'ref1', 'alt1']].merge(w, on=['gene', 'ref_pos'], how='left')
    assert (t.ref_aa == t.ref1).all(), 'reference residue mismatch with the ESM-2 residue table'
    rec = t.to_dict('records')
    t['plm_lp_wt'] = [r[f"lp_{r['ref1']}"] for r in rec]
    t['plm_lp_mut'] = [r[f"lp_{r['alt1']}"] for r in rec]
    t['plm_llr'] = t.plm_lp_mut - t.plm_lp_wt
    t['plm_entropy'] = t.entropy
    return t[['variant_id'] + PLM_COLS]


def gene_descriptor_table(genes):
    g = pd.read_csv(ANN / 'gene_features_v8.csv')
    gg = json.load(open(ANN / 'gene_feature_groups.json'))
    keep = gg['celltype'] + gg['hpo'] + gg['cgd'] + gg['impc'] + gg['network']
    cache = OUT / 'gene_descriptors_v8.csv'
    if cache.exists():                      # cached per gene; the raw constraint table is not needed again
        c = pd.read_csv(cache)
        if set(genes) <= set(c.gene):
            cols = [x for x in c.columns if x.startswith('gnomad_')]
            ct = c.loc[c.gene.isin(genes), ['gene'] + cols]
        else:
            ct = constraint_table(genes)
    else:
        ct = constraint_table(genes)
    tab = pd.DataFrame({'gene': genes}).merge(g[['gene'] + keep], on='gene', how='left').merge(ct, on='gene', how='left')
    tab['gene_desc_missing'] = tab[gg['celltype'][0]].isna().astype(float)
    constraint_cols = [c for c in ct.columns if c != 'gene']
    groups = {'hear_celltype': gg['celltype'], 'hear_hpo': gg['hpo'], 'hear_cgd': gg['cgd'], 'hear_impc': gg['impc'],
              'hear_network': gg['network'], 'hear_constraint': constraint_cols,
              'hear': keep + constraint_cols + ['gene_desc_missing']}
    return tab, groups


def assemble(v, dbnsfp=None, want=None):
    """Feature table (one row per variant_id of v) and the group definitions.
    want: names of the groups that are needed; others are skipped (so scoring a sequence-only model does not
    touch the annotation tables of groups it does not use). None = build everything."""
    def needed(*names):
        return want is None or any(n in want or any(w.startswith(n) for w in want) for n in names)
    groups = {}
    f = v[['variant_id', 'gene', 'ref_pos', 'ref1', 'alt1']].copy()
    # ---- conservation + allele frequency (dbNSFP by genomic coordinate)
    if not needed('cons', 'af'):
        dbnsfp = None
    if dbnsfp is not None:
        d = dbnsfp[['variant_id'] + [c for c in dbnsfp.columns if c.startswith(('cons_', 'af_'))]]
        f = f.merge(d, on='variant_id', how='left')
    for c in CONS_COLS:
        if c not in f:
            f[c] = np.nan
    f['cons_missing'] = f[CONS_COLS].isna().all(axis=1).astype(float)
    groups['cons'] = CONS_COLS + ['cons_missing']
    af_any = f['af_any'] if 'af_any' in f else pd.Series(np.nan, index=f.index)
    f['afx_log10'] = np.log10(af_any.fillna(0) + 1e-7)
    f['afx_present'] = f['af_present'].fillna(0).astype(float) if 'af_present' in f else af_any.notna().astype(float)
    f['afx_ge_0.005'] = (af_any.fillna(0) >= 0.005).astype(float)
    groups['af'] = ['afx_log10', 'afx_present', 'afx_ge_0.005']
    f = f.drop(columns=[c for c in f.columns if c.startswith('af_')])
    # ---- structure / sequence annotation at the residue
    if needed('struct'):
        s = pd.read_csv(ANN / 'residue_structure_table.csv', low_memory=False)
        f = f.merge(s[['gene', 'ref_pos'] + STRUCT_COLS], on=['gene', 'ref_pos'], how='left')
        groups['struct'] = STRUCT_COLS
    # ---- full-context language-model log-probabilities (frozen ESM-2 150M)
    if needed('plm'):
        f = f.merge(plm_features(v), on='variant_id', how='left')
        groups['plm'] = PLM_COLS
        assert f[PLM_COLS].notna().all().all(), 'missing language-model features'
    # ---- gene-level hearing biology
    gene_tab = None
    if needed('hear'):
        gene_tab, hg = gene_descriptor_table(sorted(f.gene.unique()))
        f = f.merge(gene_tab, on='gene', how='left')
        groups.update(hg)
    feat_cols = sorted({c for cols in groups.values() for c in cols})
    assert f.variant_id.is_unique and len(f) == len(v)
    return f[['variant_id'] + feat_cols], groups, gene_tab


def qc(f, v, groups):
    from sklearn.metrics import roc_auc_score
    f = f.merge(v[['variant_id', 'partition', 'y', 'source']], on='variant_id')
    feat_cols = sorted({c for cols in groups.values() for c in cols})
    summ = {'n_features': {k: len(c) for k, c in groups.items()}, 'missing_by_partition': {}}
    for grp in ['plm', 'cons', 'struct', 'hear', 'af']:
        raw = [c for c in groups[grp] if not c.endswith('_missing') and c not in ('afx_present',)]
        summ['missing_by_partition'][grp] = {p: float(f.loc[f.partition == p, raw].isna().all(axis=1).mean())
                                             for p in ['fit', 'dev', 'test_db', 'test_time', 'test_gene']}
    fit = f[f.partition == 'fit']
    src = (fit.source == 'dvd').astype(int)
    rows = []
    for c in feat_cols:
        x = fit[c]
        if x.notna().sum() < 50 or x.nunique() < 2:
            continue
        xf = x.fillna(x.mean())
        a = roc_auc_score(fit.y, xf)
        b = roc_auc_score(src, xf) if src.nunique() == 2 else np.nan
        m = roc_auc_score(fit.y, x.isna().astype(int)) if x.isna().any() else 0.5
        rows.append({'feature': c, 'auroc_label': max(a, 1 - a), 'auroc_source_dvd': max(b, 1 - b),
                     'auroc_label_from_missingness': max(m, 1 - m), 'missing_frac': float(x.isna().mean())})
    q = pd.DataFrame(rows).sort_values('auroc_label', ascending=False)
    q.to_csv(OUT / 'feature_qc.csv', index=False)
    summ['top_label_auroc'] = q.head(12)[['feature', 'auroc_label']].to_dict('records')
    summ['top_missingness_auroc'] = q.sort_values('auroc_label_from_missingness', ascending=False).head(6)[
        ['feature', 'auroc_label_from_missingness', 'missing_frac']].to_dict('records')
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants')
    ap.add_argument('--dbnsfp')
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.variants:
        v = pd.read_csv(a.variants, low_memory=False)
        d = pd.read_csv(a.dbnsfp, low_memory=False) if a.dbnsfp else None
        f, groups, _ = assemble(v, d)
        f.to_csv(a.out, index=False)
        print(f'wrote {a.out}: {f.shape}; missing cons {f.cons_missing.mean():.3f}')
        return
    v = pd.read_csv(REV / 'data/variants_v8.csv', low_memory=False)
    d = pd.read_csv(ANN / 'dbnsfp_v8.csv', low_memory=False)
    f, groups, gene_tab = assemble(v, d)
    f.to_csv(OUT / 'features_v8.csv', index=False)
    gene_tab.to_csv(OUT / 'gene_descriptors_v8.csv', index=False)
    json.dump(groups, open(OUT / 'feature_groups.json', 'w'), indent=1)
    summ = qc(f, v, groups)
    json.dump(summ, open(OUT / 'features_summary.json', 'w'), indent=1)
    print(json.dumps(summ['n_features']))


if __name__ == '__main__':
    main()
