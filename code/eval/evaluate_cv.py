#!/usr/bin/env python3
"""
Evaluation of the leave-gene-family-out cross-validation (every variant is scored by a model that never
saw its gene or any gene of the same homology family).

Outputs (results/)
  cv_oof_<arm>.csv        out-of-fold predictions joined with labels, gene, baselines (for figures)
  cv_overall_<arm>.csv    pooled AUROC / AUPRC (variant bootstrap and gene-cluster bootstrap CI), within-gene AUROC,
                          per method, on each method's variants and on the common subset
  cv_per_gene_<arm>.csv   genes with >= MIN_P pathogenic and >= MIN_B benign missense variants:
                          AUROC of HEARS-LM and baselines on identical variants, difference with bootstrap CI,
                          DeLong p, Benjamini-Hochberg q (across genes, per baseline)
  cv_per_gene_summary_<arm>.csv   per baseline: median per-gene AUROCs, Wilcoxon signed-rank p, wins / losses at q<0.05
  cv_inheritance_<arm>.csv        pooled and within-gene AUROC by mode of inheritance of the gene
  cv_per_residue_<arm>.csv        sensitivity analysis at the residue level (labelled residues only)

Usage: python evaluate_cv.py --arm cv_seq_ddg [--nboot 2000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from common import AF_ONLY, RESULTS, RUNS, load_baselines, load_variants  # noqa: E402
from common import panel as comparison_panel  # noqa: E402

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data'))
from paths import legacy_root  # noqa: E402

MIN_P, MIN_B = 10, 10
PANEL = Path(__file__).resolve().parents[2] / 'data/sources/gene_panel_hhl_green.csv'   # PanelApp panel 126, green genes


def inheritance_class(s):
    s = str(s).upper()
    if 'X-LINKED' in s:
        return 'X-linked'
    if 'BOTH' in s:
        return 'AD and AR'
    if 'BIALLELIC' in s:
        return 'AR'
    if 'MONOALLELIC' in s:
        return 'AD'
    return 'other'


def secondary(a, y, s, genes, ext, full, methods, eligible):
    """Results by mode of inheritance and at the residue level, for HEARS-LM and three reference predictors."""
    key3 = [m for m in ['AlphaMissense', 'REVEL', 'ESM-2 150M zero-shot'] if m in methods]
    # ---------------- inheritance
    panel = pd.read_csv(PANEL)
    inh = dict(zip(panel.gene, panel.mode_of_inheritance.map(inheritance_class)))
    cls = np.array([inh.get(g, 'other') for g in genes])
    rows = []
    for c in ['AR', 'AD', 'AD and AR', 'X-linked']:
        m = cls == c
        if m.sum() < 50 or len(np.unique(y[m])) < 2:
            continue
        for name, sc in [('HEARS-LM', s)] + [(n, ext[n].to_numpy()) for n in key3]:
            k = m & ~np.isnan(sc)
            ci = M.bootstrap_ci(y[k], sc[k], M.auroc, a.nboot)
            rows.append({'inheritance': c, 'method': name, 'n': int(k.sum()), 'n_P': int(y[k].sum()),
                         'genes': int(len(np.unique(genes[k]))), 'auroc': ci['est'], 'auroc_lo': ci['lo'],
                         'auroc_hi': ci['hi'], 'within_gene_auroc': M.within_group_auroc(y[k], sc[k], genes[k])})
    pd.DataFrame(rows).to_csv(RESULTS / f'cv_inheritance_{a.arm}.csv', index=False)
    print(pd.DataFrame(rows).round(4).to_string())

    # ---------------- residue level (labelled residues only): positive = any P/LP, negative = only B/LB
    res = full.rename(columns={'prob': 'HEARS-LM'}).copy()
    res['res'] = res.gene + ':' + res.ref_pos.astype(str)
    agg = res.groupby('res').agg(gene=('gene', 'first'), y=('y', 'max'), n_var=('y', 'size'),
                                 **{name: (name, 'mean') for name in ['HEARS-LM'] + methods})
    rows = []
    for g in eligible:
        d = agg[agg.gene == g]
        if (d.y == 1).sum() < 5 or (d.y == 0).sum() < 5:
            continue
        r = {'gene': g, 'n_residues': len(d), 'n_pos_residues': int((d.y == 1).sum())}
        for name in ['HEARS-LM'] + key3:
            k = d[name].notna()
            r[f'auroc_{name}'] = M.auroc(d.y[k], d[name][k]) if k.sum() >= 10 else np.nan
        rows.append(r)
    pd.DataFrame(rows).to_csv(RESULTS / f'cv_per_residue_{a.arm}.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--nboot', type=int, default=2000)
    ap.add_argument('--secondary-only', action='store_true',
                    help='recompute only the inheritance and residue-level tables')
    a = ap.parse_args()
    v = load_variants().set_index('variant_id')
    folds = sorted((RUNS / a.arm).glob('fold*/preds_cvtest.csv'))
    assert folds, f'no CV predictions for {a.arm}'
    oof = pd.concat([pd.read_csv(f).assign(fold=f.parent.name) for f in folds], ignore_index=True)
    assert oof.variant_id.is_unique
    oof = oof.set_index('variant_id')
    oof['gene'] = v.gene.reindex(oof.index)
    oof['ref_pos'] = v.ref_pos.reindex(oof.index)
    base, af = load_baselines()
    ext = base.reindex(oof.index)
    import json as _json
    key_baselines = comparison_panel(_json.load(open(RESULTS / 'final_model.json'))['final_arm'])   # same panel as the test sets
    full = pd.concat([oof[['y', 'prob', 'gene', 'ref_pos', 'fold'] + (['ddg_pred'] if 'ddg_pred' in oof else [])], ext], axis=1)
    full.rename(columns={'prob': 'HEARS-LM'}).to_csv(RESULTS / f'cv_oof_{a.arm}.csv')
    print(f'{a.arm}: {len(oof)} out-of-fold predictions from {len(folds)} folds; genes={oof.gene.nunique()}')

    y = oof.y.to_numpy(); s = oof.prob.to_numpy(); genes = oof.gene.to_numpy()
    methods = [m for m in key_baselines if m in ext.columns]
    af_only = -np.log10(af['af_any'].reindex(oof.index).fillna(0) + 1e-7).to_numpy()   # reference score, overall only

    if a.secondary_only:
        cnt = oof.groupby('gene').y.agg(['sum', 'size'])
        eligible = cnt[(cnt['sum'] >= MIN_P) & (cnt['size'] - cnt['sum'] >= MIN_B)].index.tolist()
        pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
        secondary(a, y, s, genes, ext, full, methods, eligible)
        return

    # ---------------- overall
    rows = []
    core_mask = ext[methods].notna().all(axis=1).to_numpy()       # variants scored by every method compared
    for subset, mask in [('all variants of the method', None), ('common subset', core_mask)]:
        for name, sc in [('HEARS-LM', s)] + [(m, ext[m].to_numpy()) for m in methods] + [(AF_ONLY, af_only)]:
            k = ~np.isnan(sc) if mask is None else (mask & ~np.isnan(sc))
            if k.sum() < 50:
                continue
            ci = M.bootstrap_ci(y[k], sc[k], M.auroc, a.nboot)
            cc = M.cluster_bootstrap_ci(y[k], sc[k], genes[k], M.auroc, a.nboot)
            cp = M.bootstrap_ci(y[k], sc[k], M.auprc, a.nboot)
            r = {'subset': subset, 'method': name, 'n': int(k.sum()), 'n_P': int(y[k].sum()),
                 'auroc': ci['est'], 'auroc_lo': ci['lo'], 'auroc_hi': ci['hi'],
                 'auroc_gene_cluster_lo': cc['lo'], 'auroc_gene_cluster_hi': cc['hi'],
                 'auprc': cp['est'], 'auprc_lo': cp['lo'], 'auprc_hi': cp['hi'],
                 'within_gene_auroc': M.within_group_auroc(y[k], sc[k], genes[k])}
            if name != 'HEARS-LM':
                pb = M.paired_bootstrap_diff(y[k], s[k], sc[k], M.auroc, a.nboot)
                dl = M.delong_test(y[k], s[k], sc[k])
                wg = M.within_group_auroc(y[k], s[k], genes[k])
                r.update({'hearslm_auroc_same_variants': dl['auc1'], 'hearslm_within_gene_same_variants': wg,
                          'diff_hearslm_minus_method': pb['diff'], 'diff_lo': pb['lo'], 'diff_hi': pb['hi'],
                          'p_delong': dl['p']})
            rows.append(r)
    ov = pd.DataFrame(rows)
    ov.to_csv(RESULTS / f'cv_overall_{a.arm}.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(ov[['subset', 'method', 'n', 'auroc', 'auroc_lo', 'auroc_hi', 'within_gene_auroc', 'diff_hearslm_minus_method',
              'p_delong']].round(4).to_string())

    # ---------------- per gene
    cnt = oof.groupby('gene').y.agg(['sum', 'size'])
    eligible = cnt[(cnt['sum'] >= MIN_P) & (cnt['size'] - cnt['sum'] >= MIN_B)].index.tolist()
    rows = []
    for g in eligible:
        m = genes == g
        yy, ss = y[m], s[m]
        base_row = {'gene': g, 'n': int(m.sum()), 'n_P': int(yy.sum()), 'n_B': int((yy == 0).sum()),
                    'fold': oof.fold[m].iloc[0]}
        ci = M.bootstrap_ci(yy, ss, M.auroc, a.nboot)
        base_row.update({'hearslm_auroc': ci['est'], 'hearslm_lo': ci['lo'], 'hearslm_hi': ci['hi']})
        for name in methods:
            b = ext[name].to_numpy()[m]
            k = ~np.isnan(b)
            r = dict(base_row); r['baseline'] = name; r['n_common'] = int(k.sum())
            if k.sum() >= 20 and (yy[k] == 1).sum() >= 5 and (yy[k] == 0).sum() >= 5:
                pb = M.paired_bootstrap_diff(yy[k], ss[k], b[k], M.auroc, a.nboot)
                dl = M.delong_test(yy[k], ss[k], b[k])
                r.update({'hearslm_auroc_common': dl['auc1'], 'baseline_auroc': dl['auc2'], 'diff': pb['diff'],
                          'diff_lo': pb['lo'], 'diff_hi': pb['hi'], 'z_delong': dl['z'], 'p_delong': dl['p']})
            rows.append(r)
    pg = pd.DataFrame(rows)
    pg['q_bh'] = np.nan
    for name in methods:
        k = pg.baseline == name
        pg.loc[k, 'q_bh'] = M.bh_fdr(pg.loc[k, 'p_delong'].to_numpy())
    pg.to_csv(RESULTS / f'cv_per_gene_{a.arm}.csv', index=False)
    srows = []
    for name in methods:
        d = pg[(pg.baseline == name) & pg['diff'].notna()]
        if len(d) < 5:
            continue
        w = wilcoxon(d.hearslm_auroc_common, d.baseline_auroc)
        srows.append({'baseline': name, 'n_genes': len(d), 'median_hearslm': d.hearslm_auroc_common.median(),
                      'median_baseline': d.baseline_auroc.median(), 'median_diff': d['diff'].median(),
                      'mean_diff': d['diff'].mean(), 'p_wilcoxon': float(w.pvalue),
                      'n_hearslm_better_q05': int(((d.q_bh < 0.05) & (d['diff'] > 0)).sum()),
                      'n_baseline_better_q05': int(((d.q_bh < 0.05) & (d['diff'] < 0)).sum()),
                      'n_hearslm_higher': int((d['diff'] > 0).sum()), 'n_baseline_higher': int((d['diff'] < 0).sum())})
    ps = pd.DataFrame(srows)
    ps.to_csv(RESULTS / f'cv_per_gene_summary_{a.arm}.csv', index=False)
    print(f'\nper-gene analysis: {len(eligible)} genes with >= {MIN_P} P and >= {MIN_B} B')
    print(ps.round(4).to_string())

    secondary(a, y, s, genes, ext, full, methods, eligible)


if __name__ == '__main__':
    main()
