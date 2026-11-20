#!/usr/bin/env python3
"""Figure 4 - leave-gene-family-out cross-validation: every gene is scored by a model that saw neither the gene
nor any homologous panel gene.
a: pooled AUROC with variant-level and gene-cluster bootstrap intervals, per method, on the common variant set.
b: per-gene AUROC of HEARS-LM against the unadapted backbone (effect of domain adaptation).
c: per-gene difference against AlphaMissense, ordered, with Benjamini-Hochberg significance.
d: AUROC by mode of inheritance."""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import GREY, GRID, INK, INK2, ROOT, C, panel_label, save

sys.path.insert(0, str(ROOT / 'code/eval'))
R = ROOT / 'results'


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else 'cv_full'
    ov = pd.read_csv(R / f'cv_overall_{arm}.csv')
    pg = pd.read_csv(R / f'cv_per_gene_{arm}.csv')
    inh = pd.read_csv(R / f'cv_inheritance_{arm}.csv') if (R / f'cv_inheritance_{arm}.csv').exists() else None
    d = ov[ov.subset == 'common subset'].sort_values('auroc', ascending=False)

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], width_ratios=[1, 1.15], hspace=0.45, wspace=0.3,
                          left=0.22, right=0.97, top=0.95, bottom=0.09)
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        c = C.get(r.method, C['HEARS-LM'] if r.method == 'HEARS-LM' else GREY)
        ax.plot([r.auroc_gene_cluster_lo, r.auroc_gene_cluster_hi], [yi, yi], lw=1.0, color=c, alpha=0.35)
        ax.plot([r.auroc_lo, r.auroc_hi], [yi, yi], lw=2.2, color=c, alpha=0.55, solid_capstyle='butt')
        ax.plot(r.auroc, yi, 'o', ms=4.5, color=c, mec='white', mew=0.5, zorder=3)
    ax.set_yticks(y, d.method, fontsize=6.3)
    ax.set_xlabel('AUROC (out-of-fold)')
    ax.set_title(f'n = {int(d.n.iloc[0]):,} variants, {int(d.n_P.iloc[0]):,} P/LP', fontsize=7)
    ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.text(0.0, -0.22, 'thick interval: bootstrap over variants; thin: bootstrap over whole genes',
            transform=ax.transAxes, fontsize=5.5, color=INK2, va='top')
    panel_label(ax, 'a', x=-0.75, y=1.1)

    ax = fig.add_subplot(gs[0, 1])
    zs = pg[(pg.baseline == 'ESM-2 150M zero-shot') & pg.baseline_auroc.notna()]
    sig = zs.q_bh < 0.05
    lo = min(zs.baseline_auroc.min(), zs.hearslm_auroc_common.min()) - 0.03
    ax.plot([lo, 1.0], [lo, 1.0], color=GREY, lw=0.8, ls='--')
    ax.scatter(zs.baseline_auroc[~sig], zs.hearslm_auroc_common[~sig], s=14, facecolor='white',
               edgecolor=C['HEARS-LM'], lw=0.8, label='not significant')
    ax.scatter(zs.baseline_auroc[sig], zs.hearslm_auroc_common[sig], s=16, color=C['HEARS-LM'], lw=0,
               label='q < 0.05')
    n_up = int((zs['diff'] > 0).sum())
    ax.set_xlabel('ESM-2 150M zero-shot AUROC (unadapted backbone)', fontsize=7)
    ax.set_ylabel('HEARS-LM AUROC', fontsize=7)
    ax.set_title(f'HEARS-LM vs unadapted ESM-2\n{n_up} of {len(zs)} genes above the diagonal', fontsize=7)
    ax.legend(frameon=False, fontsize=6, loc='lower right')
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    panel_label(ax, 'b', x=-0.2, y=1.1)

    ax = fig.add_subplot(gs[1, 0:2])
    q = pg[(pg.baseline == 'AlphaMissense') & pg.baseline_auroc.notna()].sort_values('diff')
    x = np.arange(len(q))
    colours = [C['HEARS-LM'] if v > 0 else C['AlphaMissense'] for v in q['diff']]
    ax.bar(x, q['diff'], color=colours, width=0.72, edgecolor='white', lw=0.4)
    ax.errorbar(x, q['diff'], yerr=[q['diff'] - q.diff_lo, q.diff_hi - q['diff']], fmt='none', ecolor=INK2, elinewidth=0.5,
                capsize=0)
    for xi, (_, r) in zip(x, q.iterrows()):
        if r.q_bh < 0.05:
            ax.text(xi, r.diff_hi + 0.01 if r['diff'] > 0 else r.diff_lo - 0.03, '*', ha='center', fontsize=7,
                    color=INK)
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xticks(x, q.gene, rotation=90, fontsize=5.5)
    ax.set_ylabel('$\\Delta$AUROC (HEARS-LM - AlphaMissense)', fontsize=7)
    ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    panel_label(ax, 'c', x=-0.09, y=1.04)
    save(fig, 'Fig4_crossval')

    if inh is not None and len(inh):
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        methods = inh.method.unique()
        groups = inh.inheritance.unique()
        w = 0.8 / len(methods)
        for i, m in enumerate(methods):
            d = inh[inh.method == m].set_index('inheritance').reindex(groups)
            x = np.arange(len(groups)) + (i - (len(methods) - 1) / 2) * w
            c = C.get(m, C['HEARS-LM'] if m == 'HEARS-LM' else GREY)
            ax.errorbar(x, d.auroc, yerr=[d.auroc - d.auroc_lo, d.auroc_hi - d.auroc], fmt='o', ms=4, color=c,
                        elinewidth=0.9, capsize=0, label=m, lw=0)
        ax.set_xticks(np.arange(len(groups)), [f'{g}\n(n={int(inh[inh.inheritance == g].n.iloc[0]):,})' for g in groups],
                      fontsize=6)
        ax.set_ylabel('AUROC (out-of-fold)', fontsize=7)
        ax.legend(frameon=False, fontsize=6)
        ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
        save(fig, 'FigS2_inheritance')


if __name__ == '__main__':
    main()
