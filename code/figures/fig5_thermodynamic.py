#!/usr/bin/env python3
"""Figure 5 - the thermodynamic head and orthogonal validation against experimental assays.
a: agreement of the ddG head with DDGun on the ten withheld genes, by AlphaFold2 confidence of the residue.
b: ddG head and DDGun against experimental abundance on IDENTICAL variants (those DDGun covers).
c: Spearman correlation of the pathogenicity score and two reference predictors with experimental scores
   (Human Domainome abundance assays averaged per protein, KCNQ4 current, KCNE1 function; models that never saw the gene)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import GRID, INK2, ROOT, C, panel_label, save

R = ROOT / 'results'
COL = {'HEARS-LM (pathogenicity)': C['HEARS-LM'], 'AlphaMissense': C['AlphaMissense'],
       'ESM-2 150M zero-shot': C['ESM-2 150M zero-shot']}


def main():
    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.75, 0.85, 1.3], wspace=1.0, left=0.07, right=0.98, top=0.86,
                          bottom=0.2)
    ax = fig.add_subplot(gs[0, 0])
    p = pd.read_csv(R / 'ddg_by_plddt.csv')
    p = p[p.plddt != 'no model']
    x = np.arange(len(p))
    ax.bar(x, p.spearman, width=0.65, color=C['HEARS-LM'], edgecolor='white')
    for xi, (_, r) in zip(x, p.iterrows()):
        ax.text(xi, r.spearman + 0.01, f'{r.spearman:.2f}\nn={r.n}', ha='center', va='bottom', fontsize=5.5, color=INK2)
    ax.set_xticks(x, p.plddt, fontsize=6, rotation=30, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('AlphaFold2 pLDDT', fontsize=7)
    ax.set_ylabel('Spearman, ddG head vs DDGun', fontsize=7)
    ax.set_title('distillation on\nwithheld genes', fontsize=7)
    ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    panel_label(ax, 'a', x=-0.45, y=1.08)

    ax = fig.add_subplot(gs[0, 2])
    cr = pd.read_csv(R / 'mave_correlations.csv')
    cr = cr[cr.method.isin(COL) & (cr.subset == 'common')]
    d = cr.pivot_table(index=['gene', 'assay_type'], columns='method', values='spearman')
    d = d.sort_values('HEARS-LM (pathogenicity)')
    y = np.arange(len(d))
    for m in COL:
        if m in d:
            ax.plot(d[m], y, 'o', ms=4, color=COL[m], label=m.replace(' (pathogenicity)', ''), lw=0)
    ax.set_yticks(y, [f'{g} · {a.replace("domainome ", "").replace("KCNQ4 current (homozygous)", "current").replace("KCNE1 function", "function")}' for g, a in d.index], fontsize=5.8)
    ax.set_xlabel('Spearman with the experimental score', fontsize=7)
    ax.set_title('functional and abundance assays', fontsize=7)
    ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=5.8, loc='lower right')
    panel_label(ax, 'c', x=-0.45, y=1.08)

    ax = fig.add_subplot(gs[0, 1])
    h = pd.read_csv(R / 'mave_head_vs_ddgun_same_variants.csv')
    for _, r in h.iterrows():
        ax.plot([0, 1], [r.ddgun, r['head']], color='#c9c7c0', lw=0.8, zorder=1)
    ax.scatter(np.zeros(len(h)), h.ddgun, s=14, color='#eda100', zorder=2, label='DDGun')
    ax.scatter(np.ones(len(h)), h['head'], s=14, color=C['HEARS-LM'], zorder=2, label='ddG head')
    ax.set_xticks([0, 1], ['DDGun', 'ddG head'], fontsize=6.5)
    ax.set_xlim(-0.4, 1.4)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylabel('Spearman with abundance', fontsize=7)
    ax.set_title(f'identical variants\n({len(h)} assays, n = {int(h.n.sum())})', fontsize=7)
    ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    panel_label(ax, 'b', x=-0.5, y=1.08)
    save(fig, 'Fig5_thermodynamic')


if __name__ == '__main__':
    main()
