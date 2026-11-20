#!/usr/bin/env python3
"""Figure 3 - performance on the held-out test sets.
a: AUROC of HEARS-LM and every comparator on the common variant set of each test partition, with 95% CI.
b: paired difference (HEARS-LM minus comparator) on identical variants, per partition.
c: within-gene AUROC on the same common sets (removes between-gene prevalence differences).
d: reliability of the released probabilities per partition."""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import GREY, GRID, INK, INK2, ROOT, C, panel_label, save

sys.path.insert(0, str(ROOT / 'code/eval'))

R = ROOT / 'results'
PART_LABEL = {'test_time': 'test_time\nnew ClinVar classifications', 'test_gene': 'test_gene\n10 held-out genes',
              'test_db': 'test_db\nDVD literature evidence'}
ORDER = ['test_time', 'test_gene', 'test_db']
# test sets are told apart by neutral shades and marker shape (method colours are reserved for methods)
PART_STYLE = {'test_time': ('#2b2a27', 'o'), 'test_gene': ('#8f8c85', 's'), 'test_db': ('#c9c6bd', '^')}
# how each method obtains population-frequency information (Supplementary Table S5)
AF_MARK = {'MetaRNN': '†§', 'ClinPred': '†§', 'BayesDel (with AF)': '†', 'Allele frequency only (-log10 AF)': '†',
           'AlphaMissense': '‡', 'PrimateAI': '‡', 'REVEL': '‡', 'VEST4': '‡', 'VARITY_R': '§'}


def label_of(m):
    return m + (' ' + AF_MARK[m] if m in AF_MARK else '')


def colour_of(method):
    return C.get(method, C['HEARS-LM'] if method == 'HEARS-LM' else GREY)


def main():
    cm = pd.read_csv(R / 'benchmark_common.csv')
    pw = pd.read_csv(R / 'benchmark_common_pairwise.csv')      # same variants as panel a
    rel = pd.read_csv(R / 'reliability.csv') if (R / 'reliability.csv').exists() else None
    bins = pd.read_csv(R / 'reliability_bins.csv') if (R / 'reliability_bins.csv').exists() else None
    parts = [p for p in ORDER if p in set(cm.partition)]
    order = cm[cm.partition == parts[0]].sort_values('auroc', ascending=False).method.tolist()
    order = ['HEARS-LM'] + [m for m in order if m != 'HEARS-LM']

    fig = plt.figure(figsize=(7.2, 7.4))
    gs = fig.add_gridspec(3, len(parts), height_ratios=[2.6, 2.3, 1.4], hspace=0.5, wspace=0.18,
                          left=0.26, right=0.98, top=0.95, bottom=0.07)
    for j, part in enumerate(parts):
        ax = fig.add_subplot(gs[0, j])
        d = cm[cm.partition == part].set_index('method').reindex(order).dropna(subset=['auroc'])
        y = np.arange(len(d))[::-1]
        for yi, (m, r) in zip(y, d.iterrows()):
            c = colour_of(m)
            ax.plot([r.auroc_lo, r.auroc_hi], [yi, yi], lw=1.2, color=c, alpha=0.5, solid_capstyle='round')
            ax.plot(r.auroc, yi, 'o', ms=4.5, color=c, mec='white', mew=0.5, zorder=3)
        ax.set_yticks(y, [label_of(m) for m in d.index] if j == 0 else [''] * len(d), fontsize=6.3)
        n, npos = int(d.n.iloc[0]), int(d.n_P.iloc[0])
        ax.set_title(f'{PART_LABEL[part]}\nn = {n:,} ({npos:,} P/LP)', fontsize=7)
        ax.set_xlabel('AUROC'); ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
        if j == 0:
            panel_label(ax, 'a', x=-1.0, y=1.12)

        ax = fig.add_subplot(gs[1, j])
        q = pw[(pw.partition == part) & pw.diff_hearslm_minus_method.notna()].set_index('method')
        q = q.reindex([m for m in order if m in q.index]).dropna(subset=['diff_hearslm_minus_method'])
        y = np.arange(len(q))[::-1]
        for yi, (m, r) in zip(y, q.iterrows()):
            sig = r.get('q_delong', 1) < 0.05
            c = colour_of(m)
            ax.plot([r.diff_lo, r.diff_hi], [yi, yi], lw=1.2, color=c, alpha=0.5, solid_capstyle='round')
            ax.plot(r.diff_hearslm_minus_method, yi, 'o', ms=4.5, color=c if sig else 'white', mec=c, mew=1.0, zorder=3)
        ax.axvline(0, color=INK2, lw=0.7)
        ax.set_yticks(y, [label_of(m) for m in q.index] if j == 0 else [''] * len(q), fontsize=6.3)
        ax.set_xlabel('$\\Delta$AUROC (HEARS-LM $-$ method)', fontsize=7)
        ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
        if j == 0:
            panel_label(ax, 'b', x=-1.0, y=1.08)
            ax.text(0, len(q) - 0.2, 'filled: q < 0.05 (DeLong, BH)', fontsize=5.5, color=INK2, ha='center')
            bb = ax.get_position()
            fig.text(0.02, bb.y0 - 0.055, '† uses allele frequency as an input   ‡ benign training labels drawn from '
                     'population variation   § trained on ClinVar labels', fontsize=5.5, color=INK2)

    ax = fig.add_subplot(gs[2, 0:2])
    d = cm.pivot_table(index='method', columns='partition', values='within_gene_auroc').reindex(order).dropna(how='all')
    x = np.arange(len(d))
    for i, part in enumerate(parts):
        if part in d:
            col, mk = PART_STYLE[part]
            ax.plot(x + (i - 1) * 0.22, d[part], mk, ms=4, color=col, mec='#55534e', mew=0.4, lw=0, label=part)
    ax.set_xticks(x, [label_of(m) for m in d.index], rotation=35, ha='right', fontsize=6)
    ax.set_ylabel('within-gene AUROC', fontsize=7)
    ax.legend(frameon=False, fontsize=6, ncol=3, loc='lower left')
    ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    panel_label(ax, 'c', x=-0.18, y=1.05)

    if bins is not None:
        ax = fig.add_subplot(gs[2, 2])
        ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls='--')
        for i, part in enumerate(parts):
            b = bins[bins.partition == part]
            col, mk = PART_STYLE[part]
            ax.plot(b.mean_prob, b.observed, '-' + mk, ms=3, lw=1.2, color=col, mec='#55534e', mew=0.4, label=part)
        ax.set_xlabel('predicted probability', fontsize=7); ax.set_ylabel('observed fraction P/LP', fontsize=7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
        panel_label(ax, 'd', x=-0.3, y=1.05)
    save(fig, 'Fig3_benchmark')


if __name__ == '__main__':
    main()
