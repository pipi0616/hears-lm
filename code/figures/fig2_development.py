#!/usr/bin/env python3
"""Figure 2 - model configurations on the development set (the only data used for modelling decisions).
a: AUROC (dots = seeds, line = mean +- SD); established predictors on the development variants each one scores.
b: within-gene AUROC. c: ddG-head agreement with DDGun. d: secondary design choices (seed 0)."""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import GREY, GRID, INK2, ROOT, C, panel_label, save

sys.path.insert(0, str(ROOT / 'code/eval'))
import metrics as M  # noqa: E402
from common import load_baselines, load_variants  # noqa: E402

ARMS = [('seq', 'window only'), ('seq_plm', 'window + full-context LM'), ('seq_ddg', 'window + ddG loss'),
        ('seq_plm_ddg', 'window + LM + ddG loss'), ('seq_plm_ddg_struct', '  + structure'),
        ('seq_plm_ddg_cons', '  + conservation'), ('seq_plm_ddg_hear', '  + gene-level descriptors'),
        ('full', '  + all three (full)'), ('full_af', 'full + allele frequency (diagnostic)'),
        ('frozen', 'frozen LM embeddings, no LoRA'), ('feat_only', 'descriptors only (no LM)'),
        ('feat_plm_only', 'descriptors + LM values (no fine-tuning)'), ('seq_t12', 'window only, ESM-2 35M')]
REFS = ['AlphaMissense', 'REVEL', 'ESM-2 150M zero-shot']
SECONDARY = [('seq_plm_ddg_b0.1', 'ddG weight 0.1'), ('seq_plm_ddg', 'ddG weight 0.3'),
             ('seq_plm_ddg_b1.0', 'ddG weight 1.0'), ('seq_plm_ddg_r4', 'LoRA rank 4'),
             ('seq_plm_ddg', 'LoRA rank 16'), ('seq_plm_ddg_r64', 'LoRA rank 64')]


def dotrows(ax, rows, ps, col, refs):
    y = np.arange(len(rows))[::-1]
    for yi, (kind, key, _) in zip(y, rows):
        if kind == 'arm':
            d = ps[ps.arm == key][col].dropna()
            colr = GREY if key == 'full_af' else C['HEARS-LM']
            ax.scatter(d, np.full(len(d), yi), s=8, color=colr, alpha=0.6, lw=0, zorder=3)
            if len(d) > 1:
                ax.errorbar(d.mean(), yi, xerr=d.std(ddof=1), fmt='o', ms=4, color=colr, lw=1, capsize=0, zorder=4)
            elif len(d):
                ax.plot(d.mean(), yi, 'o', ms=4, color=colr, zorder=4)
        else:
            val = refs[key]
            ax.plot(val, yi, 'D', ms=4, mfc='white', mec=C.get(key, INK2), mew=1.2, zorder=4)
    ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
    return y


def main():
    ps = pd.read_csv(ROOT / 'results/arms_per_seed_dev.csv')
    v = load_variants()
    dev = v[v.partition == 'dev']
    base, _ = load_baselines()
    b = base.reindex(dev.variant_id)
    ref_au, ref_wg = {}, {}
    for m in REFS:
        k = b[m].notna().to_numpy()
        ref_au[m] = M.auroc(dev.y.to_numpy()[k], b[m].to_numpy()[k])
        ref_wg[m] = M.within_group_auroc(dev.y.to_numpy()[k], b[m].to_numpy()[k], dev.gene.to_numpy()[k])
    rows = [('arm', a, l) for a, l in ARMS if a in set(ps.arm)] + [('ref', m, m) for m in REFS]

    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[2.2, 1], left=0.33, right=0.98, top=0.95,
                          bottom=0.1, wspace=0.2, hspace=0.55)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    y = dotrows(ax1, rows, ps, 'auroc', ref_au)
    dotrows(ax2, rows, ps, 'auroc_within_gene', ref_wg)
    ax1.set_yticks(y, [l for _, _, l in rows], fontsize=6.3)
    ax2.set_yticks(y, [''] * len(rows))
    for ax, lab in [(ax1, 'AUROC'), (ax2, 'within-gene AUROC')]:
        lo = min(ax.get_xlim()[0], 0.8)
        ax.set_xlim(lo, 1.0); ax.set_xlabel(lab)
        ax.axhline(len(REFS) - 0.5, color=GRID, lw=0.8)
    panel_label(ax1, 'a', x=-1.25, y=1.0); panel_label(ax2, 'b', x=-0.06, y=1.0)

    ax3 = fig.add_subplot(gs[0, 2])
    dd = ps[ps.ddg_spearman.notna()]
    drows = [('arm', a, l) for a, l in ARMS if a in set(dd.arm)]
    yy = dotrows(ax3, drows, dd, 'ddg_spearman', {})
    ax3.set_yticks(yy, [l.strip() for _, _, l in drows], fontsize=6)
    ax3.yaxis.tick_right()
    ax3.set_xlim(0.5, 1.0); ax3.set_xlabel('Spearman, ddG head vs DDGun')
    panel_label(ax3, 'c', x=-0.06, y=1.0)

    ax4 = fig.add_subplot(gs[1, 0:2])
    sec = [(a, l) for a, l in SECONDARY if a in set(ps.arm)]
    if 'seq_t12' in set(ps.arm):
        sec += [('seq', 'ESM-2 150M (window only)'), ('seq_t12', 'ESM-2 35M (window only)')]
    yy = np.arange(len(sec))[::-1]
    for yi, (a, _) in zip(yy, sec):
        d = ps[(ps.arm == a) & (ps.seed == 'seed0')].auroc
        ax4.plot(d, np.full(len(d), yi), 'o', ms=4, color=C['HEARS-LM'])
    ax4.set_yticks(yy, [l for _, l in sec], fontsize=6.3)
    ax4.set_xlabel('AUROC, development set (seed 0)')
    from matplotlib.ticker import MaxNLocator
    ax4.xaxis.set_major_locator(MaxNLocator(5))
    ax4.grid(axis='x', color=GRID, lw=0.5)
    panel_label(ax4, 'd', x=-0.62, y=1.0)
    save(fig, 'Fig2_development')


if __name__ == '__main__':
    main()
