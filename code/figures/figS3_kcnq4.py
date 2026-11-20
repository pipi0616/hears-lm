#!/usr/bin/env python3
"""Supplementary Figure S2 - KCNQ4 mutational landscape from a model that never saw KCNQ4 or its homologues
(cross-validation fold of KCNQ4, KCNQ1, KCNE1, KCNJ10 and KCNJ16; sequence-only configuration).
a: mean predicted pathogenicity over the 19 substitutions at each residue, with membrane topology; ticks mark
   labelled pathogenic (above) and benign (below) missense variants.
b: per-residue mean prediction against the per-residue mean normalised current measured for KCNQ4 variants (Zheng et al. 2022).
c: mean prediction by structural region."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from style import GRID, INK2, ROOT, C, panel_label, save

SEG = [('S1', 97, 118), ('S2', 130, 152), ('S3', 169, 191), ('S4', 203, 223), ('S5', 236, 258), ('pore\nH5', 271, 292),
       ('S6', 294, 322)]


def region(p):
    for name, a, b in SEG:
        if a <= p <= b:
            return 'pore (H5)' if name.startswith('pore') else ('S5-S6' if name in ('S5', 'S6') else 'voltage sensor S1-S4')
    if p < 97:
        return 'N-terminus'
    if 323 <= p <= 695:
        return 'C-terminus'
    return 'loops'


def main():
    e = pd.read_csv(ROOT / 'results/external_predictions_cv_seq_plm_ddg.csv')
    k = e[(e.gene == 'KCNQ4') & e.in_saturation]
    res = k.groupby('ref_pos').prob.mean()
    v = pd.read_csv(ROOT / 'data/variants_v8.csv', usecols=['gene', 'ref_pos', 'y_final'])
    v = v[(v.gene == 'KCNQ4') & v.y_final.notna()]
    mv = pd.read_csv(ROOT / 'results/mave_variant_scores.csv', low_memory=False)
    mv = mv[(mv.gene == 'KCNQ4')]
    mres = (-mv.groupby('ref_pos').damage.mean())          # normalised current, higher = more functional

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.55, wspace=0.35, left=0.08, right=0.98, top=0.93,
                          bottom=0.1)
    ax = fig.add_subplot(gs[0, :])
    for name, a, b in SEG:
        ax.axvspan(a, b, color='#e9f1fb', lw=0)
        ax.text((a + b) / 2, 1.04, name, ha='center', va='bottom', fontsize=5.5, color=INK2)
    sm = res.rolling(5, center=True, min_periods=1).mean()
    ax.plot(res.index, res.values, color='#b9cfee', lw=0.6)
    ax.plot(sm.index, sm.values, color=C['HEARS-LM'], lw=1.3)
    for p in v[v.y_final == 1].ref_pos.unique():
        ax.plot([p, p], [0.97, 1.0], color='#eb6834', lw=0.8)
    for p in v[v.y_final == 0].ref_pos.unique():
        ax.plot([p, p], [-0.03, 0.0], color='#1baf7a', lw=0.8)
    ax.set_xlim(1, len(res)); ax.set_ylim(-0.04, 1.12)
    ax.set_xlabel('KCNQ4 residue'); ax.set_ylabel('mean predicted\npathogenicity', fontsize=7)
    ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.text(0.995, 0.02, 'ticks: labelled P/LP (top, orange), B/LB (bottom, green)', transform=ax.transAxes,
            fontsize=5.5, color=INK2, ha='right')
    panel_label(ax, 'a', x=-0.06, y=1.05)

    ax = fig.add_subplot(gs[1, 0])
    j = pd.concat([res.rename('pred'), mres.rename('exp')], axis=1).dropna()
    ax.scatter(j.pred, j.exp, s=5, color=C['HEARS-LM'], alpha=0.5, lw=0)
    rho = spearmanr(j.pred, j.exp).statistic
    ax.set_xlabel('mean predicted pathogenicity per residue', fontsize=7)
    ax.set_ylabel('mean normalised current (experiment)', fontsize=7)
    ax.set_title(f'{len(j)} residues, Spearman {rho:.2f}', fontsize=7)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    panel_label(ax, 'b', x=-0.22, y=1.08)

    ax = fig.add_subplot(gs[1, 1])
    r = pd.DataFrame({'pred': res})
    r['region'] = [region(p) for p in r.index]
    order = ['pore (H5)', 'S5-S6', 'voltage sensor S1-S4', 'loops', 'N-terminus', 'C-terminus']
    data = [r[r.region == o].pred.to_numpy() for o in order]
    bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color='white', lw=1.2))
    for b in bp['boxes']:
        b.set_facecolor(C['HEARS-LM']); b.set_edgecolor(C['HEARS-LM'])
    ax.set_yticks(range(1, len(order) + 1), [f'{o} (n={len(d)})' for o, d in zip(order, data)], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel('mean predicted pathogenicity per residue', fontsize=7)
    ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
    panel_label(ax, 'c', x=-0.55, y=1.08)
    save(fig, 'FigS3_kcnq4_landscape')
    out = r.groupby('region').pred.agg(['size', 'mean', 'median']).reindex(order)
    out.to_csv(ROOT / 'results/kcnq4_region_scores.csv')
    print(out.round(3).to_string()); print('rho residue-level vs experiment:', round(rho, 3), 'n', len(j))


if __name__ == '__main__':
    main()
