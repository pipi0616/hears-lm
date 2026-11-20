#!/usr/bin/env python3
"""Figure 1 - study overview, drawn as a schematic with minimal text.
a: data and evaluation design (sources, labelled set, partitions and what each test set separates, cross-validation
   over whole gene families); b: model; c: from score to calibrated ACMG/AMP evidence.
Counts come from results/data_flow.json and results/table_partitions.csv; the full data flow with every exclusion is
Supplementary Figure S1 (figS1_dataflow.py)."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

from style import C, INK, INK2, ROOT, save

R = ROOT / 'results'
BLUE, BLUE_L, BLUE_D = C['HEARS-LM'], '#e3edfa', '#1d5aa3'
TRAIN, DEV = '#9cc0ec', '#cfe0f6'
PART = {'test_time': '#2b2a27', 'test_db': '#c9c6bd', 'test_gene': '#8f8c85'}      # as in Figure 3
EV = {'BP4_moderate': '#1baf7a', 'BP4_supporting': '#a8ddc6', 'indeterminate': '#e4e2dc',
      'PP3_supporting': '#f6c3ab', 'PP3_moderate': '#eb6834'}                           # as in Figure 6
PALE, EDGE = '#f4f3ef', '#b8b6ae'


def rbox(ax, x, y, w, h, fc=PALE, ec=EDGE, lw=0.7, ls='-', r=0.8, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, ls=ls,
                                zorder=z))


def txt(ax, x, y, s, fs=6.2, c=INK, ha='center', va='center', **k):
    ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, zorder=5, **k)


def arrow(ax, x0, y0, x1, y1, c=INK2, lw=0.8, ls='-'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, lw=lw, color=c, ls=ls,
                                 shrinkA=0, shrinkB=0, zorder=3))


def cylinder(ax, x, y, w, h, fc, ec):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=fc, ec='none', zorder=2))
    ax.plot([x - w / 2, x - w / 2], [y - h / 2, y + h / 2], color=ec, lw=0.7, zorder=3)
    ax.plot([x + w / 2, x + w / 2], [y - h / 2, y + h / 2], color=ec, lw=0.7, zorder=3)
    ax.add_patch(Ellipse((x, y - h / 2), w, h * 0.32, fc=fc, ec=ec, lw=0.7, zorder=2))
    ax.add_patch(Ellipse((x, y + h / 2), w, h * 0.32, fc='white', ec=ec, lw=0.7, zorder=4))


def header(ax, x, y, letter, title):
    txt(ax, x, y, letter, fs=10, ha='left', fontweight='bold')
    txt(ax, x + 3.2, y - 0.1, title, fs=7.2, ha='left', fontweight='bold', c=INK)


def panel_a(ax, flow, part):
    f = lambda n: f'{int(n):,}'  # noqa: E731
    header(ax, 1, 62, 'a', 'Data and evaluation design')
    cylinder(ax, 9.0, 55.0, 11, 4.6, '#eef3fb', '#7aa3d8'); txt(ax, 9.0, 54.6, 'ClinVar', fs=6.0)
    cylinder(ax, 9.0, 47.6, 11, 4.6, '#eef3fb', '#7aa3d8'); txt(ax, 9.0, 47.2, 'DVD', fs=6.0)
    rbox(ax, 27.5, 47.0, 18.5, 8.5, fc='white', ec=INK2)
    txt(ax, 36.75, 52.4, f"{f(flow['labelled_protein_variants'])} variants", fs=6.6, fontweight='bold')
    txt(ax, 36.75, 49.2, f"{flow['genes_with_labels']} genes · missense", fs=5.6, c=INK2)
    arrow(ax, 14.7, 54.6, 27.3, 52.2); arrow(ax, 14.7, 47.6, 27.3, 50.0)
    txt(ax, 36.75, 45.0, 'no prediction-based labels', fs=5.2, c=INK2, style='italic')
    # partitions: equal chips, grouped
    keys = [('fit', 'training', TRAIN, INK), ('dev', 'development', DEV, INK), ('test_time', 'new ClinVar', PART['test_time'], 'white'),
            ('test_db', 'literature', PART['test_db'], INK), ('test_gene', '10 genes', PART['test_gene'], 'white')]
    xs = [1, 10.2, 20.6, 29.2, 37.8]
    for (k, lab, fc, tc), x in zip(keys, xs):
        rbox(ax, x, 31, 8.2, 6.4, fc=fc, ec='none', r=0.7)
        txt(ax, x + 4.1, 35.2, lab, fs=5.0, c=tc)
        txt(ax, x + 4.1, 32.8, f(part.loc[k, 'n']), fs=5.8, c=tc, fontweight='bold')
    for x0, x1, lab in [(1, 18.4, 'fit and select'), (20.6, 46.0, 'test')]:
        ax.plot([x0, x0, x1, x1], [38.4, 39.2, 39.2, 38.4], color=INK2, lw=0.6)
        txt(ax, (x0 + x1) / 2, 40.6, lab, fs=5.6, c=INK2)
    for x, lab in zip(xs[2:], ['by time', 'by source', 'by gene']):
        txt(ax, x + 4.1, 29.6, lab, fs=5.0, c=INK2, style='italic')
    # cross-validation over whole gene families
    txt(ax, 23.5, 24.6, 'cross-validation over whole gene families', fs=5.8, c=INK)
    rng = np.random.default_rng(3)
    for k in range(5):
        x0 = 1.6 + k * 9.0
        held = k == 2
        rbox(ax, x0, 5.5, 8.0, 16.5, fc='#f7f6f2' if not held else '#ecebe6', ec=INK2 if held else EDGE,
             lw=0.9 if held else 0.6, r=0.7)
        y = 19.6
        for fam in range(4):
            n = int(rng.integers(1, 4))
            for g in range(n):
                ax.add_patch(Rectangle((x0 + 1.1 + g * 2.0, y), 1.6, 1.6, fc=PART['test_gene'] if held else TRAIN,
                                       ec='none', zorder=3))
            ax.add_patch(FancyBboxPatch((x0 + 0.8, y - 0.35), n * 2.0 + 0.2, 2.3, boxstyle='round,pad=0,rounding_size=0.5',
                                        fc='none', ec=INK2, lw=0.4, zorder=4))
            y -= 3.6
        txt(ax, x0 + 4.0, 3.6, 'held out' if held else 'train', fs=5.0, c=INK if held else INK2,
            fontweight='bold' if held else 'normal')
    txt(ax, 23.5, 0.9, f"5 folds · {flow['gene_families']['n_families']} families · "
        f"{f(part.loc['cross-validation pool', 'n'])} variants", fs=5.2, c=INK2)


def panel_b(ax):
    header(ax, 50, 62, 'b', 'Model')
    # protein, variant and window
    rbox(ax, 52, 53.6, 44, 2.0, fc='#e9e8e3', ec='none', r=1.0)
    for x in np.arange(53.2, 95.5, 1.4):
        ax.plot([x, x], [53.9, 55.3], color='white', lw=0.35, zorder=2)
    ax.add_patch(Rectangle((73.3, 53.6), 1.4, 2.0, fc=INK, ec='none', zorder=3))
    txt(ax, 74.0, 57.6, 'variant', fs=5.2, c=INK2)
    ax.plot([66.5, 66.5, 81.5, 81.5], [52.6, 51.9, 51.9, 52.6], color=INK2, lw=0.6)
    txt(ax, 74.0, 50.6, '65-residue window', fs=5.2, c=INK2)
    for y, lab, mut in [(46.0, 'wild type', False), (42.6, 'mutant', True)]:
        rbox(ax, 66.5, y, 15, 2.0, fc=BLUE_L, ec='none', r=0.5)
        ax.add_patch(Rectangle((73.3, y), 1.4, 2.0, fc=INK if mut else BLUE, ec='none', zorder=3))
        txt(ax, 65.6, y + 1.0, lab, fs=5.2, c=INK2, ha='right')
    arrow(ax, 74.0, 42.4, 74.0, 39.6)
    # adapted language model
    rbox(ax, 60.5, 30.5, 27, 8.8, fc=BLUE_L, ec=BLUE, lw=1.0, r=1.0)
    txt(ax, 74.0, 35.6, 'ESM-2 150M', fs=6.8, fontweight='bold', c=BLUE_D)
    txt(ax, 74.0, 32.6, 'low-rank adapters', fs=5.3, c=BLUE_D)
    for x in [62.2, 66.0, 81.0, 84.8]:
        rbox(ax, x, 38.8, 1.8, 1.4, fc=BLUE_D, ec='none', r=0.3, z=4)
    arrow(ax, 74.0, 30.3, 74.0, 27.4)
    for i, (lab, fc) in enumerate([(r'$h^{wt}$', '#bcd3f1'), (r'$h^{mut}$', '#8db5e8'), (r'$\Delta h$', BLUE)]):
        ax.add_patch(Rectangle((64.5 + i * 6.5, 23.6), 6.3, 3.4, fc=fc, ec='white', lw=0.6, zorder=3))
        txt(ax, 64.5 + i * 6.5 + 3.15, 25.3, lab, fs=6.0, c='white' if i == 2 else INK)
    # whole-protein language-model values and descriptors
    rng = np.random.default_rng(0)
    hm = rng.random((5, 9)) ** 2
    ax.imshow(hm, extent=(89.5, 97.5, 43.5, 47.5), cmap='Greys', vmin=0, vmax=1.4, aspect='auto', zorder=2)
    txt(ax, 93.5, 41.9, 'whole-protein', fs=5.0, c=INK2)
    txt(ax, 93.5, 40.3, 'log-probabilities', fs=5.0, c=INK2)
    for j, lab in enumerate(['conservation', 'structure', 'gene-level']):
        rbox(ax, 89.0, 35.0 - j * 3.1, 9.0, 2.5, fc=PALE, ec=EDGE, r=0.5)
        txt(ax, 93.5, 36.25 - j * 3.1, lab, fs=5.0)
    rbox(ax, 88.3, 28.2, 10.4, 20.3, fc='none', ec=EDGE, lw=0.6, ls=(0, (2, 1.5)), r=0.8)
    arrow(ax, 93.5, 28.0, 93.5, 19.4); arrow(ax, 74.0, 23.4, 74.0, 19.4)
    # fusion and heads
    rbox(ax, 60.5, 14.5, 37, 4.7, fc='white', ec=INK2, r=0.8)
    txt(ax, 79.0, 16.85, 'fusion network', fs=6.0)
    arrow(ax, 70.0, 14.3, 70.0, 10.6); arrow(ax, 89.0, 14.3, 89.0, 10.6, ls=(0, (2, 1.5)))
    rbox(ax, 61.0, 4.5, 18, 5.9, fc=BLUE_L, ec=BLUE, lw=1.0, r=0.8)
    txt(ax, 70.0, 7.45, 'P(pathogenic)', fs=6.2, fontweight='bold', c=BLUE_D)
    rbox(ax, 80.5, 4.5, 17, 5.9, fc='white', ec=INK2, ls=(0, (2, 1.5)), r=0.8)
    txt(ax, 89.0, 8.3, r'$\Delta\Delta G$', fs=6.2)
    txt(ax, 89.0, 5.9, 'auxiliary target', fs=5.0, c=INK2)


def panel_c(ax):
    header(ax, 101, 62, 'c', 'Calibrated evidence')
    txt(ax, 121.5, 55.2, 'HEARS-LM score', fs=5.8, c=INK2)
    widths = [('BP4_moderate', 5), ('BP4_supporting', 8), ('indeterminate', 13), ('PP3_supporting', 7), ('PP3_moderate', 4)]
    x = 103.0
    for k, w in widths:
        ax.add_patch(Rectangle((x, 49.5), w, 4.0, fc=EV[k], ec='white', lw=0.8, zorder=3))
        x += w
    txt(ax, 103.0, 47.6, '0', fs=5.4, c=INK2); txt(ax, 140.0, 47.6, '1', fs=5.4, c=INK2)
    txt(ax, 109.5, 45.4, 'BP4', fs=6.4, fontweight='bold', c='#137a52')
    txt(ax, 134.5, 45.4, 'PP3', fs=6.4, fontweight='bold', c='#b4471a')
    for y, k, lab in [(41.4, 'BP4_supporting', 'Supporting'), (38.9, 'BP4_moderate', 'Moderate')]:
        ax.add_patch(Rectangle((112.0, y - 0.8), 1.8, 1.6, fc=EV[k], ec='none', zorder=3))
        ax.add_patch(Rectangle((114.0, y - 0.8), 1.8, 1.6, fc=EV[k.replace('BP4', 'PP3')], ec='none', zorder=3))
        txt(ax, 117.0, y, lab, fs=5.2, c=INK2, ha='left')
    arrow(ax, 121.5, 36.4, 121.5, 31.6)
    rbox(ax, 106.5, 24.0, 30, 7.4, fc='white', ec=INK2, r=0.8)
    txt(ax, 121.5, 28.9, 'computational evidence', fs=5.8)
    txt(ax, 121.5, 26.2, 'in ACMG/AMP classification', fs=5.8)
    arrow(ax, 121.5, 23.8, 121.5, 19.4)
    rbox(ax, 106.5, 11.8, 30, 7.4, fc=BLUE_L, ec=BLUE, r=0.8)
    txt(ax, 121.5, 16.7, 'prioritised candidates', fs=5.8, c=BLUE_D)
    txt(ax, 121.5, 14.0, 'among ClinVar VUS', fs=5.8, c=BLUE_D)


def main():
    flow = json.load(open(R / 'data_flow.json'))
    part = pd.read_csv(R / 'table_partitions.csv').set_index('partition')
    fig = plt.figure(figsize=(7.2, 3.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 144); ax.set_ylim(0, 64); ax.set_aspect('equal'); ax.axis('off')
    panel_a(ax, flow, part)
    panel_b(ax)
    panel_c(ax)
    for x in [48.5, 99.5]:
        ax.plot([x, x], [2, 62], color='#e6e5e1', lw=0.8)
    save(fig, 'Fig1_design')


if __name__ == '__main__':
    main()
