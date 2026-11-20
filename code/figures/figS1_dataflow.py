#!/usr/bin/env python3
"""Supplementary Figure S1 - data flow with every exclusion (counts from results/data_flow.json and
results/table_partitions.csv)."""
import json

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from style import INK, INK2, ROOT, C, panel_label, save

R = ROOT / 'results'


def box(ax, x, y, w, h, text, fc='#f4f3ef', ec='#b8b6ae', fs=6.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.012', fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK,
            fontweight='bold' if bold else 'normal', wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, lw=0.7, color=INK2))


def main():
    flow = json.load(open(R / 'data_flow.json'))
    part = pd.read_csv(R / 'table_partitions.csv').set_index('partition')
    fig = plt.figure(figsize=(5.4, 4.6))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    f = lambda n: f'{int(n):,}'  # noqa: E731
    box(ax, 0.03, 0.82, 0.29, 0.11, f"ClinVar 2025-01 / 2026-04\n{f(flow['clinvar_2025_01_missense_records'])} / "
        f"{f(flow['clinvar_2026_04_missense_records'])} missense records\n(163 PanelApp hearing-loss genes)", fs=5.6)
    box(ax, 0.36, 0.82, 0.29, 0.11, f"Deafness Variation Database\n{f(flow['dvd_missense_records'])} missense records\n"
        f"({f(flow['canonicalisation']['dvd']['ok'])} mapped)", fs=6.2)
    box(ax, 0.69, 0.82, 0.29, 0.11, 'Excluded DVD labels\n'
        f"{f(flow['dvd_prediction_only_benign_excluded'])} prediction-only\n{f(flow['dvd_clinvar_only_not_independent'])} ClinVar-only\n"
        f"{f(flow['dvd_labels_overridden_by_consensus_vus'])} consensus VUS", fc='#fbefe9', ec='#e3b7a3', fs=5.8)
    arrow(ax, 0.65, 0.875, 0.69, 0.875)
    box(ax, 0.10, 0.62, 0.80, 0.12, 'Re-mapping to one reference protein per gene (alignment; reference residue verified)\n'
        'ClinVar >= 1 star; DVD literature / MAF / manual curation; cross-source conflicts removed\n'
        f"{f(flow['labelled_protein_variants'])} labelled missense variants in {flow['genes_with_labels']} genes", bold=False)
    arrow(ax, 0.175, 0.82, 0.30, 0.74); arrow(ax, 0.505, 0.82, 0.55, 0.74)
    cols = [('fit', 'Training\n(fit)'), ('dev', 'Development'), ('test_time', 'test_time\nnew ClinVar\n2025-01 to 2026-04'),
            ('test_db', 'test_db\nDVD literature,\nno ClinVar label'), ('test_gene', 'test_gene\n10 held-out\ngenes')]
    xs = [0.015, 0.212, 0.409, 0.606, 0.803]
    for (key, lab), x in zip(cols, xs):
        r = part.loc[key]
        fc = '#e9f1fb' if key in ('fit', 'dev') else '#f4f3ef'
        box(ax, x, 0.30, 0.168, 0.22, f"{lab}\n\nn = {f(r.n)}\nP/LP {f(r.P_LP)}\nB/LB {f(r.B_LB)}\n{int(r.genes)} genes",
            fc=fc, fs=5.8)
        arrow(ax, 0.5, 0.62, x + 0.084, 0.52)
    r = part.loc['cross-validation pool']
    box(ax, 0.10, 0.06, 0.80, 0.14, 'Leave-gene-family-out cross-validation: 5 folds of whole homology families '
        f"({flow['gene_families']['n_families']} families)\n"
        f"n = {f(r.n)} (P/LP {f(r.P_LP)} | B/LB {f(r.B_LB)}), {int(r.genes)} genes; every gene unseen when scored",
        fc='#eef6ee', ec='#a9c9a9')
    ax.text(0.5, 0.245, 'ClinVar training labels from the 2025-01 release; evaluation labels from ClinVar 2026-04',
            ha='center', fontsize=6, color=INK2, style='italic')

    save(fig, 'FigS1_dataflow')


if __name__ == '__main__':
    main()
