#!/usr/bin/env python3
"""Figure 6 - clinical use of the calibrated score.
a: local posterior probability of pathogenicity against score, with the one-sided 95% bound and the posteriors
   required for PP3/BP4 evidence strengths (Tavtigian framework as implemented by Pejaver et al. 2022).
b: fraction of temporal-test variants that reach evidence in each direction, out of sample (five-fold cross-fitting
   over the pooled test sets), HEARS-LM against comparators.
c: out-of-fold HEARS-LM score of variants classified by the ClinGen Hearing Loss VCEP, by expert classification.
d: test variants that were uncertain or conflicting in ClinVar 2025-01 and received a definitive classification by
   2026-04 - precision-recall, with the prevalence of pathogenic variants marked.
e: VUS in 2025-01 upgraded to P/LP by 2026-04 against those still VUS - precision-recall.
f: evidence level of all ClinVar 2026-04 missense variants of uncertain or conflicting significance, split by whether a
   pathogenic missense variant is already known at the same residue."""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import GREY, GRID, INK, INK2, ROOT, C, panel_label, save

sys.path.insert(0, str(ROOT / 'code/eval'))
R = ROOT / 'results'
LEVELS = ['BP4_strong', 'BP4_moderate', 'BP4_supporting', 'indeterminate', 'PP3_supporting', 'PP3_moderate',
          'PP3_strong', 'PP3_very_strong']
LEVEL_COL = {'BP4_strong': '#1b6a4a', 'BP4_moderate': '#1baf7a', 'BP4_supporting': '#a8ddc6',
             'indeterminate': '#e4e2dc', 'PP3_supporting': '#f6c3ab', 'PP3_moderate': '#eb6834',
             'PP3_strong': '#a93d12', 'PP3_very_strong': '#6d2408', 'not scored': '#f4f3ef'}


def stacked(ax, frac, labels, order):
    left = np.zeros(len(frac))
    y = np.arange(len(frac))[::-1]
    for lv in order:
        if lv not in frac:
            continue
        ax.barh(y, frac[lv], left=left, height=0.62, color=LEVEL_COL.get(lv, GREY), edgecolor='white', lw=0.6)
        left = left + frac[lv].to_numpy()
    ax.set_yticks(y, labels, fontsize=6.3)
    ax.set_xlim(0, 1)
    return y


def pr_panel(ax, rc, cv, an, title, letter, log=False):
    methods = [m for m in ['HEARS-LM', 'ClinPred', 'MetaRNN', 'AlphaMissense', 'REVEL'] if m in set(cv.method)]
    for m in methods:
        d = cv[(cv.analysis == an) & (cv.method == m) & (cv.curve == 'pr')]
        r = rc[(rc.analysis == an) & (rc.method == m)]
        if d.empty or r.empty:
            continue
        ls = '-' if m in C else ('--' if m == 'ClinPred' else ':')
        ax.plot(d.x, d.y, lw=1.4 if m == 'HEARS-LM' else 1.0, ls=ls, color=C.get(m, GREY),
                label=f'{m} ({r.iloc[0].auprc:.3f})' if log else f'{m} ({r.iloc[0].auprc:.2f})')
    r0 = rc[rc.analysis == an].iloc[0]
    ax.axhline(r0.prevalence, color=GREY, lw=0.8, ls='--')
    ax.text(0.03 if log else 0.98, r0.prevalence * (1.15 if log else 1) + (0 if log else 0.02),
            f'prevalence {r0.prevalence:.4f}' if log else f'prevalence {r0.prevalence:.2f}', fontsize=5.5, color=INK2,
            va='bottom', ha='left' if log else 'right')
    ax.set_xlabel('recall'); ax.set_ylabel('precision' + (' (log scale)' if log else ''), fontsize=7)
    ax.set_xlim(0, 1)
    if log:
        ax.set_yscale('log'); ax.set_ylim(1e-3, 1.05)
    else:
        ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=5.2, loc='upper right' if log else 'lower left', title='AUPRC', title_fontsize=5.3)
    ax.set_title(f'{title}\n(n = {int(r0.n):,}, {int(r0.n_pos):,} pathogenic)', fontsize=6.3)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    panel_label(ax, letter, x=-0.3, y=1.12)


def main():
    thr = pd.read_csv(R / 'calibration_thresholds.csv')
    cur = pd.read_csv(R / 'calibration_curves.csv')
    fig = plt.figure(figsize=(7.2, 8.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.75, wspace=0.62, left=0.14, right=0.88,
                          top=0.95, bottom=0.1)

    ax = fig.add_subplot(gs[0, 0])
    c = cur[cur.method == 'HEARS-LM']
    ax.fill_between(c.score, c.lower95, c.upper95, color=C['HEARS-LM'], alpha=0.15, lw=0)
    ax.plot(c.score, c.posterior, color=C['HEARS-LM'], lw=1.5)
    req = thr[thr.method == 'HEARS-LM'].drop_duplicates('evidence').set_index('evidence')
    for lv, style in [('PP3_supporting', ':'), ('PP3_moderate', '--'), ('PP3_strong', '-.')]:
        if lv in req.index and np.isfinite(req.loc[lv, 'required_posterior']):
            ax.axhline(req.loc[lv, 'required_posterior'], color=INK2, lw=0.6, ls=style)
            ax.text(1.0, req.loc[lv, 'required_posterior'], ' ' + lv.replace('PP3_', 'PP3 '), fontsize=5.5,
                    color=INK2, va='center')
    ax.set_xlabel('HEARS-LM score'); ax.set_ylabel('posterior probability\nof pathogenicity', fontsize=7)
    ax.set_xlim(0, 1); ax.set_yscale('log')
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    panel_label(ax, 'a', x=-0.3, y=1.08)

    ax = fig.add_subplot(gs[0, 1])
    e = pd.read_csv(R / 'evidence_crossfit.csv')
    e = e[(e.design == 'kfold') & (e.partition == 'test_time')].set_index('method')
    methods = [m for m in ['HEARS-LM', 'ClinPred', 'MetaRNN', 'REVEL', 'BayesDel (no AF)', 'VEST4', 'AlphaMissense',
                           'ESM-2 150M zero-shot'] if m in e.index]
    y = np.arange(len(methods))[::-1]
    ax.barh(y + 0.18, e.loc[methods, 'P_with_PP3'], height=0.34, color='#eb6834', label='pathogenic with PP3')
    ax.barh(y - 0.18, e.loc[methods, 'B_with_BP4'], height=0.34, color='#1baf7a', label='benign with BP4')
    for yi, m in zip(y, methods):
        ax.text(1.02, yi, f"{e.loc[m, 'B_with_PP3']:.1%} / {e.loc[m, 'P_with_BP4']:.1%}", va='center', fontsize=5.5,
                color=INK2, transform=ax.get_yaxis_transform())
    ax.text(1.02, len(methods) - 0.35, 'wrong direction\nB→PP3 / P→BP4', fontsize=5.3, color=INK2,
            transform=ax.get_yaxis_transform(), va='bottom')
    ax.set_yticks(y, methods, fontsize=6.3)
    ax.set_xlim(0, 1)
    ax.set_xlabel('fraction receiving evidence (temporal test set;\nfive-fold cross-fitted thresholds)', fontsize=6.3)
    ax.legend(frameon=False, fontsize=5.5, loc='upper center', bbox_to_anchor=(0.45, -0.3), ncol=2)
    ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
    panel_label(ax, 'b', x=-0.62, y=1.08)

    if (R / 'expert_panel_variants.csv').exists():
        ax = fig.add_subplot(gs[1, 0])
        ex = pd.read_csv(R / 'expert_panel_variants.csv')
        ex = ex[ex.y_expert.notna() & ex.hearslm_score.notna()]
        names = {'P': 'Pathogenic', 'LP': 'Likely pathogenic', 'LB': 'Likely benign', 'B': 'Benign'}
        col = {'P': '#a93d12', 'LP': '#eb6834', 'LB': '#1baf7a', 'B': '#1b6a4a'}
        rng = np.random.default_rng(0)
        order = [k for k in names if (ex.assertion == k).any()]
        for yi, k in enumerate(order[::-1]):
            d = ex[ex.assertion == k].hearslm_score.to_numpy()
            ax.scatter(d, yi + rng.uniform(-0.22, 0.22, len(d)), s=7, color=col[k], alpha=0.75, lw=0)
            ax.plot([np.median(d)] * 2, [yi - 0.32, yi + 0.32], color=INK, lw=1.3)
        ax.set_yticks(range(len(order)), [f'{names[k]} (n={int((ex.assertion == k).sum())})' for k in order[::-1]],
                      fontsize=6.3)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlabel('HEARS-LM score (out of fold)', fontsize=6.5)
        ax.set_title('ClinGen Hearing Loss VCEP classifications', fontsize=6.5)
        ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
        panel_label(ax, 'c', x=-0.62, y=1.12)

    if (R / 'reclassification_curves.csv').exists():
        rc = pd.read_csv(R / 'reclassification.csv')
        cv = pd.read_csv(R / 'reclassification_curves.csv')
        ans = list(rc.analysis.unique())
        a1 = [a for a in ans if 'vs B/LB' in a]; a2 = [a for a in ans if 'still VUS' in a]
        if a1:
            pr_panel(fig.add_subplot(gs[1, 1]), rc, cv, a1[0], 'uncertain/conflicting 2025 → P/LP vs B/LB 2026', 'd')
        if a2:
            pr_panel(fig.add_subplot(gs[2, 0]), rc, cv, a2[0], 'VUS 2025 → P/LP vs still VUS 2026', 'e', log=True)

    if (R / 'vus_catalogue.csv').exists():
        ax = fig.add_subplot(gs[2, 1])
        v = pd.read_csv(R / 'vus_catalogue.csv', usecols=['evidence', 'residue_context', 'cv_state',
                                                          'n_training_variants_gene'])
        v = v[v.n_training_variants_gene > 0]            # evidence is assigned only in genes with training variants
        ct = pd.crosstab(v.residue_context, v.evidence)
        order = ['P/LP missense at same residue', 'within 5 residues of a P/LP residue',
                 'no P/LP missense within 5 residues']
        ct = ct.reindex([o for o in order if o in ct.index])
        frac = ct.div(ct.sum(axis=1), axis=0)
        lab = {'P/LP missense at same residue': 'P/LP at same\nresidue', 'within 5 residues of a P/LP residue':
               'P/LP within\n5 residues', 'no P/LP missense within 5 residues': 'no P/LP within\n5 residues'}
        stacked(ax, frac, [f'{lab[i]} (n = {int(ct.loc[i].sum()):,})' for i in ct.index], LEVELS + ['not scored'])
        handles = [plt.Rectangle((0, 0), 1, 1, fc=LEVEL_COL[lv]) for lv in LEVELS if lv in frac]
        ax.legend(handles, [lv.replace('_', ' ') for lv in LEVELS if lv in frac], frameon=False, fontsize=5.3,
                  ncol=3, loc='upper center', bbox_to_anchor=(0.3, -0.32))
        ax.set_xlabel('fraction of ClinVar 2026-04 uncertain or conflicting\nmissense variants (genes with training variants)', fontsize=6.3)
        panel_label(ax, 'f', x=-0.62, y=1.12)
    save(fig, 'Fig6_clinical')


if __name__ == '__main__':
    main()
