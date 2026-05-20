#!/usr/bin/env python3
"""Perfect Fig 3-6 + Supp: zero overlap."""
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, re
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve

OUT = Path('/Users/pipi/Projects/QAFI_Paper/hearing_v2/figures')
ROOT = Path('/Users/pipi/Projects/QAFI_Paper/hearing_v2')
EVAL = ROOT/'eval_results'
plt.rcParams.update({'font.size': 9, 'figure.dpi': 200, 'savefig.dpi': 300,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

# =================================================================
# FIG 3 — legend ABOVE plot for panel a
# =================================================================
fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(3, 4, hspace=0.75, wspace=0.6, height_ratios=[1.6, 1, 1])

axA = fig.add_subplot(gs[0, :])
axA.text(-0.04, 1.08, 'a', fontsize=14, fontweight='bold', transform=axA.transAxes)
df_am = pd.read_csv(EVAL/'s6_held_out_only.csv')
df_zs = pd.read_csv(EVAL/'s6_held_out_vs_am_zs.csv')
df_combo = df_am.merge(df_zs[['gene','d_zs','d_zs_lo','d_zs_hi']], on='gene', how='inner')
df_combo = df_combo.sort_values('delta', ascending=False)
x = np.arange(len(df_combo))
w = 0.4
am_colors = ['#1F77B4' if w_ else '#D0D0D0' for w_ in df_combo.ours_wins]
zs_colors = ['#2E7D32' if lo > 0 else '#C8E6C9' for lo in df_combo.d_zs_lo]
axA.bar(x - w/2, df_combo.delta, w, color=am_colors, edgecolor='black', linewidth=0.4)
axA.bar(x + w/2, df_combo.d_zs, w, color=zs_colors, edgecolor='black', linewidth=0.4)
axA.errorbar(x - w/2, df_combo.delta,
              yerr=[df_combo.delta - df_combo.delta_lo, df_combo.delta_hi - df_combo.delta],
              fmt='none', color='black', capsize=2, linewidth=0.6)
axA.errorbar(x + w/2, df_combo.d_zs,
              yerr=[df_combo.d_zs - df_combo.d_zs_lo, df_combo.d_zs_hi - df_combo.d_zs],
              fmt='none', color='black', capsize=2, linewidth=0.6)
axA.axhline(0, color='black', linewidth=0.5)
axA.set_xticks(x)
axA.set_xticklabels(df_combo.gene, rotation=60, ha='right', fontsize=7.5)
axA.set_ylabel('Δ AUC (HEARS-LM − baseline)\npaired bootstrap 95% CI', fontsize=8.5)
# LEGEND ABOVE plot
axA.legend(handles=[
    plt.Rectangle((0,0),1,1,fc='#1F77B4',ec='black',label='Significant vs AlphaMissense'),
    plt.Rectangle((0,0),1,1,fc='#2E7D32',ec='black',label='Significant vs ESM-2 zero-shot'),
    plt.Rectangle((0,0),1,1,fc='#D0D0D0',ec='black',label='Tie vs AlphaMissense'),
    plt.Rectangle((0,0),1,1,fc='#C8E6C9',ec='black',label='Tie vs ESM-2 zero-shot'),
], fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
axA.set_ylim(-0.15, 0.8)

axB = fig.add_subplot(gs[1, 0])
axB.text(-0.3, 1.05, 'b', fontsize=14, fontweight='bold', transform=axB.transAxes)
tk = pd.read_csv(EVAL/'topk_residue_per_gene.csv')
ks = [5, 10, 20, 50]
om = [tk[f'P@{k}_O'].mean() for k in ks]
am_ = [tk[f'P@{k}_AM'].mean() for k in ks]
zsm = [tk[f'P@{k}_zs'].mean() for k in ks]
axB.plot(ks, om, 'o-', color='#1F77B4', label='HEARS-LM', markersize=8, linewidth=2)
axB.plot(ks, am_, 's-', color='#9467BD', label='AlphaMissense', markersize=7, linewidth=1.5)
axB.plot(ks, zsm, '^-', color='#D62728', label='ESM-2 zero-shot', markersize=7, linewidth=1.5)
axB.set_xlabel('K (top residues per gene)', fontsize=8)
axB.set_ylabel('Mean P@K across 36 genes', fontsize=8)
axB.set_xticks(ks); axB.set_ylim(0.3, 0.95)
# Legend in upper-right corner inside plot — but with framealpha=0.95 white box
axB.legend(fontsize=7, loc='lower left', framealpha=0.95)
axB.grid(True, linestyle=':', alpha=0.4)

axC = fig.add_subplot(gs[1, 1:3])
axC.text(-0.08, 1.05, 'c', fontsize=14, fontweight='bold', transform=axC.transAxes)
multi_data = {
    'HEARS-LM':       0.669, 'MutPred':         0.494,
    'DeOgen-2':        0.421, 'METARNN':         0.417,
    'PolyPhen-2':      0.411, 'REVEL':           0.403,
    'AlphaMissense':   0.372, 'PrimateAI':       0.372,
    'ESM-2 zero-shot': 0.360, 'ESM-1b':          0.116,
}
y = np.arange(len(multi_data))
colors = ['#1F77B4' if k == 'HEARS-LM' else '#B0BEC5' for k in multi_data.keys()]
axC.barh(y, list(multi_data.values()), color=colors, edgecolor='black', linewidth=0.5)
for i, (k, v) in enumerate(multi_data.items()):
    axC.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8,
             fontweight='bold' if k == 'HEARS-LM' else 'normal')
axC.set_yticks(y); axC.set_yticklabels(list(multi_data.keys()), fontsize=8)
axC.set_xlabel('Mean P@10 across 42 HHL genes', fontsize=8)
axC.set_xlim(0, 0.78)

axD = fig.add_subplot(gs[1, 3])
axD.text(-0.3, 1.05, 'd', fontsize=14, fontweight='bold', transform=axD.transAxes)
perfect_count = {'HEARS-LM': 13, 'AlphaMissense': 2, 'ESM-2 zero-shot': 2}
y = np.arange(len(perfect_count))
colors_d = ['#1F77B4','#9467BD','#D62728']
axD.barh(y, list(perfect_count.values()), color=colors_d, edgecolor='black', linewidth=0.6)
for i, v in enumerate(perfect_count.values()):
    axD.text(v + 0.2, i, f'{v}/36', va='center', fontsize=9, fontweight='bold')
axD.set_yticks(y); axD.set_yticklabels(list(perfect_count.keys()), fontsize=8)
axD.set_xlim(0, 17)
axD.set_xlabel('Genes with P@10 = 100%', fontsize=8)

axE = fig.add_subplot(gs[2, 0])
axE.text(-0.3, 1.05, 'e', fontsize=14, fontweight='bold', transform=axE.transAxes)
data_e = {
    'Syndromic (13)': 0.192,
    'AR non-syndr. (10)': 0.154,
    'AD non-syndr. (2)': 0.012,
    'X-linked (1)': -0.010,
}
colors_e = ['#FF7043','#FFA726','#42A5F5','#9CCC65']
y = np.arange(len(data_e))
axE.barh(y, list(data_e.values()), color=colors_e, edgecolor='black', linewidth=0.6)
for i, v in enumerate(data_e.values()):
    axE.text(v + (0.005 if v > 0 else -0.005), i, f'{v:+.3f}',
              va='center', ha='left' if v > 0 else 'right', fontsize=8)
axE.set_yticks(y); axE.set_yticklabels(list(data_e.keys()), fontsize=7.5)
axE.set_xlabel('Median Δ AUC (vs AlphaMissense)', fontsize=8)
axE.axvline(0, color='black', linewidth=0.5)
axE.set_xlim(-0.05, 0.25)

axF = fig.add_subplot(gs[2, 1:])
axF.text(-0.04, 1.05, 'f', fontsize=14, fontweight='bold', transform=axF.transAxes)
clinical = {
    'OTOG': 'DFNB18B', 'SLC4A11': 'CHED2 + DFNB13', 'LOXHD1': 'DFNB77',
    'USH1C': 'Usher 1C', 'ADGRV1': 'Usher 2C', 'CDH23': 'USH1D/DFNB12',
    'ALMS1': 'Alström', 'WFS1': 'Wolfram', 'OTOF': 'DFNB9 (DB-OTO trial)',
    'USH2A': 'Usher 2A', 'CHD7': 'CHARGE', 'TECTA': 'DFNB21/DFNA8/12',
}
sig_genes = df_am[df_am.ours_wins].sort_values('delta', ascending=False)
y = np.arange(len(sig_genes))
w = 0.4
axF.barh(y - w/2, sig_genes.auc_ho, w, color='#1F77B4', edgecolor='black', linewidth=0.5, label='HEARS-LM')
axF.barh(y + w/2, sig_genes.auc_am, w, color='#9467BD', edgecolor='black', linewidth=0.5, label='AlphaMissense')
for i, (g, d) in enumerate(zip(sig_genes.gene, sig_genes.delta)):
    axF.text(1.01, i, f'{g} ({clinical.get(g, "")})', va='center', fontsize=7)
    # Δ values to the LEFT of bars, not above
    axF.text(0.02, i, f'Δ=+{d:.3f}', va='center', fontsize=6.5, color='darkred', ha='left')
axF.set_yticks([])
axF.set_xlim(0, 1.55)
axF.set_xlabel('AUC (per-residue disease detection)', fontsize=8)
# Legend outside plot
axF.legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=2)

plt.savefig(OUT/'Fig3_landscape.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT/'Fig3_landscape.pdf', bbox_inches='tight')
plt.close()
print('✅ Fig 3')


# =================================================================
# FIG 4 — domain labels spaced out, text moved
# =================================================================
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 3, hspace=0.7, wspace=0.5, height_ratios=[1, 1])

axA = fig.add_subplot(gs[0, :])
axA.text(-0.04, 1.08, 'a', fontsize=14, fontweight='bold', transform=axA.transAxes)
all_p = []
for s in ['val_db','val_time','val_gene','val_func_mave']:
    df = pd.read_csv(ROOT/f'data/splits/{s}_with_features.csv', low_memory=False)
    pred = pd.read_csv(EVAL/f'{s}_predictions_t30_ep2.csv')
    df = df.merge(pred[['gene','protein_change','pred_prob']], on=['gene','protein_change'], how='left')
    sub = df[(df.gene == 'KCNQ4') & df.pred_prob.notna()]
    if len(sub) > 0:
        all_p.append(sub[['aa_pos','pred_prob','protein_change']])
kcnq = pd.concat(all_p).drop_duplicates('protein_change')
pos_p = kcnq.groupby('aa_pos').pred_prob.mean().reset_index().sort_values('aa_pos')

DISEASE = {71, 154, 162, 166, 182, 184, 188, 196, 199, 222, 229, 231, 232, 233, 234, 236,
           237, 240, 258, 260, 263, 266, 269, 271, 272, 273, 274, 275, 276, 277, 278, 280,
           281, 283, 285, 286, 287, 288, 296, 298, 309, 312, 314, 316, 348, 359, 369, 374}

# Domains: only label major ones; pore distinctly
DOMAINS = [
    ('S1', 91, 130, '#E1F5FE'),
    ('S4', 221, 250, '#4FC3F7'),
    ('Pore', 271, 295, '#FFAB91'),
    ('S6', 296, 330, '#03A9F4'),
]
ALL_TM = [(91, 130, '#E1F5FE'), (131, 180, '#B3E5FC'), (181, 220, '#81D4FA'),
          (221, 250, '#4FC3F7'), (251, 270, '#29B6F6'), (271, 295, '#FFAB91'),
          (296, 330, '#03A9F4')]
for s, e, c in ALL_TM:
    axA.axvspan(s, e, alpha=0.4 if c == '#FFAB91' else 0.25, color=c)
# Only label key domains with enough space
axA.text(110, 1.30, 'S1', ha='center', fontsize=8, fontweight='bold')
axA.text(235, 1.30, 'S4 (voltage)', ha='center', fontsize=8, fontweight='bold')
axA.text(283, 1.30, 'Pore', ha='center', fontsize=9, fontweight='bold', color='darkred')
axA.text(313, 1.30, 'S6', ha='center', fontsize=8, fontweight='bold')

axA.bar(pos_p.aa_pos.values, pos_p.pred_prob.values, width=1.5, color='gray', alpha=0.55, edgecolor='none')
pos_p['smooth'] = pos_p.pred_prob.rolling(15, center=True, min_periods=1).mean()
axA.plot(pos_p.aa_pos, pos_p.smooth, color='#1F77B4', linewidth=1.5, label='15-residue rolling mean')

disease_present = pos_p[pos_p.aa_pos.isin(DISEASE)]
axA.scatter(disease_present.aa_pos, [1.10]*len(disease_present), marker='v', s=25,
             color='red', edgecolor='black', linewidth=0.4, zorder=10,
             label=f'Disease residues (n={len(disease_present)})')

axA.set_xlim(0, 695); axA.set_ylim(0, 1.40)
axA.set_xlabel('KCNQ4 amino acid position', fontsize=8.5)
axA.set_ylabel('Mean predicted pathogenicity score', fontsize=8.5)
# Legend below x-axis OUTSIDE plot region
axA.legend(fontsize=7.5, loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=2)
axA.axhline(0.5, color='black', linewidth=0.4, linestyle=':', alpha=0.5)

# Panel B — text box BELOW violin
axB = fig.add_subplot(gs[1, 0])
axB.text(-0.25, 1.05, 'b', fontsize=14, fontweight='bold', transform=axB.transAxes)
pos_p['is_disease'] = pos_p.aa_pos.isin(DISEASE).astype(int)
dp = pos_p[pos_p.is_disease == 1].pred_prob
nd = pos_p[pos_p.is_disease == 0].pred_prob
u, pmw = mannwhitneyu(dp, nd, alternative='greater')
auc_d = roc_auc_score(pos_p.is_disease, pos_p.pred_prob)
parts = axB.violinplot([nd.values, dp.values], positions=[0, 1], widths=0.7, showmedians=True, showextrema=False)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(['#90CAF9', '#EF5350'][i]); pc.set_alpha(0.8); pc.set_edgecolor('black')
axB.set_xticks([0, 1])
axB.set_xticklabels([f'Non-disease\nn={int((pos_p.is_disease == 0).sum())}',
                      f'Disease\nn={int(pos_p.is_disease.sum())}'], fontsize=8)
axB.set_ylabel('Per-residue mean predicted pathogenicity', fontsize=8)
axB.set_ylim(0, 1.35)
# Text ABOVE violins, well clear
axB.text(0.5, 1.22, f'AUC = {auc_d:.3f}    p = {pmw:.1e}', ha='center', fontsize=8.5,
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='black'))

# Panel C — legend below
axC = fig.add_subplot(gs[1, 1])
axC.text(-0.25, 1.05, 'c', fontsize=14, fontweight='bold', transform=axC.transAxes)
tm_genes = ['KCNQ1','KCNQ4','GJB2','SLC4A11','TMC1','WFS1']
tm_pcts = [47, 50, 55, 67, 49, 27]; tm_base = [20, 20, 37, 33, 40, 26]
tm_pvals = ['2.4×10⁻⁷','4.9×10⁻⁶','4.5×10⁻⁴','0.053','0.20','0.18']
y = np.arange(len(tm_genes)); w = 0.4
axC.barh(y - w/2, tm_pcts, w, color='#EF5350', edgecolor='black', linewidth=0.5, label='Observed (high-conf P)')
axC.barh(y + w/2, tm_base, w, color='#90CAF9', edgecolor='black', linewidth=0.5, label='Baseline')
for i, (a, b, p) in enumerate(zip(tm_pcts, tm_base, tm_pvals)):
    axC.text(max(a, b) + 3, i, p, fontsize=7, va='center',
              fontweight='bold' if '10⁻' in p else 'normal')
axC.set_yticks(y); axC.set_yticklabels(tm_genes, fontsize=8)
axC.set_xlabel('% residues in transmembrane domain', fontsize=8)
axC.legend(fontsize=7, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
axC.set_xlim(0, 90)

# Panel D — legend below
axD = fig.add_subplot(gs[1, 2])
axD.text(-0.25, 1.05, 'd', fontsize=14, fontweight='bold', transform=axD.transAxes)
cluster_data = [
    ('KCNQ4', 34.5, 62.0, '<10⁻⁴', True),
    ('KCNQ1', 40.7, 54.8, '<10⁻⁴', True),
    ('OTOF', 55.8, 56.5, '0.34', False),
    ('USH2A', 80.9, 84.3, '0.24', False),
    ('CDH23', 120.8, 112.9, '0.91', False),
]
genes = [d[0] for d in cluster_data]
obs = [d[1] for d in cluster_data]; exp = [d[2] for d in cluster_data]
sigs = [d[3] for d in cluster_data]; sig_bool = [d[4] for d in cluster_data]
y = np.arange(len(genes)); w = 0.4
colors_obs = ['#EF5350' if s else '#FFAB91' for s in sig_bool]
axD.barh(y - w/2, obs, w, color=colors_obs, edgecolor='black', linewidth=0.5, label='Observed (high-conf P)')
axD.barh(y + w/2, exp, w, color='#81D4FA', edgecolor='black', linewidth=0.5, label='Random expectation')
for i, (o, e, p) in enumerate(zip(obs, exp, sigs)):
    axD.text(max(o, e) + 4, i, f'p = {p}', fontsize=7, va='center',
              fontweight='bold' if '10⁻' in p else 'normal')
axD.set_yticks(y); axD.set_yticklabels(genes, fontsize=8)
axD.set_xlabel('Mean pairwise CA distance (Å)', fontsize=8)
axD.legend(fontsize=7, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
axD.set_xlim(0, 150)

plt.savefig(OUT/'Fig4_biology.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT/'Fig4_biology.pdf', bbox_inches='tight')
plt.close()
print('✅ Fig 4')


# =================================================================
# FIG 5 — text boxes moved, legends below
# =================================================================
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 3, hspace=0.65, wspace=0.55)

recl = pd.read_csv('/Users/pipi/Projects/QAFI_Paper/hearing/data/external/clinvar_reclassifications.csv')
recl[['gene','protein_change']] = recl._key.apply(lambda k: pd.Series(k.split('|',1)))
recl = recl[recl.protein_change.str.match(r'^[A-Z][a-z]{2}\d+[A-Z][a-z]{2}$', na=False)]
upgrades = recl[(recl['2025-01']=='V') & (recl['2026-04']=='P')]
clinvar = pd.read_csv('/Users/pipi/Projects/QAFI_Paper/hearing/data/clinvar_hhl_all.csv', low_memory=False)
clinvar_vus = clinvar[clinvar.category == 'VUS'][['gene','protein_change']].drop_duplicates()
stable_vus = clinvar_vus[~(clinvar_vus.gene + '/' + clinvar_vus.protein_change.astype(str)).isin(
    set(recl.gene + '/' + recl.protein_change.astype(str)))]
all_p = []
for s in ['val_db','val_time','val_gene','val_func_mave']:
    df = pd.read_csv(ROOT/f'data/splits/{s}_with_features.csv', low_memory=False)
    pred = pd.read_csv(EVAL/f'{s}_predictions_t30_ep2.csv')
    df = df.merge(pred[['gene','protein_change','pred_prob']], on=['gene','protein_change'], how='left')
    keep = ['gene','protein_change','pred_prob']
    for c in ['esm_llr','baseline_alphamissense_score','baseline_metarnn_score','baseline_revel_score']:
        if c in df.columns: keep.append(c)
    all_p.append(df[df.pred_prob.notna()][keep])
train_pred = pd.read_csv(EVAL/'train_predictions_t30_ep2.csv')[['gene','protein_change','pred_prob']]
all_p.append(train_pred[train_pred.pred_prob.notna()])
ours = pd.concat(all_p, ignore_index=True).drop_duplicates(['gene','protein_change'])
up_pred = upgrades.merge(ours, on=['gene','protein_change'], how='inner')
stb_pred = stable_vus.merge(ours, on=['gene','protein_change'], how='inner')

axA = fig.add_subplot(gs[0, 0])
axA.text(-0.25, 1.05, 'a', fontsize=14, fontweight='bold', transform=axA.transAxes)
methods = {
    'HEARS-LM':       ('pred_prob', '#1F77B4'),
    'METARNN':        ('baseline_metarnn_score', '#FF7F0E'),
    'AlphaMissense':  ('baseline_alphamissense_score', '#9467BD'),
    'ESM-2 zero-shot':('esm_llr', '#D62728'),
    'REVEL':          ('baseline_revel_score', '#2CA02C'),
}
for name, (col, c) in methods.items():
    if col not in up_pred.columns or col not in stb_pred.columns: continue
    ups = up_pred[col].dropna().values; stbs = stb_pred[col].dropna().values
    if len(ups) < 10 or len(stbs) < 30: continue
    yy = np.concatenate([np.ones(len(ups)), np.zeros(len(stbs))])
    ss = np.concatenate([ups, stbs])
    auc = roc_auc_score(yy, ss)
    if auc < 0.5: ss = -ss; auc = 1 - auc
    fpr, tpr, _ = roc_curve(yy, ss)
    lw = 2 if name == 'HEARS-LM' else 1.2
    axA.plot(fpr, tpr, color=c, linewidth=lw, label=f'{name}: {auc:.3f}')
axA.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
axA.set_xlabel('False positive rate', fontsize=8)
axA.set_ylabel('True positive rate', fontsize=8)
axA.legend(fontsize=7, loc='lower right', title='AUC', framealpha=0.95)
axA.grid(True, linestyle=':', alpha=0.4)
axA.set_xlim(0, 1); axA.set_ylim(0, 1.02)

axB = fig.add_subplot(gs[0, 1])
axB.text(-0.25, 1.05, 'b', fontsize=14, fontweight='bold', transform=axB.transAxes)
up_v = up_pred.pred_prob.dropna().values
stb_v = stb_pred.pred_prob.dropna().values
parts = axB.violinplot([stb_v, up_v], positions=[0, 1], widths=0.7, showmedians=True, showextrema=False)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(['#90CAF9','#EF5350'][i]); pc.set_alpha(0.8); pc.set_edgecolor('black')
u, p = mannwhitneyu(up_v, stb_v, alternative='greater')
axB.set_xticks([0, 1])
axB.set_xticklabels([f'Stable VUS\n(n={len(stb_v)})', f'Future P upgrades\n(n={len(up_v)})'], fontsize=8)
axB.set_ylabel('Predicted pathogenicity score', fontsize=8)
axB.set_ylim(0, 1.25)
# Text ABOVE violins, no overlap
axB.text(0.5, 1.13, f'stable {stb_v.mean():.3f}   upgrade {up_v.mean():.3f}   p={p:.1e}',
          ha='center', fontsize=7.5,
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='black'))
axB.axhline(0.5, color='gray', linewidth=0.5, linestyle='--')

axC = fig.add_subplot(gs[0, 2])
axC.text(-0.25, 1.05, 'c', fontsize=14, fontweight='bold', transform=axC.transAxes)
prosp_aucs = {
    'HEARS-LM':       0.748, 'METARNN':       0.732,
    'AlphaMissense':  0.722, 'ESM-2 zero-shot':0.701,
    'REVEL':          0.608,
}
y = np.arange(len(prosp_aucs))
colors_c = ['#1F77B4','#FF7F0E','#9467BD','#D62728','#2CA02C']
axC.barh(y, list(prosp_aucs.values()), color=colors_c, edgecolor='black', linewidth=0.6)
for i, (k, v) in enumerate(prosp_aucs.items()):
    axC.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9,
              fontweight='bold' if k == 'HEARS-LM' else 'normal')
axC.set_yticks(y); axC.set_yticklabels(list(prosp_aucs.keys()), fontsize=8)
axC.set_xlabel('AUC for prospective upgrade prediction', fontsize=8)
axC.set_xlim(0.55, 0.78)

# Panel D - text moved above bars
axD = fig.add_subplot(gs[1, 0])
axD.text(-0.25, 1.05, 'd', fontsize=14, fontweight='bold', transform=axD.transAxes)
acmg_data = [
    ('No BP4\n(n=154)', 0.648, '#90A4AE'),
    ('BP4 applied\n(n=13)', 0.209, '#42A5F5'),
    ('No PP3\n(n=89)', 0.497, '#90A4AE'),
    ('PP3 applied\n(n=78)', 0.747, '#EF5350'),
]
x = np.arange(len(acmg_data))
means = [v[1] for v in acmg_data]
colors_d = [v[2] for v in acmg_data]
axD.bar(x, means, color=colors_d, edgecolor='black', linewidth=0.6)
for i, (l, m, _) in enumerate(acmg_data):
    axD.text(i, m + 0.02, f'{m:.2f}', ha='center', fontsize=9, fontweight='bold')
axD.set_xticks(x); axD.set_xticklabels([v[0] for v in acmg_data], fontsize=7)
axD.set_ylabel('Mean predicted pathogenicity', fontsize=8)
axD.set_ylim(0, 1.15)
# Text in TOP area, above all bars
axD.text(0.5, 1.05, 'BP4: p = 8.4×10⁻⁷   PP3: p = 1.2×10⁻⁶', ha='center', fontsize=8,
          transform=axD.transAxes,
          bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))
axD.axhline(0.5, color='gray', linewidth=0.5, linestyle='--')

axE = fig.add_subplot(gs[1, 1])
axE.text(-0.25, 1.05, 'e', fontsize=14, fontweight='bold', transform=axE.transAxes)
clingen_data = {
    'val_db subset\n(n=29)': 0.648,
    'val_time subset\n(n=43)': 0.912,
    'val_gene subset\n(n=21)': 0.971,
    'Combined held-out\n(n=66)': 0.901,
}
y = np.arange(len(clingen_data))
colors_e = ['#FFAB91','#FFD54F','#A5D6A7','#1F77B4']
axE.barh(y, list(clingen_data.values()), color=colors_e, edgecolor='black', linewidth=0.6)
for i, (k, v) in enumerate(clingen_data.items()):
    axE.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9,
              fontweight='bold' if 'Combined' in k else 'normal')
axE.set_yticks(y); axE.set_yticklabels(list(clingen_data.keys()), fontsize=8)
axE.set_xlabel('AUC on ClinGen HL-VCEP variants', fontsize=8)
axE.set_xlim(0.5, 1.0)

axF = fig.add_subplot(gs[1, 2])
axF.text(-0.07, 1.05, 'f', fontsize=14, fontweight='bold', transform=axF.transAxes)
axF.axis('off')
kcnq4_rules = [
    ['Tyr270', 'Pore helix N-term', 'supporting', '✓'],
    ['Trp275', 'Pore region', 'moderate', '✓'],
    ['Trp276', 'Pore region', 'moderate', '✓'],
    ['Trp277', 'Pore region', 'moderate', '✓'],
    ['Gly285', 'Selectivity filter', 'moderate', '✓'],
    ['Gly296', 'Pore region', 'supporting', '✗'],
]
data = [['Residue', 'Region', 'ACMG', 'Found']] + kcnq4_rules
# Table with more vertical room for notes
table = axF.table(cellText=data[1:], colLabels=data[0],
                   colWidths=[0.22, 0.36, 0.22, 0.16], loc='upper center', cellLoc='center',
                   bbox=[0, 0.3, 1, 0.65])
table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.5)
for j in range(4):
    table[(0, j)].set_text_props(weight='bold')
    table[(0, j)].set_facecolor('#F5F5F5')
for i in range(1, 7):
    c = '#C8E6C9' if data[i][3] == '✓' else '#FFCDD2'
    table[(i, 3)].set_facecolor(c)
# Note BELOW the table, not overlapping
axF.text(0.5, 0.05, 'OR = 88, p = 4.7×10⁻⁶\n5 of 6 specific residues recovered',
          ha='center', fontsize=8, transform=axF.transAxes,
          bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))

plt.savefig(OUT/'Fig5_prospective.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT/'Fig5_prospective.pdf', bbox_inches='tight')
plt.close()
print('✅ Fig 5')


# =================================================================
# FIG 6
# =================================================================
fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(3, 3, hspace=0.65, wspace=0.5)

axA = fig.add_subplot(gs[0, :])
axA.text(-0.04, 1.08, 'a', fontsize=14, fontweight='bold', transform=axA.transAxes)
all_p = []
for s in ['val_db','val_time','val_gene','val_func_mave']:
    df = pd.read_csv(ROOT/f'data/splits/{s}_with_features.csv', low_memory=False)
    pred = pd.read_csv(EVAL/f'{s}_predictions_t30_ep2.csv')
    df = df.merge(pred[['gene','protein_change','pred_prob']], on=['gene','protein_change'], how='left')
    sub = df[(df.gene == 'OTOF') & df.pred_prob.notna()]
    if len(sub) > 0:
        all_p.append(sub[['aa_pos','pred_prob','protein_change']])
otof = pd.concat(all_p).drop_duplicates('protein_change')
otof_pos = otof.groupby('aa_pos').pred_prob.mean().reset_index().sort_values('aa_pos')
clinvar = pd.read_csv('/Users/pipi/Projects/QAFI_Paper/hearing/data/clinvar_hhl_all.csv', low_memory=False)
def parse_pos(p):
    if pd.isna(p): return None
    m = re.match(r'^[A-Z][a-z]{2}?(\d+)', str(p))
    return int(m.group(1)) if m else None
clinvar['aa_pos'] = clinvar.protein_change.apply(parse_pos)
otof_cv = clinvar[clinvar.gene == 'OTOF']
plp_pos = set(otof_cv[otof_cv.category == 'P_LP'].aa_pos.dropna().astype(int))

OTOF_DOMAINS = [
    ('C2-A', 1, 132, '#E3F2FD'), ('C2-B', 371, 489, '#BBDEFB'),
    ('C2-C', 575, 691, '#90CAF9'), ('C2-D', 910, 1029, '#64B5F6'),
    ('C2-E', 1060, 1178, '#42A5F5'), ('C2-F', 1700, 1822, '#2196F3'),
    ('TM', 1964, 1984, '#FF6F00'),
]
for n, s, e, c in OTOF_DOMAINS:
    axA.axvspan(s, e, color=c, alpha=0.55)
    axA.text((s+e)/2, 1.22, n, ha='center', fontsize=8, fontweight='bold')

axA.bar(otof_pos.aa_pos.values, otof_pos.pred_prob.values, width=2.5, color='gray', alpha=0.55, edgecolor='none')
otof_pos['smooth'] = otof_pos.pred_prob.rolling(30, center=True, min_periods=1).mean()
axA.plot(otof_pos.aa_pos, otof_pos.smooth, color='#0D47A1', linewidth=1.5, label='30-residue rolling mean')

plp_present = otof_pos[otof_pos.aa_pos.isin(plp_pos)]
axA.scatter(plp_present.aa_pos, [1.10]*len(plp_present), marker='v', s=18,
             color='red', edgecolor='black', linewidth=0.3, zorder=10,
             label=f'ClinVar P/LP positions (n={len(plp_present)})')

axA.set_xlim(0, 1997); axA.set_ylim(0, 1.40)
axA.set_xlabel('OTOF amino acid position', fontsize=8.5)
axA.set_ylabel('Mean predicted pathogenicity score', fontsize=8.5)
axA.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)
axA.axhline(0.5, color='black', linewidth=0.4, linestyle=':', alpha=0.5)

axB = fig.add_subplot(gs[1, 0])
axB.text(-0.25, 1.05, 'b', fontsize=14, fontweight='bold', transform=axB.transAxes)
otof_aucs = {
    'HEARS-LM':       0.950,
    'AlphaMissense':  0.752,
    'ESM-2 zero-shot':0.626,
}
y = np.arange(len(otof_aucs))
colors_b = ['#1F77B4','#9467BD','#D62728']
axB.barh(y, list(otof_aucs.values()), color=colors_b, edgecolor='black', linewidth=0.6)
for i, v in enumerate(otof_aucs.values()):
    axB.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')
axB.set_yticks(y); axB.set_yticklabels(list(otof_aucs.keys()), fontsize=8)
axB.set_xlim(0, 1.15)
axB.set_xlabel('AUC (OTOF P/LP residue detection)', fontsize=8)
# Note BELOW plot
axB.text(0.5, -0.35, 'Δ vs AlphaMissense = +0.198\nΔ vs ESM-2 zero-shot = +0.324',
          transform=axB.transAxes, fontsize=7.5, ha='center',
          bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))

# Panel C
axC = fig.add_subplot(gs[1, 1:])
axC.text(-0.04, 1.05, 'c', fontsize=14, fontweight='bold', transform=axC.transAxes)
vus_data = [
    ('WFS1',    471, 59, 'Wolfram'),
    ('USH2A',   1783, 50, 'Usher 2A'),
    ('CDH23',   1216, 43, 'USH1D'),
    ('LOXHD1',  423, 22, 'DFNB77'),
    ('OTOF',    295, 16, 'DFNB9 + DB-OTO'),
    ('TECTA',   446, 14, 'DFNB21'),
    ('ALMS1',   1954, 8, 'Alström'),
    ('CHD7',    855, 7, 'CHARGE'),
    ('ADGRV1',  1380, 5, 'USH2C'),
    ('USH1C',   242, 2, 'Usher 1C'),
    ('SLC4A11', 115, 1, 'CHED2'),
    ('OTOG',    375, 0, 'DFNB18B'),
]
genes = [v[0] for v in vus_data]
n_vus = [v[1] for v in vus_data]
n_reclass = [v[2] for v in vus_data]
diseases = [v[3] for v in vus_data]
y = np.arange(len(genes))
axC.barh(y, n_reclass, color='#7E57C2', edgecolor='black', linewidth=0.5)
for i, (g, v, r, d) in enumerate(zip(genes, n_vus, n_reclass, diseases)):
    axC.text(r + 1, i, f' {r} / {v} VUS — {d}', va='center', fontsize=7.5)
axC.set_yticks(y); axC.set_yticklabels(genes, fontsize=8)
axC.set_xlabel('High-confidence VUS reclassifications', fontsize=8)
axC.set_xlim(0, max(n_reclass)*1.8)

# Panel D
axD = fig.add_subplot(gs[2, 0])
axD.text(-0.25, 1.05, 'd', fontsize=14, fontweight='bold', transform=axD.transAxes)
labels = ['At known P/LP\nposition\n(181, 80%)', 'Truly novel\nposition\n(26, 11%)', 'Ultra-novel\n>5 aa\n(20, 9%)']
sizes = [181, 26, 20]
colors_d = ['#90CAF9','#FFA726','#EF5350']
wedges, texts = axD.pie(sizes, labels=labels, colors=colors_d, startangle=90,
                          explode=(0, 0.06, 0.1), textprops={'fontsize': 7},
                          wedgeprops=dict(edgecolor='black', linewidth=0.6),
                          labeldistance=1.2)

# Panel E
axE = fig.add_subplot(gs[2, 1:])
axE.text(-0.04, 1.05, 'e', fontsize=14, fontweight='bold', transform=axE.transAxes)
axE.axis('off')
ultra = [
    ('USH2A', 3425, 'Usher 2A'), ('CDH23', 2720, 'USH1D'),
    ('USH2A', 1368, 'Usher 2A'), ('CDH23', 2844, 'USH1D'),
    ('CDH23', 864, 'USH1D'),     ('CDH23', 1583, 'USH1D'),
    ('CDH23', 1610, 'USH1D'),    ('LOXHD1', 577, 'DFNB77'),
    ('WFS1', 779, 'Wolfram'),    ('WFS1', 351, 'Wolfram'),
    ('WFS1', 177, 'Wolfram'),    ('WFS1', 494, 'Wolfram'),
    ('WFS1', 528, 'Wolfram'),    ('CHD7', 1352, 'CHARGE'),
    ('ADGRV1', 5970, 'USH2C'),   ('TECTA', 1314, 'DFNB21'),
    ('TECTA', 1993, 'DFNB21'),   ('TECTA', 1352, 'DFNB21'),
]
ncols, n_per_col = 3, 6
header = ['Gene','Position','Disease']*ncols
table_data = [header]
for i in range(n_per_col):
    row = []
    for c in range(ncols):
        idx = c * n_per_col + i
        if idx < len(ultra):
            g, p, d = ultra[idx]
            row.extend([g, str(p), d])
        else:
            row.extend(['', '', ''])
    table_data.append(row)
table = axE.table(cellText=table_data[1:], colLabels=table_data[0],
                   colWidths=[0.10, 0.08, 0.14]*3, loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1, 1.6)
for j in range(9):
    table[(0, j)].set_text_props(weight='bold')
    table[(0, j)].set_facecolor('#F5F5F5')

plt.savefig(OUT/'Fig6_otof_clinical.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT/'Fig6_otof_clinical.pdf', bbox_inches='tight')
plt.close()
print('✅ Fig 6')
