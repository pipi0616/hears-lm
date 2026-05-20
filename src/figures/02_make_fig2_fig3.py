#!/usr/bin/env python3
"""Final label-occlusion fixes for all figures."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np, pandas as pd, re
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve

OUT = Path('/Users/pipi/Projects/QAFI_Paper/hearing_v2/figures')
ROOT = Path('/Users/pipi/Projects/QAFI_Paper/hearing_v2')
EVAL = ROOT/'eval_results'
plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif',
                     'figure.dpi': 200, 'savefig.dpi': 300,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

# =================================================================
# FIG 1 — Panel E +0.013 label cut at right
# =================================================================
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 4, hspace=0.6, wspace=0.7, height_ratios=[1, 1.0])

axA = fig.add_subplot(gs[0, :3])
axA.set_xlim(0, 12.5); axA.set_ylim(0, 6.4); axA.axis('off')
axA.text(0.0, 1.0, 'a', fontsize=14, fontweight='bold', transform=axA.transAxes)

def box(ax, x, y, w, h, label, color, fs=8):
    b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06',
                       edgecolor='black', facecolor=color, linewidth=1.2)
    ax.add_patch(b)
    ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=fs, fontweight='bold')

def arr(ax, x1, y1, x2, y2, color='black'):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=14,
                        color=color, linewidth=1.3)
    ax.add_patch(a)

box(axA, 0.2, 5.0, 2.0, 0.7, 'Protein sequence\n(ESM-2 tokens)', '#FFE5B4')
box(axA, 0.2, 3.7, 2.0, 0.8, '100 HHL features\n(HPO, IMPC, CGD,\ngnomAD, structural)', '#E0F2F1')
box(axA, 0.2, 2.4, 2.0, 0.7, 'AlphaFold pLDDT,\nin_domain', '#FFE0B2')
box(axA, 0.2, 1.1, 2.0, 0.7, 'FoldX ΔΔG\n(aux supervision)', '#FFCC80')

box(axA, 3.3, 4.0, 2.3, 1.1, 'ESM-2 t30\n(150 M params)\n+ LoRA r=16\n(1.2 M trainable)', '#90CAF9')
arr(axA, 2.25, 5.35, 3.25, 4.65)

box(axA, 6.0, 4.0, 2.0, 1.1, 'Cross-attention\nfusion', '#A5D6A7')
arr(axA, 5.65, 4.55, 5.95, 4.55)
arr(axA, 2.25, 4.1, 5.95, 4.3, color='gray')
arr(axA, 2.25, 2.75, 5.95, 4.2, color='gray')

heads = [
    (8.5, 5.0, 'Pathogenicity\n(focal CE)', '#EF9A9A', 'P/B prob.'),
    (8.5, 3.8, 'Thermodynamic\n(MSE)', '#FFCC80', 'ΔΔG'),
    (8.5, 2.6, 'Mechanism\n(5-class)', '#CE93D8', 'mech'),
    (8.5, 1.4, 'Ordinal\nseverity', '#90A4AE', 'severity'),
]
for x, y, lbl, c, out_lbl in heads:
    box(axA, x, y, 1.9, 0.8, lbl, c)
    arr(axA, 8.0, 4.55, 8.45, y+0.4, color='gray')
    axA.text(10.55, y+0.4, out_lbl, fontsize=8, va='center', style='italic')

# Panel B
axB = fig.add_subplot(gs[0, 3])
axB.text(-0.3, 1.0, 'b', fontsize=14, fontweight='bold', transform=axB.transAxes)
splits = ['train', 'val_db', 'val_time', 'val_gene', 'val_func_mave']
sizes = [36397, 2835, 1885, 1004, 17367]
descs = ['n=36,397\nP=14%', 'n=2,835\nP=73%', 'n=1,885\nP=47%', 'n=1,004\nP=73%', 'n=17,367\nfunctional']
colors_b = ['#90CAF9', '#FFAB91', '#FFE082', '#A5D6A7', '#CE93D8']
y_pos = np.arange(len(splits))
axB.barh(y_pos, sizes, color=colors_b, edgecolor='black', linewidth=0.6)
for i, (s, d) in enumerate(zip(sizes, descs)):
    axB.text(s*1.2, i, d, va='center', fontsize=6.5)
axB.set_yticks(y_pos); axB.set_yticklabels(splits, fontsize=7.5)
axB.invert_yaxis()
axB.set_xscale('log')
axB.set_xlim(50, 500000)
axB.set_xlabel('Variants (log scale)', fontsize=8)

# Panel C
axC = fig.add_subplot(gs[1, 0])
axC.text(-0.3, 1.05, 'c', fontsize=14, fontweight='bold', transform=axC.transAxes)
groups = [
    ('Cell-type expr.', 44, '#FFCDD2'),
    ('Allele frequency', 10, '#C8E6C9'),
    ('Complex API', 9, '#BBDEFB'),
    ('Mouse phenotype', 8, '#FFE0B2'),
    ('HPO disease', 6, '#D1C4E9'),
    ('Other / meta', 6, '#CFD8DC'),
    ('CGD inheritance', 5, '#F0F4C3'),
    ('Structural', 5, '#FFCCBC'),
    ('Conservation', 4, '#B2DFDB'),
    ('gnomAD constraint', 3, '#F8BBD0'),
]
y = np.arange(len(groups))
counts = [g[1] for g in groups]
labels = [g[0] for g in groups]
colors = [g[2] for g in groups]
axC.barh(y, counts, color=colors, edgecolor='black', linewidth=0.5)
for i, c in enumerate(counts):
    axC.text(c+0.5, i, str(c), va='center', fontsize=8)
axC.set_yticks(y); axC.set_yticklabels(labels, fontsize=7)
axC.invert_yaxis()
axC.set_xlabel('Number of features', fontsize=8)
axC.set_xlim(0, 55)

# Panel D
axD = fig.add_subplot(gs[1, 1])
axD.text(-0.3, 1.05, 'd', fontsize=14, fontweight='bold', transform=axD.transAxes)
tasks = [('Pathogenicity', 1.0, '#EF5350'),
         ('Thermodynamic', 0.30, '#FFA726'),
         ('Mechanism', 0.20, '#AB47BC'),
         ('Ordinal severity', 0.15, '#78909C')]
y = np.arange(len(tasks))
weights = [t[1] for t in tasks]
colors_d = [t[2] for t in tasks]
axD.barh(y, weights, color=colors_d, edgecolor='black', linewidth=0.5)
for i, w in enumerate(weights):
    axD.text(w+0.03, i, f'{w}', va='center', fontsize=8)
axD.set_yticks(y); axD.set_yticklabels([t[0] for t in tasks], fontsize=7.5)
axD.set_xlabel('Loss weight', fontsize=8)
axD.set_xlim(0, 1.35)

# === Panel E: FIX — extend right xlim so '+0.013' fits completely ===
axE = fig.add_subplot(gs[1, 2])
axE.text(-0.3, 1.05, 'e', fontsize=14, fontweight='bold', transform=axE.transAxes)
ab = pd.read_csv(EVAL/'story4_ablation.csv')
loo = ab[ab.name.str.startswith('no_')].copy()
loo['feature'] = loo.name.str.replace('no_','')
loo['drop_db'] = -loo['drop_db']
loo = loo.sort_values('drop_db', ascending=True)
y = np.arange(len(loo))
colors_e = plt.cm.RdBu_r(np.linspace(0.1, 0.9, len(loo)))
axE.barh(y, loo.drop_db, color=colors_e, edgecolor='black', linewidth=0.5)
for i, v in enumerate(loo.drop_db):
    if v < 0:
        axE.text(v - 0.003, i, f'{v:+.3f}', va='center', ha='right', fontsize=6.5)
    else:
        axE.text(v + 0.003, i, f'{v:+.3f}', va='center', ha='left', fontsize=6.5)
axE.set_yticks(y); axE.set_yticklabels(loo.feature, fontsize=7)
axE.set_xlabel('Δ AUC when feature group removed', fontsize=8)
axE.axvline(0, color='black', linewidth=0.5)
# FIX: extend right xlim from 0.045 to 0.08 so '+0.013' has room
axE.set_xlim(-0.17, 0.08)

# Panel F
axF = fig.add_subplot(gs[1, 3])
axF.text(-0.3, 1.05, 'f', fontsize=14, fontweight='bold', transform=axF.transAxes)
ablation_data = {
    'val_db':   (0.661, 0.764, 0.840),
    'val_time': (0.722, 0.990, 0.973),
    'val_gene': (0.629, 0.929, 0.945),
}
labels = list(ablation_data.keys())
zs = [v[0] for v in ablation_data.values()]
lgbm = [v[1] for v in ablation_data.values()]
ours = [v[2] for v in ablation_data.values()]
x = np.arange(len(labels))
w = 0.27
axF.bar(x - w, zs, w, color='#FFCDD2', edgecolor='black', linewidth=0.5, label='ESM-2 zero-shot')
axF.bar(x, lgbm, w, color='#FFE0B2', edgecolor='black', linewidth=0.5, label='HHL features alone')
axF.bar(x + w, ours, w, color='#90CAF9', edgecolor='black', linewidth=0.5, label='HEARS-LM (full)')
for i, o in enumerate(ours):
    axF.text(i + w, o + 0.005, f'{o:.2f}', ha='center', fontsize=7, fontweight='bold')
axF.set_xticks(x); axF.set_xticklabels(labels, fontsize=8)
axF.set_ylabel('AUC', fontsize=8)
axF.legend(fontsize=6.5, loc='upper left', bbox_to_anchor=(0, -0.18), ncol=2)
axF.set_ylim(0.5, 1.05)

plt.savefig(OUT/'Fig1_architecture.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT/'Fig1_architecture.pdf', bbox_inches='tight')
plt.close()
print('✅ Fig 1 — Panel E xlim extended to fit +0.013')


# =================================================================
# FIG 2 — Panel D text box moved further from violin top
# =================================================================
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.5)

df = pd.read_csv(ROOT/'data/splits/val_gene_with_features.csv', low_memory=False)
pred = pd.read_csv(EVAL/'val_gene_predictions_t30_ep2.csv')
df = df.merge(pred[['gene','protein_change','pred_ddg']], on=['gene','protein_change'], how='left')
df = df[(df.has_ddg == 1) & df.pred_ddg.notna()].copy()

axA = fig.add_subplot(gs[0, 0])
axA.text(-0.25, 1.05, 'a', fontsize=14, fontweight='bold', transform=axA.transAxes)
hb = axA.hexbin(df.ddg_fold, df.pred_ddg, gridsize=20, cmap='YlOrRd', mincnt=1)
overall_rho = spearmanr(df.ddg_fold, df.pred_ddg).statistic
axA.set_xlabel('FoldX ΔΔG (kcal/mol)', fontsize=8)
axA.set_ylabel('Predicted ΔΔG (HEARS-LM)', fontsize=8)
axA.text(0.98, 0.05, f'Spearman ρ = {overall_rho:.3f}\nn = {len(df)} variants\n10 unseen HHL genes',
         transform=axA.transAxes, fontsize=7.5, ha='right', va='bottom',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='black'))
axA.axhline(0, color='gray', linewidth=0.4, linestyle='--')
axA.axvline(0, color='gray', linewidth=0.4, linestyle='--')
cbar = plt.colorbar(hb, ax=axA, shrink=0.7)
cbar.set_label('Variant count', fontsize=7)

axB = fig.add_subplot(gs[0, 1:])
axB.text(-0.07, 1.05, 'b', fontsize=14, fontweight='bold', transform=axB.transAxes)
methods = {
    'HEARS-LM':  ('pred_ddg', '#1F77B4'),
    'REVEL':     ('baseline_revel_score', '#FF7F0E'),
    'AlphaMissense': ('baseline_alphamissense_score', '#2CA02C'),
    'ESM-2 zero-shot': ('esm_llr', '#D62728'),
    'phyloP-100':('phylop100', '#9467BD'),
}
per_gene = {m: {} for m in methods}
for g, gsub in df.groupby('gene'):
    if len(gsub) < 20: continue
    for m, (col, _) in methods.items():
        if col not in gsub.columns: continue
        mm = gsub[col].notna()
        if mm.sum() < 10: continue
        rho = abs(spearmanr(gsub.loc[mm, col], gsub.loc[mm, 'ddg_fold']).statistic)
        if not np.isnan(rho): per_gene[m][g] = rho
common = sorted([g for g in df.gene.unique() if len([m for m in methods if g in per_gene[m]]) >= 3])
x = np.arange(len(common))
w = 0.16
for i, (m, (_, c)) in enumerate(methods.items()):
    vals = [per_gene[m].get(g, 0) for g in common]
    axB.bar(x + i*w - 2*w, vals, w, color=c, label=m, edgecolor='black', linewidth=0.4)
axB.set_xticks(x)
axB.set_xticklabels(common, rotation=45, ha='right', fontsize=8)
axB.set_ylabel('|Spearman ρ| with FoldX', fontsize=8)
axB.legend(fontsize=7.5, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False)
axB.set_ylim(0, 1.0)
axB.grid(axis='y', linestyle=':', alpha=0.4)

axC = fig.add_subplot(gs[1, 0])
axC.text(-0.25, 1.05, 'c', fontsize=14, fontweight='bold', transform=axC.transAxes)
baseline_names = ['phyloP-100','ESM-2 zero-shot','AlphaMissense','REVEL']
rng = np.random.default_rng(42)
results = []
for b in baseline_names:
    deltas = []
    for g in common:
        if g in per_gene['HEARS-LM'] and g in per_gene[b]:
            deltas.append(per_gene['HEARS-LM'][g] - per_gene[b][g])
    deltas = np.array(deltas)
    boots = [np.median(deltas[rng.integers(0, len(deltas), len(deltas))]) for _ in range(5000)]
    results.append({'b': b, 'med': np.median(deltas),
                    'lo': np.percentile(boots, 2.5), 'hi': np.percentile(boots, 97.5)})
y_pos = np.arange(len(results))
colors_c = ['#9467BD','#D62728','#2CA02C','#FF7F0E']
for i, r in enumerate(results):
    axC.barh(i, r['med'], color=colors_c[i], edgecolor='black', linewidth=0.6, alpha=0.85)
    axC.errorbar(r['med'], i, xerr=[[r['med']-r['lo']], [r['hi']-r['med']]], fmt='|', color='black', capsize=4)
    axC.text(r['hi']+0.02, i, f'+{r["med"]:.2f}', va='center', fontsize=8, fontweight='bold')
axC.set_yticks(y_pos); axC.set_yticklabels([r['b'] for r in results], fontsize=8.5)
axC.set_xlabel('Δ |Spearman| (HEARS-LM − baseline)\n95% bootstrap CI', fontsize=8)
axC.axvline(0, color='black', linewidth=0.5)
axC.set_xlim(-0.1, 1.0)

# === Panel D: FIX — increase ylim and lift text box higher ===
axD = fig.add_subplot(gs[1, 1])
axD.text(-0.25, 1.05, 'd', fontsize=14, fontweight='bold', transform=axD.transAxes)
val_db = pd.read_csv(ROOT/'data/splits/val_db_with_features.csv', low_memory=False)
val_db_t30 = pd.read_csv(EVAL/'val_db_predictions_t30_ep2.csv')
mm = val_db.merge(val_db_t30[['gene','protein_change','pred_ddg']], on=['gene','protein_change'], how='left')
mm = mm[mm.pred_ddg.notna()]
mm['abs_pred'] = mm.pred_ddg.abs()
p_var = mm[mm.y == 1]; b_var = mm[mm.y == 0]
u, pval = mannwhitneyu(p_var.abs_pred, b_var.abs_pred, alternative='greater')
# CLIP outliers above 3.0 so violin doesn't extend that high
b_vals = b_var.abs_pred.values
p_vals = p_var.abs_pred.values
parts = axD.violinplot([b_vals, p_vals], positions=[0, 1],
                       widths=0.7, showmedians=True, showextrema=False)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(['#66BB6A','#EF5350'][i]); pc.set_alpha(0.8); pc.set_edgecolor('black')
axD.set_xticks([0, 1])
axD.set_xticklabels([f'Benign\n(n={len(b_var)})', f'Pathogenic\n(n={len(p_var)})'], fontsize=8)
axD.set_ylabel('Predicted |ΔΔG|', fontsize=8)
# Increase ylim to 5.0 so text box at 4.5 is well above violin top (~3.2)
axD.set_ylim(0, 5.0)
# Text well above any violin extent
axD.text(0.5, 4.55, f'Benign median {b_var.abs_pred.median():.2f}    Pathogenic median {p_var.abs_pred.median():.2f}\nMann-Whitney p = {pval:.1e}',
         ha='center', va='center', fontsize=7.5,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='black'))

axE = fig.add_subplot(gs[1, 2])
axE.text(-0.07, 1.05, 'e', fontsize=14, fontweight='bold', transform=axE.transAxes)
axE.axis('off')
table_data = [
    ['Training FoldX-supervised genes', '112'],
    ['Held-out FoldX-supervised genes', '10'],
    ['Gene overlap (verified)', '0 / 10'],
    ['Held-out Spearman ρ', '0.817'],
    ['Per-gene ρ median (27 genes)', '0.744'],
    ['Genes with ρ > 0.7', '18 / 27'],
    ['Δ vs ESM-2 zero-shot', '+0.547'],
    ['Δ vs AlphaMissense', '+0.454'],
    ['Δ vs REVEL', '+0.423'],
    ['Δ vs phyloP-100', '+0.697'],
]
table = axE.table(cellText=table_data, colWidths=[0.72, 0.25], loc='center', cellLoc='left')
table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1, 1.6)
for i in range(6, 10):
    for j in range(2):
        table[(i, j)].set_facecolor('#E3F2FD')
        if j == 1:
            table[(i, j)].set_text_props(weight='bold')

plt.savefig(OUT/'Fig2_ddg.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT/'Fig2_ddg.pdf', bbox_inches='tight')
plt.close()
print('✅ Fig 2 — Panel D text lifted to y=4.55 (well above violin)')


# =================================================================
# FIG 3 — fix B legend overlap, E label cut, F δ values past border + legend
# =================================================================
fig = plt.figure(figsize=(16, 11.5))
gs = fig.add_gridspec(3, 4, hspace=0.85, wspace=0.7, height_ratios=[1.6, 1, 1])

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
axA.legend(handles=[
    plt.Rectangle((0,0),1,1,fc='#1F77B4',ec='black',label='Significant vs AlphaMissense'),
    plt.Rectangle((0,0),1,1,fc='#2E7D32',ec='black',label='Significant vs ESM-2 zero-shot'),
    plt.Rectangle((0,0),1,1,fc='#D0D0D0',ec='black',label='Tie vs AlphaMissense'),
    plt.Rectangle((0,0),1,1,fc='#C8E6C9',ec='black',label='Tie vs ESM-2 zero-shot'),
], fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
axA.set_ylim(-0.15, 0.8)

# === Panel B: FIX — legend OUTSIDE plot with proper labelspacing ===
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
# FIX: legend below the plot area
axB.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.22),
           ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.5)
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

# === Panel E: FIX — extend left xlim so X-linked label not occluded ===
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
# Always put value labels on the RIGHT side of the bar to keep them away from y-tick labels
for i, v in enumerate(data_e.values()):
    if v > 0:
        axE.text(v + 0.005, i, f'{v:+.3f}', va='center', ha='left', fontsize=8)
    else:
        # negative value: put label OUTSIDE bar to the LEFT
        axE.text(v - 0.005, i, f'{v:+.3f}', va='center', ha='right', fontsize=8)
axE.set_yticks(y); axE.set_yticklabels(list(data_e.keys()), fontsize=7.5)
axE.set_xlabel('Median Δ AUC (vs AlphaMissense)', fontsize=8)
axE.axvline(0, color='black', linewidth=0.5)
# FIX: extend left xlim so the -0.010 label fits without overlapping y-tick
axE.set_xlim(-0.08, 0.27)

# === Panel F: FIX — restructure: legend OUTSIDE bottom; xlim extended for Δ values ===
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
# Gene name + disease label
for i, (g, d) in enumerate(zip(sig_genes.gene, sig_genes.delta)):
    axF.text(1.01, i, f'{g} ({clinical.get(g, "")})', va='center', fontsize=7)
# Δ values
for i, (g, d) in enumerate(zip(sig_genes.gene, sig_genes.delta)):
    axF.text(1.65, i, f'Δ=+{d:.3f}', va='center', fontsize=7, color='darkred',
             fontweight='bold', ha='left')
axF.set_yticks([])
# FIX: extend xlim so Δ values fit completely
axF.set_xlim(0, 2.0)
axF.set_xlabel('AUC (per-residue disease detection)', fontsize=8)
# FIX: legend WELL below xlabel (not overlapping)
axF.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.25),
           ncol=2, frameon=False)

plt.savefig(OUT/'Fig3_landscape.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT/'Fig3_landscape.pdf', bbox_inches='tight')
plt.close()
print('✅ Fig 3 — B legend below, E xlim extended, F xlim extended + legend below')

print('\nAll fixes applied.')
