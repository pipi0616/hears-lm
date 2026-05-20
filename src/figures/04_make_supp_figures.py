#!/usr/bin/env python3
"""Regen all supp figs, no titles, journal style + fix Fig 3e overlap."""
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

OUT = Path('/Users/pipi/Projects/QAFI_Paper/hearing_v2/figures')
ROOT = Path('/Users/pipi/Projects/QAFI_Paper/hearing_v2')
EVAL = ROOT/'eval_results'
plt.rcParams.update({'font.size': 9, 'figure.dpi': 200, 'savefig.dpi': 300,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

# ========== Supp Fig S1: AF dominance + ceiling ==========
fig = plt.figure(figsize=(14, 5))
gs = fig.add_gridspec(1, 3, wspace=0.55)

axA = fig.add_subplot(gs[0])
axA.text(-0.18, 1.05, 'a', fontsize=14, fontweight='bold', transform=axA.transAxes)
af_data = {
    'val_db\n(AF=0)': (0, 0.840),
    'val_time\n(recent)': (0.987, 0.973),
    'val_gene\n(new genes)': (0.900, 0.945),
}
x = np.arange(len(af_data))
af_aucs = [v[0] for v in af_data.values()]
ours_aucs = [v[1] for v in af_data.values()]
w = 0.35
axA.bar(x - w/2, af_aucs, w, color='#FF7043', edgecolor='black', linewidth=0.6, label='Allele frequency alone')
axA.bar(x + w/2, ours_aucs, w, color='#1F77B4', edgecolor='black', linewidth=0.6, label='HEARS-LM')
axA.set_xticks(x); axA.set_xticklabels(list(af_data.keys()), fontsize=8)
axA.set_ylabel('AUC', fontsize=9)
# FIX: legend ABOVE plot, no overlap with bar tops
axA.legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
axA.set_ylim(0, 1.05)

axB = fig.add_subplot(gs[1])
axB.text(-0.2, 1.05, 'b', fontsize=14, fontweight='bold', transform=axB.transAxes)
# FIX: shorten labels so they don't get cut by panel A's right edge
ceiling = {
    'METARNN (28 tools)': 0.898,
    'LGBM + ext. predictors': 0.895,
    'LGBM HHL + HEARS-LM': 0.872,
    'HEARS-LM alone': 0.840,
    'LGBM HHL feats only': 0.823,
    'LGBM HHL (no PLM)': 0.779,
}
colors_b = ['#9E9E9E','#BDBDBD','#90A4AE','#1F77B4','#B0BEC5','#CFD8DC']
y = np.arange(len(ceiling))
axB.barh(y, list(ceiling.values()), color=colors_b, edgecolor='black', linewidth=0.6)
for i, (k, v) in enumerate(ceiling.items()):
    axB.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8,
              fontweight='bold' if 'HEARS-LM' in k else 'normal')
axB.set_yticks(y); axB.set_yticklabels(list(ceiling.keys()), fontsize=7.5)
axB.set_xlabel('val_db AUC (AF-independent)', fontsize=8)
axB.set_xlim(0.7, 0.93)

axC = fig.add_subplot(gs[2])
axC.text(-0.2, 1.05, 'c', fontsize=14, fontweight='bold', transform=axC.transAxes)
splits_data = {'val_db': (2068, 767), 'val_time': (891, 994), 'val_gene': (736, 268)}
labels = list(splits_data.keys())
p_vals = [v[0] for v in splits_data.values()]
b_vals = [v[1] for v in splits_data.values()]
y = np.arange(len(labels))
axC.barh(y, p_vals, color='#EF5350', edgecolor='black', linewidth=0.6, label='Pathogenic')
axC.barh(y, b_vals, color='#66BB6A', edgecolor='black', linewidth=0.6, label='Benign', left=p_vals)
for i, (p, b) in enumerate(zip(p_vals, b_vals)):
    axC.text(p+b+50, i, f' P={p}, B={b}', va='center', fontsize=7)
axC.set_yticks(y); axC.set_yticklabels(labels, fontsize=8)
axC.set_xlabel('Number of variants', fontsize=8)
axC.legend(fontsize=7)

plt.savefig(OUT/'FigS1_af_ceiling.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT/'FigS1_af_ceiling.pdf', bbox_inches='tight')
plt.close()

# ========== S2: Top-K per gene table ==========
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111); ax.axis('off')
tk = pd.read_csv(EVAL/'topk_residue_per_gene.csv').sort_values('p_lp', ascending=False)
data = [['Gene', 'Residues evaluated', 'P/LP residues', 'Ours P@10', 'AM P@10', 'ESM-zs P@10', 'Δ vs AM', 'Δ vs ESM-zs']]
for _, r in tk.iterrows():
    data.append([
        r['gene'], f'{int(r["n_res"])}', f'{int(r["p_lp"])}',
        f'{r["P@10_O"]:.2f}',
        f'{r["P@10_AM"]:.2f}' if pd.notna(r["P@10_AM"]) else '—',
        f'{r["P@10_zs"]:.2f}',
        f'{r["P@10_O"]-r["P@10_AM"]:+.2f}' if pd.notna(r["P@10_AM"]) else '—',
        f'{r["P@10_O"]-r["P@10_zs"]:+.2f}',
    ])
table = ax.table(cellText=data[1:], colLabels=data[0],
                  colWidths=[0.13, 0.13, 0.11, 0.12, 0.10, 0.13, 0.11, 0.12],
                  loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(7); table.scale(1, 1.2)
for j in range(8):
    table[(0, j)].set_text_props(weight='bold', color='white')
    table[(0, j)].set_facecolor('#1F77B4')
for i in range(1, len(data)):
    try:
        p10 = float(data[i][3])
        if p10 == 1.0:
            for j in range(8):
                table[(i, j)].set_facecolor('#C8E6C9')
    except: pass
plt.savefig(OUT/'FigS2_topk_per_gene.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT/'FigS2_topk_per_gene.pdf', bbox_inches='tight')
plt.close()

# ========== S3: Multi-baseline ==========
fig = plt.figure(figsize=(11, 5))
ax = fig.add_subplot(111)
multi_data = {
    'HEARS-LM':       0.669, 'MutPred':         0.494,
    'DeOgen-2':        0.421, 'METARNN':         0.417,
    'PolyPhen-2':      0.411, 'REVEL':           0.403,
    'AlphaMissense':   0.372, 'PrimateAI':       0.372,
    'ESM-2 zero-shot': 0.360, 'ESM-1b':          0.116,
}
y = np.arange(len(multi_data))
colors = ['#1F77B4' if k == 'HEARS-LM' else '#B0BEC5' for k in multi_data.keys()]
ax.barh(y, list(multi_data.values()), color=colors, edgecolor='black', linewidth=0.5)
for i, (k, v) in enumerate(multi_data.items()):
    ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9,
             fontweight='bold' if k == 'HEARS-LM' else 'normal')
ax.set_yticks(y); ax.set_yticklabels(list(multi_data.keys()), fontsize=9)
ax.set_xlabel('Mean P@10 across 42 HHL genes', fontsize=9)
ax.set_xlim(0, 0.75)
plt.savefig(OUT/'FigS3_multi_baseline.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT/'FigS3_multi_baseline.pdf', bbox_inches='tight')
plt.close()

# ========== S4: Inheritance + 3D detail ==========
fig = plt.figure(figsize=(13, 5))
gs = fig.add_gridspec(1, 2, wspace=0.35)

axA = fig.add_subplot(gs[0])
axA.text(-0.18, 1.05, 'a', fontsize=14, fontweight='bold', transform=axA.transAxes)
inh_data = pd.DataFrame({
    'category': ['Syndromic\n(13 genes)','AR non-syndr.\n(10 genes)','AD non-syndr.\n(2 genes)','X-linked\n(1 gene)'],
    'median_delta': [0.192, 0.154, 0.012, -0.010],
    'sig_wins': [7, 5, 0, 0],
    'total': [13, 10, 2, 1],
})
x = np.arange(len(inh_data))
axA.bar(x, inh_data.median_delta, color=['#EF5350','#FFA726','#42A5F5','#9E9E9E'], edgecolor='black', linewidth=0.6)
for i, r in inh_data.iterrows():
    axA.text(i, r.median_delta + 0.005, f'{r.sig_wins}/{r.total} significant\nΔ = {r.median_delta:+.3f}',
              ha='center', fontsize=8, va='bottom' if r.median_delta > 0 else 'top')
axA.set_xticks(x); axA.set_xticklabels(inh_data.category, fontsize=8)
axA.set_ylabel('Median Δ AUC (Ours − AlphaMissense)', fontsize=9)
axA.axhline(0, color='black', linewidth=0.5)

axB = fig.add_subplot(gs[1])
axB.text(-0.18, 1.05, 'b', fontsize=14, fontweight='bold', transform=axB.transAxes)
gene_data = ['KCNQ4','KCNQ1','TECTA','OPA1','OTOF','USH2A','CDH23','CHD7']
obs_dist = [34.5, 40.7, 63.6, 61.8, 55.8, 80.9, 120.8, 81.9]
random_dist = [62.0, 54.8, 66.8, 63.6, 56.5, 84.3, 112.9, 62.1]
sigs = ['p<10⁻⁴','p<10⁻⁴','p=0.10','p=0.41','p=0.34','p=0.24','p=0.91','p=0.98']
x = np.arange(len(gene_data)); w = 0.4
axB.bar(x - w/2, obs_dist, w, color='#FF7043', edgecolor='black', linewidth=0.5, label='Observed (high-conf P)')
axB.bar(x + w/2, random_dist, w, color='#81D4FA', edgecolor='black', linewidth=0.5, label='Random expectation')
for i, s in enumerate(sigs):
    axB.text(i, max(obs_dist[i], random_dist[i]) + 4, s, ha='center', fontsize=7,
              fontweight='bold' if '10⁻⁴' in s else 'normal')
axB.set_xticks(x); axB.set_xticklabels(gene_data, fontsize=8)
axB.set_ylabel('Mean pairwise CA distance (Å)', fontsize=9)
axB.legend(fontsize=8)

plt.savefig(OUT/'FigS4_inheritance_3d.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT/'FigS4_inheritance_3d.pdf', bbox_inches='tight')
plt.close()

# ========== S5: Calibration ==========
fig = plt.figure(figsize=(13, 5))
gs = fig.add_gridspec(1, 3, wspace=0.35)
for i, sname in enumerate(['val_db','val_time','val_gene']):
    ax = fig.add_subplot(gs[i])
    ax.text(-0.18, 1.05, chr(ord('a')+i), fontsize=14, fontweight='bold', transform=ax.transAxes)
    df = pd.read_csv(ROOT/f'data/splits/{sname}_with_features.csv', low_memory=False)
    p30 = pd.read_csv(EVAL/f'{sname}_predictions_t30_ep2.csv')
    df = df.merge(p30[['gene','protein_change','pred_prob']], on=['gene','protein_change'], how='left')
    df = df.dropna(subset=['pred_prob'])
    bins = np.linspace(0, 1, 11)
    bin_means, bin_actuals = [], []
    for j in range(10):
        mask = (df.pred_prob >= bins[j]) & (df.pred_prob < bins[j+1])
        if mask.sum() < 5: continue
        bin_means.append(df.loc[mask, 'pred_prob'].mean())
        bin_actuals.append(df.loc[mask, 'y'].mean())
    ax.plot(bin_means, bin_actuals, 'o-', color='#1F77B4', markersize=8, linewidth=2, label='HEARS-LM')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, label='Perfect calibration')
    ax.set_xlabel('Mean predicted probability', fontsize=8)
    ax.set_ylabel('Actual P-rate', fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_xlabel(f'Mean predicted (on {sname})', fontsize=8)
plt.savefig(OUT/'FigS5_calibration.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT/'FigS5_calibration.pdf', bbox_inches='tight')
plt.close()

# ========== S6: DVD retrain (mark as preliminary, ep0 only) ==========
fig = plt.figure(figsize=(11, 4.5))
gs = fig.add_gridspec(1, 2, wspace=0.35)
axA = fig.add_subplot(gs[0])
axA.text(-0.18, 1.05, 'a', fontsize=14, fontweight='bold', transform=axA.transAxes)
splits_ret = ['val_db', 'val_time', 'val_gene']
orig = [0.8396, 0.9731, 0.9448]
dvdexp = [0.8442, 0.9683, 0.9313]
x = np.arange(3); w = 0.35
axA.bar(x - w/2, orig, w, color='#90CAF9', edgecolor='black', linewidth=0.6, label='Original (3 epochs)')
axA.bar(x + w/2, dvdexp, w, color='#42A5F5', edgecolor='black', linewidth=0.6, label='Expanded training (1 epoch)')
for i, (o, d) in enumerate(zip(orig, dvdexp)):
    delta = d - o
    color = 'green' if delta > 0 else 'red'
    axA.text(i, max(o, d) + 0.005, f'Δ = {delta:+.3f}', ha='center', fontsize=8, color=color)
axA.set_xticks(x); axA.set_xticklabels(splits_ret, fontsize=9)
axA.set_ylabel('AUC', fontsize=9)
axA.set_ylim(0.8, 1.0)
axA.legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

axB = fig.add_subplot(gs[1])
axB.text(-0.18, 1.05, 'b', fontsize=14, fontweight='bold', transform=axB.transAxes)
per_gene_change = {
    'OPA1': 0.073, 'TECTA': 0.067, 'HSD17B4': 0.061, 'MYO7A': 0.043,
    'COL11A2': 0.007, 'KCNQ1': 0.002, 'USH2A': 0.001,
    'ADGRV1': -0.002, 'ALMS1': -0.007, 'WFS1': -0.009,
    'SLC26A4': -0.014, 'CHD7': -0.014, 'CDH23': -0.021,
    'KCNE1': -0.026, 'PCDH15': -0.028, 'MYO15A': -0.032,
    'GJB2': -0.041, 'COL11A1': -0.044, 'TMPRSS3': -0.059, 'SLC4A11': -0.093,
}
genes = list(per_gene_change.keys())
deltas = list(per_gene_change.values())
y = np.arange(len(genes))
colors_b = ['#66BB6A' if d > 0 else '#EF5350' for d in deltas]
axB.barh(y, deltas, color=colors_b, edgecolor='black', linewidth=0.5)
axB.set_yticks(y); axB.set_yticklabels(genes, fontsize=7)
axB.set_xlabel('Δ AUC on val_db (expanded training − original)', fontsize=8)
axB.axvline(0, color='black', linewidth=0.5)

plt.savefig(OUT/'FigS6_dvd_retrain.png', bbox_inches='tight', dpi=300)
plt.savefig(OUT/'FigS6_dvd_retrain.pdf', bbox_inches='tight')
plt.close()

print('✅ All supp figs regenerated')
