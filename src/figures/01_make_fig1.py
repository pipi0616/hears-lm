#!/usr/bin/env python3
"""Fig 1a as icon-driven schematic — minimal text, real visual icons."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon, Ellipse, PathPatch
from matplotlib.path import Path
import numpy as np, pandas as pd
from pathlib import Path as PathlibPath

OUT = PathlibPath('/Users/pipi/Projects/QAFI_Paper/hearing_v2/figures')
ROOT = PathlibPath('/Users/pipi/Projects/QAFI_Paper/hearing_v2')
EVAL = ROOT/'eval_results'
plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif',
                     'figure.dpi': 200, 'savefig.dpi': 300,
                     'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.linewidth': 0.6})

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.7, height_ratios=[1.05, 1.0])

# ====================================================================
# PANEL A — icon-driven schematic
# ====================================================================
axA = fig.add_subplot(gs[0, :3])
axA.set_xlim(0, 100); axA.set_ylim(0, 100); axA.axis('off')
axA.text(0.0, 1.0, 'a', fontsize=14, fontweight='bold', transform=axA.transAxes)

def rb(ax, x, y, w, h, fc, ec, lw=1.0, zorder=2):
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6,rounding_size=1.5',
                         edgecolor=ec, facecolor=fc, linewidth=lw, zorder=zorder)
    ax.add_patch(box)

def arr(ax, x1, y1, x2, y2, color='#455A64', lw=1.2, hw=5, hl=6, alpha=0.8):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=f'->,head_width={hw},head_length={hl}',
                        color=color, linewidth=lw, alpha=alpha, zorder=4)
    ax.add_patch(a)

# ===================== LEFT COL — INPUTS =====================
INP_X, INP_W = 1.5, 18
input_y_centers = [78, 58, 38, 18]

# --- Input 1: Protein sequence (icon = chain of beads) ---
y = input_y_centers[0]; box_h = 12
rb(axA, INP_X, y - box_h/2, INP_W, box_h, '#FFF8E1', '#F9A825', lw=1.2)
# bead chain icon (centered)
n_beads = 7
cx_start = INP_X + 4
cx_end = INP_X + INP_W - 4
bead_xs = np.linspace(cx_start, cx_end, n_beads)
bead_y = y - 1.5
# connecting line
axA.plot(bead_xs, [bead_y]*n_beads, color='#F57F17', linewidth=1.4, zorder=3, solid_capstyle='round')
for i, bx in enumerate(bead_xs):
    cc = plt.cm.YlOrBr(0.35 + 0.4 * (i % 3) / 3)
    axA.add_patch(Circle((bx, bead_y), 0.95, facecolor=cc, edgecolor='#E65100', linewidth=0.6, zorder=4))
axA.text(INP_X + INP_W/2, y + 3.3, 'Protein sequence',
         ha='center', fontsize=9, fontweight='bold', color='#E65100')

# --- Input 2: HHL biological features (icon = 10x10 grid) ---
y = input_y_centers[1]
rb(axA, INP_X, y - box_h/2, INP_W, box_h, '#E8F5E9', '#43A047', lw=1.2)
# 10x10 grid of small squares
grid_n = 10
grid_w = 6.5
grid_x0 = INP_X + INP_W/2 - grid_w/2
grid_y0 = y - 2.7
cell = grid_w / grid_n
np.random.seed(3)
cat_color = ['#A5D6A7','#81C784','#FFCC80','#FFAB91','#90CAF9',
             '#B39DDB','#F48FB1','#BCAAA4','#80CBC4','#FFE082']
for i in range(grid_n):
    for j in range(grid_n):
        c = cat_color[j % len(cat_color)] if np.random.rand() > 0.15 else '#ECEFF1'
        axA.add_patch(Rectangle((grid_x0 + j*cell, grid_y0 + i*cell), cell*0.85, cell*0.85,
                                facecolor=c, edgecolor='none', zorder=3))
axA.text(INP_X + INP_W/2, y + 3.3, 'HHL biological features',
         ha='center', fontsize=9, fontweight='bold', color='#1B5E20')

# --- Input 3: Structure context (icon = α-helix cartoon) ---
y = input_y_centers[2]
rb(axA, INP_X, y - box_h/2, INP_W, box_h, '#FFF3E0', '#FB8C00', lw=1.2)
# Draw a stylized helix (sine + shading)
hx_x = np.linspace(INP_X + 3.5, INP_X + INP_W - 3.5, 80)
hx_y = y - 1.5 + 1.5 * np.sin((hx_x - INP_X) * 1.5)
# back ribbon (lighter, behind)
axA.plot(hx_x, hx_y + 0.6, color='#FFCC80', linewidth=4.5, zorder=3, solid_capstyle='round', alpha=0.9)
# front ribbon (darker)
axA.plot(hx_x, hx_y, color='#FB8C00', linewidth=3.2, zorder=4, solid_capstyle='round')
axA.text(INP_X + INP_W/2, y + 3.3, 'Structure context',
         ha='center', fontsize=9, fontweight='bold', color='#E65100')

# --- Input 4: FoldX ΔΔG (icon = energy curve) ---
y = input_y_centers[3]
rb(axA, INP_X, y - box_h/2, INP_W, box_h, '#FCE4EC', '#D81B60', lw=1.2)
# Energy landscape: a U-shape parabola
ec_x = np.linspace(INP_X + 3, INP_X + INP_W - 3, 100)
ec_y = y - 4 + 4.5 * ((ec_x - INP_X - INP_W/2) / (INP_W/2))**2
axA.plot(ec_x, ec_y, color='#AD1457', linewidth=2.0, zorder=3, solid_capstyle='round')
# Min dot
axA.add_patch(Circle((INP_X + INP_W/2, y - 4), 0.7, facecolor='#D81B60', edgecolor='white', linewidth=0.5, zorder=4))
# Δ arrow indicator (up arrow on right side)
axA.annotate('', xy=(INP_X + INP_W - 4.5, y - 0.5), xytext=(INP_X + INP_W - 4.5, y - 3.8),
             arrowprops=dict(arrowstyle='->,head_width=0.3,head_length=0.4', color='#AD1457', lw=1.0), zorder=4)
axA.text(INP_X + INP_W/2, y + 3.3, 'FoldX ΔΔG',
         ha='center', fontsize=9, fontweight='bold', color='#AD1457')


# ===================== CENTER — Pre-trained PLM =====================
ESM_X, ESM_Y, ESM_W, ESM_H = 28, 11, 24, 78

# Outer container
rb(axA, ESM_X, ESM_Y, ESM_W, ESM_H, '#E3F2FD', '#1565C0', lw=1.6)

# Top title
axA.text(ESM_X + ESM_W/2, ESM_Y + ESM_H - 5, 'ESM-2 t30',
         ha='center', fontsize=11.5, fontweight='bold', color='#0D47A1')

# Stack of layers (4 layers with 3D perspective)
stack_n = 4
stack_w = ESM_W - 8
stack_h = 5.5
stack_x = ESM_X + 4
stack_y_top = ESM_Y + ESM_H - 14
stack_gap = 1.5
shift = 1.6  # perspective offset
for i in range(stack_n):
    yy = stack_y_top - i * (stack_h + stack_gap)
    # shadow
    sh = Rectangle((stack_x + shift, yy - 0.6), stack_w, stack_h,
                   facecolor='#90CAF9', edgecolor='#1565C0', linewidth=0.4, alpha=0.7, zorder=3)
    axA.add_patch(sh)
    # front
    pt = FancyBboxPatch((stack_x, yy), stack_w, stack_h,
                        boxstyle='round,pad=0,rounding_size=0.6',
                        facecolor='#BBDEFB', edgecolor='#1565C0', linewidth=0.7, zorder=4)
    axA.add_patch(pt)

# "…" between stack and next
axA.text(stack_x + stack_w/2, stack_y_top - stack_n*(stack_h+stack_gap) + 1.5,
         '⋮', ha='center', va='center', fontsize=18, color='#1565C0', zorder=4)

# "×30" floating badge near stack
badge_x = ESM_X + ESM_W - 4.5
badge_y = ESM_Y + ESM_H - 10.5
axA.add_patch(Circle((badge_x, badge_y), 3.2,
                     facecolor='#1565C0', edgecolor='white', linewidth=1.3, zorder=6))
axA.text(badge_x, badge_y, '×30', ha='center', va='center',
         fontsize=8.5, fontweight='bold', color='white', zorder=7)

# LoRA adapter (small box attached at bottom)
lora_y = ESM_Y + 5
lora_w = ESM_W - 8
lora_h = 7
rb(axA, ESM_X + 4, lora_y, lora_w, lora_h, '#FFE0B2', '#EF6C00', lw=1.1)
# two small triangles inside representing A*B^T low-rank decomposition
tri_y = lora_y + lora_h/2
tri_h = 3
tx = ESM_X + 4 + lora_w/2
# triangle A (down)
tri_a = Polygon([[tx - 4, tri_y + tri_h/2],
                 [tx - 0.5, tri_y - tri_h/2],
                 [tx - 0.5, tri_y + tri_h/2]],
                facecolor='#FB8C00', edgecolor='#E65100', linewidth=0.5, zorder=4)
axA.add_patch(tri_a)
# triangle B (up)
tri_b = Polygon([[tx + 0.5, tri_y - tri_h/2],
                 [tx + 4, tri_y + tri_h/2],
                 [tx + 0.5, tri_y + tri_h/2]],
                facecolor='#FB8C00', edgecolor='#E65100', linewidth=0.5, zorder=4)
axA.add_patch(tri_b)
axA.text(ESM_X + ESM_W/2, lora_y - 2.2, 'LoRA',
         ha='center', va='center', fontsize=8.5, fontweight='bold', color='#E65100')


# ===================== CROSS-ATTENTION FUSION =====================
FUS_X, FUS_Y, FUS_W, FUS_H = 58, 35, 16, 30
rb(axA, FUS_X, FUS_Y, FUS_W, FUS_H, '#E8F5E9', '#2E7D32', lw=1.6)

axA.text(FUS_X + FUS_W/2, FUS_Y + FUS_H - 4.5, 'Cross-',
         ha='center', fontsize=10, fontweight='bold', color='#1B5E20')
axA.text(FUS_X + FUS_W/2, FUS_Y + FUS_H - 8.5, 'attention',
         ha='center', fontsize=10, fontweight='bold', color='#1B5E20')

# Tiny attention-matrix icon
am_n = 5
am_x = FUS_X + FUS_W/2 - 4
am_y = FUS_Y + 6
am_cell = 1.6
np.random.seed(7)
am_val = np.random.rand(am_n, am_n) * 0.5
for ii in range(am_n):
    for jj in range(am_n):
        if abs(ii-jj) <= 1: am_val[ii, jj] = 0.6 + 0.4*np.random.rand()
am_val[1, 3] = 0.95
for ii in range(am_n):
    for jj in range(am_n):
        col = plt.cm.Greens(0.2 + 0.7 * am_val[ii, jj])
        axA.add_patch(Rectangle((am_x + jj*am_cell, am_y + ii*am_cell), am_cell*0.92, am_cell*0.92,
                                facecolor=col, edgecolor='#2E7D32', linewidth=0.3, zorder=4))


# ===================== RIGHT COL — Multi-task heads =====================
HEAD_X, HEAD_W = 82, 16
head_y_centers = [78, 58, 38, 18]

# --- Head 1: Pathogenicity (icon = bimodal hist mini) ---
y = head_y_centers[0]; box_h = 12
rb(axA, HEAD_X, y - box_h/2, HEAD_W, box_h, '#FFEBEE', '#C62828', lw=1.2)
# bimodal mini-histogram
np.random.seed(1)
hg = np.r_[np.random.normal(0.18, 0.08, 200), np.random.normal(0.82, 0.10, 200)]
hh, edges = np.histogram(hg, bins=14, range=(0, 1))
hh = hh / hh.max() * 4.5
mh_x0 = HEAD_X + 2.5; mh_w = HEAD_W - 5
bw = mh_w / 14
for i, hi in enumerate(hh):
    col = '#66BB6A' if edges[i] < 0.5 else '#EF5350'
    axA.add_patch(Rectangle((mh_x0 + i*bw, y - 4), bw*0.85, hi,
                            facecolor=col, edgecolor='none', alpha=0.9, zorder=3))
axA.text(HEAD_X + HEAD_W/2, y + 3.3, 'Pathogenicity',
         ha='center', fontsize=9, fontweight='bold', color='#B71C1C')

# --- Head 2: Thermodynamic ΔΔG (icon = scatter + diagonal) ---
y = head_y_centers[1]
rb(axA, HEAD_X, y - box_h/2, HEAD_W, box_h, '#FFF3E0', '#E65100', lw=1.2)
np.random.seed(2)
sx_n = 18
sx = np.random.uniform(0, 1, sx_n)
sy = sx + np.random.normal(0, 0.13, sx_n)
sy = np.clip(sy, 0, 1)
sct_x0 = HEAD_X + 3; sct_y0 = y - 4.5
sct_w = HEAD_W - 6; sct_h = 6.5
# diagonal line
axA.plot([sct_x0, sct_x0 + sct_w], [sct_y0, sct_y0 + sct_h],
         color='#BF360C', linewidth=0.8, linestyle='--', alpha=0.7, zorder=3)
# points
axA.scatter(sct_x0 + sx*sct_w, sct_y0 + sy*sct_h, s=10, color='#EF6C00',
            edgecolor='none', alpha=0.85, zorder=4)
axA.text(HEAD_X + HEAD_W/2, y + 3.3, 'Thermodynamic',
         ha='center', fontsize=9, fontweight='bold', color='#BF360C')

# --- Head 3: Mechanism (icon = 5 colored chips) ---
y = head_y_centers[2]
rb(axA, HEAD_X, y - box_h/2, HEAD_W, box_h, '#F3E5F5', '#6A1B9A', lw=1.2)
chip_colors = ['#8E24AA', '#AB47BC', '#BA68C8', '#CE93D8', '#E1BEE7']
chip_w = (HEAD_W - 5) / 5
for i, c in enumerate(chip_colors):
    ch_x = HEAD_X + 2.5 + i * chip_w
    axA.add_patch(FancyBboxPatch((ch_x, y - 3.2), chip_w*0.85, 5,
                                 boxstyle='round,pad=0,rounding_size=0.3',
                                 facecolor=c, edgecolor='#4A148C', linewidth=0.4, zorder=3))
axA.text(HEAD_X + HEAD_W/2, y + 3.3, 'Mechanism',
         ha='center', fontsize=9, fontweight='bold', color='#4A148C')

# --- Head 4: Severity (icon = 5-step ladder/gradient circles) ---
y = head_y_centers[3]
rb(axA, HEAD_X, y - box_h/2, HEAD_W, box_h, '#ECEFF1', '#455A64', lw=1.2)
sev_cx_start = HEAD_X + 3
sev_cx_end = HEAD_X + HEAD_W - 3
sev_xs = np.linspace(sev_cx_start, sev_cx_end, 5)
for i, sx in enumerate(sev_xs):
    col = plt.cm.RdYlGn_r((i + 0.5) / 5)
    axA.add_patch(Circle((sx, y - 1.5), 1.3, facecolor=col, edgecolor='#263238', linewidth=0.6, zorder=3))
axA.text(HEAD_X + HEAD_W/2, y + 3.3, 'Severity',
         ha='center', fontsize=9, fontweight='bold', color='#263238')


# ===================== ARROWS =====================
# Inputs → ESM-2 (color-coded by input)
input_colors = ['#F9A825', '#43A047', '#FB8C00', '#D81B60']
# converge to different heights of ESM-2 stack
esm_targets = [ESM_Y + ESM_H*0.88, ESM_Y + ESM_H*0.70,
               ESM_Y + ESM_H*0.50, lora_y + lora_h/2]
for ic, ty, ac in zip(input_y_centers, esm_targets, input_colors):
    arr(axA, INP_X + INP_W + 0.5, ic, ESM_X - 0.5, ty, color=ac, lw=1.4, hw=5, hl=6, alpha=0.9)

# ESM-2 → Fusion
arr(axA, ESM_X + ESM_W + 0.5, ESM_Y + ESM_H*0.55,
    FUS_X - 0.5, FUS_Y + FUS_H/2, color='#1565C0', lw=2.0, hw=6, hl=7, alpha=0.95)

# Fusion → 4 heads
for hy in head_y_centers:
    arr(axA, FUS_X + FUS_W + 0.5, FUS_Y + FUS_H/2,
        HEAD_X - 0.5, hy, color='#455A64', lw=1.2, hw=4.5, hl=5.5, alpha=0.8)

# Stage labels at bottom
stage_y = 3
for cx, lbl, col in [(INP_X + INP_W/2, 'Inputs', '#616161'),
                     (ESM_X + ESM_W/2, 'Pre-trained PLM', '#0D47A1'),
                     (FUS_X + FUS_W/2, 'Fusion', '#1B5E20'),
                     (HEAD_X + HEAD_W/2, 'Multi-task heads', '#37474F')]:
    axA.text(cx, stage_y, lbl, ha='center', va='center', fontsize=9.5,
             color=col, fontweight='bold')


# ====================================================================
# Panels B-F unchanged
# ====================================================================
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
axE.set_xlim(-0.17, 0.08)

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
print('✅ Fig 1a — icon-driven schematic, minimal text')
