"""Shared figure style: validated categorical colours (light mode), fonts, helpers."""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)

# validated with the dataviz palette checker (light surface): blue, orange, aqua, violet - fixed order
C = {'HEARS-LM': '#2a78d6', 'AlphaMissense': '#eb6834', 'REVEL': '#1baf7a', 'ESM-2 150M zero-shot': '#4a3aa7'}
GREY = '#9a9893'
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e6e5e1'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'lines.linewidth': 1.5,
    'savefig.dpi': 300, 'figure.dpi': 150, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def colour(method):
    return C.get(method, GREY)


def panel_label(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='left', color=INK)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(FIG / f'{name}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('saved', FIG / f'{name}.pdf')
