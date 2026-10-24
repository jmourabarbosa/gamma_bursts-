"""Supplementary figure: single-dip test, NMDA model vs STP model.

Each trial has one dip of the common input (0-1,000 ms in 25 ms steps, 300 trials per length, 12,300 per model),
starting 500 ms after cue onset, at each model's fitted depth, and no other dips.
a, fraction of trials with the memory held against dip length (the network itself, no read-out), 95% Wilson intervals.
b, the same trials read out as the data: memory held against Off time in the delay (fraction of the 91 delay time
   points Off x 900 ms), in 10 bins of 0.1 of the delay.

usage: python figS_single_dip.py <off_vs_memory dir with {nmda,fac}/single_dip_readout_trials.npz> <out.png>
models: figures/off_vs_memory on branch paper-model (paper_model/single_dip_readout.py)
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODELS, OUT = sys.argv[1:3]
NPT, W = 91, 900
INK, C_STP, GREY = '#000000', '#2b83ba', '#7f7f7f'
EDGES = np.r_[np.arange(0, 1.0, 0.1), 1.0001]
CENT = (EDGES[:-1] + EDGES[1:]) / 2
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'],
                     'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK,
                     'lines.linewidth': 0.9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def letter(ax, txt, x=-0.3, y=1.04):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=10, weight='bold', va='bottom')


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


fig = plt.figure(figsize=(4.8, 2.25))
gs = fig.add_gridspec(1, 2, wspace=0.5, left=0.12, right=0.97, top=0.82, bottom=0.21)
a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
for v, name, col in (('nmda', 'NMDA model', INK), ('fac', 'STP model', C_STP)):
    d = np.load(f'{MODELS}/{v}/single_dip_readout_trials.npz')
    L, held = d['dip_length_ms'], d['memory_held'].astype(bool)
    frac = np.round(d['off_time_s'] / 0.01) / NPT
    u = np.unique(L)
    k = np.array([held[L == x].sum() for x in u]); n = np.array([(L == x).sum() for x in u])
    lo, hi = wilson(k, n)
    a.fill_between(u, lo, hi, color=col, alpha=0.18, lw=0)
    a.plot(u, k / n, '-', color=col, lw=1.2, label=name)
    i = np.digitize(frac, EDGES) - 1
    k = np.array([held[i == j].sum() for j in range(len(CENT))]); n = np.array([(i == j).sum() for j in range(len(CENT))])
    lo, hi = wilson(k, n)
    b.fill_between(W * CENT, lo, hi, color=col, alpha=0.18, lw=0)
    b.plot(W * CENT, k / n, 'o-', color=col, ms=3)
for ax in (a, b):
    ax.set_ylim(0, 1.03); ax.set_yticks([0, 0.5, 1], ['0', '', '1']); ax.yaxis.labelpad = -6
    ax.set_ylabel('Fraction memory held')
a.set_xlim(0, 1000); a.set_xticks([0, 250, 500, 750, 1000], ['0', '', '500', '', '1000'])
a.set_xlabel('Dip length (ms)')
a.legend(loc='center right', handlelength=1.2, fontsize=7, frameon=False)
b.set_xlim(0, W); b.set_xticks([0, 300, 600, 900]); b.set_xlabel('Off time in delay (ms)')
a.set_title('Network', fontsize=8, pad=4); b.set_title('Read out as the data', fontsize=8, pad=4)
letter(a, 'a'); letter(b, 'b')
fig.savefig(OUT, dpi=300)
