"""Paper figure: memory held vs time Off in the delay, NMDA model vs facilitation (STP) model, in two simulations.

Top row (a-c): each fitted model with its own random dips, 100,000 trials per model
    (figures/off_vs_memory/{nmda,fac}/natural_readout_trials.npz, from paper_model.natural_readout).
Bottom row (d-f): the single-dip test, one dip of 0-1,000 ms per trial (25 ms steps, 300 trials each), 500 ms after cue
    onset, at each model's fitted depth (single_dip_readout_trials.npz, from paper_model.single_dip_readout).
Both are read out with the all-units read-out; run each producer with and without PM_VARIANT=fac first.
a, d: memory held vs time Off (500-1,400 ms), sliding windows as the data (off_vs_memory.sliding), 95% Wilson intervals.
b, c, e, f: time Off of trials with the memory held (grey) or lost (red), NMDA model and STP model, each group normalised
   to its own number of trials. Lost trials are drawn only if there are at least N_MIN of them.
Output: figures/paper_model/fig4_off_time_models.{png,svg} and fig4_off_time_models_data.csv.

usage: python -m paper_model.fig_off_time_models
"""
import csv
import os

import numpy as np

from .make_figures import plt, save, letter, INK, GREY, C_ON, OUT
from .off_vs_memory import sliding

MODELS = (('nmda', 'NMDA model', INK), ('fac', 'STP model', '#2b83ba'))
SIMS = (('natural_readout_trials.npz', 'Fitted dips', 'abc', (0.75, 1.005), 'lower left'),
        ('single_dip_readout_trials.npz', 'Single dip, 0-1 s', 'def', (-0.02, 1.02), 'center right'))
T = 0.91                       # delay window 500-1,400 ms: 91 time points of 10 ms
BINS = np.linspace(0, T, 27)   # 35 ms bins
N_MIN = 20                     # fewest lost trials for which their histogram is drawn


def load(v, src):
    d = np.load(os.path.join('figures', 'off_vs_memory', v, src))
    return d['off_time_s'], d['memory_held'].astype(bool)


def row(fig, gs, src, title, letters, ylim, loc, rows):
    a = fig.add_subplot(gs[0])
    for j, (v, label, col) in enumerate(MODELS):
        off, mem = load(v, src)
        c, p, lo, hi, n = sliding(off, mem, width=0.025 * T, step=0.0025 * T, start=0.05 * T, top=T)
        a.fill_between(c, lo, hi, color=col, alpha=0.18, lw=0)
        a.plot(c, p, color=col, label=label)
        rows += [dict(panel=letters[0], model=label, x_off_time_s=round(x, 5), memory_held=q, ci_lo=l, ci_hi=h,
                      n_trials=int(k)) for x, q, l, h, k in zip(c, p, lo, hi, n)]
        h = fig.add_subplot(gs[j + 1])
        hh, _ = np.histogram(off[mem], BINS)
        hl, _ = np.histogram(off[~mem], BINS)
        nh, nl = max(mem.sum(), 1), max((~mem).sum(), 1)
        h.bar(BINS[:-1], hh / nh, width=np.diff(BINS), align='edge', color=GREY, alpha=0.75, lw=0, label='Memory held')
        if (~mem).sum() >= N_MIN:   # a histogram of a handful of lost trials is not shown
            h.bar(BINS[:-1], hl / nl, width=np.diff(BINS), align='edge', color=C_ON, alpha=0.75, lw=0, label='Memory lost')
        rows += [dict(panel=letters[j + 1], model=label, bin_lo_s=round(b0, 4), bin_hi_s=round(b1, 4), n_held=int(x),
                      n_lost=int(y), frac_held=x / nh, frac_lost=y / nl) for b0, b1, x, y in zip(BINS[:-1], BINS[1:], hh, hl)]
        h.set_xlim(0, T); h.set_xlabel('Time Off in delay (s)'); h.set_ylabel('Fraction of trials in group')
        h.set_title(f'{label}: {mem.sum():,} held, {(~mem).sum():,} lost', fontsize=7.5, pad=3)
        h.legend(loc='upper center', handlelength=1, fontsize=7)
        letter(h, letters[j + 1], x=-0.3)
    a.set_xlim(0, T); a.set_ylim(*ylim)
    a.set_xlabel('Time Off in delay (s)'); a.set_ylabel('Memory held\n(fraction of trials)')
    a.set_title(title, fontsize=8, weight='bold', loc='left', pad=3)
    a.legend(loc=loc, handlelength=1.2)
    letter(a, letters[0], x=-0.3)


def main():
    fig = plt.figure(figsize=(7.2, 4.6))
    outer = fig.add_gridspec(2, 1, hspace=0.6, left=0.1, right=0.98, bottom=0.1, top=0.94)
    rows = []
    for k, (src, title, letters, ylim, loc) in enumerate(SIMS):
        row(fig, outer[k].subgridspec(1, 3, wspace=0.5), src, title, letters, ylim, loc, rows)
    os.makedirs(OUT, exist_ok=True)
    save(fig, os.path.join(OUT, 'fig4_off_time_models.png'))
    keys = ['panel', 'model', 'x_off_time_s', 'memory_held', 'ci_lo', 'ci_hi', 'n_trials', 'bin_lo_s', 'bin_hi_s',
            'n_held', 'n_lost', 'frac_held', 'frac_lost']
    with open(os.path.join(OUT, 'fig4_off_time_models_data.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
