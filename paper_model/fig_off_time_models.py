"""Paper figure: memory held vs time Off in the delay, NMDA model vs facilitation (STP) model.

Figure 4 (fig4_off_time_models): each fitted model with its own random dips, 20,000 trials per model
    (figures/off_vs_memory/{nmda,fac}/natural_readout_trials.npz, from paper_model.natural_readout).
Supplementary (figS_off_time_models_single_dip): the single-dip test, one dip of 0-1,000 ms per trial, 500 ms after
    cue onset, at each model's fitted depth (single_dip_readout_trials.npz, from paper_model.single_dip_readout).
Both are read out with the all-units read-out; run each producer with and without PM_VARIANT=fac first.
a: memory held vs Off time (500-1,400 ms), sliding windows as the data (off_vs_memory.sliding), 95% Wilson intervals.
b, c: Off time of trials with the memory held (grey) or lost (red), NMDA model and STP model, each group normalised to
   its own number of trials (Figure 4: lost trials are few) or to all trials of the model (supplementary).
   Lost trials are drawn only if there are at least N_MIN of them.
Output: figures/paper_model/<name>.{png,svg} and <name>_data.csv.

usage: python -m paper_model.fig_off_time_models
"""
import csv
import os

import numpy as np

from .make_figures import plt, save, letter, INK, GREY, C_ON, OUT
from .off_vs_memory import sliding

MODELS = (('nmda', 'NMDA model', INK), ('fac', 'STP model', '#2b83ba'))
T = 0.91                       # delay window 500-1,400 ms: 91 time points of 10 ms
BINS = np.linspace(0, T, 27)   # 35 ms bins
N_MIN = 20                     # fewest lost trials for which their histogram is drawn


def load(v, src):
    d = np.load(os.path.join('figures', 'off_vs_memory', v, src))
    return d['off_time_s'], d['memory_held'].astype(bool)


def make(src, name, within, ylim_a):
    fig = plt.figure(figsize=(7.2, 2.3))
    gs = fig.add_gridspec(1, 3, wspace=0.45, left=0.07, right=0.98, bottom=0.2, top=0.86)
    a = fig.add_subplot(gs[0])
    rows = []
    for j, (v, label, col) in enumerate(MODELS):
        off, mem = load(v, src)
        c, p, lo, hi, n = sliding(off, mem, width=0.025 * T, step=0.0025 * T, start=0.05 * T, top=T)
        a.fill_between(c, lo, hi, color=col, alpha=0.18, lw=0)
        a.plot(c, p, color=col, label=label)
        rows += [dict(panel='a', model=label, x_off_time_s=round(x, 5), memory_held=q, ci_lo=l, ci_hi=h, n_trials=int(k))
                 for x, q, l, h, k in zip(c, p, lo, hi, n)]
        h = fig.add_subplot(gs[j + 1])
        hh, _ = np.histogram(off[mem], BINS)
        hl, _ = np.histogram(off[~mem], BINS)
        nh, nl = (max(mem.sum(), 1), max((~mem).sum(), 1)) if within else (len(mem), len(mem))
        h.bar(BINS[:-1], hh / nh, width=np.diff(BINS), align='edge', color=GREY, alpha=0.75, lw=0, label='Memory held')
        if (~mem).sum() >= N_MIN:   # a histogram of a handful of lost trials is not shown
            h.bar(BINS[:-1], hl / nl, width=np.diff(BINS), align='edge', color=C_ON, alpha=0.75, lw=0, label='Memory lost')
        rows += [dict(panel='bc'[j], model=label, bin_lo_s=round(b0, 4), bin_hi_s=round(b1, 4), n_held=int(x), n_lost=int(y),
                     frac_held=x / nh, frac_lost=y / nl)
                 for b0, b1, x, y in zip(BINS[:-1], BINS[1:], hh, hl)]
        h.set_xlim(0, T); h.set_xlabel('Time Off in delay (s)'); h.set_ylabel('Fraction of trials' + (' in group' if within else ''))
        h.set_title(f'{label}: {mem.sum()} held, {(~mem).sum()} lost', fontsize=7.5, pad=3)
        h.legend(loc='upper center', handlelength=1, fontsize=7)
        letter(h, 'bc'[j], x=-0.3)
    a.set_xlim(0, T); a.set_ylim(*ylim_a)
    a.set_xlabel('Time Off in delay (s)'); a.set_ylabel('Memory held (fraction of trials)')
    a.legend(loc='lower left', handlelength=1.2)
    letter(a, 'a', x=-0.3)
    os.makedirs(OUT, exist_ok=True)
    save(fig, os.path.join(OUT, name + '.png'))
    keys = ['panel', 'model', 'x_off_time_s', 'memory_held', 'ci_lo', 'ci_hi', 'n_trials', 'bin_lo_s', 'bin_hi_s', 'n_held', 'n_lost', 'frac_held', 'frac_lost']
    with open(os.path.join(OUT, name + '_data.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def main():
    make('natural_readout_trials.npz', 'fig4_off_time_models', within=True, ylim_a=(0.75, 1.005))
    make('single_dip_readout_trials.npz', 'figS_off_time_models_single_dip', within=False, ylim_a=(-0.02, 1.02))


if __name__ == '__main__':
    main()
