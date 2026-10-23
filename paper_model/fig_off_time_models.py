"""Paper figure: memory held vs time Off in the delay, NMDA model vs facilitation (STP) model.

Input: figures/off_vs_memory/{nmda,fac}/single_dip_readout_trials.npz, written by
    python -m paper_model.single_dip_readout  and  PM_VARIANT=fac python -m paper_model.single_dip_readout
(one dip of 0-1,000 ms per trial, 500 ms after cue onset, at each model's fitted depth; all-units read-out).
a: memory held vs Off time (500-1,400 ms), sliding windows as the data (off_vs_memory.sliding), 95% Wilson intervals.
b, c: Off time of trials with the memory held (grey) or lost (red), NMDA model and STP model.
Output: figures/paper_model/fig4_off_time_models.{png,svg} and fig4_off_time_models_data.csv.

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


def load(v):
    d = np.load(os.path.join('figures', 'off_vs_memory', v, 'single_dip_readout_trials.npz'))
    return d['off_time_s'], d['memory_held'].astype(bool), float(d['dip_depth'])


def main():
    fig = plt.figure(figsize=(7.2, 2.3))
    gs = fig.add_gridspec(1, 3, wspace=0.45, left=0.07, right=0.98, bottom=0.2, top=0.86)
    a = fig.add_subplot(gs[0])
    rows = []
    for j, (v, name, col) in enumerate(MODELS):
        off, mem, depth = load(v)
        c, p, lo, hi, n = sliding(off, mem, width=0.025 * T, step=0.0025 * T, start=0.05 * T, top=T)
        a.fill_between(c, lo, hi, color=col, alpha=0.18, lw=0)
        a.plot(c, p, color=col, label=name)
        rows += [dict(panel='a', model=name, x_off_time_s=round(x, 5), memory_held=q, ci_lo=l, ci_hi=h, n_trials=int(k))
                 for x, q, l, h, k in zip(c, p, lo, hi, n)]
        h = fig.add_subplot(gs[j + 1])
        hh, _ = np.histogram(off[mem], BINS)
        hl, _ = np.histogram(off[~mem], BINS)
        h.bar(BINS[:-1], hh, width=np.diff(BINS), align='edge', color=GREY, alpha=0.75, lw=0, label='Memory held')
        h.bar(BINS[:-1], hl, width=np.diff(BINS), align='edge', color=C_ON, alpha=0.75, lw=0, label='Memory lost')
        rows += [dict(panel='bc'[j], model=name, bin_lo_s=round(b0, 4), bin_hi_s=round(b1, 4), n_held=int(x), n_lost=int(y))
                 for b0, b1, x, y in zip(BINS[:-1], BINS[1:], hh, hl)]
        h.set_xlim(0, T); h.set_xlabel('Time Off in delay (s)'); h.set_ylabel('Trials')
        h.set_title(f'{name}: {mem.sum()} held, {(~mem).sum()} lost', fontsize=7.5, pad=3)
        h.legend(loc='upper center', handlelength=1, fontsize=7)
        letter(h, 'bc'[j], x=-0.3)
    a.set_xlim(0, T); a.set_ylim(-0.02, 1.02)
    a.set_xlabel('Time Off in delay (s)'); a.set_ylabel('Memory held (fraction of trials)')
    a.legend(loc='center right', handlelength=1.2)
    letter(a, 'a', x=-0.3)
    os.makedirs(OUT, exist_ok=True)
    save(fig, os.path.join(OUT, 'fig4_off_time_models.png'))
    keys = ['panel', 'model', 'x_off_time_s', 'memory_held', 'ci_lo', 'ci_hi', 'n_trials', 'bin_lo_s', 'bin_hi_s', 'n_held', 'n_lost']
    with open(os.path.join(OUT, 'fig4_off_time_models_data.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
