"""Single-dip trials (single_dip.py) passed through the all-units read-out: memory vs time Off in the delay.

Every trial has one dip of length 0-1,000 ms starting 500 ms after cue onset (fitted depth, no other dips). Each trial
is read out with the fitted model's all-units gain and noise and labelled as in readout.py; Off time = Off time points
in 500-1,400 ms x 10 ms. Memory = NMDA criterion at the end of the delay. Sliding windows as off_vs_memory.py.

usage: python -m paper_model.single_dip_readout
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import model as M, readout as R
from .off_vs_memory import OUT, RHO, reference, sliding
from .single_dip import simulate_dip, LENGTHS, N_TR


def main():
    ses, _ = R.load_data()
    gain = reference(ses)
    p = M.params()
    rng = np.random.default_rng(5)
    T = 91 * R.STEP / 1000
    off, mem, length = [], [], []
    thr = None
    for i, L in enumerate(LENGTHS):
        sim = simulate_dip(p, L, p['dip_depth'], N_TR, seed=50000 + i)
        dw, tz = R.rate_diff(sim)
        delay, valid = (tz >= 500) & (tz <= 1400), (tz >= 0) & (tz <= 1400)
        if thr is None:
            thr = R.null_threshold(dw.shape, RHO, rng)
        state = R.label(gain * dw + R.noise(dw.shape, RHO, rng), valid, thr)
        off.append((state[:, delay] == -1).mean(1) * T)
        mem.append(M.memory_correct(sim))
        length.append(np.full(N_TR, L))
    off, mem, length = np.concatenate(off), np.concatenate(mem), np.concatenate(length)
    c, pr, lo, hi, n = sliding(off, mem, width=0.025 * T, step=0.0025 * T, start=0.05 * T, top=T)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].fill_between(c, lo, hi, color='0.85', lw=0); ax[0].plot(c, pr, 'k', lw=1)
    ax[0].set_xlim(0, T); ax[0].set_ylim(-0.02, 1.02)
    ax[0].set_xlabel('time Off in the delay (s, window centre)'); ax[0].set_ylabel('memory held (fraction of trials)')
    ax[0].set_title(f'{len(mem)} trials, {(~mem).sum()} errors; all-units read-out', fontsize=9)
    u = np.unique(length)
    q = np.array([np.percentile(off[length == v], (10, 50, 90)) for v in u])
    ax[1].fill_between(u / 1000, q[:, 0], q[:, 2], color='0.85', lw=0, label='10-90%')
    ax[1].plot(u / 1000, q[:, 1], 'k', lw=1, label='median')
    ax[1].plot([0, 0.9], [0, 0.9], ':', color='0.5', lw=0.8, label='Off time = dip length')
    ax2 = ax[1].twinx(); ax2.plot(u / 1000, [mem[length == v].mean() for v in u], color='C3', lw=1); ax2.set_ylim(-0.02, 1.02)
    ax2.set_ylabel('memory held', color='C3')
    ax[1].set_xlabel('dip length (s)'); ax[1].set_ylabel('time Off in the delay (s)'); ax[1].legend(fontsize=7, loc='upper left')
    ax[1].set_title('read-out Off time vs dip length', fontsize=9)
    bins = np.linspace(0, T, 31)
    ax[2].hist(off[mem], bins, color='0.4', alpha=0.7, label='memory held')
    ax[2].hist(off[~mem], bins, color='C3', alpha=0.7, label='memory lost')
    ax[2].set_xlabel('time Off in the delay (s)'); ax[2].set_ylabel('trials'); ax[2].legend(fontsize=7)
    ax[2].set_title(f'Off time: held {off[mem].mean():.2f} s, lost {off[~mem].mean():.2f} s', fontsize=9)
    fig.suptitle('One dip per trial (0-1 s, 500 ms after cue onset, fitted depth), read out with the all-units pipeline', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'memory_vs_off_time_single_dip.png'), dpi=200)
    out = dict(off_time_held=float(off[mem].mean()), off_time_lost=float(off[~mem].mean()),
               by_length={int(v): dict(off_median=float(np.median(off[length == v])), memory=float(mem[length == v].mean())) for v in u})
    json.dump(out, open(os.path.join(OUT, 'single_dip_readout.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'by_length'}))
    print(' '.join(f"{k}:{v['off_median']:.2f}/{v['memory']:.2f}" for k, v in out['by_length'].items()))


if __name__ == '__main__':
    main()
