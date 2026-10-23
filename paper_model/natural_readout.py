"""Fitted model with its own random dips, read out like the data: per-trial time Off in the delay and memory held.

Trials of the fitted model (variant.py: NMDA or STP) with the fitted dip statistics, read out with the fitted model's
all-units gain and noise and labelled as in readout.py. Off time = Off time points in 500-1,400 ms x 10 ms; memory held
= memory_correct of the model at the end of the delay. Also the total dip time: time in 500-1,400 ms (dip_time_s) and in the whole delay after the cue,
50-1,650 ms (dip_time_delay_s), during which the common input is below the middle of the dip (1 - depth / 2).
Per-trial On/Off statistics as the data's behaviour analysis (delay 500-1,400 ms, 91 points): frac_off, n_off (Off
periods, runs clipped to the delay), off_len_ms (mean Off-period length, NaN without Off periods) and frac_on.
Writes <OUT>/natural_readout_trials.npz.

usage: [PM_VARIANT=fac] python -m paper_model.natural_readout
"""
import os

import numpy as np

from . import readout as R, variant as M
from .off_vs_memory import RHO, reference

N_TRIALS, BATCH = 100000, 1000


def main():
    ses, _ = R.load_data()
    gain = reference(ses)
    os.makedirs(M.OUT, exist_ok=True)
    rng = np.random.default_rng(9)
    T = 91 * R.STEP / 1000
    off, mem, dip, dip_all, st, thr = [], [], [], [], [], None
    p = M.params()
    for b in range(N_TRIALS // BATCH):
        sim = M.simulate(p, BATCH, seed=70000 + b)
        dw, tz = R.rate_diff(sim)
        delay, valid = (tz >= 500) & (tz <= 1400), (tz >= 0) & (tz <= 1400)
        if thr is None:
            thr = R.null_threshold(dw.shape, RHO, rng)
        state = R.label(gain * dw + R.noise(dw.shape, RHO, rng), valid, thr)
        sd = state[:, delay]
        off.append((sd == -1).mean(1) * T)
        lens = [[(b - a) * R.STEP for a, b in R.runs(r == -1)] for r in sd]
        st.append(np.c_[(sd == -1).mean(1), [len(x) for x in lens], [np.mean(x) if x else np.nan for x in lens],
                        (sd == 1).mean(1)])
        mem.append(M.memory_correct(sim))
        in_dip = sim['m'] < 1 - p['dip_depth'] / 2           # 1 ms resolution
        t = sim['t']
        dip.append(in_dip[:, (t >= 500) & (t <= 1400)].sum(1) / 1000)
        dip_all.append(in_dip[:, t >= p['cue_dur']].sum(1) / 1000)
    off, mem = np.concatenate(off), np.concatenate(mem)
    dip, dip_all, st = np.concatenate(dip), np.concatenate(dip_all), np.concatenate(st)
    np.savez(os.path.join(M.OUT, 'natural_readout_trials.npz'), off_time_s=off, memory_held=mem, dip_time_s=dip,
             dip_time_delay_s=dip_all, frac_off=st[:, 0], n_off=st[:, 1], off_len_ms=st[:, 2], frac_on=st[:, 3])
    print(f"{M.NAME}: {len(mem)} trials, memory {mem.mean():.4f}, Off time held {off[mem].mean():.3f} s"
          + (f", lost {off[~mem].mean():.3f} s" if (~mem).any() else '')
          + f"; dip time held {dip[mem].mean():.3f} s" + (f", lost {dip[~mem].mean():.3f} s" if (~mem).any() else ''))


if __name__ == '__main__':
    main()
