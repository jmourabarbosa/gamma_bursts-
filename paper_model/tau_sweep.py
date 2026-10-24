"""Fitted model with its slow time constant changed (all other parameters and the read-out fixed): per-trial fraction of
the delay Off and memory held, for fitting the time constant to the monkeys' performance (fit_tau_behaviour.py).

PM_VARIANT=nmda (default): tau_N of the NMDA gating varied. PM_VARIANT=fac: tau_f of the STP facilitation varied
(FIT_FAC otherwise; resting facilitation, and so the effective recurrent strength, also changes with tau_f).
Trials as natural_readout.py (own random dips, all-units read-out with the fitted model's gain).
Writes figures/off_vs_memory/<variant>/tau_sweep/tau_<value>.npz with frac_off and memory_held.

usage: [PM_VARIANT=fac] python -m paper_model.tau_sweep 60 80 100 ...
"""
import os
import sys

import numpy as np

from . import readout as R, variant as M
from .off_vs_memory import RHO, reference

KEY = 'tau_f' if M.NAME == 'fac' else 'tau_N'
OUT = os.path.join(M.OUT, 'tau_sweep')
N_TRIALS, BATCH = 30000, 1000


def run(tau, gain):
    rng = np.random.default_rng(int(tau))
    p = M.params(**{KEY: float(tau)})
    frac, mem, thr = [], [], None
    for b in range(N_TRIALS // BATCH):
        sim = M.simulate(p, BATCH, seed=90000 + b)
        dw, tz = R.rate_diff(sim)
        delay, valid = (tz >= 500) & (tz <= 1400), (tz >= 0) & (tz <= 1400)
        if thr is None:
            thr = R.null_threshold(dw.shape, RHO, rng)
        state = R.label(gain * dw + R.noise(dw.shape, RHO, rng), valid, thr)
        frac.append((state[:, delay] == -1).mean(1))
        mem.append(M.memory_correct(sim))
    frac, mem = np.concatenate(frac), np.concatenate(mem)
    np.savez(os.path.join(OUT, f'tau_{int(tau)}.npz'), frac_off=frac, memory_held=mem)
    print(f'{M.NAME} {KEY} {tau:g}: memory {mem.mean():.4f}, frac Off {frac.mean():.3f}', flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    ses, _ = R.load_data()
    gain = reference(ses)
    for tau in sys.argv[1:]:
        run(float(tau), gain)


if __name__ == '__main__':
    main()
