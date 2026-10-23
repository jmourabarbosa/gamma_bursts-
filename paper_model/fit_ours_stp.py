"""CMA-ES fit of model_ours_stp (our model with STP in place of NMDA) to the all-units read-out, as fit.py.

Fitted: the same seven parameters as fit.py, with facilitation tau_f in place of tau_N. Loss as fit_stp.py (fit.py's
loss plus the resting-state penalty over 1 s before the cue). w_s range wider than in fit.py because resting
facilitation (u ~ 0.5 at 1 Hz) already makes the recurrent synapses stronger than with NMDA.

usage: python -m paper_model.fit_ours_stp --out fit_ours_stp_run0 --seed 0 [--no-depression]
"""
import argparse
import json
import os

import cma
import numpy as np

from . import model_ours_stp as H, readout as R
from .fit import SET
from .fit_stp import loss, T_PRE

R.M = H
BOUNDS = dict(tau_f=(100.0, 3000.0), w_s=(0.5, 1.7), dip_rate=(0.5, 12.0), dip_mean=(20.0, 300.0),
              dip_shape=(0.7, 10.0), dip_depth=(0.0, 1.0), sigma=(0.005, 0.04))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--maxfev', type=int, default=1000)
    ap.add_argument('--no-depression', action='store_true', help='facilitation only (x = 1)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ses, hist = R.load_data()
    lo, hi = np.array(list(BOUNDS.values())).T
    theta = lambda u: dict(zip(BOUNDS, lo + (hi - lo) * np.clip(u, 0, 1)))
    best = [np.inf]

    def f(u):
        th = theta(u)
        out = R.evaluate(H.params(t_pre=T_PRE, depression=0.0 if a.no_depression else 1.0, **th), ses, n_trials=792, seed=a.seed, sets=(SET,))
        L = loss(out, ses, hist)
        L += 10 * np.sum(np.clip(u - 1, 0, None) ** 2 + np.clip(-u, 0, None) ** 2)   # stay inside the bounds
        if L < best[0]:
            best[0] = L
            json.dump(dict(theta=th, loss=L, mem=out['mem'], spont=out['spont']),
                      open(os.path.join(a.out, 'best.json'), 'w'), indent=1)
        return L

    u0 = np.random.default_rng(a.seed).uniform(0.25, 0.75, len(BOUNDS))
    es = cma.CMAEvolutionStrategy(u0, 0.25, dict(seed=a.seed + 1, maxfevals=a.maxfev, verbose=-9, popsize=12))
    while not es.stop():
        U = es.ask()
        es.tell(U, [f(u) for u in U])
        print(f'{es.countevals:5d} best loss {best[0]:.2f}', flush=True)


if __name__ == '__main__':
    main()
