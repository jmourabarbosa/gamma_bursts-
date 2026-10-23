"""CMA-ES fit of the binary STP model (model_stp.FIT) to the all-units read-out only, as fit.py.

Fitted: background input I_b, noise sigma, cue amplitude and the four dip parameters. Loss as fit.py (six On/Off
statistics, 50 x Hellinger^2 of the On and Off length distributions, memory >= 98%), plus a resting-state penalty:
in 1 s before the cue (attention on, no dips) neither cluster may emit a population spike (> 10 Hz) in more than 2% of
trials, since without a cue there is no memory.

usage: python -m paper_model.fit_stp --out fit_stp_run0 --seed 0
"""
import argparse
import json
import os

import cma
import numpy as np

from . import model_stp as S, readout as R
from .fit import STATS, SET

R.M = S      # read out the STP model exactly like the paper model
BOUNDS = dict(I_b=(3.0, 12.0), sigma=(0.0, 1.5), cue_amp=(5.0, 40.0), dip_rate=(0.5, 12.0), dip_mean=(20.0, 300.0),
              dip_shape=(0.7, 10.0), dip_depth=(0.0, 1.0))
T_PRE = 1000.0


def loss(out, ses, hist):
    r, d = out['sets'][SET], ses[SET]
    L = np.mean([((r[mk] - d[dk].mean()) / sc) ** 2 for dk, mk, sc in STATS])
    for st in ('on', 'off'):
        p, q = r['h_' + st][2:], hist[SET, st][2:]
        L += 50 * np.sum((np.sqrt(p / max(p.sum(), 1)) - np.sqrt(q / q.sum())) ** 2)
    sim = out['sim']
    pre = (sim['t'] >= -T_PRE + 100) & (sim['t'] < 0)
    spont = ((sim['rc'][:, pre] > 10) | (sim['ru'][:, pre] > 10)).any(1).mean()
    out['spont'] = float(spont)
    L += 2e4 * max(0.0, spont - 0.02) ** 2
    return L + 2e4 * max(0.0, 0.98 - out['mem']) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--maxfev', type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ses, hist = R.load_data()
    lo, hi = np.array(list(BOUNDS.values())).T
    theta = lambda u: dict(zip(BOUNDS, lo + (hi - lo) * np.clip(u, 0, 1)))
    best = [np.inf]

    def f(u):
        th = theta(u)
        out = R.evaluate(S.params(t_pre=T_PRE, **th), ses, n_trials=792, seed=a.seed, sets=(SET,))
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
