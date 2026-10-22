"""CMA-ES fit of the model's dynamics (model.FIT) to the all-units read-out only.

Loss: mean squared scaled error of six On/Off statistics, plus 100 x Hellinger^2 between model and data On and Off
length distributions (states >= 40 ms), plus a memory penalty 2e4 x max(0, 0.98 - memory correct)^2.

usage: python -m paper_model.fit --out fit_run0 --seed 0
"""
import argparse
import json
import os

import cma
import numpy as np

from . import model as M, readout as R

SET = 'all'
BOUNDS = dict(dip_rate=(0.5, 12.0), dip_mean=(20.0, 300.0), dip_shape=(0.7, 10.0), dip_depth=(0.02, 1.0), w_ff=(0.0, 1.0))
# (data column, model key, scale)
STATS = (('off_per_trial', 'n_off', 0.3), ('off_mean_ms', 'off_mean', 20.0), ('frac_off', 'frac_off', 0.05),
         ('on_per_trial', 'n_on', 0.3), ('on_mean_ms', 'on_mean', 20.0), ('frac_on', 'frac_on', 0.05))


def loss(out, ses, hist):
    r, d = out['sets'][SET], ses[SET]
    L = np.mean([((r[mk] - d[dk].mean()) / sc) ** 2 for dk, mk, sc in STATS])
    for st in ('on', 'off'):
        p, q = r['h_' + st][2:], hist[SET, st][2:]
        L += 50 * np.sum((np.sqrt(p / max(p.sum(), 1)) - np.sqrt(q / q.sum())) ** 2)
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
        L = loss(R.evaluate(M.params(**th), ses, n_trials=792, seed=a.seed, sets=(SET,)), ses, hist)
        L += 10 * np.sum(np.clip(u - 1, 0, None) ** 2 + np.clip(-u, 0, None) ** 2)   # stay inside the bounds
        if L < best[0]:
            best[0] = L
            json.dump(dict(theta=th, loss=L), open(os.path.join(a.out, 'best.json'), 'w'), indent=1)
        return L

    u0 = np.random.default_rng(a.seed).uniform(0.25, 0.75, len(BOUNDS))
    es = cma.CMAEvolutionStrategy(u0, 0.25, dict(seed=a.seed + 1, maxfevals=a.maxfev, verbose=-9, popsize=12))
    while not es.stop():
        U = es.ask()
        es.tell(U, [f(u) for u in U])
        print(f'{es.countevals:5d} best loss {best[0]:.2f}', flush=True)


if __name__ == '__main__':
    main()
