"""CMA-ES fit of the model's dynamics (model.BOUNDS) to the read-out table of every unit set.

Loss: for every unit set, the mean squared scaled error of the six On/Off statistics (readout.STATS) plus
100 x Hellinger^2 between model and data On and Off length histograms (averaged over sets); plus a memory penalty
2e4 x max(0, 0.98 - memory correct)^2. The read-out gains are not fitted (anchored by mu_on, readout.py).

usage: python -m paper_model.fit --out fit_run0 --seed 0 [--maxfev 1000] [--fit-set 'all']
"""
import argparse
import json
import os

import numpy as np

from . import model as M, readout as R

MEM_MIN = 0.98


def hellinger2(p, q):
    return 0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)


def loss(out, summ, hist, sets=R.SETS):
    terms = {'/'.join(st): float(np.mean([((out['sets'][st][mk] - summ[st][dk]) / sc) ** 2 for dk, mk, sc in R.STATS]))
             for st in sets}
    for st in sets:
        r, h = out['sets'][st], hist[st]
        terms['hist/' + '/'.join(st)] = 100 / len(sets) * (hellinger2(r['h_off'], h['off']) + hellinger2(r['h_on'], h['on']))
    terms['mem'] = 2e4 * max(0.0, MEM_MIN - out['mem']) ** 2
    return float(sum(terms.values())), terms


def main():
    import cma
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--maxfev', type=int, default=1000)
    ap.add_argument('--n-trials', type=int, default=792)
    ap.add_argument('--x0', action='store_true', help='start from the current best fit (model.FIT)')
    ap.add_argument('--fit-set', default=None, choices=list(R.LABELS),
                    help='fit to this unit set only (default: all 7); the others are then predictions')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summ, hist = R.load_data()
    sets = R.SETS if a.fit_set is None else (R.SETS[R.LABELS.index(a.fit_set)],)
    names = list(M.BOUNDS)
    lo, hi = np.array([M.BOUNDS[k] for k in names]).T
    theta = lambda u: {k: float(l + (h - l) * np.clip(x, 0, 1)) for k, l, h, x in zip(names, lo, hi, u)}
    best = [np.inf]

    def f(u):
        th = theta(u)
        out = R.evaluate(th, n_trials=a.n_trials, seed=a.seed, summ=summ, sets=sets)
        L, terms = loss(out, summ, hist, sets)
        L += 10 * float(np.sum(np.clip(u - 1, 0, None) ** 2 + np.clip(-u, 0, None) ** 2))   # stay inside bounds
        if L < best[0]:
            best[0] = L
            json.dump(dict(theta=th, loss=L, terms=terms, mem=out['mem']), open(os.path.join(a.out, 'best.json'), 'w'), indent=1)
        return L

    u0 = np.random.default_rng(a.seed).uniform(0.25, 0.75, len(names))
    if a.x0:
        u0 = np.clip([(M.FIT[k] - l) / (h - l) for k, l, h in zip(names, lo, hi)], 0.02, 0.98)
    es = cma.CMAEvolutionStrategy(u0, 0.15 if a.x0 else 0.25, dict(seed=a.seed + 1, maxfevals=a.maxfev, verbose=-9, popsize=12))
    while not es.stop():
        U = es.ask()
        es.tell(U, [f(u) for u in U])
        print(f'{es.countevals:5d} best loss {best[0]:.2f}', flush=True)


if __name__ == '__main__':
    main()
