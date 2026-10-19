"""Fit the mean-field model to Panichello et al. (2024) Fig. 4c with CMA-ES on the fast surrogate.

usage: python scripts/mf_fit.py --out data/mf_fit/run0 --seed 0 [--fix tau_N=75] [--maxfev 1500]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import cma  # noqa: E402
from meanfield import fit as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fix', nargs='*', default=[])
    ap.add_argument('--maxfev', type=int, default=1500)
    ap.add_argument('--n-sessions', type=int, default=4)
    ap.add_argument('--x0', default=None, help='json file with a starting theta')
    ap.add_argument('--sigma0', type=float, default=0.25)
    ap.add_argument('--n-trials', type=int, default=1056, help='trials per evaluation (rates read-out)')
    ap.add_argument('--readout', choices=('decoder', 'rates', 'silence'), default='decoder',
                    help='decoder: recorded cells + fast decoder, full loss; rates: pool rates + one gain, '
                         'state statistics (histograms and counts) + memory requirement; silence: '
                         'read-out-free, states = network attractor states, same loss')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fixed = {k: float(v) for k, v in (s.split('=') for s in a.fix)}
    if a.readout == 'rates':
        from meanfield import rate_states
        bounds, evaluate, keep = F.BOUNDS_RATES, lambda th: rate_states.evaluate(th, n_trials=a.n_trials, seed=a.seed), F.TERMS_STATES
    elif a.readout == 'silence':
        from meanfield import rate_states
        bounds = {k: v for k, v in F.BOUNDS_RATES.items() if k != 'log10_c'}
        evaluate, keep = lambda th: rate_states.evaluate_silence(th, n_trials=a.n_trials, seed=a.seed), F.TERMS_STATES
    else:
        bounds, evaluate, keep = F.BOUNDS, lambda th: F.evaluate(th, n_sessions=a.n_sessions, seed=a.seed), None
    names = [k for k in bounds if k not in fixed]
    lo = np.array([bounds[k][0] for k in names]); hi = np.array([bounds[k][1] for k in names])
    rng = np.random.default_rng(a.seed)
    if a.x0:
        th0 = json.load(open(a.x0))
        u0 = np.array([(th0[k] - l) / (h - l) for k, l, h in zip(names, lo, hi)])
    else:
        u0 = rng.uniform(0.25, 0.75, len(names))

    def theta(u):
        return dict(fixed, **{k: float(l + (h - l) * np.clip(x, 0, 1)) for k, l, h, x in zip(names, lo, hi, u)})

    log = open(os.path.join(a.out, 'log.jsonl'), 'a')
    best = [np.inf, None]

    def f(u):
        th = theta(u)
        s = evaluate(th)
        L, parts = F.loss(s, parts=True, keep_terms=keep)
        L += 10 * float(np.sum(np.clip(u - 1, 0, None) ** 2 + np.clip(-u, 0, None) ** 2))   # stay in bounds
        row = dict(theta=th, loss=L, parts=parts, **{k: s[k] for k in ('n_on', 'n_off', 'on_mean', 'off_mean',
                                                                        'on_median', 'off_median', 'acc', 'mem')})
        log.write(json.dumps(row) + '\n'); log.flush()
        if L < best[0]:
            best[0], best[1] = L, row
            json.dump(row, open(os.path.join(a.out, 'best.json'), 'w'), indent=1)
        return L

    es = cma.CMAEvolutionStrategy(u0, a.sigma0, dict(seed=a.seed + 1, maxfevals=a.maxfev, verbose=-9, popsize=12))
    t0 = time.time()
    while not es.stop():
        U = es.ask()
        es.tell(U, [f(u) for u in U])
        b = best[1]
        print(f'{es.countevals:5d} {time.time() - t0:6.0f}s best {best[0]:.3f} | On {b["n_on"]:.2f}/{b["on_mean"]:.0f} '
              f'Off {b["n_off"]:.2f}/{b["off_mean"]:.0f} acc {b["acc"]:.2f} mem {b["mem"]:.2f} | '
              + ' '.join(f'{k}={v:.3g}' for k, v in b['theta'].items()), flush=True)
    json.dump(dict(xbest=theta(es.result.xbest), xmean=theta(es.result.xfavorite), best=best[1]),
              open(os.path.join(a.out, 'result.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
