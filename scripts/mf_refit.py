"""CMA-ES refit of the rate-model dynamics to the recorded read-out table at lambda = 10,000
(meanfield.refit: gain per unit set anchored by mu_on, several sets at once, memory constraint).

usage: python scripts/mf_refit.py --out data/mf_refit/run0 --seed 0 [--maxfev 1000]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import cma  # noqa: E402
from meanfield import fit as F, refit as RF  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--maxfev', type=int, default=1000)
    ap.add_argument('--n-trials', type=int, default=792)
    ap.add_argument('--x0', default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tg, hist = RF.load_targets()
    bounds = {k: v for k, v in F.BOUNDS_RATES.items() if k != 'log10_c'}
    names = list(bounds)
    lo = np.array([bounds[k][0] for k in names]); hi = np.array([bounds[k][1] for k in names])
    rng = np.random.default_rng(a.seed)
    u0 = rng.uniform(0.25, 0.75, len(names))
    if a.x0:
        th0 = json.load(open(a.x0))
        u0 = np.array([(th0[k] - l) / (h - l) for k, l, h in zip(names, lo, hi)])
    theta = lambda u: {k: float(l + (h - l) * np.clip(x, 0, 1)) for k, l, h, x in zip(names, lo, hi, u)}
    log = open(os.path.join(a.out, 'log.jsonl'), 'a')
    best = [np.inf, None]

    def f(u):
        th = theta(u)
        out = RF.evaluate(th, tg, n_trials=a.n_trials, seed=a.seed)
        L, parts = RF.loss(out, tg, hist, parts=True)
        L += 10 * float(np.sum(np.clip(u - 1, 0, None) ** 2 + np.clip(-u, 0, None) ** 2))
        row = dict(theta=th, loss=L, parts=parts, mem=out['mem'],
                   sets={'/'.join(s): {k: v for k, v in r.items() if not k.startswith('h_')} for s, r in out['sets'].items()})
        log.write(json.dumps(row) + '\n'); log.flush()
        if L < best[0]:
            best[0], best[1] = L, row
            json.dump(row, open(os.path.join(a.out, 'best.json'), 'w'), indent=1)
        return L

    es = cma.CMAEvolutionStrategy(u0, 0.25, dict(seed=a.seed + 1, maxfevals=a.maxfev, verbose=-9, popsize=12))
    t0 = time.time()
    while not es.stop():
        U = es.ask()
        es.tell(U, [f(u) for u in U])
        b = best[1]; t20 = b['sets']['top/20']
        print(f'{es.countevals:5d} {time.time() - t0:6.0f}s best {best[0]:.2f} mem {b["mem"]:.3f} | top20 Off {t20["n_off"]:.2f}/'
              f'{t20["off_mean"]:.0f} On {t20["n_on"]:.2f}/{t20["on_mean"]:.0f} | '
              + ' '.join(f'{k}={v:.3g}' for k, v in b['theta'].items()), flush=True)
    json.dump(dict(xbest=theta(es.result.xbest), best=best[1]), open(os.path.join(a.out, 'result.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
