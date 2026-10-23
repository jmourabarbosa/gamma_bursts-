"""Per-session fit of the NMDA model: the dynamics (w_s, sigma and the four dip parameters; tau_N fixed at 150 ms)
fitted to one session's six all-units On/Off statistics, with that session's read-out gain (from its mu_on).

Loss: mean squared scaled error of the six statistics (as fit.py, no length histograms, which exist only pooled),
plus the memory penalty 2e4 x max(0, 0.98 - memory correct)^2. CMA-ES started from the global fit (model.FIT).
Writes <out>/<session>.json with the fitted parameters, loss, memory and the read-out gain.

usage: python -m paper_model.fit_session --out fits_session --sessions 210921 220425 ...
"""
import argparse
import csv
import json
import os

import cma
import numpy as np

from . import model as M, readout as R
from .fit import STATS

BOUNDS = dict(w_s=(1.30, 1.70), dip_rate=(0.5, 15.0), dip_mean=(20.0, 300.0), dip_shape=(0.7, 10.0),
              dip_depth=(0.3, 1.0), sigma=(0.005, 0.04))
N_TRIALS, MAXFEV = 528, 300


def session_data(session):
    rows = [r for r in csv.DictReader(open(os.path.join(R.DATA, 'readout_per_session_data_lam10000.csv')))
            if r['set'] == 'all' and r['session'] == session]
    return {'all': {c: np.array([float(rows[0][c])]) for c in rows[0] if c not in ('session', 'set', 'n', 'draw')}}


def loss(out, ses):
    r, d = out['sets']['all'], ses['all']
    return np.mean([((r[mk] - d[dk].mean()) / sc) ** 2 for dk, mk, sc in STATS]) + 2e4 * max(0.0, 0.98 - out['mem']) ** 2


def fit(session, out_dir, seed=0):
    ses = session_data(session)
    lo, hi = np.array(list(BOUNDS.values())).T
    theta = lambda u: dict(zip(BOUNDS, lo + (hi - lo) * np.clip(u, 0, 1)))
    best = dict(loss=np.inf)

    def f(u):
        th = theta(u)
        out = R.evaluate(M.params(**th), ses, n_trials=N_TRIALS, seed=seed, sets=('all',))
        L = loss(out, ses) + 10 * np.sum(np.clip(u - 1, 0, None) ** 2 + np.clip(-u, 0, None) ** 2)
        if L < best['loss']:
            r = out['sets']['all']
            best.update(session=session, theta=th, loss=float(L), mem=out['mem'], gain=r['gain'],
                        model={mk: float(r[mk]) for _, mk, _ in STATS}, data={dk: float(ses['all'][dk][0]) for dk, _, _ in STATS})
            json.dump(best, open(os.path.join(out_dir, f'{session}.json'), 'w'), indent=1)
        return L

    u0 = np.clip([(M.FIT[k] - l) / (h - l) for k, (l, h) in BOUNDS.items()], 0.05, 0.95)
    es = cma.CMAEvolutionStrategy(u0, 0.15, dict(seed=seed + 1, maxfevals=MAXFEV, verbose=-9, popsize=10))
    while not es.stop():
        U = es.ask()
        es.tell(U, [f(u) for u in U])
    print(f"{session}: loss {best['loss']:.2f}, memory {best['mem']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--sessions', nargs='+', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for s in a.sessions:
        fit(s, a.out)


if __name__ == '__main__':
    main()
