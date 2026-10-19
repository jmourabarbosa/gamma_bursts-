"""Which parameters set On and Off lengths? One-at-a-time sweeps around a fitted parameter set
(fast pipeline, common random numbers), plus log-log elasticities at the fit.

usage: python scripts/mf_sensitivity.py --best data/mf_fit/best.json --out data/mf_sens.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanfield import fit as F  # noqa: E402

KEYS = ('on_mean', 'on_median', 'off_mean', 'off_median', 'n_on', 'n_off', 'acc', 'mem')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--best', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=7)
    ap.add_argument('--n-sessions', type=int, default=4)
    ap.add_argument('--params', nargs='*', default=list(F.BOUNDS))
    a = ap.parse_args()
    th0 = json.load(open(a.best))['theta']
    res = dict(theta=th0, base={k: v for k, v in F.evaluate(th0, a.n_sessions, seed=50).items() if k in KEYS}, sweeps={})
    for name in a.params:
        lo, hi = F.BOUNDS[name]
        vals = np.linspace(lo, hi, a.n)
        rows = []
        for v in vals:
            s = F.evaluate(dict(th0, **{name: float(v)}), a.n_sessions, seed=50)
            rows.append(dict(value=float(v), **{k: s[k] for k in KEYS}))
        res['sweeps'][name] = rows
        print(name, ' '.join(f'{r["value"]:.3g}:On{r["on_mean"]:.0f}/Off{r["off_mean"]:.0f}' for r in rows), flush=True)
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
