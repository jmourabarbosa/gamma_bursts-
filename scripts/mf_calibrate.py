"""Validate the fast surrogate (linearised decoder, meanfield.surrogate) against the full Panichello
pipeline (L2 logistic decoder with approximate leave-one-out, meanfield.decoder.alo_confidence).

Sessions are simulated at random parameter sets; both pipelines decode the same Poisson counts with
the same shuffles (lam = 1000, 20 shuffles); On/Off statistics are saved for comparison.

usage: python scripts/mf_calibrate.py --n 10 --seed0 0 --out data/mf_calib
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanfield import model as M, recording as R, surrogate as S  # noqa: E402

RANGES = dict(tau_N=(50, 150), w_s=(1.35, 1.65), dip_rate=(1.0, 6.0), dip_mean=(40, 220), dip_shape=(1.0, 4.0),
              dip_depth=(0.5, 1.0), sigma=(0.01, 0.03), f_sel=(0.005, 0.05), tune_sd=(0.0, 0.03))


def draw(rng):
    return {k: float(rng.uniform(*v)) for k, v in RANGES.items()}


def generate(out, n, seed0, n_shuffle, lam):
    os.makedirs(out, exist_ok=True)
    for i in range(seed0, seed0 + n):
        f = os.path.join(out, f'val{i:03d}.json')
        if os.path.exists(f):
            continue
        rng = np.random.default_rng(1000 + i)
        th = draw(rng)
        rec = dict(R.REC_DEFAULTS, f_sel=th['f_sel'], tune_sd=th['tune_sd'])
        p = M.params(**{k: v for k, v in th.items() if k not in rec})
        ses = R.simulate_session(p, rec, seed=10 * i + 1)
        row = dict(theta=th, mem=float(M.memory_correct(ses['sim']).mean()))
        for name, res in (('full', R.full_pipeline(ses, n_shuffle=n_shuffle, lam=lam, seed=i)),
                          ('fast', S.fast_pipeline(ses, lam=lam, n_shuffle=n_shuffle, seed=i))):
            st = res['stats']
            row[name] = dict(acc=res['acc'], thr=float(res['thr']), n_on=float(st['n_on'].mean()), n_off=float(st['n_off'].mean()),
                             on_dur=st['on_dur'].tolist(), off_dur=st['off_dur'].tolist())
        json.dump(row, open(f, 'w'))
        print(i, ' | '.join(f'{k}: acc {row[k]["acc"]:.2f} On {row[k]["n_on"]:.2f}/{np.mean(row[k]["on_dur"] or [0]):.0f} '
                            f'Off {row[k]["n_off"]:.2f}/{np.mean(row[k]["off_dur"] or [0]):.0f}' for k in ('full', 'fast')), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--n-shuffle', type=int, default=20)
    ap.add_argument('--lam', type=float, default=S.LAM)
    ap.add_argument('--out', default='data/mf_calib')
    a = ap.parse_args()
    generate(a.out, a.n, a.seed0, a.n_shuffle, a.lam)


if __name__ == '__main__':
    main()
