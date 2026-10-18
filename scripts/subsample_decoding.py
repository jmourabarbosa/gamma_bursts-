"""Panichello-style On/Off analysis as a function of the number of recorded neurons.

For each sample size K, draw `n_draws` random samples of K neurons (from all cells,
E and I; with the model's own fraction of selective cells) and run the full pipeline
(gamma_wm.panichello). Also reports the ground truth: the full-network readout (cued
minus uncued pool rate) during decoder Off states.

usage: python scripts/subsample_decoding.py --data data/session --out figures/subsampling \
           --sizes 25 50 100 200 400 800 --n-draws 2
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_wm import panichello as pa  # noqa: E402
from gamma_wm.network import A, B  # noqa: E402


def load(prefix):
    sdirs = sorted(glob.glob(prefix + '*'))
    meta = json.load(open(os.path.join(sdirs[0], 'meta.json')))
    group = np.load(os.path.join(sdirs[0], 'net.npz'))['group']
    trials = []
    for sd in sdirs:
        for f in sorted(glob.glob(os.path.join(sd, 'trial*.npz'))):
            d = dict(np.load(f))
            d['cue'] = str(d['cue'])
            trials.append(d)
    return meta, group, trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sizes', type=int, nargs='+', default=[25, 50, 100, 200, 400, 800])
    ap.add_argument('--n-draws', type=int, default=2)
    ap.add_argument('--n-shuffle', type=int, default=20)
    ap.add_argument('--lam', type=float, default=10.0, help='L2 penalty of the decoder')
    ap.add_argument('--frac-selective', type=float, default=None,
                    help='if set, draw this fraction of each sample from the memory pools '
                         '(the rest from non-selective E and I cells), e.g. 0.44 as in Panichello')
    ap.add_argument('--pool', default='all', choices=['all', 'E'], help='cells to sample from')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meta, group, trials = load(args.data)
    T = meta['task']
    t_cue = T['t_pre']
    t_don = t_cue + T['cue_dur']
    n_t = int(t_don + T['t_delay'])
    y = np.array([d['cue'] == 'A' for d in trials], int)
    # ground-truth readout, 100 ms boxcar, sign relative to cue
    truth = []
    for d in trials:
        r = d['rates'][:n_t]
        diff = (r[:, A] - r[:, B]) * (1 if d['cue'] == 'A' else -1)
        truth.append(np.convolve(diff, np.ones(100) / 100, mode='same'))
    truth = np.array(truth)
    cells = np.arange(len(group)) if args.pool == 'all' else np.where(group < 3)[0]
    rng = np.random.default_rng(123)
    rows = []
    for K in args.sizes:
        for dr in range(args.n_draws):
            if args.frac_selective is None:
                sample = np.sort(rng.choice(cells, min(K, len(cells)), replace=False))
            else:
                sel_cells = np.where(np.isin(group, [A, B]))[0]
                other = np.setdiff1d(cells, sel_cells)
                n_s = min(int(round(args.frac_selective * K)), len(sel_cells))
                sample = np.sort(np.r_[rng.choice(sel_cells, n_s, replace=False),
                                       rng.choice(other, K - n_s, replace=False)])
            n_sel = int(np.isin(group[sample], [A, B]).sum())
            res = pa.run(trials, sample, y, n_t, t_cue, delay_start=t_don + 500,
                         count_window=(t_cue, t_cue + 1500), n_shuffle=args.n_shuffle, seed=K * 10 + dr,
                         lam=args.lam)
            st = res['stats']
            tb = res['t'].astype(int)
            off = res['state'] == -1
            truth_off = truth[:, np.clip(tb, 0, n_t - 1)][off]
            row = dict(K=K, draw=dr, lam=args.lam, frac_selective=args.frac_selective, n_selective=n_sel, acc_delay=res['acc_delay'], acc_on=res['acc_on'],
                       acc_off=res['acc_off'], frac_on=res['frac_on_delay'], frac_off=res['frac_off_delay'],
                       on_per_trial=float(st['n_on'].mean()), off_per_trial=float(st['n_off'].mean()),
                       on_mean=float(st['on_dur'].mean()) if len(st['on_dur']) else np.nan,
                       on_median=float(np.median(st['on_dur'])) if len(st['on_dur']) else np.nan,
                       off_mean=float(st['off_dur'].mean()) if len(st['off_dur']) else np.nan,
                       off_median=float(np.median(st['off_dur'])) if len(st['off_dur']) else np.nan,
                       mixture_mbit=res['mixture_mbit'],
                       truth_correct_in_off=float((truth_off > 0).mean()) if truth_off.size else np.nan,
                       **{('split_' + k): v for k, v in res.get('split_half', {}).items()})
            rows.append(row)
            print(json.dumps(row), flush=True)
            json.dump(rows, open(os.path.join(args.out, 'subsampling.json'), 'w'), indent=1)
            if dr == 0:
                np.savez_compressed(os.path.join(args.out, f'states_K{K}.npz'), conf=res['conf'], state=res['state'],
                                    t=res['t'], y=y, sample=sample, on_dur=st['on_dur'], off_dur=st['off_dur'])
    json.dump(rows, open(os.path.join(args.out, 'subsampling.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
