"""Cross-validated (held-out) tuning curves in On vs Off states, as Panichello et al. 2024
Fig. 4d,e, for a binary task.

For a random recorded sample of K cells: split it in two halves; label On/Off states with
the decoder on one half (gamma_wm.panichello); for every neuron of the other half compute
the mean z-scored activity (z across trials, per time bin) over that trial's On (or Off)
delay bins, separately for each cue; align to the neuron's preferred cue (largest On+Off
response) and subtract the same quantity computed with shuffled cue labels. Then swap
halves. Also reports the held-out population rate in On / Off relative to the pre-cue
baseline.

usage: python scripts/heldout_tuning.py --data data/session --sizes 25 100 400 --out figures/heldout_model1
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_wm import panichello as pa  # noqa: E402
from gamma_wm.analysis import loo_confidence_dual  # noqa: E402
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


def state_means(Z, state, y, lab, dmask):
    """Per held-out neuron: mean z over each trial's bins in state `lab` (delay only),
    then mean over trials of each cue. Returns (n_neurons, 2) [cue B, cue A] (nan if none)."""
    out = np.full((Z.shape[1], 2), np.nan)
    per = []
    for k in range(Z.shape[0]):
        m = (state[k] == lab) & dmask
        per.append(Z[k][:, m].mean(1) if m.sum() >= 1 else np.full(Z.shape[1], np.nan))
    per = np.array(per)
    for c in (0, 1):
        sel = (y == c) & ~np.isnan(per[:, 0])
        if sel.sum():
            out[:, c] = per[sel].mean(0)
    return out


def aligned(on, off):
    """Align to preferred cue (argmax of On+Off): returns arrays (n, 2) [pref, nonpref]."""
    pref = np.nanargmax(np.nan_to_num(on, nan=-1e9) + np.nan_to_num(off, nan=-1e9), axis=1)
    idx = np.arange(len(pref))
    return (np.c_[on[idx, pref], on[idx, 1 - pref]], np.c_[off[idx, pref], off[idx, 1 - pref]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sizes', type=int, nargs='+', default=[25, 100, 400])
    ap.add_argument('--n-shuffle', type=int, default=10)
    ap.add_argument('--n-tuning-shuffle', type=int, default=20)
    ap.add_argument('--lam', type=float, default=10.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meta, group, trials = load(args.data)
    T = meta['task']
    t_cue = T['t_pre']
    t_don = t_cue + T['cue_dur']
    n_t = int(t_don + T['t_delay'])
    y = np.array([d['cue'] == 'A' for d in trials], int)
    rng = np.random.default_rng(7)
    results = []
    for K in args.sizes:
        sample = np.sort(rng.choice(len(group), K, replace=False))
        Call, tc = pa.counts_matrix(trials, sample, n_t, 100, 10)
        # baseline: pre-cue counts per neuron
        base = Call[:, :, (tc >= 200) & (tc < t_cue - 50)]
        keep = tc >= t_cue - 100
        C, tk = Call[:, :, keep], tc[keep]
        valid = tk >= t_cue
        dmask = tk >= t_don + 500
        perm = rng.permutation(K)
        halves = (perm[: K // 2], perm[K // 2:])
        res = dict(K=K, n_selective=int(np.isin(group[sample], [A, B]).sum()),
                   on_pref=[], on_non=[], off_pref=[], off_non=[], rate_on=[], rate_off=[], n_on_bins=0, n_off_bins=0)
        for h in (0, 1):
            lab_idx, test_idx = halves[h], halves[1 - h]
            Cl = np.ascontiguousarray(C[:, lab_idx])
            conf = loo_confidence_dual(Cl, y, args.lam)
            nulls = np.array([loo_confidence_dual(Cl, y, args.lam, shuffle_rng=np.random.default_rng(1000 * K + 10 * h + s))
                              for s in range(args.n_shuffle)])
            state, _, _ = pa.label_states(conf, nulls, valid)
            res['n_on_bins'] += int(((state == 1) & dmask).sum())
            res['n_off_bins'] += int(((state == -1) & dmask).sum())
            Xt = C[:, test_idx].astype(float)
            # z-score across trials per neuron and time bin
            Z = (Xt - Xt.mean(0, keepdims=True)) / (Xt.std(0, keepdims=True) + 1e-9)
            on = state_means(Z, state, y, 1, dmask)
            off = state_means(Z, state, y, -1, dmask)
            on_a, off_a = aligned(on, off)
            # shuffled-label null for the alignment bias
            nul_on, nul_off = [], []
            for s in range(args.n_tuning_shuffle):
                ys = np.random.default_rng(5000 + s).permutation(y)
                o1, o2 = aligned(state_means(Z, state, ys, 1, dmask), state_means(Z, state, ys, -1, dmask))
                nul_on.append(o1)
                nul_off.append(o2)
            on_c = on_a - np.nanmean(nul_on, 0)
            off_c = off_a - np.nanmean(nul_off, 0)
            res['on_pref'] += list(on_c[:, 0])
            res['on_non'] += list(on_c[:, 1])
            res['off_pref'] += list(off_c[:, 0])
            res['off_non'] += list(off_c[:, 1])
            # population rate in On / Off relative to the pre-cue baseline (sd units)
            bmu = base[:, test_idx].mean((0, 2))
            bsd = base[:, test_idx].std((0, 2)) + 1e-9
            for lab, key in ((1, 'rate_on'), (-1, 'rate_off')):
                vals = []
                for k in range(len(trials)):
                    m = (state[k] == lab) & dmask
                    if m.sum():
                        vals.append(((Xt[k][:, m].mean(1) - bmu) / bsd).mean())
                res[key].append(float(np.mean(vals)) if vals else np.nan)
        for key in ('on_pref', 'on_non', 'off_pref', 'off_non'):
            v = np.array(res[key], float)
            res[key + '_mean'] = float(np.nanmean(v))
            res[key + '_sem'] = float(np.nanstd(v) / np.sqrt(np.sum(~np.isnan(v))))
            res[key] = [float(x) for x in v]
        res['rate_on'] = float(np.nanmean(res['rate_on']))
        res['rate_off'] = float(np.nanmean(res['rate_off']))
        print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}), flush=True)
        results.append(res)
        json.dump(results, open(os.path.join(args.out, 'heldout_tuning.json'), 'w'))


if __name__ == '__main__':
    main()
