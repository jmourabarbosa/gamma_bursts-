"""Compare On/Off state-length histograms (Panichello et al. 2024 Fig. 2c) and memory survival
across models that differ in one parameter (e.g. the NMDA decay time constant).

For each model, decode several independent uniform random samples of K cells (Panichello
pipeline), pool the On and Off state lengths over the delay, and plot the histograms. Also
reports memory correct at the end of the delay and whether the cued pool restarts first after
pool pauses.

usage: python scripts/nmda_sweep.py --data data/ctrl_nmda50_session data/ouT_sd0.45_session \
           --labels "NMDA 50 ms" "NMDA 300 ms" --out figures/nmda_sweep
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_wm import panichello as pa  # noqa: E402
from gamma_wm.network import A, B  # noqa: E402
from heldout_tuning import load  # noqa: E402

C_ON, C_OFF, INK, INK2 = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
plt.rcParams.update({'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 8,
                     'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'legend.frameon': False})


def pool_pauses(trials, n_t, d0):
    """Pool pauses (both pools < 2 Hz, 20 ms smoothing, >= 50 ms) in the delay and whether the
    cued pool leads in the 100 ms after each pause."""
    out = []
    for d in trials:
        r = d['rates'][:n_t].astype(float)
        C, U = (A, B) if d['cue'] == 'A' else (B, A)
        cp, up = uniform_filter1d(r[:, C], 20), uniform_filter1d(r[:, U], 20)
        m = np.r_[0, ((cp[d0:] < 2) & (up[d0:] < 2)).astype(int), 0]
        for a, b in zip(np.where(np.diff(m) == 1)[0] + d0, np.where(np.diff(m) == -1)[0] + d0):
            if b - a >= 50 and b + 100 <= n_t:
                out.append((b - a, bool(cp[b:b + 100].sum() > up[b:b + 100].sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--K', type=int, default=100)
    ap.add_argument('--draws', type=int, default=5)
    ap.add_argument('--n-shuffle', type=int, default=10)
    ap.add_argument('--lam', type=float, default=10.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = []
    for prefix, lab in zip(args.data, args.labels):
        meta, group, trials = load(prefix)
        T = meta['task']
        t_cue, t_don = T['t_pre'], T['t_pre'] + T['cue_dur']
        n_t = int(t_don + T['t_delay'])
        y = np.array([d['cue'] == 'A' for d in trials], int)
        on_d, off_d, n_on, n_off, acc = [], [], [], [], []
        for s in range(args.draws):
            sample = np.sort(np.random.default_rng(100 + s).choice(len(group), args.K, replace=False))
            r = pa.run(trials, sample, y, n_t, t_cue, delay_start=t_don + 500, count_window=(t_don, n_t),
                       n_shuffle=args.n_shuffle, seed=s + 1, lam=args.lam, split_half=False)
            st = r['stats']
            on_d += list(st['on_dur'])
            off_d += list(st['off_dur'])
            n_on.append(float(st['n_on'].mean()))
            n_off.append(float(st['n_off'].mean()))
            acc.append(float(r['acc_delay']))
        end_ok = np.mean([(d['rates'][n_t - 300:n_t, A].mean() > d['rates'][n_t - 300:n_t, B].mean()) == (d['cue'] == 'A')
                          for d in trials])
        pp = pool_pauses(trials, n_t, int(t_don + 500))
        row = dict(label=lab, data=prefix, tau_nmda=meta['params'].get('tau_nmda'), noise_sd=T.get('ext_noise_sd', 0),
                   acc_delay=float(np.mean(acc)), memory_correct_end=float(end_ok),
                   off_per_trial=float(np.mean(n_off)), on_per_trial=float(np.mean(n_on)),
                   off_mean=float(np.mean(off_d)) if off_d else None, off_median=float(np.median(off_d)) if off_d else None,
                   on_mean=float(np.mean(on_d)) if on_d else None, on_median=float(np.median(on_d)) if on_d else None,
                   n_off_states=len(off_d), n_on_states=len(on_d),
                   n_pool_pauses=len(pp), cued_restarts_first=float(np.mean([p[1] for p in pp])) if pp else None,
                   off_dur=[float(v) for v in off_d], on_dur=[float(v) for v in on_d])
        print(json.dumps({k: v for k, v in row.items() if not isinstance(v, list)}), flush=True)
        res.append(row)
        json.dump(res, open(os.path.join(args.out, 'sweep.json'), 'w'))

    n = len(res)
    fig, ax = plt.subplots(n, 2, figsize=(7, 1.9 * n + 0.4), squeeze=False, gridspec_kw=dict(hspace=0.75, wspace=0.3))
    bins = np.arange(0, 820, 20)
    for i, r in enumerate(res):
        for j, (key, col, name) in enumerate((('off_dur', C_OFF, 'Off'), ('on_dur', C_ON, 'On'))):
            a = ax[i, j]
            v = np.array(r[key])
            if len(v):
                a.hist(np.clip(v, 0, bins[-1] - 1), bins, color=col, density=True)
            a.set_xlim(0, 800)
            a.set_yticks([])
            mean = f'{v.mean():.0f}' if len(v) else '–'
            a.set_title(f'{r["label"]}: {name} (n={len(v)}, mean {mean} ms)', fontsize=8)
            if i == n - 1:
                a.set_xlabel('state length (ms)')
        ax[i, 0].text(-0.08, 0.5, f'memory {r["memory_correct_end"]:.0%}\ndecoding {r["acc_delay"]:.0%}',
                      transform=ax[i, 0].transAxes, ha='right', va='center', fontsize=7.5, color=INK2)
    fig.savefig(os.path.join(args.out, 'state_lengths.png'), dpi=150, bbox_inches='tight')
    print('wrote', os.path.join(args.out, 'state_lengths.png'))


if __name__ == '__main__':
    main()
