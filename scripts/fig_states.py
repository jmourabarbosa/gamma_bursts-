"""One-model summary figure in the style of Panichello et al. 2024 Fig. 2:
(a) trial-averaged decoding accuracy through the delay, (b) single-trial decoder confidence with
On/Off states for a set of consecutive trials (not hand-picked), (c) Off and On state-length
histograms pooled over several random recorded samples (from scripts/nmda_sweep.py output).

usage: python scripts/fig_states.py --data data/ctrl_nmda100_session --sweep figures/nmda_sweep/sweep.json \
           --label "NMDA 100 ms" --out figures/model_nmda100
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_wm import panichello as pa  # noqa: E402
from gamma_wm.network import A, B  # noqa: E402
from heldout_tuning import load  # noqa: E402

C_ON, C_OFF, INK, INK2 = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
plt.rcParams.update({'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
                     'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 8,
                     'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'legend.frameon': False})


def runs(mask):
    a = np.r_[False, mask, False].astype(int)
    return list(zip(np.where(np.diff(a) == 1)[0], np.where(np.diff(a) == -1)[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--sweep', required=True, help='sweep.json from nmda_sweep.py (pooled state lengths)')
    ap.add_argument('--label', required=True, help='label of this model in sweep.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--K', type=int, default=100)
    ap.add_argument('--draw', type=int, default=0, help='which random sample to show (same seeds as nmda_sweep)')
    ap.add_argument('--n-show', type=int, default=12)
    ap.add_argument('--lam', type=float, default=10.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meta, group, trials = load(args.data)
    T = meta['task']
    t_cue, t_don = T['t_pre'], T['t_pre'] + T['cue_dur']
    n_t = int(t_don + T['t_delay'])
    y = np.array([d['cue'] == 'A' for d in trials], int)
    sample = np.sort(np.random.default_rng(100 + args.draw).choice(len(group), args.K, replace=False))
    r = pa.run(trials, sample, y, n_t, t_cue, delay_start=t_don + 500, count_window=(t_don, n_t),
               n_shuffle=10, seed=args.draw + 1, lam=args.lam, split_half=False)
    conf, state, t = r['conf'], r['state'], r['t']
    sw = next(s for s in json.load(open(args.sweep)) if s['label'] == args.label)
    n_pool = int(np.isin(group[sample], [A, B]).sum())

    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2.6, 1.1], hspace=0.45, wspace=0.25)
    # (a) trial-averaged decoding
    a = fig.add_subplot(gs[0, :])
    acc = (conf > 0.5).mean(0)
    a.plot(t, acc, color=INK, lw=1.2)
    a.axhline(0.5, color=INK2, ls=':', lw=0.8)
    a.axvspan(t_cue, t_don, color='#e4e3df', lw=0)
    a.set_ylim(0.3, 1.02)
    a.set_xlim(t[0], t[-1])
    a.set_ylabel('decoding accuracy\n(all trials)')
    a.set_title(f'{args.label}: trial-averaged decoding {sw["acc_delay"]:.0%} in the delay '
                f'({args.K} random cells, {n_pool} memory cell{'s' if n_pool != 1 else ''}; grey = cue)')
    # (b) single trials: first n-show trials, no selection
    a = fig.add_subplot(gs[1, :])
    for j in range(args.n_show):
        base = (args.n_show - 1 - j) * 1.25
        s = np.where(t >= t_cue, state[j], 0)
        for lab, col in ((1, C_ON), (-1, C_OFF)):
            for i0, i1 in runs(s == lab):
                a.axvspan(t[i0] - 5, t[i1 - 1] + 5, ymin=0, ymax=1, color=col, alpha=0, lw=0)
                a.fill_between([t[i0] - 5, t[i1 - 1] + 5], base, base + 1, color=col, alpha=0.3, lw=0)
        a.plot(t, base + conf[j], color=INK, lw=0.8)
        a.plot([t[0], t[-1]], [base + 0.5] * 2, color=INK2, ls=':', lw=0.6)
        d = trials[j]
        ok = (d['rates'][n_t - 300:n_t, A].mean() > d['rates'][n_t - 300:n_t, B].mean()) == (d['cue'] == 'A')
        a.text(t[-1] + 15, base + 0.5, f'trial {j + 1}' + ('' if ok else ' (memory lost)'), fontsize=7,
               va='center', color=INK2)
    a.axvline(t_cue, color=INK2, lw=0.6)
    a.axvline(t_don, color=INK2, lw=0.6)
    a.set_xlim(t[0], t[-1])
    a.set_yticks([])
    a.set_xlabel('time (ms)')
    a.set_title(f'First {args.n_show} trials (not selected): decoder confidence in the correct cue (0–1 per row); '
                'blue = On, orange = Off')
    # (c) histograms
    bins = np.arange(0, 820, 20)
    for k, (key, col, name) in enumerate((('off_dur', C_OFF, 'Off'), ('on_dur', C_ON, 'On'))):
        a = fig.add_subplot(gs[2, k])
        v = np.array(sw[key])
        a.hist(np.clip(v, 0, bins[-1] - 1), bins, color=col, density=True)
        a.set_xlim(0, 800)
        a.set_xlabel('state length (ms)')
        a.set_ylabel('probability density')
        a.set_title(f'{name} states: n = {len(v)}, mean {v.mean():.0f} ms, median {np.median(v):.0f} ms',
                    fontsize=8)
    fig.text(0.5, 0.005, f'Histograms pooled over 5 random samples of {args.K} cells; the last bin collects '
             'states ≥ 800 ms. Memory correct at end of delay: '
             f'{sw["memory_correct_end"]:.0%} of {len(trials)} trials.', ha='center', fontsize=7.5, color=INK2)
    fig.savefig(os.path.join(args.out, 'states.png'), dpi=150, bbox_inches='tight')
    print('wrote', os.path.join(args.out, 'states.png'))


if __name__ == '__main__':
    main()
