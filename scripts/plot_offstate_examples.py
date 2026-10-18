"""Example trials: decoder confidence from a small recorded sample with On/Off states,
next to the full-network memory readout (ground truth) and the population raster.

usage: python scripts/plot_offstate_examples.py --data data/session \
           --states figures/subsampling_v1/part_25_1600/states_K25.npz --out figures/subsampling_v1/examples_K25.png
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_wm.network import A, B  # noqa: E402

C_ON, C_OFF, C_A, C_B, INK, INK2 = '#2a78d6', '#eb6834', '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
plt.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'legend.frameon': False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--states', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--trials', type=int, nargs='+', default=None)
    args = ap.parse_args()
    sdirs = sorted(glob.glob(args.data + '*'))
    meta = json.load(open(os.path.join(sdirs[0], 'meta.json')))
    group = np.load(os.path.join(sdirs[0], 'net.npz'))['group']
    trials = []
    for sd in sdirs:
        for f in sorted(glob.glob(os.path.join(sd, 'trial*.npz'))):
            d = dict(np.load(f))
            d['cue'] = str(d['cue'])
            trials.append(d)
    st = np.load(args.states)
    conf, state, t, sample = st['conf'], st['state'], st['t'], st['sample']
    T = meta['task']
    t_cue = T['t_pre']
    n_t = int(t_cue + T['cue_dur'] + T['t_delay'])
    ks = args.trials or list(np.argsort(-(state == -1).sum(1))[:3])
    fig = plt.figure(figsize=(10, 3.0 * len(ks) + 0.6))
    subs = fig.subfigures(len(ks), 1, hspace=0.06)
    subs = np.atleast_1d(subs)
    ax = []
    for sf in subs:
        ax += list(sf.subplots(3, 1, sharex=True, gridspec_kw=dict(height_ratios=[1.0, 0.7, 1.0], hspace=0.12)))
    ax = np.array(ax)
    sel = np.isin(group[sample], [A, B])
    for j, k in enumerate(ks):
        d = trials[k]
        a0, a1, a2 = ax[3 * j], ax[3 * j + 1], ax[3 * j + 2]
        a0.plot(t, conf[k], color=INK, lw=1)
        a0.fill_between(t, 0, 1, where=state[k] == 1, color=C_ON, alpha=0.25, lw=0, step='mid')
        a0.fill_between(t, 0, 1, where=state[k] == -1, color=C_OFF, alpha=0.3, lw=0, step='mid')
        a0.axhline(0.5, color=INK2, ls=':', lw=0.8)
        a0.set_ylim(0, 1)
        a0.set_ylabel('decoder\nconfidence')
        a0.set_title(f'Trial {k} (cue {d["cue"]}): {len(sample)} recorded cells ({sel.sum()} selective); '
                     'blue = On, orange = Off')
        lut = np.full(len(group), -1)
        lut[sample] = np.arange(len(sample))
        m = np.isin(d['spk_i'], sample)
        cols = np.where(sel[lut[d['spk_i'][m]]], C_A, INK2)
        a1.scatter(d['spk_t'][m], lut[d['spk_i'][m]], s=2, c=cols, lw=0)
        a1.set_yticks([])
        a1.set_ylabel('recorded\ncells')
        r = d['rates'][:n_t]
        sm = lambda x: np.convolve(x, np.ones(100) / 100, mode='same')  # noqa: E731
        cued, unc = (A, B) if d['cue'] == 'A' else (B, A)
        a2.plot(np.arange(n_t), sm(r[:, cued]), color=C_A, label='cued pool (all cells)')
        a2.plot(np.arange(n_t), sm(r[:, unc]), color=C_B, label='uncued pool (all cells)')
        for (s_, e_) in _runs(state[k] == -1):
            a2.axvspan(t[s_], t[min(e_, len(t) - 1)], color=C_OFF, alpha=0.15, lw=0)
        a2.set_ylabel('pool rate\n(Hz, 100 ms)')
        a2.legend(fontsize=7.5, loc='upper right', ncol=2)
        for a in (a0, a1, a2):
            a.axvline(t_cue, color=INK2, lw=0.8)
    ax[-1].set_xlabel('time (ms); vertical line = cue onset')
    for a in ax:
        a.set_xlim(t_cue - 100, n_t)
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print('wrote', args.out)


def _runs(mask):
    a = np.r_[False, mask, False].astype(int)
    return list(zip(np.where(np.diff(a) == 1)[0], np.where(np.diff(a) == -1)[0]))


if __name__ == '__main__':
    main()
