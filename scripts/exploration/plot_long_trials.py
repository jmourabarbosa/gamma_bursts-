"""Plot very long trials: cued and uncued pool rates (1 s bins) for each trial, grouped by
noise level, plus the fraction of trials in which the cued pool is ahead over time.

usage: python scripts/exploration/plot_long_trials.py --in data/long60_sd0 data/long60_sd0.3 data/long60 \
           --labels "no noise" "noise SD 0.3" "noise SD 0.45" --out figures/long_trials.png
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from gamma_wm.network import A, B  # noqa: E402

C_CUE, C_UNC, INK, INK2 = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
plt.rcParams.update({'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 8,
                     'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'legend.frameon': False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    groups = []
    for d in args.inp:
        tr = []
        for f in sorted(glob.glob(os.path.join(d, '*.npz'))):
            z = np.load(f)
            r = z['rates'].astype(float)
            C, U = (A, B) if str(z['cue']) == 'A' else (B, A)
            n = (r.shape[0] - 1150) // 1000
            rc = r[1150:1150 + n * 1000, C].reshape(n, 1000).mean(1)
            ru = r[1150:1150 + n * 1000, U].reshape(n, 1000).mean(1)
            tr.append((rc, ru))
        groups.append(tr)
    nrow = max(len(g) for g in groups)
    fig = plt.figure(figsize=(4.2 * len(groups), 0.75 * nrow + 2.6))
    gs = fig.add_gridspec(nrow + 2, len(groups), hspace=0.35, wspace=0.25)
    for j, (g, lab) in enumerate(zip(groups, args.labels)):
        for i, (rc, ru) in enumerate(g):
            a = fig.add_subplot(gs[i, j])
            x = np.arange(len(rc)) + 0.5
            a.plot(x, rc, color=C_CUE, lw=1)
            a.plot(x, ru, color=C_UNC, lw=1)
            a.set_ylim(-5, 150)
            a.set_yticks([0, 100])
            a.set_xlim(0, len(rc))
            if i < len(g) - 1:
                a.set_xticklabels([])
            if i == 0:
                a.set_title(f'{lab}: pool rates (Hz), 1 s bins')
                if j == 0:
                    a.legend(['cued pool', 'uncued pool'], loc='upper right', ncol=2, fontsize=7)
        a.set_xlabel('time in delay (s)')
        a = fig.add_subplot(gs[nrow:, j])
        rc = np.array([t[0] for t in g])
        ru = np.array([t[1] for t in g])
        x = np.arange(rc.shape[1]) + 0.5
        a.plot(x, (rc > ru + 10).mean(0), color=C_CUE, label='cued pool ahead')
        a.plot(x, (ru > rc + 10).mean(0), color=C_UNC, label='uncued pool ahead')
        a.plot(x, (np.abs(rc - ru) <= 10).mean(0), color=INK2, ls=':', label='both pools off')
        a.set_ylim(-0.05, 1.05)
        a.set_xlim(0, rc.shape[1])
        a.set_xlabel('time in delay (s)')
        a.set_ylabel(f'fraction of {len(g)} trials')
        a.legend(fontsize=7, loc='center right')
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
