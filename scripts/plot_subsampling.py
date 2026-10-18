"""Figure: Panichello-style On/Off statistics vs number of recorded neurons.

usage: python scripts/plot_subsampling.py --in figures/subsampling_v1 --title "..."
Collects every subsampling.json below --in and writes subsampling.png / .json there.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C1, C2, INK, INK2 = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
plt.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'lines.linewidth': 1.4, 'legend.frameon': False})
# Panichello et al. 2024 reference values
P_OFF_PER_TRIAL, P_ON_PER_TRIAL, P_OFF_MEAN, P_ON_MEAN = 3.74, 2.35, 146.1, 192.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--title', default='')
    args = ap.parse_args()
    rows = []
    for f in glob.glob(os.path.join(args.inp, '**', 'subsampling.json'), recursive=True):
        rows += json.load(open(f))
    rows.sort(key=lambda r: (r['K'], r['draw']))
    json.dump(rows, open(os.path.join(args.inp, 'subsampling_all.json'), 'w'), indent=1)
    Ks = sorted(set(r['K'] for r in rows))

    def agg(key):
        m = np.array([np.nanmean([r[key] for r in rows if r['K'] == K]) for K in Ks])
        return m

    fig, ax = plt.subplots(2, 3, figsize=(11.5, 6.2))
    ax = ax.ravel()
    panels = [
        ('off_per_trial', 'on_per_trial', 'states per trial (cue + 1.5 s)', P_OFF_PER_TRIAL, P_ON_PER_TRIAL),
        ('off_mean', 'on_mean', 'mean state duration (ms)', P_OFF_MEAN, P_ON_MEAN),
        ('acc_off', 'acc_on', 'decoding accuracy within state', None, None),
    ]
    for a, (koff, kon, lab, poff, pon) in zip(ax[:3], panels):
        for k, c, name in ((koff, C2, 'Off'), (kon, C1, 'On')):
            a.plot(Ks, agg(k), 'o-', color=c, label=name, ms=5)
            for r in rows:
                a.plot(r['K'], r[k], '.', color=c, alpha=0.4, ms=4)
        if poff is not None:
            a.axhline(poff, color=C2, ls='--', lw=0.9)
            a.axhline(pon, color=C1, ls='--', lw=0.9)
        a.set_xscale('log')
        a.set_xlabel('recorded neurons')
        a.set_ylabel(lab)
        a.legend(fontsize=8)
    ax[0].set_title('On/Off states (dashed: Panichello)')
    ax[2].axhline(0.5, color=INK2, ls=':', lw=0.8)
    a = ax[3]
    a.plot(Ks, agg('truth_correct_in_off'), 'o-', color=INK, ms=5)
    a.set_ylim(0, 1.05)
    a.axhline(0.5, color=INK2, ls=':', lw=0.8)
    a.set_xscale('log')
    a.set_xlabel('recorded neurons')
    a.set_ylabel('full-network memory correct')
    a.set_title('Ground truth during decoder Off states')
    a = ax[4]
    if 'split_ev_on' in rows[0]:
        a.plot(Ks, agg('split_ev_on'), 'o-', color=C1, label='On', ms=5)
        a.plot(Ks, agg('split_ev_off'), 'o-', color=C2, label='Off', ms=5)
        a.set_xscale('log')
        a.set_xlabel('recorded neurons')
        a.set_ylabel('cue-explained variance')
        a.set_title('Held-out half: tuning in On vs Off')
        a.legend(fontsize=8)
    a = ax[5]
    a.plot(Ks, agg('mixture_mbit'), 'o-', color=INK, ms=5)
    a.axhline(0, color=INK2, ls=':', lw=0.8)
    a.set_xscale('log')
    a.set_xlabel('recorded neurons')
    a.set_ylabel('2- minus 1-state (mbit/sample)')
    a.set_title('Two-state mixture test')
    for a in ax:
        a.set_xticks(Ks)
        a.set_xticklabels([str(k) for k in Ks])
        a.minorticks_off()
    fig.suptitle(args.title, x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(args.inp, 'subsampling.png'), dpi=150, bbox_inches='tight')
    print('wrote', os.path.join(args.inp, 'subsampling.png'))


if __name__ == '__main__':
    main()
