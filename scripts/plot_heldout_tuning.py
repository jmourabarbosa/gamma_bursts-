"""Plot held-out tuning curves in On vs Off states (Panichello Fig. 4d style) for several
recorded-sample sizes, plus held-out population rate in On / Off (Fig. 4e style).

usage: python scripts/plot_heldout_tuning.py --in figures/heldout_model1_a figures/heldout_model1_b \
           --title "..." --out figures/heldout_model1.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C_ON, C_OFF, INK, INK2 = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
plt.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'legend.frameon': False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', nargs='+', required=True)
    ap.add_argument('--title', default='')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rows = []
    for d in args.inp:
        rows += json.load(open(os.path.join(d, 'heldout_tuning.json')))
    rows.sort(key=lambda r: r['K'])
    n = len(rows)
    fig, ax = plt.subplots(1, n + 1, figsize=(3.1 * (n + 1), 3.4), gridspec_kw=dict(wspace=0.45))
    lo = min(min(r['on_non_mean'] - r['on_non_sem'], r['off_non_mean'] - r['off_non_sem']) for r in rows)
    hi = max(max(r['on_pref_mean'] + r['on_pref_sem'], r['off_pref_mean'] + r['off_pref_sem']) for r in rows)
    pad = 0.15 * (hi - lo)
    for a, r in zip(ax[:n], rows):
        for st, c, lab in (('on', C_ON, 'On'), ('off', C_OFF, 'Off')):
            m = [r[f'{st}_non_mean'], r[f'{st}_pref_mean']]
            e = [r[f'{st}_non_sem'], r[f'{st}_pref_sem']]
            a.errorbar([0, 1], m, yerr=e, color=c, marker='o', ms=6, lw=1.6, capsize=3, label=lab)
        a.axhline(0, color=INK2, ls=':', lw=0.8)
        a.set_xticks([0, 1], ['non-\npreferred', 'preferred'])
        a.set_xlim(-0.4, 1.4)
        a.set_ylim(lo - pad, hi + pad)
        on_d = r['on_pref_mean'] - r['on_non_mean']
        off_d = r['off_pref_mean'] - r['off_non_mean']
        a.set_title(f'{r["K"]} recorded cells ({r["n_selective"]} selective)', fontsize=9)
        a.text(0.5, 0.04, f'pref − non: On {on_d:.2f}, Off {off_d:.2f}', transform=a.transAxes,
               ha='center', fontsize=7.5, color=INK2)
        if a is ax[0]:
            a.set_ylabel('held-out cells, delay activity\n(z-score, shuffle-corrected)')
            a.legend(fontsize=8, loc='upper left')
    a = ax[n]
    Ks = [r['K'] for r in rows]
    x = np.arange(n)
    a.bar(x - 0.18, [r['rate_on'] for r in rows], 0.34, color=C_ON, label='On')
    a.bar(x + 0.18, [r['rate_off'] for r in rows], 0.34, color=C_OFF, label='Off')
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xticks(x, [str(k) for k in Ks])
    a.set_xlabel('recorded cells')
    a.set_ylabel('held-out population rate\n(s.d. above pre-cue baseline)')
    a.set_title('Firing rate by state', fontsize=9)
    a.legend(fontsize=8)
    fig.suptitle(args.title, x=0.01, y=1.04, ha='left', fontweight='bold')
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
