"""Figures for the mean-field model (figures/meanfield/).

usage: python scripts/mf_figures.py mechanism
       python scripts/mf_figures.py calibration --calib data/mf_calib
       python scripts/mf_figures.py fit --best data/mf_fit/best.json
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanfield import explore as E, fit as F, model as M  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), '..', 'figures', 'meanfield')
C_ON, C_OFF, INK, INK2, MUTED = '#eb6834', '#2a78d6', '#0b0b0b', '#52514e', '#b5b4ae'
TAU_C = {50: '#b3aaf0', 100: '#7a6be0', 150: '#3d2ea8'}          # one hue, light -> dark
plt.rcParams.update({'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 8, 'axes.edgecolor': INK2,
                     'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'legend.frameon': False, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK})


def paper_bars(ax, key, col):
    tgt = F.target_hist()[key][:-1] / 20
    ax.bar(F.EDGES[:-1], tgt, width=18, align='edge', color=col, alpha=0.35, linewidth=0, label='Panichello 2024')


def fig_mechanism():
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.4, height_ratios=[1, 1.15])
    for j, (key, col, n) in enumerate((('on', C_ON, 3890), ('off', C_OFF, 6174))):
        ax = fig.add_subplot(gs[0, j])
        paper_bars(ax, key, col)
        t = F.target_hist()[key]
        ax.set_title(f'Target: {key.capitalize()} states (n = {n:,})')
        ax.set_xlim(0, 620); ax.set_xlabel('state length (ms)'); ax.set_ylabel('probability / ms')
        ax.text(0.97, 0.93, f'{100 * t[-1]:.1f}% > 600 ms', transform=ax.transAxes, ha='right', color=INK2, fontsize=7)
    # memory survival after one dip
    ax = fig.add_subplot(gs[0, 2])
    D = [0, 50, 100, 150, 200, 250, 300, 400, 500, 600, 800]
    for tau, c in TAU_C.items():
        rows = E.one_dip(M.params(tau_N=tau, sigma=0.02), D, depth=0.8, sigma=0.02, n_rep=60)
        ax.plot(D, [r['A_high'] for r in rows], '-o', color=c, ms=3, lw=2, label=f'τ_N {tau} ms')
        ax.plot(D, [r['B_high'] for r in rows], '--', color=c, lw=1)
    ax.set_ylim(-0.03, 1.03); ax.set_xlabel('dip length (ms), drive −80%'); ax.set_ylabel('fraction of trials')
    ax.set_title('Who restarts after a dip')
    ax.legend(fontsize=6.5, loc='upper right', title='cued (solid) / uncued (dashed)', title_fontsize=6.5)
    # example trial
    p = M.params(sigma=0.02, dip_rate=3.0, dip_mean=120.0, dip_depth=0.8)
    s = M.simulate(p, [0] * 8, seed=4)
    k = int(np.argmax([(s['m'][i] < 0.5).sum() for i in range(8)]))
    t = np.arange(s['rA'].shape[1]) - s['t_cue']
    ax = fig.add_subplot(gs[1, :])
    ax.fill_between(t, 0, 140, where=s['m'][k] < 0.6, color=MUTED, alpha=0.35, lw=0, label='input dip')
    ax.plot(t, s['rA'][k], color=INK, lw=1.5, label='cued pool A')
    ax.plot(t, s['rB'][k], color=C_OFF, lw=1.5, label='uncued pool B')
    ax.plot(t, 10 * s['bg'][k], color=MUTED, lw=1, label='background ×10')
    ax2 = ax.twinx()
    ax2.spines['right'].set_visible(True)
    ax2.plot(t, s['sA'][k], color=INK, lw=1, ls=':'); ax2.plot(t, s['sB'][k], color=C_OFF, lw=1, ls=':')
    ax2.set_ylim(0, 1.0); ax2.set_ylabel('NMDA gating s (dotted)', color=INK2)
    ax.axvspan(0, 50, color=C_ON, alpha=0.25, lw=0)
    ax.text(55, 140, 'cue', ha='left', fontsize=7, color=INK2)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, 150)
    ax.set_xlabel('time from cue onset (ms)'); ax.set_ylabel('rate (Hz)')
    ax.set_title('Example trial: τ_N 100 ms, dips 3/s, mean 120 ms, −80%', pad=22)
    ax.legend(loc='lower left', ncol=4, fontsize=6.5, bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, 'fig1_targets_mechanism.png')
    fig.savefig(f, dpi=170, bbox_inches='tight')
    print('wrote', f)


def fig_surrogate(calib='data/mf_calib'):
    import glob
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(calib, 'val*.json')))]
    fig, ax = plt.subplots(1, 5, figsize=(7.4, 2.0), gridspec_kw=dict(wspace=0.55))
    items = (('acc', 'decoding accuracy', lambda r: r['acc']), ('n_on', 'On per trial', lambda r: r['n_on']),
             ('n_off', 'Off per trial', lambda r: r['n_off']),
             ('on', 'On mean (ms)', lambda r: np.mean(r['on_dur']) if r['on_dur'] else np.nan),
             ('off', 'Off mean (ms)', lambda r: np.mean(r['off_dur']) if r['off_dur'] else np.nan))
    tgt = dict(acc=0.70, n_on=2.35, n_off=3.74, on=192.4, off=146.1)
    for a, (k, lab, g) in zip(ax, items):
        x = np.array([g(r['full']) for r in rows]); y = np.array([g(r['fast']) for r in rows])
        m = ~np.isnan(x) & ~np.isnan(y)
        lo, hi = np.nanmin(np.r_[x, y]), np.nanmax(np.r_[x, y])
        a.plot([lo, hi], [lo, hi], color=MUTED, lw=1)
        a.scatter(x[m], y[m], s=10, color=INK, edgecolor='#fcfcfb', linewidth=0.5, zorder=3)
        a.axvline(tgt[k], color=C_ON if k in ('n_on', 'on') else C_OFF if k in ('n_off', 'off') else INK2, lw=1, ls=':')
        a.set_title(lab, fontsize=8); a.set_xlabel('full pipeline')
        a.text(0.04, 0.96, f'r = {np.corrcoef(x[m], y[m])[0, 1]:.3f}', transform=a.transAxes, va='top', fontsize=7, color=INK2)
    ax[0].set_ylabel('fast surrogate')
    fig.suptitle(f'Fast surrogate (linearised decoder) vs full pipeline (logistic, approx. LOO): {len(rows)} sessions at '
                 'random parameters, λ = 1000; dotted = paper value', fontsize=8, x=0.07, ha='left', y=1.08, color=INK)
    f = os.path.join(OUT, 'fig2_surrogate_validation.png')
    fig.savefig(f, dpi=170, bbox_inches='tight')
    print('wrote', f)


if __name__ == '__main__':
    {'mechanism': fig_mechanism, 'surrogate': fig_surrogate}[sys.argv[1]](*sys.argv[2:])
