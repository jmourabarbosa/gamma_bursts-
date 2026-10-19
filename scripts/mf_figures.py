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
    methods = [m for m in ('fast', 'expected') if all(m in r for r in rows)]
    names = dict(fast='spike-based fast decoder', expected='expected z (no spikes)')
    fig, axs = plt.subplots(len(methods), 5, figsize=(7.4, 2.1 * len(methods)), squeeze=False,
                            gridspec_kw=dict(wspace=0.55, hspace=0.7))
    for ax, meth in zip(axs, methods):
        _surrogate_row(ax, rows, meth, names[meth])
    fig.suptitle(f'Approximations vs the full pipeline (logistic decoder, approx. LOO): {len(rows)} sessions at random '
                 'parameters, λ = 1000; dotted = paper value', fontsize=8, x=0.07, ha='left', y=1.02, color=INK)
    f = os.path.join(OUT, 'fig2_surrogate_validation.png')
    fig.savefig(f, dpi=170, bbox_inches='tight')
    print('wrote', f)


def _surrogate_row(ax, rows, meth, label):
    items = (('acc', 'decoding accuracy', lambda r: r['acc']), ('n_on', 'On per trial', lambda r: r['n_on']),
             ('n_off', 'Off per trial', lambda r: r['n_off']),
             ('on', 'On mean (ms)', lambda r: np.mean(r['on_dur']) if r['on_dur'] else np.nan),
             ('off', 'Off mean (ms)', lambda r: np.mean(r['off_dur']) if r['off_dur'] else np.nan))
    tgt = dict(acc=0.70, n_on=2.35, n_off=3.74, on=192.4, off=146.1)
    for a, (k, lab, g) in zip(ax, items):
        x = np.array([g(r['full']) for r in rows]); y = np.array([g(r[meth]) for r in rows])
        m = ~np.isnan(x) & ~np.isnan(y)
        lo, hi = np.nanmin(np.r_[x, y]), np.nanmax(np.r_[x, y])
        a.plot([lo, hi], [lo, hi], color=MUTED, lw=1)
        a.scatter(x[m], y[m], s=10, color=INK, edgecolor='#fcfcfb', linewidth=0.5, zorder=3)
        a.axvline(tgt[k], color=C_ON if k in ('n_on', 'on') else C_OFF if k in ('n_off', 'off') else INK2, lw=1, ls=':')
        a.set_title(lab, fontsize=8); a.set_xlabel('full pipeline')
        a.text(0.04, 0.96, f'r = {np.corrcoef(x[m], y[m])[0, 1]:.3f}', transform=a.transAxes, va='top', fontsize=7, color=INK2)
    ax[0].set_ylabel(label, fontsize=7.5)


def _load_val(val):
    import glob
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(val, 'ses*.json')))]
    out = {}
    for name in ('full', 'fast'):
        g = lambda k: np.concatenate([np.asarray(r[name][k], float) for r in rows])
        out[name] = dict(on_dur=g('on_dur'), off_dur=g('off_dur'), n_on=g('n_on'), n_off=g('n_off'),
                         acc=float(np.mean([r[name]['acc'] for r in rows])))
    out['mem'] = float(np.mean([r['mem'] for r in rows]))
    out['n_sessions'] = len(rows)
    out['theta'] = rows[0]['theta']
    return out


def fig_fit(val='data/mf_validate'):
    v = _load_val(val)
    d = v['full']
    tgt = F.target_hist()
    fig, ax = plt.subplots(1, 3, figsize=(8.2, 2.6), gridspec_kw=dict(wspace=0.38, width_ratios=[1, 1, 1.05]))
    for a, key, col, name, n_p in ((ax[0], 'on', C_ON, 'On', 3890), (ax[1], 'off', C_OFF, 'Off', 6174)):
        a.bar(F.EDGES[:-1], tgt[key][:-1] / 20, width=18, align='edge', color=col, alpha=0.35, lw=0,
              label=f'Panichello 2024 (n = {n_p:,})')
        dur = d[f'{key}_dur']
        h = F.model_hist(dur)[:-1] / 20
        a.step(np.r_[F.EDGES[:-1], F.EDGES[-1]], np.r_[h, h[-1]], where='post', color=INK, lw=1.6,
               label=f'model (n = {len(dur):,})')
        a.set_xlim(0, 620); a.set_xlabel('state length (ms)'); a.set_ylabel('probability / ms')
        H = F.hellinger2(F.model_hist(dur), tgt[key])
        a.set_title(f'{name} states')
        a.text(0.97, 0.55, f'mean {dur.mean():.0f} ms (paper {F.TARGET[key + "_mean"]:.0f})\n'
                           f'median {np.median(dur):.0f} ms (paper {F.TARGET[key + "_median"]:.0f})\nHellinger² {H:.3f}',
               transform=a.transAxes, ha='right', va='top', fontsize=6.5, color=INK2)
        a.legend(fontsize=6.5, loc='upper right')
    a = ax[2]; a.axis('off')
    rows = [('', 'model', 'paper'),
            ('On per trial', f'{d["n_on"].mean():.2f}', '2.35'),
            ('Off per trial', f'{d["n_off"].mean():.2f}', '3.74'),
            ('decoding 0.5–1.4 s', f'{d["acc"]:.0%}', '~70%'),
            ('memory correct', f'{v["mem"]:.1%}', '≥95% req.'),
            ('trials', f'{len(d["n_on"]):,}', '1,653')]
    for i, r in enumerate(rows):
        for j, c in enumerate(r):
            a.text([0.0, 0.66, 0.9][j], 0.92 - 0.14 * i, c, fontsize=7, ha='left' if j == 0 else 'center',
                   color=INK if i else INK2, weight='bold' if i == 0 else 'normal', transform=a.transAxes)
    th = v['theta']
    a.text(0.0, -0.08, f'τ_N {th["tau_N"]:.0f} ms · dips {th["dip_rate"]:.1f}/s, mean {th["dip_mean"]:.0f} ms, '
                       f'k {th["dip_shape"]:.1f}, −{100 * th["dip_depth"]:.0f}%\nσ {th["sigma"]:.3f} · w_s {th["w_s"]:.2f} · '
                       f'f_sel {th["f_sel"]:.3f} · tune {th["tune_sd"]:.3f}', fontsize=6.3, color=INK2, transform=a.transAxes, va='top')
    fig.suptitle(f'Best fit, full pipeline (logistic decoder): {v["n_sessions"]} sessions × 132 trials × 329 units',
                 fontsize=8, x=0.07, ha='left', y=1.04)
    f = os.path.join(OUT, 'fig3_fit_histograms.png')
    fig.savefig(f, dpi=170, bbox_inches='tight')
    print('wrote', f)


def fig_sensitivity(sens='data/mf_sens.json'):
    r = json.load(open(sens))
    names = list(r['sweeps'])
    lab = dict(tau_N='τ_N (ms)', w_s='w_s (NMDA strength)', dip_rate='dip rate (1/s)', dip_mean='dip mean (ms)',
               dip_shape='dip shape k', dip_depth='dip depth δ', sigma='noise σ', f_sel='memory-cell fraction',
               tune_sd='background tuning SD')
    fig, ax = plt.subplots(2, len(names), figsize=(1.25 * len(names) + 0.6, 3.6), sharey='row',
                           gridspec_kw=dict(hspace=0.35, wspace=0.12))
    for j, n in enumerate(names):
        rows = r['sweeps'][n]
        x = [q['value'] for q in rows]
        a = ax[0, j]
        a.plot(x, [q['on_mean'] for q in rows], '-o', color=C_ON, ms=2.5, lw=1.6, label='On')
        a.plot(x, [q['off_mean'] for q in rows], '-o', color=C_OFF, ms=2.5, lw=1.6, label='Off')
        a.axvline(r['theta'][n], color=MUTED, lw=1, ls=':')
        a.set_title(lab[n], fontsize=7, loc='center', weight='normal')
        b = ax[1, j]
        b.plot(x, [q['n_on'] for q in rows], '-o', color=C_ON, ms=2.5, lw=1.6)
        b.plot(x, [q['n_off'] for q in rows], '-o', color=C_OFF, ms=2.5, lw=1.6)
        b.plot(x, [4 * q['mem'] for q in rows], '--', color=INK2, lw=1)
        b.axvline(r['theta'][n], color=MUTED, lw=1, ls=':')
        b.tick_params(axis='x', labelsize=6)
        a.tick_params(axis='x', labelsize=6)
    ax[0, 0].set_ylabel('mean length (ms)'); ax[1, 0].set_ylabel('states per trial\n(dashed: 4 × memory)')
    ax[0, 0].legend(fontsize=6.5, loc='upper left')
    fig.suptitle('One-at-a-time sweeps around the best fit (dotted line); fast pipeline', fontsize=8, x=0.05, ha='left', y=0.99)
    f = os.path.join(OUT, 'fig4_sensitivity.png')
    fig.savefig(f, dpi=170, bbox_inches='tight')
    print('wrote', f)


if __name__ == '__main__':
    {'mechanism': fig_mechanism, 'surrogate': fig_surrogate, 'fit': fig_fit, 'sensitivity': fig_sensitivity}[sys.argv[1]](*sys.argv[2:])
