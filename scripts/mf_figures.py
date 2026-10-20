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


# user's analysis of the recordings (all units vs top-k units by decoder weight)
DATA_TOPK = [dict(label='all', acc=0.676, n_on=1.8, on=185, n_off=3.1, off=155),
             dict(label='top 50', acc=0.720, n_on=2.5, on=210, n_off=2.8, off=130),
             dict(label='top 20', acc=0.721, n_on=2.6, on=219, n_off=2.75, off=122),
             dict(label='top 10', acc=0.710, n_on=2.7, on=220, n_off=2.8, off=117)]


def fig_readout(best='data/mf_fit_rates/run0/best.json', out='data/mf_readout_curve.json'):
    """Fixed dynamics, read-out gain varied: state statistics vs decoding accuracy."""
    from meanfield import rate_states as RS
    th = json.load(open(best))['theta']
    if os.path.exists(out) and json.load(open(out))['theta'] == th:
        rows = json.load(open(out))['rows']
    else:
        rows = []
        for lc in np.linspace(-2.5, 1.5, 17):
            ss = [RS.evaluate(dict(th, log10_c=float(lc)), seed=sd) for sd in (11, 12)]
            rows.append(dict(c=float(10 ** lc), **{k: float(np.mean([x[k] for x in ss]))
                                                   for k in ('acc', 'n_on', 'n_off', 'on_mean', 'off_mean')}))
        json.dump(dict(theta=th, rows=rows), open(out, 'w'))
    c_fit = 10 ** th['log10_c']
    fig, ax = plt.subplots(1, 4, figsize=(7.6, 2.0), gridspec_kw=dict(wspace=0.5))
    items = (('n_on', 'n_on', 'On per trial', C_ON, 2.35), ('on_mean', 'on', 'On length (ms)', C_ON, 192.4),
             ('n_off', 'n_off', 'Off per trial', C_OFF, 3.74), ('off_mean', 'off', 'Off length (ms)', C_OFF, 146.1))
    acc = np.array([r['acc'] for r in rows])
    for a, (km, kd, lab, col, paper) in zip(ax, items):
        y = np.array([r[km] for r in rows])
        a.plot(acc, y, '-', color=col, lw=2, zorder=2)
        a.plot(acc, y, 'o', color=col, ms=2.5, zorder=2)
        i = int(np.argmin(np.abs(np.log([r['c'] for r in rows]) - np.log(c_fit))))
        a.plot(acc[i], y[i], 'o', ms=7, mfc='none', mec=INK, mew=1.2, zorder=3)
        a.annotate('perfect\nread-out', (acc[-1], y[-1]), xytext=(-4, 0), textcoords='offset points', ha='right',
                   va='center', fontsize=6, color=INK2)
        a.plot(0.70, paper, 'D', color=MUTED, ms=4.5, zorder=3)
        for d in DATA_TOPK:
            a.plot(d['acc'], d[kd], 's', ms=4, color=INK if d['label'] == 'all' else '#fcfcfb', mec=INK, mew=1, zorder=4)
        a.set_title(lab, fontsize=8); a.set_xlabel('decoding accuracy'); a.set_xlim(0.55, 1.0)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK2, lw=2, marker='o', ms=2.5, label='model, read-out gain varied'),
         Line2D([], [], ls='', marker='o', ms=7, mfc='none', mec=INK, label='fitted gain'),
         Line2D([], [], ls='', marker='s', ms=4, color=INK, label='data, all units'),
         Line2D([], [], ls='', marker='s', ms=4, mfc='#fcfcfb', mec=INK, label='data, top 50/20/10'),
         Line2D([], [], ls='', marker='D', ms=4.5, color=MUTED, label='paper (Fig. 4)')]
    fig.legend(handles=h, loc='lower center', ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, -0.2))
    f = os.path.join(OUT, 'fig5_readout_curve.png')
    fig.savefig(f, dpi=170, bbox_inches='tight')
    print('wrote', f)
    for r in rows:
        print({k: round(v, 3) for k, v in r.items()})


def fig_dynamics_fits(free='data/mf_fit_rates', silence='data/mf_fit_silence', out='data/mf_dyn_fresh.json'):
    """Best dynamics-only fits (free read-out gain; read-out-free) vs the paper, on fresh seeds."""
    import glob
    from meanfield import rate_states as RS
    best = {}
    for name, d in (('free', free), ('silence', silence)):
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, 'run*', 'best.json')))]
        best[name] = dict(runs=runs, best=min(runs, key=lambda r: r['loss']))
    res = {}
    for name, ev in (('free', RS.evaluate), ('silence', RS.evaluate_silence)):
        th = best[name]['best']['theta']
        s = ev(th, n_trials=2112, seed=99)
        res[name] = dict(theta=th, on_dur=s['on_dur'].tolist(), off_dur=s['off_dur'].tolist(),
                         **{k: s[k] for k in ('n_on', 'n_off', 'on_mean', 'off_mean', 'on_median', 'off_median', 'acc', 'mem')},
                         loss=F.loss(s, keep_terms=F.TERMS_STATES),
                         spread={k: [r['theta'][k] for r in best[name]['runs']] for k in th},
                         run_loss=[r['loss'] for r in best[name]['runs']])
    json.dump(res, open(out, 'w'), indent=1)
    tgt = F.target_hist()
    fig, ax = plt.subplots(2, 2, figsize=(6.4, 4.4), gridspec_kw=dict(hspace=0.75, wspace=0.3))
    titles = dict(free='Fit 1: dynamics + read-out gain (gain fitted)', silence='Fit 2: read-out-free (network states)')
    for i, name in enumerate(('free', 'silence')):
        r = res[name]
        for j, (key, col, lab) in enumerate((('on', C_ON, 'On'), ('off', C_OFF, 'Off'))):
            a = ax[i, j]
            a.bar(F.EDGES[:-1], tgt[key][:-1] / 20, width=18, align='edge', color=col, alpha=0.35, lw=0, label='paper')
            dur = np.array(r[f'{key}_dur'])
            h = F.model_hist(dur)[:-1] / 20
            a.step(np.r_[F.EDGES[:-1], F.EDGES[-1]], np.r_[h, h[-1]], where='post', color=INK, lw=1.6, label='model')
            a.set_xlim(0, 620)
            a.text(0.97, 0.95, f'{r["n_" + key]:.2f}/trial (paper {F.TARGET["n_" + key]:.2f})\n'
                               f'mean {r[key + "_mean"]:.0f} ms (paper {F.TARGET[key + "_mean"]:.0f})',
                   transform=a.transAxes, ha='right', va='top', fontsize=6.5, color=INK2)
            a.set_title(f'{lab} states', fontsize=8, loc='left')
            if i == 1:
                a.set_xlabel('state length (ms)')
            if j == 0:
                a.set_ylabel('probability / ms')
        th = r['theta']
        g = f' · gain {10 ** th["log10_c"]:.3f} z/Hz' if 'log10_c' in th else ''
        ax[i, 0].text(0, 1.32, f'{titles[name]} — loss {r["loss"]:.1f}, memory {r["mem"]:.1%} (fresh seeds)',
                      transform=ax[i, 0].transAxes, fontsize=7.5, weight='bold', color=INK)
        ax[i, 0].text(0, 1.18, f'τ_N {th["tau_N"]:.0f} ms · w_s {th["w_s"]:.2f} · dips {th["dip_rate"]:.1f}/s, mean '
                               f'{th["dip_mean"]:.0f} ms, k {th["dip_shape"]:.1f}, −{100 * th["dip_depth"]:.0f}% · σ {th["sigma"]:.3f}{g}',
                      transform=ax[i, 0].transAxes, fontsize=6.5, color=INK2)
    ax[0, 1].legend(fontsize=6.5, loc='center right')
    f = os.path.join(OUT, 'fig6_dynamics_fits.png')
    fig.savefig(f, dpi=170, bbox_inches='tight')
    print('wrote', f)
    for name in res:
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name].items() if k not in ('on_dur', 'off_dur', 'theta')})


def _gain_curve(th, out, n_trials=1056):
    """Fixed dynamics, gain varied; states counted 0-1,400 ms after cue onset (as in the data analysis)."""
    from meanfield import rate_states as RS
    th = {k: v for k, v in th.items() if k != 'log10_c'}
    if os.path.exists(out) and json.load(open(out))['theta'] == th:
        return json.load(open(out))['rows']
    p = M.params(**th); p['t_delay'] = 1600.0
    rows = []
    for lc in np.linspace(-2.5, 1.5, 17):
        acc = []
        stats = []
        for sd in (11, 12):
            sim = M.simulate(p, np.tile([0, 1], n_trials // 2), seed=sd)
            st, z, state, tc, thr = RS.states(sim, 10 ** lc, np.random.default_rng(sd + 1), delay=(1350., 1350.))
            dm = (tc >= sim['t_cue'] + 500) & (tc < sim['t_cue'] + 1400)
            stats.append(dict(acc=float((z[:, dm] > 0).mean()), n_on=float(st['n_on'].mean()), n_off=float(st['n_off'].mean()),
                              on_mean=float(st['on_dur'].mean()) if len(st['on_dur']) else np.nan,
                              off_mean=float(st['off_dur'].mean()) if len(st['off_dur']) else np.nan,
                              frac_on=float((state[:, dm] == 1).mean()), frac_off=float((state[:, dm] == -1).mean())))
        rows.append(dict(c=float(10 ** lc), **{k: float(np.nanmean([x[k] for x in stats])) for k in stats[0]}))
    json.dump(dict(theta=th, rows=rows), open(out, 'w'))
    return rows


def fig_readout_data(data='meanfield/data/readout_summary_data.csv', dyn='data/mf_dyn_fresh.json', lam_label='30',
                     out='fig7_readout_data.png'):
    import csv
    D = list(csv.DictReader(open(data)))
    res = json.load(open(dyn))
    curves = dict(free=_gain_curve(res['free']['theta'], 'data/mf_curve_free.json'),
                  silence=_gain_curve(res['silence']['theta'], 'data/mf_curve_silence.json'))
    c_fit = 10 ** res['free']['theta']['log10_c']
    items = (('off_per_trial', 'n_off', 'Off per trial', C_OFF), ('off_mean_ms', 'off_mean', 'Off mean (ms)', C_OFF),
             ('frac_off', 'frac_off', 'fraction of delay Off', C_OFF),
             ('on_per_trial', 'n_on', 'On per trial', C_ON), ('on_mean_ms', 'on_mean', 'On mean (ms)', C_ON),
             ('frac_on', 'frac_on', 'fraction of delay On', C_ON))
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.3), gridspec_kw=dict(wspace=0.42, hspace=0.55))
    for a, (kd, km, lab, col) in zip(axs.ravel(), items):
        for name, ls, lw in (('free', '-', 2.0), ('silence', '--', 1.4)):
            rows = curves[name]
            acc = np.array([r['acc'] for r in rows]); y = np.array([r[km] for r in rows])
            a.plot(acc, y, ls, color=col, lw=lw, zorder=2)
            if name == 'free':
                i = int(np.argmin(np.abs(np.log([r['c'] for r in rows]) - np.log(c_fit))))
                a.plot(acc[i], y[i], 'o', ms=6, mfc='none', mec=INK, mew=1.1, zorder=3)
        for st, mk, fc in (('random', 'o', MUTED), ('top', 's', '#fcfcfb')):
            rr = [r for r in D if r['set'] == st]
            x = [float(r['accuracy']) for r in rr]; y = [float(r[kd]) for r in rr]
            e = [float(r[kd + '_sem']) for r in rr]
            a.plot(x, y, '-', color=INK2, lw=0.8, zorder=3)
            a.errorbar(x, y, yerr=e, fmt=mk, ms=4, mfc=fc, mec=INK, ecolor=INK2, elinewidth=0.8, capsize=0, zorder=4)
            for r, xx, yy in zip(rr, x, y):
                if r['k'] in ('5', '200'):
                    a.annotate(r['k'], (xx, yy), xytext=(3, 3), textcoords='offset points', fontsize=5.5, color=INK2)
        al = [r for r in D if r['set'] == 'all'][0]
        a.errorbar(float(al['accuracy']), float(al[kd]), yerr=float(al[kd + '_sem']), fmt='D', ms=5, color=INK, zorder=5)
        a.set_title(lab, fontsize=8); a.set_xlim(0.5, 1.0)
        if a in axs[1]:
            a.set_xlabel('decoding accuracy (500–1,400 ms)')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK2, lw=2, label='model fit 1 dynamics, gain varied'),
         Line2D([], [], color=INK2, lw=1.4, ls='--', label='model fit 2 dynamics, gain varied'),
         Line2D([], [], ls='', marker='o', ms=6, mfc='none', mec=INK, label='fit 1 gain'),
         Line2D([], [], ls='', marker='D', ms=5, color=INK, label='data, all units'),
         Line2D([], [], ls='', marker='s', ms=4, mfc='#fcfcfb', mec=INK, label='data, top k'),
         Line2D([], [], ls='', marker='o', ms=4, color=MUTED, mec=INK, label='data, random k')]
    fig.legend(handles=h, loc='lower center', ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(f'States vs read-out: data (25 sessions, λ = {lam_label}, ± session SEM; k = 5…200) and model (gain varied); '
                 'states counted 0–1,400 ms', fontsize=7.5, x=0.06, ha='left', y=0.99)
    f = os.path.join(OUT, out)
    fig.savefig(f, dpi=170, bbox_inches='tight')
    print('wrote', f)


def fig_spiking(dec='data/sp_start_decode', label='spiking starting point', out='fig8_spiking_start.png', mem=None):
    """Spiking run vs data (lambda = 30): Off and On length histograms for all units and top 20, plus a table."""
    import csv
    import glob
    ses = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(dec, 'sp_*.json')))]
    H = {}
    for r in csv.DictReader(open('meanfield/data/readout_histograms_data.csv')):
        H.setdefault((r['set'], r['k'], r['state']), []).append(float(r['count']))
    S = json.load(open(os.path.join(dec, 'summary.json')))
    D = {(r['set'], r['k']): r for r in csv.DictReader(open('meanfield/data/readout_summary_data.csv'))}
    edges = np.arange(0, 620, 20)
    fig, ax = plt.subplots(2, 3, figsize=(7.6, 4.2), gridspec_kw=dict(wspace=0.35, hspace=0.6, width_ratios=[1, 1, 0.95]))
    for i, (st, dk, name) in enumerate((('all', ('all', 'session count'), 'all units (329)'), ('top', ('top', '20'), 'top 20 units'))):
        for j, (state, col) in enumerate((('off', C_OFF), ('on', C_ON))):
            a = ax[i, j]
            hd = np.array(H[(dk[0], dk[1], state)]); hd = hd[:-1] / hd.sum()
            a.bar(edges[:-1], hd / 20, width=18, align='edge', color=col, alpha=0.35, lw=0, label='data')
            dur = np.concatenate([[d for dr in [x for x in s['results'] if x['set'] == st and x['lam'] == 30.0][0]['draws']
                                   for d in dr[state + '_dur']] for s in ses])
            h, _ = np.histogram(np.minimum(dur, 10000), np.r_[edges, np.inf]); h = h[:-1] / max(1, len(dur))
            a.step(edges, np.r_[h, h[-1]] / 20, where='post', color=INK, lw=1.5, label='spiking')
            a.set_xlim(0, 620); a.set_title(f'{state.capitalize()} — {name}', fontsize=8)
            if i == 1:
                a.set_xlabel('state length (ms)')
        ax[i, 0].set_ylabel('probability / ms')
        ax[0, 0].legend(fontsize=6.5)
        a = ax[i, 2]; a.axis('off')
        v = S[f'{st}|{20 if st == "top" else 329}|30.0']; d = D[dk]
        rows = [('', 'spiking', 'data'), ('accuracy', f'{v["accuracy"][0]:.2f}', f'{float(d["accuracy"]):.2f}'),
                ('Off / trial', f'{v["off_per_trial"][0]:.2f}', f'{float(d["off_per_trial"]):.2f}'),
                ('Off mean ms', f'{v["off_mean_ms"][0]:.0f}', f'{float(d["off_mean_ms"]):.0f}'),
                ('frac Off', f'{v["frac_off"][0]:.2f}', f'{float(d["frac_off"]):.2f}'),
                ('On / trial', f'{v["on_per_trial"][0]:.2f}', f'{float(d["on_per_trial"]):.2f}'),
                ('On mean ms', f'{v["on_mean_ms"][0]:.0f}', f'{float(d["on_mean_ms"]):.0f}')]
        for r_, row in enumerate(rows):
            for c_, txt in enumerate(row):
                a.text([0.0, 0.62, 0.88][c_], 0.95 - 0.13 * r_, txt, transform=a.transAxes, fontsize=7,
                       ha='left' if c_ == 0 else 'center', color=INK if r_ else INK2, weight='bold' if r_ == 0 else 'normal')
    title = f'{label}: 4 sessions × 132 trials, λ = 30, states 0–1,400 ms'
    if mem is not None:
        title += f'; memory correct {mem:.0%} (required ≥ 95%)'
    fig.suptitle(title, fontsize=8, x=0.06, ha='left', y=0.99)
    f = os.path.join(OUT, out)
    fig.savefig(f, dpi=170, bbox_inches='tight')
    print('wrote', f)


def _runs(mask):
    a = np.r_[False, mask, False].astype(int)
    return list(zip(np.where(np.diff(a) == 1)[0], np.where(np.diff(a) == -1)[0]))


def fig_spiking_decoding(tr='data/sp_start_traces', label='Spiking starting point', out='fig9_spiking_decoding.png',
                         session=0, n_show=12):
    """Trial-averaged decoding through the delay and consecutive single trials with On/Off states."""
    import glob
    S = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(tr, '*_traces.npz')))]
    t = S[0]['t_rel']
    fig = plt.figure(figsize=(7.6, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.8, 3.4], hspace=0.5, wspace=0.12)
    # (a) accuracy over time, mean +- SEM over sessions
    a = fig.add_subplot(gs[0, :])
    for key, col, lab in (('all', INK, 'all units (329)'), ('top', C_ON, 'top 20 units')):
        acc = np.array([(x[key + '_conf'] > 0.5).mean(0) for x in S])
        m, e = acc.mean(0), acc.std(0) / np.sqrt(len(S))
        a.fill_between(t, m - e, m + e, color=col, alpha=0.2, lw=0)
        a.plot(t, m, color=col, lw=1.4, label=lab)
    a.axhline(0.5, color=INK2, ls=':', lw=0.8); a.axvspan(0, 50, color='#e4e3df', lw=0)
    a.axvspan(500, 1400, ymin=0, ymax=0.04, color=MUTED, lw=0)
    a.set_xlim(t[0], t[-1]); a.set_ylim(0.3, 1.02); a.set_ylabel('decoding accuracy')
    a.legend(fontsize=7, loc='lower right', ncol=2)
    n_tr = sum(len(x['y']) for x in S)
    a.set_title(f'{label}: decoding across all {n_tr} trials ({len(S)} sessions; λ = 30; grey = cue; '
                'bar = 500–1,400 ms)', fontsize=8.5)
    # (b) fraction of trials On / Off over time (all units)
    a = fig.add_subplot(gs[1, :])
    for lab_, col, name in ((1, C_ON, 'On'), (-1, C_OFF, 'Off')):
        fr = np.array([(x['all_state'] == lab_).mean(0) for x in S])
        a.plot(t, fr.mean(0), color=col, lw=1.4, label=name)
    a.axvspan(0, 50, color='#e4e3df', lw=0); a.set_xlim(t[0], t[-1]); a.set_ylim(0, 1)
    a.set_ylabel('fraction of trials'); a.legend(fontsize=7, loc='upper right', ncol=2)
    a.set_title('Fraction of trials in each state (all units)', fontsize=8.5)
    # (c) consecutive single trials
    x = S[session]
    r = x['rates']
    for col_i, (key, name) in enumerate((('all', 'all units (329)'), ('top', 'top 20 units'))):
        a = fig.add_subplot(gs[2, col_i])
        for j in range(n_show):
            base = (n_show - 1 - j) * 1.25
            st = x[key + '_state'][j]
            for lab_, c in ((1, C_ON), (-1, C_OFF)):
                for i0, i1 in _runs(st == lab_):
                    a.fill_between([t[i0] - 5, t[i1 - 1] + 5], base, base + 1, color=c, alpha=0.35, lw=0)
            a.plot(t, base + x[key + '_conf'][j], color=INK, lw=0.7)
            a.plot([t[0], t[-1]], [base + 0.5] * 2, color=INK2, ls=':', lw=0.5)
            if col_i == 1:
                cA = 0 if x['y'][j] == 1 else 1
                ok = r[j, -300:, cA].mean() > r[j, -300:, 1 - cA].mean()
                a.text(t[-1] + 20, base + 0.5, f'trial {j + 1}' + ('' if ok else '\nmemory lost'), fontsize=6.5,
                       va='center', color=INK2 if ok else C_ON)
        a.axvline(0, color=INK2, lw=0.5); a.axvline(50, color=INK2, lw=0.5)
        a.set_xlim(t[0], t[-1]); a.set_yticks([]); a.set_xlabel('time from cue onset (ms)')
        a.set_title(f'{name}: first {n_show} trials of session {session + 1} (not selected)', fontsize=8)
    fig.text(0.06, 0.035, 'Decoder confidence in the correct cue (0–1 per row, dotted = 0.5); shading: orange = On, '
             'blue = Off. "memory lost": uncued pool ahead at the end of the delay.', fontsize=7, color=INK2)
    f = os.path.join(OUT, out)
    fig.savefig(f, dpi=170, bbox_inches='tight')
    print('wrote', f)


if __name__ == '__main__':
    {'mechanism': fig_mechanism, 'surrogate': fig_surrogate, 'fit': fig_fit, 'sensitivity': fig_sensitivity, 'readout': fig_readout, 'dynamics': fig_dynamics_fits, 'readout_data': fig_readout_data, 'spiking': fig_spiking, 'spiking_decoding': fig_spiking_decoding}[sys.argv[1]](*sys.argv[2:])
