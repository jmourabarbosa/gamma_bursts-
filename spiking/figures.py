"""Figures of the spiking ring network against the data (PNG + SVG in spiking/figures/).

    python -m spiking.figures            # after run.py sessions / nodips / single_dip / offtime

fig_example_trial     raster of the E population sorted by preferred angle, the common input, decoded z (all units)
                      with On/Off shading
fig_unit_sets         On/Off statistics of the 7 decoded populations, model (with dips, and the no-dip control) vs data
fig_histograms        On and Off length histograms (all units) vs data
fig_single_dip        memory held against the length of a single dip (cut-off)
fig_memory_off_time   memory held against time Off in 500-1,400 ms (all-units read-out), as the paper's Figure 3
fig_populations       decoder z of all units, top 5 and random 5 on the same trials
fig_tuning            delay tuning of the recorded cells (why top-unit decoders are not better than random ones)
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from paper_model import readout as PR  # noqa: E402
from . import network as N, readout as R, run as RU  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), 'figures')
RES = RU.OUT
ORDER = ('top 5', 'top 20', 'top 50', 'rand 5', 'rand 20', 'rand 50', 'all')
C_ON, C_OFF, INK, GREY, LIGHT, C_CTRL = '#e8474a', '#3cb4c9', '#000000', '#7f7f7f', '#d9d9d9', '#9467bd'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'],
                     'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'lines.linewidth': 0.9, 'legend.frameon': False,
                     'legend.fontsize': 7, 'svg.fonttype': 'none', 'pdf.fonttype': 42})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300)
    fig.savefig(os.path.join(OUT, name + '.svg'))
    plt.close(fig)


def letter(ax, txt, x=-0.25, y=1.04):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=10, weight='bold', va='bottom')


def load_sessions(tag):
    f = os.path.join(RES, f'sessions_{tag}.json')
    if not os.path.exists(f):
        return None
    rows = json.load(open(f))['rows']
    return {lab: {k: np.array([r['sets'][lab][k] for r in rows], float) for k in rows[0]['sets'][lab]
                  if not k.startswith('h_')} | {k: np.sum([r['sets'][lab][k] for r in rows], 0)
                                               for k in rows[0]['sets'][lab] if k.startswith('h_')}
            for lab in rows[0]['sets']}, rows


def fig_unit_sets(model, ctrl, data):
    panels = (('off_per_trial', 'n_off', 'Off states per trial', C_OFF, None),
              ('off_median_ms', 'off_median', 'Off length, median (ms)', C_OFF, 300),
              ('frac_off', 'frac_off', 'Fraction of delay Off', C_OFF, 1),
              ('on_per_trial', 'n_on', 'On states per trial', C_ON, None),
              ('on_median_ms', 'on_median', 'On length, median (ms)', C_ON, 300),
              ('frac_on', 'frac_on', 'Fraction of delay On', C_ON, 1))
    x = np.arange(len(ORDER))
    fig, axs = plt.subplots(2, 3, figsize=(6.6, 4.2), gridspec_kw=dict(hspace=0.75, wspace=0.6, left=0.1, right=0.98,
                                                                       top=0.93, bottom=0.13))
    for k, (a, (dk, mk, name, col, top)) in enumerate(zip(axs.ravel(), panels)):
        v = [data[lab][dk] for lab in ORDER]
        a.errorbar(x - 0.2, [s.mean() for s in v], yerr=[2 * s.std(ddof=1) / np.sqrt(len(s)) for s in v], fmt='o',
                   ms=3, color=INK, elinewidth=0.6, capsize=0, label='Data')
        m = [model[lab][mk] for lab in ORDER]
        a.errorbar(x + 0.05, [np.nanmean(s) for s in m], yerr=[2 * np.nanstd(s, ddof=1) / np.sqrt(len(s)) for s in m],
                   fmt='o', ms=3, color=col, elinewidth=0.6, capsize=0, label='Model')
        if ctrl is not None:
            c = [ctrl[lab][mk] for lab in ORDER]
            a.plot(x + 0.28, [np.nanmean(s) for s in c], 'x', ms=3.5, color=C_CTRL, mew=0.8, label='No dips')
        a.axvline(2.5, color=GREY, ls=(0, (3, 2)), lw=0.5); a.axvline(5.5, color=GREY, ls=(0, (3, 2)), lw=0.5)
        a.set_xlim(-0.6, len(ORDER) - 0.4)
        a.set_xticks(x); a.set_xticklabels(ORDER, rotation=60, ha='right', rotation_mode='anchor')
        a.set_ylim(0, top or np.ceil(a.get_ylim()[1]))
        a.set_ylabel(name)
        letter(a, 'abcdef'[k], x=-0.45)
        if k in (0, 3):
            a.legend(loc='upper right' if k == 3 else 'lower right', handletextpad=0.1, borderaxespad=0.1)
    save(fig, 'fig_unit_sets')


def fig_histograms(model, data_hist, ctrl=None):
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.0), gridspec_kw=dict(wspace=0.4, left=0.1, right=0.98, bottom=0.22, top=0.88))
    e = np.arange(31) * 20
    for a, st, col in ((axs[0], 'off', C_OFF), (axs[1], 'on', C_ON)):
        d = data_hist['all', st]
        m = model['all'][f'h_{st}'].astype(float)
        a.bar(e[:30] + 10, d[:30] / d.sum(), width=20, color=LIGHT, edgecolor='none', label='Data')
        a.step(e[:30] + 10, m[:30] / m.sum(), where='mid', color=col, lw=1.1, label='Model')
        if ctrl is not None:
            c = ctrl['all'][f'h_{st}'].astype(float)
            if c.sum() > 0:
                a.step(e[:30] + 10, c[:30] / c.sum(), where='mid', color=C_CTRL, lw=0.7, ls='--', label=f'No dips ({int(c.sum())})')
        a.set_xlim(0, 600)
        a.set_xlabel(f'{st.capitalize()} length (ms)')
        a.set_ylabel('Fraction of states')
        a.text(0.98, 0.98, f'>600 ms: data {d[30] / d.sum():.2f}, model {m[30] / m.sum():.2f}', transform=a.transAxes,
               ha='right', va='top', fontsize=6, color=GREY)
        a.legend(loc='center right')
    letter(axs[0], 'a'); letter(axs[1], 'b')
    save(fig, 'fig_histograms')


def fig_single_dip():
    f = os.path.join(RES, 'single_dip.json')
    if not os.path.exists(f):
        return
    r = json.load(open(f))
    L = np.array(r['lengths'])
    fig, a = plt.subplots(figsize=(3.0, 2.2), gridspec_kw=dict(left=0.17, right=0.97, bottom=0.2, top=0.93))
    a.plot(L, r['held'], 'o-', color=INK, ms=3, label='Held (end of delay)')
    a.plot(L, r['held_probe'], 's--', color=GREY, ms=2.5, lw=0.7, label='Bump restarts at cue (probe)')
    a.plot(L, r['wrong_probe'], '^:', color=C_ON, ms=2.5, lw=0.7, label='Restarts elsewhere (probe)')
    a.axvline(2.5 * N.P['tau_N'], color=C_OFF, lw=0.6, ls=(0, (3, 2)))
    a.text(2.5 * N.P['tau_N'] + 8, 0.5, r'2.5 $\tau_N$', color=C_OFF, fontsize=7)
    a.set_xlabel('Length of the single dip (ms)'); a.set_ylabel('Fraction of trials')
    a.set_ylim(-0.03, 1.03)
    a.legend(loc='center left', fontsize=6)
    save(fig, 'fig_single_dip')


def fig_memory_off_time():
    f = os.path.join(RES, 'sessions_dips_trials.npz')
    if not os.path.exists(f):
        return
    d = np.load(f)
    ks = sorted(k[:-len('_off_time_all_trials')] for k in d if k.endswith('_off_time_all_trials'))
    off = np.concatenate([d[k + '_off_time_all_trials'] for k in ks])
    held = np.concatenate([d[k + '_held'] for k in ks])
    edges = np.linspace(0, 0.91, 11)
    b = np.clip(np.digitize(off, edges) - 1, 0, 9)
    cen = (edges[:-1] + edges[1:]) / 2
    n = np.bincount(b, minlength=10)
    k = np.bincount(b, weights=held, minlength=10)
    p = k / np.maximum(n, 1)
    se = np.sqrt(p * (1 - p) / np.maximum(n, 1))
    ok = n >= 10
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.1), gridspec_kw=dict(wspace=0.45, left=0.11, right=0.98, bottom=0.22, top=0.9))
    a = axs[0]
    a.errorbar(cen[ok], p[ok], yerr=2 * se[ok], fmt='o-', ms=3, color=INK, elinewidth=0.6, capsize=0, label='Model')
    # data: fraction correct per bin of time Off (behaviour_per_trial_correct_and_error.csv)
    import csv
    rows = list(csv.DictReader(open(os.path.join(PR.DATA, 'behaviour_per_trial_correct_and_error.csv'))))
    fo = np.array([float(r['frac_off']) for r in rows]) * 0.91
    cor = np.array([float(r['correct']) for r in rows])
    bd = np.clip(np.digitize(fo, edges) - 1, 0, 9)
    nd = np.bincount(bd, minlength=10); pd = np.bincount(bd, weights=cor, minlength=10) / np.maximum(nd, 1)
    a.plot(cen[nd >= 10], pd[nd >= 10], 's-', ms=2.5, color=GREY, lw=0.7, label='Data (fraction correct)')
    a.set_xlabel('Time Off in 500-1,400 ms (s)'); a.set_ylabel('Memory held / correct')
    a.set_ylim(0.0, 1.02); a.legend(loc='lower left', fontsize=6)
    a = axs[1]
    bins = np.linspace(0, 0.91, 14)
    for msk, c, lab in ((held, GREY, f'held ({held.sum()})'), (~held, C_ON, f'lost ({(~held).sum()})')):
        if msk.sum() >= 10:
            a.hist(off[msk], bins, density=True, histtype='step', color=c, label=lab)
    a.set_xlabel('Time Off in 500-1,400 ms (s)'); a.set_ylabel('Density')
    a.legend(fontsize=6)
    letter(axs[0], 'a'); letter(axs[1], 'b')
    save(fig, 'fig_memory_off_time')


def fig_example_trial():
    f = os.path.join(RES, 'example_trials.npz')
    if not os.path.exists(f):
        return
    d = np.load(f)
    k = int(d['pick'])
    fig, axs = plt.subplots(3, 1, figsize=(5.2, 4.4), sharex=True,
                            gridspec_kw=dict(height_ratios=[3, 0.8, 1.4], hspace=0.12, left=0.13, right=0.97, top=0.96, bottom=0.1))
    r = d['raster'][k]
    t, i = np.nonzero(r)
    t = d['t1'][t]
    axs[0].plot(t, i * 360 / r.shape[1], ',', color=INK, alpha=0.6, rasterized=True)
    axs[0].axhline(float(d['cue'][k]) * 45, color=C_ON, lw=0.5, ls=':')
    axs[0].set_ylabel('E cells, preferred angle (deg)'); axs[0].set_yticks([0, 90, 180, 270, 360])
    axs[0].axvspan(0, 50, color=LIGHT, lw=0, zorder=0)
    axs[1].plot(d['t1'], d['m'][k], color=GREY, lw=0.7, label='m(t)')
    axs[1].plot(d['t1'], d['m_eff'][k], color=INK, lw=0.8, label='drive to E cells (with kick)')
    axs[1].set_ylabel('Common\ninput'); axs[1].legend(loc='upper right', fontsize=6, ncol=2)
    z, st, tz = d['z'], d['state'], d['tz']
    a = axs[2]
    a.fill_between(tz, -3, 12, where=st == 1, color=C_ON, alpha=0.25, lw=0, step='mid')
    a.fill_between(tz, -3, 12, where=st == -1, color=C_OFF, alpha=0.3, lw=0, step='mid')
    a.plot(tz, z, color=INK, lw=0.8)
    a.axhline(1.96, color=GREY, lw=0.5, ls=':'); a.axhline(0.84, color=GREY, lw=0.5, ls=':')
    a.set_ylabel('Decoder z\n(all units)'); a.set_xlabel('Time from cue onset (ms)')
    a.set_xlim(-300, 1500); a.set_ylim(-3, max(8, np.nanmax(z) + 1))
    for ax, l in zip(axs, 'abc'):
        letter(ax, l, x=-0.12)
    save(fig, 'fig_example_trial')


def fig_populations(n=4):
    """Decoder z of three populations (all units, top 5, random 5) on the same trials, with the common input."""
    f = os.path.join(RES, 'sessions_dips_trials.npz')
    if not os.path.exists(f):
        return
    d = np.load(f)
    tz, held, cue = d['tz'], d['s0_held'], d['s0_cue']
    corr = np.where(held)[0]
    ks = [k for k in range(min(20, len(corr))) if corr[k] < len(d['s0_m'])][:n]
    t1 = np.arange(d['s0_m'].shape[1]) - RU.TUNED['t_pre']
    fig, axs = plt.subplots(len(ks), 1, figsize=(5.6, 1.35 * len(ks) + 0.4), sharex=True,
                            gridspec_kw=dict(hspace=0.25, left=0.11, right=0.97, top=0.95, bottom=0.1))
    for a, k in zip(axs, ks):
        i = corr[k]
        m = d['s0_m'][i]
        a.fill_between(t1, -3, 14, where=m < 0.75, color=LIGHT, lw=0, step='mid')
        for lab, col in (('all', INK), ('top 5', C_ON), ('rand 5', C_OFF)):
            a.plot(tz, d[f's0_z_{lab}'][k], color=col, lw=0.8, label=lab)
        a.axhline(1.96, color=GREY, lw=0.4, ls=':'); a.axhline(0.84, color=GREY, lw=0.4, ls=':')
        a.set_ylim(-3, 14); a.set_ylabel('z')
        a.text(0.01, 0.97, f'cue {cue[i] * 45} deg', transform=a.transAxes, fontsize=6, va='top')
    axs[0].legend(loc='upper right', ncol=3, fontsize=6)
    axs[-1].set_xlim(-100, 1450); axs[-1].set_xlabel('Time from cue onset (ms); grey: dips')
    save(fig, 'fig_populations')


def fig_tuning():
    """Delay rates of the recorded cells of session 0 against cue, and the ANOVA top cells of one pair."""
    f = os.path.join(RES, 'sessions_dips_trials.npz')
    if not os.path.exists(f):
        return
    d = np.load(f)
    C, held, cue, rec = d['s0_counts'], d['s0_held'], d['s0_cue'], d['s0_rec']
    X, tz = R.windows(C[held], np.arange(C.shape[1]) * 10.0 - RU.TUNED['t_pre'])
    cu = cue[held]
    dl = (tz >= 500) & (tz <= 1400)
    rate = X[:, dl].mean(1) * 10
    tun = np.array([rate[cu == c].mean(0) for c in range(8)])
    isE = rec < N.P['N_E']
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.2), gridspec_kw=dict(wspace=0.55, left=0.11, right=0.98, bottom=0.22, top=0.86))
    a = axs[0]
    o = np.argsort(rec[isE])
    im = a.imshow(tun[:, isE][:, o].T, aspect='auto', cmap='Greys', extent=(-22.5, 337.5, 360, 0), vmax=40)
    a.set_xlabel('Cue (deg)'); a.set_ylabel('Recorded E cell (pref. angle, deg)')
    a.set_xticks([0, 90, 180, 270]); plt.colorbar(im, ax=a, label='Delay rate (Hz)', fraction=0.05)
    a = axs[1]
    pk, bs = tun.max(0), tun.min(0)
    a.plot(bs[isE], pk[isE], '.', ms=2, color=INK, label='E')
    a.plot(bs[~isE], pk[~isE], '.', ms=2, color=C_ON, label='I')
    a.set_xlabel('Lowest cue rate (Hz)'); a.set_ylabel('Highest cue rate (Hz)'); a.legend(fontsize=6)
    a = axs[2]
    tot = X[:, (tz >= 0) & (tz <= 1400)].sum(1)
    m = cu % 4 == 0
    top = np.argsort(-R.anova_f(tot[m], cu[m]))[:5]
    for j in top:
        a.plot(np.arange(8) * 45, tun[:, j], 'o-', ms=2, lw=0.7)
    a.axvline(0, color=GREY, lw=0.5, ls=':'); a.axvline(180, color=GREY, lw=0.5, ls=':')
    a.set_xlabel('Cue (deg)'); a.set_ylabel('Delay rate (Hz)'); a.set_xticks([0, 90, 180, 270])
    a.set_title('Top 5 cells for the 0 vs 180 deg pair', fontsize=7)
    for a, l, x in zip(axs, 'abc', (-0.5, -0.3, -0.3)):
        letter(a, l, x=x)
    save(fig, 'fig_tuning')


def table(model, rows, ctrl, data):
    """Markdown table model (mean over sessions) vs data (mean over 25 sessions), written to results/table.md."""
    cols = (('Off/trial', 'n_off', 'off_per_trial', 2), ('Off median', 'off_median', 'off_median_ms', 0),
            ('frac Off', 'frac_off', 'frac_off', 2), ('On/trial', 'n_on', 'on_per_trial', 2),
            ('On median', 'on_median', 'on_median_ms', 0), ('frac On', 'frac_on', 'frac_on', 2), ('mu_on', 'mu_on', 'mu_on', 2))
    lines = ['| population | ' + ' | '.join(c[0] for c in cols) + ' |', '|---|' + '---|' * len(cols)]
    for lab in ORDER:
        lines.append(f'| {lab} | ' + ' | '.join(f'{np.nanmean(model[lab][mk]):.{d}f} ({data[lab][dk].mean():.{d}f})'
                                                 for _, mk, dk, d in cols) + ' |')
    if ctrl is not None:
        lines += ['', 'No-dip control (model):', '', lines[0], lines[1]]
        for lab in ORDER:
            lines.append(f'| {lab} | ' + ' | '.join(f'{np.nanmean(ctrl[lab][mk]):.{d}f}' for _, mk, _, d in cols) + ' |')
    mem = {k: np.mean([r[k] for r in rows]) for k in ('mem_held', 'mem_held_rate', 'mem_held_nmda', 'mem_held_probe',
                                                      'wrong_place', 'wrong_probe')}
    lines += ['', f'Sessions: {len(rows)} x {rows[0]["n_trials"]} trials. Memory held (end of delay, rate or NMDA): '
              f'{mem["mem_held"]:.3f}; rate bump active and at the cue: {mem["mem_held_rate"]:.3f}; NMDA criterion: '
              f'{mem["mem_held_nmda"]:.3f}; bump restarts at the cue in the probe: {mem["mem_held_probe"]:.3f}; '
              f'wrong place at the end of the delay: {mem["wrong_place"]:.3f}; restarts elsewhere in the probe: '
              f'{mem["wrong_probe"]:.3f}.']
    open(os.path.join(RES, 'table.md'), 'w').write('Model (data in brackets); lengths in ms.\n\n' + '\n'.join(lines) + '\n')


def main():
    ses, hist = PR.load_data()
    m = load_sessions('dips')
    c = load_sessions('nodips')
    ctrl = c[0] if c else None
    if m:
        fig_unit_sets(m[0], ctrl, ses)
        fig_histograms(m[0], hist, ctrl)
        table(m[0], m[1], ctrl, ses)
    fig_single_dip()
    fig_memory_off_time()
    fig_example_trial()
    fig_populations()
    fig_tuning()


if __name__ == '__main__':
    main()
