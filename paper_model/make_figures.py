"""Paper figures of the best-fit model.

Figure 1: A, an example trial: 1 s of delay with a constant common input, then dips (pool rates and NMDA gating;
          common input; all-units decoding z with On/Off labels). B: all-units Off and On length distributions
          (states 0-1,400 ms after cue onset), model vs data.
Figure 2: On/Off statistics of every unit set, model vs data. The dynamics were fitted to the all-units set only;
          the other sets are predicted by changing the read-out (gain from the data's mu_on, noise share).

usage: python -m paper_model.make_figures       (writes figures/paper_model/)
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from . import model as M, readout as R  # noqa: E402

OUT = 'figures/paper_model'
FIT_SET = 'all'                # unit set the dynamics were fitted to, and the read-out shown in Figure 1
QUIET, DIPS = 1000, 1500       # example trial: delay without dips (ms after cue onset), then with dips
C_ON, C_OFF, INK, INK2, MUTED, C_UN = '#eb6834', '#2a78d6', '#0b0b0b', '#52514e', '#b5b4ae', '#7a6be0'
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 8, 'axes.edgecolor': INK2,
                     'axes.titlesize': 8, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'legend.frameon': False, 'xtick.color': INK2, 'ytick.color': INK2})


def example_trial(gain, n=300, seed=11):
    """Long trials (QUIET ms without dips, then DIPS ms with dips) read out with FIT_SET's gain and noise; returns
    the most representative one: memory held, On throughout the quiet delay, 3-5 Off states once dips start."""
    sim = M.simulate(M.params(t_delay=QUIET + DIPS - M.FIXED['cue_dur']), n, seed=seed + 1, quiet=QUIET)
    rng = np.random.default_rng(seed)
    dw, tz = R.rate_diff(sim, t_from=-250)
    z = gain * dw + R.noise(dw.shape, R.RHO['all'], rng)
    state = R.label(z, tz >= 0, R.null_threshold(dw.shape, R.RHO['all'], rng))
    ok = M.memory_correct(sim) & (sim['rc'][:, -300:].max(1) > 80) & (sim['sc'][:, -1] > 0.4)
    quiet = (tz >= 300) & (tz < QUIET)
    best, score = None, np.inf
    for k in np.where(ok)[0]:
        off = [(b - a) * R.STEP for a, b in R.runs(state[k] == -1) if tz[a] >= QUIET]
        on = [(b - a) * R.STEP for a, b in R.runs(state[k] == 1) if tz[a] >= QUIET]
        if (state[k, quiet] == -1).any() or not 3 <= len(off) <= 5 or len(on) < 2 or max(off) > 300:
            continue
        sc = abs(np.median(off) - 110) / 30 + abs(np.median(on) - 160) / 40 + (state[k, quiet] != 1).mean() * 5
        if sc < score:
            best, score = k, sc
    return {key: v[best] for key, v in sim.items() if key != 't'}, sim['t'], tz, z[best], state[best]


def shade(ax, tz, state):
    for lab, c in ((1, C_ON), (-1, C_OFF)):
        for a, b in R.runs(state == lab):
            ax.axvspan(tz[a] - R.STEP / 2, tz[b - 1] + R.STEP / 2, color=c, alpha=0.18, lw=0)


def figure1(out, hist, path):
    r = out['sets'][FIT_SET]
    tr, t, tz, z, state = example_trial(r['gain'])
    fig = plt.figure(figsize=(10.0, 4.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 0.35, 1.8], hspace=0.12, left=0.06, right=0.74, top=0.86, bottom=0.15)
    gh = fig.add_gridspec(2, 1, hspace=0.35, left=0.82, right=0.99, top=0.86, bottom=0.15)
    a, b, d = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    shade(a, tz, state)
    a.axvspan(0, M.FIXED['cue_dur'], color=INK, alpha=0.12, lw=0)
    a.text(60, 0.04, 'cue', fontsize=6.5, color=INK2)
    a.plot(t, tr['rc'] / M.FIXED['r_max'], color=INK, lw=1.0)     # rates relative to the sigmoid ceiling
    a.plot(t, tr['ru'] / M.FIXED['r_max'], color=C_UN, lw=1.0)
    a.plot(t, tr['sc'], color=INK, lw=1.0, ls=':')
    a.plot(t, tr['su'], color=C_UN, lw=1.0, ls=':')
    a.set_ylim(0, 1.07); a.set_yticks([0, 0.5, 1], ['0', '', '1']); a.set_ylabel('rate (a.u.), NMDA gating')
    a.text(QUIET + 15, 1.0, 'dips start', fontsize=6.5, color=INK2, va='top')
    a.legend(handles=[Line2D([], [], color=INK, label='cued pool'), Line2D([], [], color=C_UN, label='uncued pool'),
                      Line2D([], [], color=INK2, ls=':', label='NMDA gating'),
                      plt.Rectangle((0, 0), 1, 1, color=C_ON, alpha=0.35, lw=0, label='On'),
                      plt.Rectangle((0, 0), 1, 1, color=C_OFF, alpha=0.35, lw=0, label='Off')],
             loc='lower left', bbox_to_anchor=(0, 1.0), ncol=5, fontsize=7, borderaxespad=0.2, handlelength=1.5)
    a.text(-0.08, 1.09, 'A', transform=a.transAxes, fontsize=9, weight='bold')
    shade(d, tz, state)
    d.plot(tz, z, color=INK, lw=0.9)
    for y, c in ((0, INK2), (1.96, C_ON), (0.84, C_OFF)):
        d.axhline(y, color=c, ls='--', lw=0.7)
    d.set_ylim(z.min() - 0.3, z.max() + 0.3); d.set_yticks(np.arange(0, z.max() + 0.3, 2))
    d.set_ylabel('decoding z\n(all units)')
    b.plot(t, tr['m'], color=INK2, lw=0.8)     # common input, minimal
    b.set_ylim(0, 1.1); b.set_yticks([]); b.spines['left'].set_visible(False); b.spines['bottom'].set_visible(False)
    b.tick_params(bottom=False)
    b.text(-0.01, 0.5, 'input', transform=b.transAxes, ha='right', va='center', fontsize=7, color=INK2)
    d.set_xlabel('time from cue onset (ms)')
    for ax in (a, d, b):
        ax.axvline(QUIET, color=INK2, ls='--', lw=0.7)
        ax.set_xlim(-300, QUIET + DIPS)
    for ax in (a, b):
        ax.tick_params(labelbottom=False)
    edges = np.arange(0, 620, 20)
    for j, (name, c) in enumerate((('off', C_OFF), ('on', C_ON))):
        h = fig.add_subplot(gh[j], sharex=fig.axes[-1] if j else None, sharey=fig.axes[-1] if j else None)
        hm, hd = r['h_' + name], hist[FIT_SET, name]
        h.bar(edges[:-1], hm[:-1] / hm.sum(), width=18, align='edge', color=c, alpha=0.8, lw=0)
        h.step(edges, np.r_[hd[:-1], hd[-2]] / hd.sum(), where='post', color=INK, lw=1.0)
        h.set_xlim(0, 620); h.set_xticks([0, 200, 400, 600]); h.tick_params(labelsize=6.5)
        h.set_ylabel('fraction of states', fontsize=7)
        h.text(0.97, 0.88, f'{name.capitalize()} states', transform=h.transAxes, ha='right', color=c, weight='bold')
        if j == 0:
            h.tick_params(labelbottom=False)
            h.text(-0.3, 1.09, 'B', transform=h.transAxes, fontsize=9, weight='bold')
            h.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MUTED, lw=0, label='model'),
                              Line2D([], [], color=INK, label='data')], loc='center right', fontsize=6.5)
        else:
            h.set_xlabel('state length (ms)')
    fig.savefig(path, dpi=300)


def figure2(out, ses, path):
    x = np.arange(len(R.SETS))
    jit = np.random.default_rng(0).uniform(-0.18, 0.18, 25)
    panels = (('off_per_trial', 'n_off', 'Off states per trial', C_OFF, None),
              ('off_median_ms', 'off_median', 'Off length, median (ms)', C_OFF, 250),
              ('frac_off', 'frac_off', 'fraction of delay Off', C_OFF, 1),
              ('on_per_trial', 'n_on', 'On states per trial', C_ON, None),
              ('on_median_ms', 'on_median', 'On length, median (ms)', C_ON, 250),
              ('frac_on', 'frac_on', 'fraction of delay On', C_ON, 1))
    fig, axs = plt.subplots(2, 3, figsize=(6.6, 4.3), gridspec_kw=dict(hspace=0.75, wspace=0.42))
    for a, (dk, mk, name, col, top) in zip(axs.ravel(), panels):
        for i, lab in enumerate(R.SETS):
            v = ses[lab][dk]
            a.plot(i + jit[:len(v)], v, '.', ms=3, color=INK2, alpha=0.35, mew=0, zorder=1)
        v = [ses[lab][dk] for lab in R.SETS]
        a.errorbar(x, [s.mean() for s in v], yerr=[2 * s.std(ddof=1) / np.sqrt(len(s)) for s in v], fmt='s', ms=4,
                   mfc='white', mec=INK, ecolor=INK2, elinewidth=0.8, zorder=3)
        a.plot(x, [out['sets'][lab][mk] for lab in R.SETS], 'o', ms=4.5, color=col, zorder=2)
        j = list(R.SETS).index(FIT_SET)
        a.axvspan(j - 0.45, j + 0.45, color=MUTED, alpha=0.3, lw=0, zorder=0)
        a.axvline(2.5, color=MUTED, lw=0.6); a.axvline(5.5, color=MUTED, lw=0.6)
        a.set_xticks(x); a.set_xticklabels(R.SETS, rotation=60, fontsize=6.5)
        a.get_xticklabels()[j].set_fontweight('bold')
        a.set_ylim(0, top or a.get_ylim()[1] * 1.05)
        a.set_title(name); a.tick_params(labelsize=6.5)
    fig.legend(handles=[Line2D([], [], ls='', marker='s', ms=4, mfc='white', mec=INK, label='data (mean ± 2 SEM over 25 sessions, ≈95% CI)'),
                        Line2D([], [], ls='', marker='.', ms=5, color=INK2, alpha=0.4, label='sessions'),
                        Line2D([], [], ls='', marker='o', ms=4.5, color=INK2, label='model'),
                        plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.3, lw=0, label='fitted (all units); other sets predicted')],
               loc='lower left', bbox_to_anchor=(0.02, 0.93), ncol=4, fontsize=6.5)
    fig.savefig(path, dpi=300, bbox_inches='tight')


def main():
    os.makedirs(OUT, exist_ok=True)
    ses, hist = R.load_data()
    out = R.evaluate(M.params(), ses, n_trials=2112, seed=1)
    figure1(out, hist, os.path.join(OUT, 'fig1_trial_histograms.png'))
    figure2(out, ses, os.path.join(OUT, 'fig2_unit_sets.png'))
    keys = ('gain', 'n_off', 'off_mean', 'off_median', 'frac_off', 'n_on', 'on_mean', 'on_median', 'frac_on')
    stats = dict(fit=M.FIT, memory_correct=out['mem'],
                 sets={lab: {k: float(r[k]) for k in keys} for lab, r in out['sets'].items()})
    json.dump(stats, open(os.path.join(OUT, 'model_stats.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
