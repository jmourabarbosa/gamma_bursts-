"""Paper figures of the best-fit model.

Figure 1: A, an example trial: 2 s of delay with a constant common input, then dips switched on (pool rates
          and NMDA gating, decoding z from all units, with On/Off labels; common input below). B, right of A: all-unit Off and On
          length histograms (counts, states 0-1,400 ms after cue onset, as in the fit).
Figure 2: On/Off statistics of every unit set, model vs data.

usage: python -m paper_model.make_figures [--n-trials 2112] [--seed 1] [--out figures/paper_model]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from . import model as M, readout as R  # noqa: E402

C_ON, C_OFF, INK, INK2, MUTED, C_UN = '#eb6834', '#2a78d6', '#0b0b0b', '#52514e', '#b5b4ae', '#7a6be0'
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 8, 'axes.edgecolor': INK2,
                     'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'legend.frameon': False, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK})


FIG1_SET = ('all', 'session count')   # unit set whose read-out Figure 1 shows
QUIET, DIPS = 2000, 1500      # example trial: delay without dips (ms after cue onset), then with dips


def example_trials(gain, n=300, seed=11):
    """Long trials: a delay of QUIET ms with a constant common input, then DIPS ms with dips. Returns the
    simulation, read-out z (gain and noise of FIG1_SET) and On/Off labels over the whole delay."""
    p = M.params(t_delay=QUIET + DIPS - M.FIXED['cue_dur'])
    rng = np.random.default_rng(seed)
    dt = p['dt']
    n_t = int(round((p['t_pre'] + p['cue_dur'] + p['t_delay']) / dt))
    i0 = int(round((p['t_pre'] + QUIET) / dt))
    m = np.ones((n, n_t))
    m[:, i0:] = M.common_input(n, n_t - i0, dt, p['dip_rate'], p['dip_mean'], p['dip_depth'], p['dip_shape'], rng)
    m[:, i0:i0 + int(10 / dt)] = 1.0          # no dip already running at the switch
    sim = M.simulate(p, np.zeros(n, int), seed=seed + 1, m=m)
    dw, tz = R.rate_diff(sim)
    z = gain * dw + R.noise(dw.shape, R.RHO[FIG1_SET[0]], rng)
    thr = R.null_threshold(dw.shape, R.RHO[FIG1_SET[0]], rng)
    return sim, tz, z, R.label(z, tz >= 0, thr)


def pick_trial(sim, tz, state):
    """Representative trial: memory held, On throughout the quiet delay, 3-5 Off states once dips start,
    with typical lengths."""
    ok = M.memory_correct(sim)
    quiet = (tz >= 300) & (tz < QUIET)
    best, score = 0, np.inf
    for k in range(len(ok)):
        st = state[k]
        off = [(b - a) * R.STEP for a, b in R.runs(st == -1) if tz[a] >= QUIET]
        on = [(b - a) * R.STEP for a, b in R.runs(st == 1) if tz[a] >= QUIET]
        strong = sim['rA'][k][-300:].max() > 80 and sim['sA'][k][-1] > 0.4 and min(
            sim['sA'][k][int(sim['t_cue'] + QUIET):].min(), 1) > 0.25     # memory clearly held through the dips
        if not ok[k] or not strong or (st[quiet] == -1).any() or not 3 <= len(off) <= 5 or len(on) < 2 or max(off) > 300:
            continue
        sc = abs(np.median(off) - 110) / 30 + abs(np.median(on) - 160) / 40 + (st[quiet] != 1).mean() * 5
        if sc < score:
            best, score = k, sc
    return best


def figure1(out, path):
    r = out['sets'][FIG1_SET]
    sim, tz, z, state = example_trials(r['gain'])
    k = pick_trial(sim, tz, state)
    t = np.arange(sim['rA'].shape[1]) - sim['t_cue']
    fig = plt.figure(figsize=(10.0, 4.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.6, 1.3, 0.9], hspace=0.15, left=0.06, right=0.58, top=0.86, bottom=0.15)
    gh = fig.add_gridspec(1, 2, wspace=0.1, left=0.7, right=0.99, top=0.86, bottom=0.15)
    a = fig.add_subplot(gs[0])
    for lab, c in ((1, C_ON), (-1, C_OFF)):
        for i0, i1 in R.runs(state[k] == lab):
            a.axvspan(tz[i0] - R.STEP / 2, tz[i1 - 1] + R.STEP / 2, color=c, alpha=0.18, lw=0)
    a.axvspan(0, M.FIXED['cue_dur'], color=INK, alpha=0.12, lw=0)
    a.text(60, 6, 'cue', fontsize=6.5, color=INK2)
    a.plot(t, sim['rA'][k], color=INK, lw=1.0)
    a.plot(t, sim['rB'][k], color=C_UN, lw=1.0)
    a.set_xlim(-300, QUIET + DIPS); a.set_ylim(0, 160); a.set_ylabel('rate (Hz)')
    a.tick_params(labelbottom=False)
    a2 = a.twinx(); a2.spines['right'].set_visible(True)
    a2.plot(t, sim['sA'][k], color=INK, lw=1.0, ls=':')
    a2.plot(t, sim['sB'][k], color=C_UN, lw=1.0, ls=':')
    a2.set_ylim(0, 1); a2.set_ylabel('NMDA gating s (dotted)')
    a.legend(handles=[Line2D([], [], color=INK, lw=1.2, label='cued pool'), Line2D([], [], color=C_UN, lw=1.2, label='uncued pool'),
                      Line2D([], [], color=INK2, lw=1.0, ls=':', label='NMDA gating'),
                      plt.Rectangle((0, 0), 1, 1, color=C_ON, alpha=0.35, lw=0, label='On'),
                      plt.Rectangle((0, 0), 1, 1, color=C_OFF, alpha=0.35, lw=0, label='Off')],
             loc='lower left', bbox_to_anchor=(0, 1.0), ncol=5, fontsize=7, borderaxespad=0.2, handlelength=1.5)
    a.text(-0.08, 1.09, 'A', transform=a.transAxes, fontsize=9, weight='bold')
    d = fig.add_subplot(gs[1], sharex=a)
    for lab, c in ((1, C_ON), (-1, C_OFF)):
        for i0, i1 in R.runs(state[k] == lab):
            d.axvspan(tz[i0] - R.STEP / 2, tz[i1 - 1] + R.STEP / 2, color=c, alpha=0.18, lw=0)
    d.plot(tz, z[k], color=INK, lw=0.9)
    d.axhline(1.96, color=C_ON, ls=':', lw=0.8); d.axhline(0.84, color=C_OFF, ls=':', lw=0.8)
    d.set_ylim(-3, 9); d.set_ylabel('decoding z\n(all units)'); d.tick_params(labelbottom=False)
    b = fig.add_subplot(gs[2], sharex=a)
    b.plot(t, sim['m'][k], color=INK2, lw=1.0)
    b.set_ylim(0, 1.15); b.set_yticks([0, 1]); b.set_ylabel('common\ninput m(t)')
    b.set_xlabel('time from cue onset (ms)')
    for ax_ in (a, d, b):
        ax_.axvline(QUIET, color=INK2, ls='--', lw=0.7)
    a.text(QUIET + 15, 150, 'dips start', fontsize=6.5, color=INK2, va='top')
    # all-unit state lengths (states 0-1,400 ms after cue onset), model vs data, as fractions of all states
    edges = np.arange(0, 620, 20)
    hs = []
    for j, (lab, c, name) in enumerate(((-1, C_OFF, 'Off'), (1, C_ON, 'On'))):
        h = fig.add_subplot(gh[0, j], sharey=hs[0] if hs else None)
        dur = [(e - b0) * R.STEP for row in r['state'] for b0, e in R.runs(row == lab)]
        h.bar(edges[:-1], np.histogram(dur, edges)[0] / len(dur), width=18, align='edge', color=c, alpha=0.8, lw=0)
        dc, n = R.data_counts(FIG1_SET, name.lower())
        h.step(edges, np.r_[dc, dc[-1]] / n, where='post', color=INK, lw=1.0)
        h.set_xlim(0, 620); h.set_xticks([0, 200, 400, 600])
        h.set_title(f'{name} states', fontsize=8, color=c, loc='center')
        h.set_xlabel('state length (ms)'); h.tick_params(labelsize=6.5)
        if j == 0:
            h.set_ylabel('fraction of states', fontsize=7)
            h.text(-0.42, 1.09, 'B', transform=h.transAxes, fontsize=9, weight='bold')
            h.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MUTED, lw=0, label='model'),
                              Line2D([], [], color=INK, lw=1.0, label='data')], loc='upper right', fontsize=6.5)
        else:
            h.tick_params(labelleft=False)
        hs.append(h)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return k


def figure2(out, summ, path):
    x = np.arange(len(R.SETS))
    panels = (('off_per_trial', 'n_off', 'Off states per trial', C_OFF), ('off_mean_ms', 'off_mean', 'Off length, mean (ms)', C_OFF),
              ('frac_off', 'frac_off', 'fraction of delay Off', C_OFF),
              ('on_per_trial', 'n_on', 'On states per trial', C_ON), ('on_mean_ms', 'on_mean', 'On length, mean (ms)', C_ON),
              ('frac_on', 'frac_on', 'fraction of delay On', C_ON))
    fig, axs = plt.subplots(2, 3, figsize=(6.6, 4.3), gridspec_kw=dict(hspace=0.75, wspace=0.42))
    for a, (dk, mk, name, col) in zip(axs.ravel(), panels):
        a.set_title(name, fontsize=8)
        if mk is None:
            a.bar([0], [100 * out['mem']], width=0.5, color=MUTED)
            a.axhline(95, color=INK2, ls=':', lw=0.8)
            a.text(0.3, 85, 'required ≥ 95%', fontsize=6.5, color=INK2, va='top')
            a.text(0, 100 * out['mem'] + 1, f"{100 * out['mem']:.1f}%", ha='center', fontsize=7, va='bottom')
            a.set_xlim(-0.8, 1.6); a.set_ylim(0, 105); a.set_xticks([0]); a.set_xticklabels(['model'], fontsize=7)
            a.set_ylabel('% of trials', fontsize=7)
            continue
        if dk:
            a.errorbar(x, [summ[s][dk] for s in R.SETS], yerr=[summ[s][dk + '_sem'] for s in R.SETS], fmt='s', ms=4,
                       mfc='white', mec=INK, ecolor=INK2, elinewidth=0.8, zorder=3)
        a.plot(x, [out['sets'][s][mk] for s in R.SETS], 'o', ms=4.5, color=col, zorder=2)
        a.set_xticks(x); a.set_xticklabels(R.LABELS, rotation=60, fontsize=6.5)
        a.axvline(2.5, color=MUTED, lw=0.6); a.axvline(5.5, color=MUTED, lw=0.6)
        a.set_ylim(0, a.get_ylim()[1] * 1.05); a.tick_params(labelsize=6.5)
    fig.legend(handles=[Line2D([], [], ls='', marker='s', ms=4, mfc='white', mec=INK, label='data (mean ± SEM over 25 sessions)'),
                        Line2D([], [], ls='', marker='o', ms=4.5, color=INK2, label='model')],
               loc='lower left', bbox_to_anchor=(0.1, 0.93), ncol=2, fontsize=7)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-trials', type=int, default=2112)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='figures/paper_model')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summ, hist = R.load_data()
    out = R.evaluate(n_trials=a.n_trials, seed=a.seed, summ=summ, keep=True)
    k = figure1(out, os.path.join(a.out, 'fig1_trial_histograms.png'))
    figure2(out, summ, os.path.join(a.out, 'fig2_fit_all_sets.png'))
    res = dict(theta=M.FIT, n_trials=a.n_trials, seed=a.seed, example_trial=int(k), memory_correct=out['mem'],
               sets={'/'.join(s): {kk: v for kk, v in r.items() if kk not in ('z', 'state', 'h_on', 'h_off')} for s, r in out['sets'].items()})
    json.dump(res, open(os.path.join(a.out, 'model_stats.json'), 'w'), indent=1)
    print(json.dumps(dict(memory_correct=out['mem'], example_trial=int(k)), indent=1))


if __name__ == '__main__':
    main()
