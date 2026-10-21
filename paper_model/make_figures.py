"""Paper figures of the best-fit model.

Figure 1: a representative single trial (pool rates with the top-20 On/Off labels, and the common input below)
          and the model's top-20 Off and On length histograms.
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


def pick_trial(out):
    """A representative trial: memory held, 3-4 Off and 2-4 On states in 0-1,400 ms, with typical lengths."""
    sim, r = out['sim'], out['sets'][('top', '20')]
    ok = M.memory_correct(sim)
    best, score = 0, np.inf
    for k in range(len(ok)):
        off = [(b - a) * R.STEP for a, b in R.runs(r['state'][k] == -1)]
        on = [(b - a) * R.STEP for a, b in R.runs(r['state'][k] == 1)]
        if not ok[k] or not 3 <= len(off) <= 4 or not 2 <= len(on) <= 4 or max(off) > 300:
            continue
        s = abs(np.median(off) - 110) / 30 + abs(np.median(on) - 160) / 40 + (r['state'][k][r['state'][k] != 0].size < 100)
        if s < score:
            best, score = k, s
    return best


def figure1(out, summ, hist, path):
    sim, r = out['sim'], out['sets'][('top', '20')]
    k = pick_trial(out)
    cued = sim['cued'][k]
    rc, ru = (sim['rA'][k], sim['rB'][k]) if cued == 0 else (sim['rB'][k], sim['rA'][k])
    t = np.arange(len(rc)) - sim['t_cue']
    tz, st = out['tz'], r['state'][k]
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.3, 1], height_ratios=[2.2, 1], wspace=0.32, hspace=0.18,
                          left=0.08, right=0.98, top=0.86, bottom=0.15)
    a = fig.add_subplot(gs[0, 0])
    for lab, c in ((1, C_ON), (-1, C_OFF)):
        for i0, i1 in R.runs(st == lab):
            a.axvspan(tz[i0] - R.STEP / 2, tz[i1 - 1] + R.STEP / 2, color=c, alpha=0.18, lw=0)
    a.axvspan(0, M.FIXED['cue_dur'], color=INK, alpha=0.12, lw=0)
    a.text(25, 140, 'cue', ha='center', fontsize=6.5, color=INK2, rotation=90)
    a.plot(t, rc, color=INK, lw=1.2, label='cued pool')
    a.plot(t, ru, color=C_UN, lw=1.2, label='uncued pool')
    a.set_xlim(-300, 1500); a.set_ylim(0, 160); a.set_ylabel('rate (Hz)')
    a.tick_params(labelbottom=False)
    a.legend(handles=[Line2D([], [], color=INK, lw=1.2, label='cued pool'), Line2D([], [], color=C_UN, lw=1.2, label='uncued pool'),
                      plt.Rectangle((0, 0), 1, 1, color=C_ON, alpha=0.35, lw=0, label='On'),
                      plt.Rectangle((0, 0), 1, 1, color=C_OFF, alpha=0.35, lw=0, label='Off')],
             loc='lower left', bbox_to_anchor=(0, 1.0), ncol=4, fontsize=7, borderaxespad=0.2, handlelength=1.5)
    a.set_title('A', loc='left', x=-0.12, y=1.06)
    b = fig.add_subplot(gs[1, 0], sharex=a)
    b.plot(t, sim['m'][k], color=INK2, lw=1.1)
    b.set_ylim(0, 1.1); b.set_yticks([0, 1]); b.set_ylabel('common\ninput m(t)')
    b.set_xlabel('time from cue onset (ms)')
    for j, (s, c, name) in enumerate((('off', C_OFF, 'Off'), ('on', C_ON, 'On'))):
        h = fig.add_subplot(gs[j, 1]) if j == 0 else fig.add_subplot(gs[1, 1])
        lab = 1 if s == 'on' else -1
        dur = np.array([(b - a) * R.STEP for row in r['state'] for a, b in R.runs(row == lab)], float)
        edges = np.arange(0, 620, 20)
        hm = np.histogram(dur, edges)[0] / len(dur) / 20
        h.bar(edges[:-1], hm, width=18, align='edge', color=c, alpha=0.8, lw=0)
        h.set_xlim(0, 620); h.set_ylim(0, hm.max() * 1.1)
        h.set_xticks([0, 200, 400, 600])
        h.text(0.97, 0.9, f'{name} states', transform=h.transAxes, ha='right', fontsize=7.5, color=c, weight='bold')
        h.tick_params(labelsize=6.5)
        h.set_ylabel('prob. / ms', fontsize=7)
        if j == 0:
            h.tick_params(labelbottom=False)
            h.set_title('B', loc='left', x=-0.3, y=1.06)
        else:
            h.set_xlabel('length (ms)')
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
    k = figure1(out, summ, hist, os.path.join(a.out, 'fig1_trial_histograms.png'))
    figure2(out, summ, os.path.join(a.out, 'fig2_fit_all_sets.png'))
    res = dict(theta=M.FIT, n_trials=a.n_trials, seed=a.seed, example_trial=int(k), memory_correct=out['mem'],
               sets={'/'.join(s): {kk: v for kk, v in r.items() if kk not in ('z', 'state', 'h_on', 'h_off')} for s, r in out['sets'].items()})
    json.dump(res, open(os.path.join(a.out, 'model_stats.json'), 'w'), indent=1)
    print(json.dumps(dict(memory_correct=out['mem'], example_trial=int(k)), indent=1))


if __name__ == '__main__':
    main()
