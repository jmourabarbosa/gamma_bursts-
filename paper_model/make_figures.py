"""Paper figures of the best-fit model.

Figure 1: a, an example trial: 1 s of delay with a constant common input (attention signal), then dips (pool
          rates and NMDA gating; attention signal; all-units decoding z with On/Off labels). B: the model's
          all-units Off and On length distributions (states 0-1,400 ms after cue onset).
Figure 2: On/Off statistics of every unit set, model vs data. The dynamics were fitted to the all-units set only;
          the other sets are predicted by changing the read-out (gain from the data's mu_on, noise share).

usage: python -m paper_model.make_figures       (writes PNG and SVG files to figures/paper_model/)
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
CUE = 75                       # example trial only: longer cue for display (fitted trials: 50 ms, as in the task)
# journal style (Nature Neuroscience-like): Arial-metric font, 6-7 pt text, thin black axes with outward ticks,
# lowercase bold panel letters, sentence-case labels, text legends, grey epoch bars
C_ON, C_OFF, INK, GREY, LIGHT, C_UN = '#e8474a', '#3cb4c9', '#000000', '#7f7f7f', '#d9d9d9', '#8c8c8c'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'],
                     'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK,
                     'axes.titlesize': 8, 'axes.titleweight': 'normal', 'lines.linewidth': 0.9,
                     'legend.frameon': False, 'legend.fontsize': 7.5, 'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def save(fig, path):
    """PNG for viewing and SVG (editable text, e.g. in Illustrator or Inkscape)."""
    fig.savefig(path, dpi=600)
    fig.savefig(path.replace('.png', '.svg'))


def letter(ax, txt, x=-0.08, y=1.06):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=10, weight='bold', va='bottom')


def example_trial(gain, n=300, seed=11):
    """Long trials (QUIET ms without dips, then DIPS ms with dips) read out with FIT_SET's gain and noise; returns
    the most representative one: memory held, On throughout the quiet delay, 3-5 Off states once dips start."""
    sim = M.simulate(M.params(cue_dur=CUE, t_delay=QUIET + DIPS - CUE), n, seed=seed + 1, quiet=QUIET)
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
            ax.axvspan(tz[a] - R.STEP / 2, tz[b - 1] + R.STEP / 2, color=c, alpha=0.15, lw=0)


def figure1(out, path):
    r = out['sets'][FIT_SET]
    tr, t, tz, z, state = example_trial(r['gain'])
    fig = plt.figure(figsize=(7.2, 3.0))     # double column, 183 mm
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 0.35, 2.0], hspace=0.22, left=0.08, right=0.71, top=0.9, bottom=0.14)
    gh = fig.add_gridspec(2, 1, hspace=0.45, left=0.85, right=0.99, top=0.9, bottom=0.14)
    a, b, d = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    shade(a, tz, state)
    a.axvspan(0, CUE, color=LIGHT, lw=0)
    a.plot(t, tr['rc'] / M.FIXED['r_max'], color=INK)     # rates relative to the sigmoid ceiling (a.u.)
    a.plot(t, tr['ru'] / M.FIXED['r_max'], color=C_UN)
    a.plot(t, tr['sc'], color=INK, ls=(0, (1, 1.5)))
    a.plot(t, tr['su'], color=C_UN, ls=(0, (1, 1.5)))
    a.set_ylim(0, 1); a.set_yticks([0, 0.5, 1], ['0', '', '1']); a.set_ylabel('Rate / NMDA')
    xa = a.get_xaxis_transform()     # x in data, y in axes fraction
    for dx, label in ((1, 'With attention fluctuations'), (-1, 'Without attention fluctuations')):   # above the panel
        a.annotate('', xy=(QUIET + dx * 160, 1.06), xytext=(QUIET + dx * 15, 1.06), xycoords=xa, textcoords=xa,
                   arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.6, mutation_scale=5), annotation_clip=False)
        a.text(QUIET + dx * 180, 1.06, label, transform=xa, fontsize=7.5, va='center', ha='left' if dx > 0 else 'right')
    # direct labels on the traces, in the steady delay before the dips
    a.text(500, tr['sc'][(t > 400) & (t < 600)].max() + 0.04, 'NMDA', fontsize=7.5, ha='center', va='bottom')
    a.text(500, tr['rc'][(t > 400) & (t < 600)].min() / M.FIXED['r_max'] - 0.04, 'Rate', fontsize=7.5, ha='center', va='top')
    off0 = R.runs(state == -1)[0]
    on1 = next((a0, b0) for a0, b0 in R.runs(state == 1) if tz[a0] > QUIET)    # first On state with attention fluctuations
    a.text((tz[on1[0]] + tz[on1[1] - 1]) / 2, 0.96, 'On', color=C_ON, fontsize=7.5, va='top', ha='center')
    a.text((tz[off0[0]] + tz[off0[1] - 1]) / 2, 0.96, 'Off', color=C_OFF, fontsize=7.5, va='top', ha='center')
    letter(a, 'a')
    b.plot(t, tr['m'], color=INK, lw=0.7)     # common input, minimal
    b.set_ylim(0, 1.1); b.set_yticks([]); b.spines['left'].set_visible(False); b.spines['bottom'].set_visible(False)
    b.tick_params(bottom=False)
    b.text(-0.01, 0.5, 'Attention\nsignal', transform=b.transAxes, ha='right', va='center', fontsize=7.5)
    shade(d, tz, state)
    d.plot(tz, z, color=INK)
    for y, c, txt in ((0, GREY, ''), (1.96, C_ON, 'P < 0.05'), (0.84, C_OFF, 'P > 0.20')):
        d.axhline(y, color=c, ls=(0, (3, 2)), lw=0.6)
        d.text(1.005, y, txt, transform=d.get_yaxis_transform(), color=c, fontsize=7.5, va='center')
    d.set_ylim(z.min() - 0.3, 6); d.set_yticks([0, 3, 6], ['0', '', '6'])
    d.set_ylabel('Decoding z')
    d.set_title(r'$z(t) = c\,[r_\mathrm{cued}(t) - r_\mathrm{uncued}(t)] + \eta(t)$', fontsize=7.5, loc='left', pad=2)
    d.set_xlabel('Time from cue onset (ms)')
    for ax in (a, b, d):
        ax.axvline(QUIET, color=GREY, ls=(0, (3, 2)), lw=0.6)
        ax.set_xlim(-300, QUIET + DIPS)
    for ax in (a, b):
        ax.tick_params(labelbottom=False)
    edges = np.arange(0, 620, 20)
    for j, (name, c) in enumerate((('off', C_OFF), ('on', C_ON))):
        h = fig.add_subplot(gh[j], sharex=fig.axes[-1] if j else None)
        hm = r['h_' + name]
        h.bar(edges[:-1], hm[:-1] / hm.sum(), width=20, align='edge', color=c, edgecolor='white', lw=0.3)
        h.set_xlim(0, 620); h.set_xticks([0, 200, 400, 600])
        h.set_ylim(0, 0.25); h.set_yticks([0, 0.125, 0.25], ['0', '', '0.25'])
        h.set_ylabel('Fraction\nof states', labelpad=-14)   # beside the unlabelled middle tick
        h.text(0.97, 0.9, f'{name.capitalize()} states', transform=h.transAxes, ha='right', va='top', color=c, fontsize=7.5)
        if j == 0:
            h.tick_params(labelbottom=False)
            letter(h, 'b', x=-0.55, y=1.0)
        else:
            h.set_xlabel('State length (ms)')
    save(fig, path)


def figure2(out, ses, path):
    x = np.arange(len(R.SETS))
    jit = np.random.default_rng(0).uniform(-0.18, 0.18, 25)
    panels = (('off_per_trial', 'n_off', 'Off states per trial', C_OFF, None),
              ('off_median_ms', 'off_median', 'Off length, median (ms)', C_OFF, 250),
              ('frac_off', 'frac_off', 'Fraction of delay Off', C_OFF, 1),
              ('on_per_trial', 'n_on', 'On states per trial', C_ON, None),
              ('on_median_ms', 'on_median', 'On length, median (ms)', C_ON, 250),
              ('frac_on', 'frac_on', 'Fraction of delay On', C_ON, 1))
    fig, axs = plt.subplots(2, 3, figsize=(6.2, 4.0), gridspec_kw=dict(hspace=0.8, wspace=0.6, left=0.11, right=0.98,
                                                                       top=0.92, bottom=0.12))
    j = list(R.SETS).index(FIT_SET)
    for k, (a, (dk, mk, name, col, top)) in enumerate(zip(axs.ravel(), panels)):
        a.axvspan(j - 0.45, j + 0.45, color=LIGHT, lw=0, zorder=0)
        for i, lab in enumerate(R.SETS):
            v = ses[lab][dk]
            a.plot(i + jit[:len(v)], v, '.', ms=2.5, color=GREY, alpha=0.5, mew=0, zorder=1)
        v = [ses[lab][dk] for lab in R.SETS]
        a.errorbar(x, [s.mean() for s in v], yerr=[2 * s.std(ddof=1) / np.sqrt(len(s)) for s in v], fmt='o', ms=3,
                   color=INK, elinewidth=0.6, capsize=0, zorder=3)
        a.plot(x + 0.25, [out['sets'][lab][mk] for lab in R.SETS], 'o', ms=3, color=col, zorder=2)
        a.axvline(2.5, color=GREY, ls=(0, (3, 2)), lw=0.5); a.axvline(5.5, color=GREY, ls=(0, (3, 2)), lw=0.5)
        a.set_xlim(-0.6, len(R.SETS) - 0.4)
        a.set_xticks(x); a.set_xticklabels(R.SETS, rotation=60, ha='right', rotation_mode='anchor', fontsize=7)
        a.set_ylim(0, top or np.ceil(a.get_ylim()[1]))     # top limit on an integer tick
        if top == 1:
            a.set_yticks([0, 0.5, 1], ['0', '', '1'])
        a.set_ylabel(name)
        letter(a, 'abcdef'[k], x=-0.5, y=1.02)
    for a in axs.ravel():     # fitted (inside the grey bar) vs predicted (bracket over the other sets)
        a.text(j, 0.97, 'Fitted', transform=a.get_xaxis_transform(), ha='center', va='top', rotation=90, fontsize=7)
        a.text(2.5, 1.1, 'Predicted', transform=a.get_xaxis_transform(), ha='center', va='bottom', fontsize=7)
        a.plot([0, 0, 5, 5], [1.06, 1.09, 1.09, 1.06], color=INK, lw=0.5, transform=a.get_xaxis_transform(), clip_on=False)
    for a, col, y0, va in ((axs[0, 0], C_OFF, 0.05, 'bottom'), (axs[1, 0], C_ON, 0.97, 'top')):   # legends in empty corners
        for n, (txt, c) in enumerate((('Data', INK), ('Model', col), ('Sessions', GREY))):
            y = y0 + (2 - n) * 0.09 if va == 'bottom' else y0 - n * 0.09
            a.text(0.5, y, txt, color=c, transform=a.transAxes, va=va, fontsize=7)
    save(fig, path)


def main():
    os.makedirs(OUT, exist_ok=True)
    ses, _ = R.load_data()
    out = R.evaluate(M.params(), ses, n_trials=2112, seed=1)
    figure1(out, os.path.join(OUT, 'fig1_trial_histograms.png'))
    figure2(out, ses, os.path.join(OUT, 'fig2_unit_sets.png'))
    keys = ('gain', 'n_off', 'off_mean', 'off_median', 'frac_off', 'n_on', 'on_mean', 'on_median', 'frac_on')
    stats = dict(fit=M.FIT, memory_correct=out['mem'],
                 sets={lab: {k: float(r[k]) for k in keys} for lab, r in out['sets'].items()})
    json.dump(stats, open(os.path.join(OUT, 'model_stats.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
