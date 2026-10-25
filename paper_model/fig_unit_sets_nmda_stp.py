"""Supplementary Figure 2: Figure 2 layout with the NMDA model (circles) and the STP model (squares, facilitation only),
and the fit loss of each model per decoded population (g) and summed over the 7 populations (h).

Both models fitted to the all-units read-out only; the other six decoded populations are predicted by changing only
the read-out (gain from the data's mu_on, noise share), as in Figure 2. Off statistics in cyan, On statistics in red.

usage: python -m paper_model.fig_unit_sets_nmda_stp     (writes figures/paper_model/figS_unit_sets_nmda_stp.{png,svg,csv})
"""
import csv
import os

import numpy as np

from . import model as M, model_ours_stp as H, readout as R
from .fig_unit_sets_compare import loss_per_set, plain_log
from .make_figures import plt, save, letter, INK, GREY, LIGHT, OUT, FIT_SET, C_ON, C_OFF

BAR = {'NMDA model': '#4d4d4d', 'STP model': '#b3b3b3'}

MODELS = (('NMDA model', M, M.params(), 'o', 0.2), ('STP model', H, H.params(**H.FIT_FAC), 's', 0.36))
PANELS = (('off_per_trial', 'n_off', 'Off states per trial', C_OFF, None),
          ('off_median_ms', 'off_median', 'Off length, median (ms)', C_OFF, 250),
          ('frac_off', 'frac_off', 'Fraction of delay Off', C_OFF, 1),
          ('on_per_trial', 'n_on', 'On states per trial', C_ON, None),
          ('on_median_ms', 'on_median', 'On length, median (ms)', C_ON, 250),
          ('frac_on', 'frac_on', 'Fraction of delay On', C_ON, 1))


def main():
    ses, hist = R.load_data()
    outs = []
    for name, mod, p, _, _ in MODELS:
        R.M = mod
        outs.append(R.evaluate(p, ses, n_trials=2112, seed=1))
    R.M = M
    x = np.arange(len(R.SETS))
    jit = np.random.default_rng(0).uniform(-0.18, 0.18, 25)
    fig = plt.figure(figsize=(6.4, 6.2))
    gs = fig.add_gridspec(3, 3, hspace=0.95, wspace=0.6, left=0.11, right=0.98, top=0.95, bottom=0.08,
                          height_ratios=[1, 1, 0.9])
    axs = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
    j = list(R.SETS).index(FIT_SET)
    for k, (a, (dk, mk, label, col, top)) in enumerate(zip(axs.ravel(), PANELS)):
        a.axvspan(j - 0.45, j + 0.5, color=LIGHT, lw=0, zorder=0)
        for i, lab in enumerate(R.SETS):
            v = ses[lab][dk]
            a.plot(i + jit[:len(v)], v, '.', ms=2.5, color=GREY, alpha=0.5, mew=0, zorder=1)
        v = [ses[lab][dk] for lab in R.SETS]
        a.errorbar(x, [s.mean() for s in v], yerr=[2 * s.std(ddof=1) / np.sqrt(len(s)) for s in v], fmt='o', ms=3,
                   color=INK, elinewidth=0.6, capsize=0, zorder=3)
        for (name, _, _, mk_, dx), out in zip(MODELS, outs):
            a.plot(x + dx, [out['sets'][lab][mk] for lab in R.SETS], mk_, ms=3, color=col, mew=0, zorder=2)
        a.axvline(2.5, color=GREY, ls=(0, (3, 2)), lw=0.5); a.axvline(5.5, color=GREY, ls=(0, (3, 2)), lw=0.5)
        a.set_xlim(-0.6, len(R.SETS) - 0.4)
        a.set_xticks(x); a.set_xticklabels(R.SETS, rotation=60, ha='right', rotation_mode='anchor', fontsize=7)
        a.set_ylim(0, top or np.ceil(a.get_ylim()[1]))
        if top == 1:
            a.set_yticks([0, 0.5, 1], ['0', '', '1'])
        a.set_ylabel(label)
        letter(a, 'abcdef'[k], x=-0.5, y=1.02)
        a.text(j + 0.03, 0.97, 'Fitted', transform=a.get_xaxis_transform(), ha='center', va='top', rotation=90, fontsize=7)
        a.text(2.5, 1.1, 'Predicted', transform=a.get_xaxis_transform(), ha='center', va='bottom', fontsize=7)
        a.plot([0, 0, 5, 5], [1.06, 1.09, 1.09, 1.06], color=INK, lw=0.5, transform=a.get_xaxis_transform(), clip_on=False)
    for a, col, y0, va in ((axs[0, 0], C_OFF, 0.05, 'bottom'), (axs[1, 0], C_ON, 0.97, 'top')):   # legends in empty corners
        items = (('Data', INK), ('\u25cf NMDA', col), ('\u25a0 STP', col), ('Sessions', GREY))
        for n, (txt, c) in enumerate(items):
            y = y0 + (len(items) - 1 - n) * 0.09 if va == 'bottom' else y0 - n * 0.09
            a.text(0.5, y, txt, color=c, transform=a.transAxes, va=va, fontsize=7)
    losses = [{lab: loss_per_set(o['sets'][lab], ses[lab], hist, lab) for lab in R.SETS} for o in outs]
    g = fig.add_subplot(gs[2, :2])
    w = 0.36
    for m, ((name, *_), ls) in enumerate(zip(MODELS, losses)):
        g.bar(x + (m - 0.5) * w, [ls[lab] for lab in R.SETS], width=w, color=BAR[name], lw=0, label=name)
    g.set_yscale('log'); g.set_xticks(x); g.set_xticklabels(R.SETS, rotation=60, ha='right', rotation_mode='anchor', fontsize=7)
    g.axvline(2.5, color=GREY, ls=(0, (3, 2)), lw=0.5); g.axvline(5.5, color=GREY, ls=(0, (3, 2)), lw=0.5)
    g.set_ylabel('Fit loss (log)'); g.set_xlim(-0.6, len(R.SETS) - 0.4)
    plain_log(g, 1)
    g.set_ylim(1, 100); plain_log(g, 1)
    for n, (name, *_) in enumerate(MODELS):
        g.text(0.02, 0.97 - n * 0.11, name, color=BAR[name], transform=g.transAxes, ha='left', va='top', fontsize=7)
    letter(g, 'g', x=-0.2, y=1.02)
    h = fig.add_subplot(gs[2, 2])
    tot = [sum(ls.values()) for ls in losses]
    h.bar(range(len(MODELS)), tot, color=[BAR[n] for n, *_ in MODELS], lw=0, width=0.6)
    for i, v in enumerate(tot):
        h.text(i, v * 1.15, f'{v:.0f}', ha='center', va='bottom', fontsize=7)
    h.set_yscale('log'); h.set_ylim(10, max(tot) * 6)
    plain_log(h, 10)
    h.set_xlim(-0.7, len(MODELS) - 0.3)
    h.set_xticks(range(len(MODELS)), ['NMDA', 'STP'], fontsize=7)
    h.set_ylabel('Fit loss, sum over\n7 populations (log)')
    letter(h, 'h', x=-0.45, y=1.02)
    save(fig, os.path.join(OUT, 'figS_unit_sets_nmda_stp.png'))
    print('loss per set', [{k: round(v, 1) for k, v in ls.items()} for ls in losses], 'sum', [round(v) for v in tot])
    with open(os.path.join(OUT, 'figS_unit_sets_nmda_stp.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'set', 'memory'] + [mk for _, mk, _, _, _ in PANELS])
        for (name, *_), out in zip(MODELS, outs):
            for lab in R.SETS:
                w.writerow([name, lab, out['mem']] + [out['sets'][lab][mk] for _, mk, _, _, _ in PANELS])
    for (name, *_), out in zip(MODELS, outs):
        print(name, 'memory', round(out['mem'], 4), {lab: (round(out['sets'][lab]['n_off'], 2), round(out['sets'][lab]['frac_on'], 2)) for lab in R.SETS})


if __name__ == '__main__':
    main()
