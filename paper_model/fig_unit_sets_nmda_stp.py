"""Supplementary Figure 2: Figure 2 layout with the NMDA model (circles) and the STP model (squares, facilitation only).

Both models fitted to the all-units read-out only; the other six decoded populations are predicted by changing only
the read-out (gain from the data's mu_on, noise share), as in Figure 2. Off statistics in cyan, On statistics in red.

usage: python -m paper_model.fig_unit_sets_nmda_stp     (writes figures/paper_model/figS_unit_sets_nmda_stp.{png,svg,csv})
"""
import csv
import os

import numpy as np
from matplotlib.lines import Line2D

from . import model as M, model_ours_stp as H, readout as R
from .make_figures import plt, save, letter, INK, GREY, LIGHT, OUT, FIT_SET, C_ON, C_OFF

MODELS = (('NMDA model', M, M.params(), 'o', 0.2), ('STP model', H, H.params(**H.FIT_FAC), 's', 0.36))
PANELS = (('off_per_trial', 'n_off', 'Off states per trial', C_OFF, None),
          ('off_median_ms', 'off_median', 'Off length, median (ms)', C_OFF, 250),
          ('frac_off', 'frac_off', 'Fraction of delay Off', C_OFF, 1),
          ('on_per_trial', 'n_on', 'On states per trial', C_ON, None),
          ('on_median_ms', 'on_median', 'On length, median (ms)', C_ON, 250),
          ('frac_on', 'frac_on', 'Fraction of delay On', C_ON, 1))


def main():
    ses, _ = R.load_data()
    outs = []
    for name, mod, p, _, _ in MODELS:
        R.M = mod
        outs.append(R.evaluate(p, ses, n_trials=2112, seed=1))
    R.M = M
    x = np.arange(len(R.SETS))
    jit = np.random.default_rng(0).uniform(-0.18, 0.18, 25)
    fig, axs = plt.subplots(2, 3, figsize=(6.4, 4.2), gridspec_kw=dict(hspace=0.8, wspace=0.6, left=0.11, right=0.98,
                                                                       top=0.86, bottom=0.12))
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
    hand = [Line2D([], [], ls='', marker='o', ms=3.5, color=INK, label='Data (mean ± 2 SEM)'),
            Line2D([], [], ls='', marker='.', ms=5, color=GREY, label='Sessions'),
            Line2D([], [], ls='', marker='o', ms=3.5, color=GREY, mec='none', label='NMDA model'),
            Line2D([], [], ls='', marker='s', ms=3.5, color=GREY, mec='none', label='STP model')]
    fig.legend(handles=hand, loc='upper center', ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.55, 1.0),
               handletextpad=0.3, columnspacing=1.5)
    save(fig, os.path.join(OUT, 'figS_unit_sets_nmda_stp.png'))
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
