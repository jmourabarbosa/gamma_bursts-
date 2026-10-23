"""Figure 2 layout with three models: On/Off statistics of the 7 unit sets, data vs the NMDA model, the STP model with
facilitation only, and the STP model with facilitation and depression. All fitted to the all-units read-out only;
the other six sets are predicted by the read-out (gain from the data's mu_on, noise share), as in Figure 2.

usage: python -m paper_model.fig_unit_sets_compare     (writes figures/paper_model/figS_unit_sets_models.{png,svg,csv})
"""
import csv
import os

import numpy as np

from . import model as M, model_ours_stp as H, readout as R
from .fit import STATS
from .make_figures import plt, save, letter, INK, GREY, LIGHT, OUT, FIT_SET

MODELS = (('NMDA model', M, M.params(), '#d7301f'),
          ('STP, facilitation', H, H.params(**H.FIT_FAC), '#2b83ba'),
          ('STP, facilitation + depression', H, H.params(**H.FIT_DEP), '#7b3294'))
PANELS = (('off_per_trial', 'n_off', 'Off states per trial', None),
          ('off_median_ms', 'off_median', 'Off length, median (ms)', 250),
          ('frac_off', 'frac_off', 'Fraction of delay Off', 1),
          ('on_per_trial', 'n_on', 'On states per trial', None),
          ('on_median_ms', 'on_median', 'On length, median (ms)', 250),
          ('frac_on', 'frac_on', 'Fraction of delay On', 1))


def loss_per_set(r, d, h, lab):
    st = np.mean([((r[mk] - d[dk].mean()) / sc) ** 2 for dk, mk, sc in STATS])
    return st + sum(50 * np.sum((np.sqrt(r['h_' + s][2:] / max(r['h_' + s][2:].sum(), 1)) -
                                 np.sqrt(h[lab, s][2:] / h[lab, s][2:].sum())) ** 2) for s in ('on', 'off'))


def main():
    ses, hist = R.load_data()
    outs = []
    for name, mod, p, col in MODELS:
        R.M = mod
        outs.append(R.evaluate(p, ses, n_trials=2112, seed=1))
    R.M = M
    x = np.arange(len(R.SETS))
    jit = np.random.default_rng(0).uniform(-0.18, 0.18, 25)
    fig, axs = plt.subplots(2, 3, figsize=(6.6, 4.3), gridspec_kw=dict(hspace=0.8, wspace=0.6, left=0.13, right=0.98,
                                                                       top=0.84, bottom=0.12))
    j = list(R.SETS).index(FIT_SET)
    for k, (a, (dk, mk, label, top)) in enumerate(zip(axs.ravel(), PANELS)):
        a.axvspan(j - 0.45, j + 0.45, color=LIGHT, lw=0, zorder=0)
        for i, lab in enumerate(R.SETS):
            v = ses[lab][dk]
            a.plot(i - 0.12 + jit[:len(v)] * 0.6, v, '.', ms=2, color=GREY, alpha=0.4, mew=0, zorder=1)
        v = [ses[lab][dk] for lab in R.SETS]
        a.errorbar(x - 0.12, [s.mean() for s in v], yerr=[2 * s.std(ddof=1) / np.sqrt(len(s)) for s in v], fmt='o',
                   ms=3, color=INK, elinewidth=0.6, capsize=0, zorder=3)
        for m, ((name, _, _, col), out) in enumerate(zip(MODELS, outs)):
            a.plot(x + 0.1 + 0.11 * m, [out['sets'][lab][mk] for lab in R.SETS], 'o', ms=2.6, color=col, zorder=2)
        a.axvline(2.5, color=GREY, ls=(0, (3, 2)), lw=0.5); a.axvline(5.5, color=GREY, ls=(0, (3, 2)), lw=0.5)
        a.set_xlim(-0.6, len(R.SETS) - 0.4)
        a.set_xticks(x); a.set_xticklabels(R.SETS, rotation=60, ha='right', rotation_mode='anchor', fontsize=7)
        a.set_ylim(0, top or np.ceil(a.get_ylim()[1]))
        if top == 1:
            a.set_yticks([0, 0.5, 1], ['0', '', '1'])
        a.set_ylabel(label)
        letter(a, 'abcdef'[k], x=-0.45, y=1.02)
        a.text(j, 0.97, 'Fitted', transform=a.get_xaxis_transform(), ha='center', va='top', rotation=90, fontsize=7)
    items = [('Data (mean ± 2 SEM)', INK)] + [(f"{n} ({sum(loss_per_set(o['sets'][lab], ses[lab], hist, lab) for lab in R.SETS):.0f})", c)
                                         for (n, _, _, c), o in zip(MODELS, outs)]
    for n, (txt, c) in enumerate(items):
        fig.text(0.13 + 0.42 * (n % 2), 0.975 - 0.035 * (n // 2), txt, color=c, fontsize=7, va='center')
    fig.text(0.13, 0.9, 'Loss summed over the 7 sets in brackets; models fitted to "all" only', fontsize=6.5, color=GREY)
    save(fig, os.path.join(OUT, 'figS_unit_sets_models.png'))
    rows = []
    for (name, _, _, _), out in zip(MODELS, outs):
        for lab in R.SETS:
            r = out['sets'][lab]
            rows.append(dict(model=name, set=lab, memory=out['mem'], loss=loss_per_set(r, ses[lab], hist, lab),
                             **{mk: r[mk] for _, mk, _, _ in PANELS}))
    for lab in R.SETS:
        rows.append(dict(model='data', set=lab, **{mk: ses[lab][dk].mean() for dk, mk, _, _ in PANELS}))
    with open(os.path.join(OUT, 'figS_unit_sets_models.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, ['model', 'set', 'memory', 'loss'] + [mk for _, mk, _, _ in PANELS])
        w.writeheader()
        w.writerows(rows)
    for (name, _, _, _), out in zip(MODELS, outs):
        print(f"{name:32s} memory {out['mem']:.3f} | " + ' '.join(
            f"{lab} {loss_per_set(out['sets'][lab], ses[lab], hist, lab):.1f}" for lab in R.SETS))


if __name__ == '__main__':
    main()
