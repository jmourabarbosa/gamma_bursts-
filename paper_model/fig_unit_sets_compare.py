"""Figure 2 layout with three models: On/Off statistics of the 7 unit sets, data vs the NMDA model, the STP model with
facilitation only, and the STP model with facilitation and depression. All fitted to the all-units read-out only;
the other six sets are predicted by the read-out (gain from the data's mu_on, noise share), as in Figure 2.

g, h: fit loss (six statistics plus the length-histogram term, as fit.py) per set and summed over the 7 sets.

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
    fig = plt.figure(figsize=(6.6, 6.3))
    gs = fig.add_gridspec(3, 3, hspace=0.95, wspace=0.6, left=0.13, right=0.98, top=0.88, bottom=0.08, height_ratios=[1, 1, 0.9])
    axs = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
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
        fig.text(0.13 + 0.42 * (n % 2), 0.98 - 0.025 * (n // 2), txt, color=c, fontsize=7, va='center')
    fig.text(0.13, 0.925, 'Loss summed over the 7 sets in brackets; models fitted to "all" only', fontsize=6.5, color=GREY)
    losses = [{lab: loss_per_set(o['sets'][lab], ses[lab], hist, lab) for lab in R.SETS} for o in outs]
    g = fig.add_subplot(gs[2, :2])
    w = 0.26
    for m, ((name, _, _, col), ls) in enumerate(zip(MODELS, losses)):
        g.bar(x + (m - 1) * w, [ls[lab] for lab in R.SETS], width=w, color=col, lw=0)
    g.set_xticks(x); g.set_xticklabels(R.SETS, rotation=60, ha='right', rotation_mode='anchor', fontsize=7)
    g.axvline(2.5, color=GREY, ls=(0, (3, 2)), lw=0.5); g.axvline(5.5, color=GREY, ls=(0, (3, 2)), lw=0.5)
    g.set_ylabel('Fit loss'); g.set_xlim(-0.6, len(R.SETS) - 0.4)
    letter(g, 'g', x=-0.2, y=1.02)
    h = fig.add_subplot(gs[2, 2])
    tot = [sum(ls.values()) for ls in losses]
    h.bar(range(len(MODELS)), tot, color=[c for _, _, _, c in MODELS], lw=0)
    for i, v in enumerate(tot):
        h.text(i, v + max(tot) * 0.02, f'{v:.0f}', ha='center', va='bottom', fontsize=7)
    h.set_ylim(0, max(tot) * 1.15)
    h.set_xticks(range(len(MODELS)), ['NMDA', 'STP\nfacil.', 'STP facil.\n+ depr.'], fontsize=6)
    h.set_ylabel('Fit loss, sum over\n7 populations')
    letter(h, 'h', x=-0.45, y=1.02)
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
