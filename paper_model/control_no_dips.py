"""Control: how many On/Off states does the read-out noise alone produce?

The fitted NMDA model (model.FIT) with a constant common input (no dips), read out exactly as in Figure 2 for each of
the 7 unit sets (same noise model, cluster-mass threshold and labelling), in two ways:
- fixed gain: each set's read-out gain from the fitted model (the identical read-out applied to a network that never
  falls silent);
- recalibrated gain: the gain set again from the data's mu_on (readout.evaluate on the no-dip model).
(With dips, the two coincide: the fitted model's gain is the one calibrated to the data.)
g, h: fit loss (fit.py: six statistics plus the length-histogram term) per population and summed over the 7.
Output: figures/paper_model/figS_control_no_dips.{png,svg} and figS_control_no_dips.json.

usage: python -m paper_model.control_no_dips
"""
import json
import os

import numpy as np

from . import model as M, readout as R
from .make_figures import plt, save, letter, INK, GREY, LIGHT, C_ON, C_OFF, OUT, FIT_SET
from .fig_unit_sets_compare import loss_per_set

N_TRIALS = 2112
PANELS = (('off_per_trial', 'n_off', 'Off states per trial', None), ('off_median_ms', 'off_median', 'Off length, median (ms)', 600),
          ('frac_off', 'frac_off', 'Fraction of delay Off', 1), ('on_per_trial', 'n_on', 'On states per trial', None),
          ('on_median_ms', 'on_median', 'On length, median (ms)', 1500), ('frac_on', 'frac_on', 'Fraction of delay On', 1))


def fixed_gain(sim, gains, seed=2):
    """Read out sim with the given gain per set, as readout.evaluate does otherwise."""
    dw, tz = R.rate_diff(sim)
    rng = np.random.default_rng(seed * 100 + 2)
    delay, valid = (tz >= 500) & (tz <= 1400), (tz >= 0) & (tz <= 1400)
    cache, res = {}, {}
    for lab in R.SETS:
        rho = R.RHO[R.SETS[lab][0]]
        if rho not in cache:
            cache[rho] = R.noise(dw.shape, rho, rng), R.null_threshold(dw.shape, rho, rng)
        eta, thr = cache[rho]
        res[lab] = R.stats(R.label(gains[lab] * dw + eta, valid, thr), delay)
    return res


def plain_log(ax, bottom):
    """Log y axis from `bottom`, with plain tick labels at 1, 3, 10, 30, ..."""
    ax.set_ylim(bottom=bottom)
    top = ax.get_ylim()[1]
    ticks = [t for t in (1, 3, 10, 30, 100, 300, 1000, 3000, 10000) if bottom <= t <= top]
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())


def main():
    R.M = M
    ses, hist = R.load_data()
    fit = R.evaluate(M.params(), ses, n_trials=N_TRIALS, seed=1)
    gains = {lab: r['gain'] for lab, r in fit['sets'].items()}
    sim0 = M.simulate(M.params(dip_depth=0.0), N_TRIALS, seed=101)
    fixed = fixed_gain(sim0, gains)
    recal = R.evaluate(M.params(dip_depth=0.0), ses, n_trials=N_TRIALS, seed=1)
    conds = (('Fitted model (dips)', {l: fit['sets'][l] for l in R.SETS}, 'o', '#4d4d4d'),
             ('No dips, same gain', fixed, '^', '#8c8c8c'),
             ('No dips, recalibrated', {l: recal['sets'][l] for l in R.SETS}, 'D', '#c8c8c8'))
    PCOL = (C_OFF, C_OFF, C_OFF, C_ON, C_ON, C_ON)

    x = np.arange(len(R.SETS))
    jit = np.random.default_rng(0).uniform(-0.18, 0.18, 25)
    fig = plt.figure(figsize=(6.4, 6.2))
    gs = fig.add_gridspec(3, 3, hspace=0.95, wspace=0.6, left=0.11, right=0.98, top=0.95, bottom=0.08, height_ratios=[1, 1, 0.9])
    axs = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
    j = list(R.SETS).index(FIT_SET)
    for k, (a, (dk, mk, label, top)) in enumerate(zip(axs.ravel(), PANELS)):
        col = PCOL[k]
        a.axvspan(j - 0.45, j + 0.55, color=LIGHT, lw=0, zorder=0)
        for i, lab in enumerate(R.SETS):
            v = ses[lab][dk]
            a.plot(i + jit[:len(v)], v, '.', ms=2.5, color=GREY, alpha=0.5, mew=0, zorder=1)
        v = [ses[lab][dk] for lab in R.SETS]
        a.errorbar(x, [s.mean() for s in v], yerr=[2 * s.std(ddof=1) / np.sqrt(len(s)) for s in v], fmt='o', ms=3,
                   color=INK, elinewidth=0.6, capsize=0, zorder=3)
        for m, (name, res, mk_, _) in enumerate(conds):
            a.plot(x + 0.2 + 0.14 * m, [res[lab][mk] for lab in R.SETS], mk_, ms=3 if mk_ == 'o' else 2.6, color=col,
                   mew=0, zorder=2)
        a.axvline(2.5, color=GREY, ls=(0, (3, 2)), lw=0.5); a.axvline(5.5, color=GREY, ls=(0, (3, 2)), lw=0.5)
        a.set_xlim(-0.6, len(R.SETS) - 0.4)
        a.set_xticks(x); a.set_xticklabels(R.SETS, rotation=60, ha='right', rotation_mode='anchor', fontsize=7)
        a.set_ylim(0, top or np.ceil(a.get_ylim()[1]))
        if top == 1:
            a.set_yticks([0, 0.5, 1], ['0', '', '1'])
        a.set_ylabel(label)
        letter(a, 'abcdef'[k], x=-0.5, y=1.02)
        a.text(j + 0.05, 0.97, 'Fitted', transform=a.get_xaxis_transform(), ha='center', va='top', rotation=90, fontsize=7)
        a.text(2.5, 1.1, 'Predicted', transform=a.get_xaxis_transform(), ha='center', va='bottom', fontsize=7)
        a.plot([0, 0, 5, 5], [1.06, 1.09, 1.09, 1.06], color=INK, lw=0.5, transform=a.get_xaxis_transform(), clip_on=False)
    short = {'Fitted model (dips)': 'Dips (fitted model)', 'No dips, same gain': 'No dips, same gain',
             'No dips, recalibrated': 'No dips, recalibrated'}
    for a, col, x0, y0 in ((axs[0, 1], C_OFF, 0.06, 0.95), (axs[1, 1], C_ON, 0.06, 0.8)):   # legends in empty areas
        a.text(x0, y0, 'Data', color=INK, transform=a.transAxes, va='center', fontsize=6.5)
        for n, (name, _, mk_, _) in enumerate(conds):
            y = y0 - (n + 1) * 0.09
            a.plot([x0 + 0.025], [y], mk_, ms=3 if mk_ == 'o' else 2.6, color=col, mew=0, transform=a.transAxes, clip_on=False)
            a.text(x0 + 0.06, y, short[name], color=col, transform=a.transAxes, va='center', fontsize=6.5)
        a.text(x0, y0 - 4 * 0.09, 'Sessions', color=GREY, transform=a.transAxes, va='center', fontsize=6.5)
    losses = {name: {lab: float(loss_per_set(res[lab], ses[lab], hist, lab)) for lab in R.SETS} for name, res, _, _ in conds}
    g = fig.add_subplot(gs[2, :2])
    w = 0.26
    for m, (name, _, _, bc) in enumerate(conds):
        g.bar(x + (m - 1) * w, [losses[name][lab] for lab in R.SETS], width=w, color=bc, lw=0)
    g.set_yscale('log'); g.set_xticks(x); g.set_xticklabels(R.SETS, rotation=60, ha='right', rotation_mode='anchor', fontsize=7)
    g.axvline(2.5, color=GREY, ls=(0, (3, 2)), lw=0.5); g.axvline(5.5, color=GREY, ls=(0, (3, 2)), lw=0.5)
    g.set_ylabel('Fit loss (log)'); g.set_xlim(-0.6, len(R.SETS) - 0.4)
    g.set_ylim(1, 30000); plain_log(g, 1)
    for n, (name, _, _, bc) in enumerate(conds):
        g.text(0.02, 0.97 - n * 0.11, name, color=bc if n < 2 else '#a6a6a6', transform=g.transAxes, ha='left', va='top', fontsize=7)
    letter(g, 'g', x=-0.2, y=1.02)
    h = fig.add_subplot(gs[2, 2])
    tot = [sum(losses[name].values()) for name, _, _, _ in conds]
    h.bar(range(3), tot, color=[bc for _, _, _, bc in conds], lw=0, width=0.6)
    for i, v in enumerate(tot):
        h.text(i, v * 1.15, f'{v:.0f}', ha='center', va='bottom', fontsize=7)
    h.set_yscale('log'); h.set_ylim(10, max(tot) * 6)
    plain_log(h, 10)
    h.set_xticks(range(3), ['Dips', 'No dips\nsame gain', 'No dips\nrecalib.'], fontsize=6)
    h.set_ylabel('Fit loss, sum over\n7 populations (log)')
    letter(h, 'h', x=-0.45, y=1.02)
    save(fig, os.path.join(OUT, 'figS_control_no_dips.png'))
    keys = ('gain', 'n_off', 'off_median', 'off_mean', 'frac_off', 'n_on', 'on_median', 'on_mean', 'frac_on')
    out = {name: {lab: {k: float(res[lab][k]) for k in keys if k in res[lab]} for lab in R.SETS} for name, res, _, _ in conds}
    out['loss'] = {name: dict(**losses[name], total=sum(losses[name].values())) for name, _, _, _ in conds}
    out['memory'] = dict(fitted=fit['mem'], no_dips=recal['mem'])
    out['data'] = {lab: {mk: float(ses[lab][dk].mean()) for dk, mk, _, _ in PANELS} for lab in R.SETS}
    json.dump(out, open(os.path.join(OUT, 'figS_control_no_dips.json'), 'w'), indent=1)
    for name, res, _, _ in conds:
        print(name)
        for lab in R.SETS:
            r = res[lab]
            print(f"  {lab:8s} gain {r.get('gain', gains[lab]):.3f} | Off {r['n_off']:.2f}/trial, median {r['off_median']:.0f} ms, frac {r['frac_off']:.2f}"
                  f" | On {r['n_on']:.2f}/trial, median {r['on_median']:.0f} ms, frac {r['frac_on']:.2f}")
    print('memory', out['memory'])
    for name, _, _, _ in conds:
        print(f"loss {name}: total {out['loss'][name]['total']:.0f} | " + ' '.join(f"{lab} {losses[name][lab]:.0f}" for lab in R.SETS))


if __name__ == '__main__':
    main()
