"""Control: how many On/Off states does the read-out noise alone produce?

The fitted NMDA model (model.FIT) with a constant common input (no dips), read out exactly as in Figure 2 for each of
the 7 unit sets (same noise model, cluster-mass threshold and labelling), in two ways:
- fixed gain: each set's read-out gain from the fitted model (the identical read-out applied to a network that never
  falls silent);
- recalibrated gain: the gain set again from the data's mu_on (readout.evaluate on the no-dip model).
Output: figures/paper_model/figS_control_no_dips.{png,svg} and figS_control_no_dips.json.

usage: python -m paper_model.control_no_dips
"""
import json
import os

import numpy as np

from . import model as M, readout as R
from .make_figures import plt, save, letter, INK, GREY, LIGHT, C_ON, C_OFF, OUT, FIT_SET

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


def main():
    R.M = M
    ses, _ = R.load_data()
    fit = R.evaluate(M.params(), ses, n_trials=N_TRIALS, seed=1)
    gains = {lab: r['gain'] for lab, r in fit['sets'].items()}
    sim0 = M.simulate(M.params(dip_depth=0.0), N_TRIALS, seed=101)
    fixed = fixed_gain(sim0, gains)
    recal = R.evaluate(M.params(dip_depth=0.0), ses, n_trials=N_TRIALS, seed=1)
    conds = (('Fitted model (dips)', {l: fit['sets'][l] for l in R.SETS}, '#d7301f'),
             ('No dips, same gain', fixed, '#2b83ba'),
             ('No dips, gain recalibrated', {l: recal['sets'][l] for l in R.SETS}, '#7b3294'))

    x = np.arange(len(R.SETS))
    fig, axs = plt.subplots(2, 3, figsize=(6.6, 4.3), gridspec_kw=dict(hspace=0.8, wspace=0.6, left=0.13, right=0.98, top=0.84, bottom=0.12))
    j = list(R.SETS).index(FIT_SET)
    for k, (a, (dk, mk, label, top)) in enumerate(zip(axs.ravel(), PANELS)):
        a.axvspan(j - 0.45, j + 0.45, color=LIGHT, lw=0, zorder=0)
        v = [ses[lab][dk] for lab in R.SETS]
        a.errorbar(x - 0.15, [s.mean() for s in v], yerr=[2 * s.std(ddof=1) / np.sqrt(len(s)) for s in v], fmt='o', ms=3,
                   color=INK, elinewidth=0.6, capsize=0, zorder=3)
        for m, (name, res, col) in enumerate(conds):
            a.plot(x + 0.05 + 0.12 * m, [res[lab][mk] for lab in R.SETS], 'o', ms=2.6, color=col, zorder=2)
        a.axvline(2.5, color=GREY, ls=(0, (3, 2)), lw=0.5); a.axvline(5.5, color=GREY, ls=(0, (3, 2)), lw=0.5)
        a.set_xlim(-0.6, len(R.SETS) - 0.4)
        a.set_xticks(x); a.set_xticklabels(R.SETS, rotation=60, ha='right', rotation_mode='anchor', fontsize=7)
        a.set_ylim(0, top or np.ceil(a.get_ylim()[1]))
        if top == 1:
            a.set_yticks([0, 0.5, 1], ['0', '', '1'])
        a.set_ylabel(label)
        letter(a, 'abcdef'[k], x=-0.45, y=1.02)
    items = [('Data (mean ± 2 SEM)', INK)] + [(n, c) for n, _, c in conds]
    for n, (txt, c) in enumerate(items):
        fig.text(0.13 + 0.42 * (n % 2), 0.975 - 0.035 * (n // 2), txt, color=c, fontsize=7, va='center')
    save(fig, os.path.join(OUT, 'figS_control_no_dips.png'))
    keys = ('gain', 'n_off', 'off_median', 'off_mean', 'frac_off', 'n_on', 'on_median', 'on_mean', 'frac_on')
    out = {name: {lab: {k: float(res[lab][k]) for k in keys if k in res[lab]} for lab in R.SETS} for name, res, _ in conds}
    out['memory'] = dict(fitted=fit['mem'], no_dips=recal['mem'])
    out['data'] = {lab: {mk: float(ses[lab][dk].mean()) for dk, mk, _, _ in PANELS} for lab in R.SETS}
    json.dump(out, open(os.path.join(OUT, 'figS_control_no_dips.json'), 'w'), indent=1)
    for name, res, _ in conds:
        print(name)
        for lab in R.SETS:
            r = res[lab]
            print(f"  {lab:8s} gain {r.get('gain', gains[lab]):.3f} | Off {r['n_off']:.2f}/trial, median {r['off_median']:.0f} ms, frac {r['frac_off']:.2f}"
                  f" | On {r['n_on']:.2f}/trial, median {r['on_median']:.0f} ms, frac {r['frac_on']:.2f}")
    print('memory', out['memory'])


if __name__ == '__main__':
    main()
