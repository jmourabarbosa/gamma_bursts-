"""Fit the NMDA time constant to the monkeys' pooled performance against time Off, with a bootstrap.

All 15,468 trials pooled (data/behaviour_per_trial_correct_and_error.csv). Model curve P(memory held | Off fraction)
for each tau_N from the global fit with its own dips (tau_sweep.py, 20-1,500 ms), in 10 bins of the Off fraction,
interpolated in log tau (fit_tau_behaviour.model_curves).
Link: P(correct) = p0 x [P(held) + G x (1 - P(held))]: a lost memory is a guess among the 8 cues (G = 1/8), p0 is the
performance when the memory is held (one value for all trials). Maximum likelihood over tau_N and p0.
Bootstrap: sessions resampled with replacement N_BOOT times (all trials of each drawn session pooled), refitted;
distribution of the best tau_N.
Output: figures/paper_model/fig_tau_fit_pooled.{png,svg} and fig_tau_fit_pooled.csv (one row per bootstrap sample).

usage: [PM_VARIANT=os] python -m paper_model.fit_tau_pooled      (NMDA models only: nmda or os)
"""
import csv
import os

import numpy as np

from .fit_tau_behaviour import DATA, EDGES, NB, P0, bins, model_curves, wilson
from .make_figures import plt, save, letter, INK, OUT
from . import variant as V

SUFFIX = '' if V.NAME == 'nmda' else f'_{V.NAME}'

G = 1 / 8           # chance of a correct response after the memory is lost (8 cue locations)
N_BOOT = 1000
COL = '#d7301f'


def best(k, n, Hf):
    """Best (tau index, p0 index) and the log-likelihood over (tau, p0), for correct counts k and trials n per bin."""
    p = np.clip(P0[None, :, None] * (Hf + G * (1 - Hf))[:, None, :], 1e-9, 1 - 1e-9)
    ll = (k * np.log(p) + (n - k) * np.log(1 - p)).sum(-1)
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    return i, j, ll


def main():
    rows = list(csv.DictReader(open(DATA)))
    b = bins(np.array([float(r['frac_off']) for r in rows]))
    c = np.array([int(r['correct']) for r in rows])
    sess = np.array([r['session'] for r in rows])
    names = np.unique(sess)
    ks = np.array([np.bincount(b[sess == s], weights=c[sess == s], minlength=NB) for s in names])   # per session
    ns = np.array([np.bincount(b[sess == s], minlength=NB) for s in names])
    k = np.bincount(b, weights=c, minlength=NB); n = np.bincount(b, minlength=NB)
    q = k.sum() / n.sum()
    ll0 = k.sum() * np.log(q) + (n - k).sum() * np.log(1 - q)
    fine, Hf = model_curves(V.NAME)[2:]
    i, j, ll = best(k, n, Hf)
    tau, p0, prof = fine[i], P0[j], ll.max(1) - ll.max()
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(names), len(names))
        ib, jb, llb = best(ks[idx].sum(0), ns[idx].sum(0), Hf)
        boot.append((fine[ib], P0[jb], llb.max()))
    t = np.array([x[0] for x in boot])
    med, lo95, hi95 = np.percentile(t, (50, 2.5, 97.5))

    fig = plt.figure(figsize=(7.2, 2.4))
    gs = fig.add_gridspec(1, 3, wspace=0.55, left=0.08, right=0.98, top=0.85, bottom=0.2)
    cen = 900 * (EDGES[:-1] + np.minimum(EDGES[1:], 1)) / 2
    a = fig.add_subplot(gs[0])
    pk = k / n; lo, hi = wilson(k, n)
    a.errorbar(cen, pk, yerr=[pk - lo, hi - pk], fmt='o', ms=3, color=INK, elinewidth=0.6, capsize=0, label='Monkeys, pooled')
    a.plot(cen, p0 * (Hf[i] + G * (1 - Hf[i])), color=COL, label=rf'NMDA model, $\tau_N$ = {tau:.0f} ms')
    a.set_xlim(0, 900); a.set_xticks([0, 300, 600, 900]); a.set_ylim(0.5, 1.0)
    a.set_xlabel('Off time in delay (ms)'); a.set_ylabel('Fraction correct')
    a.legend(fontsize=6, loc='lower left', handlelength=1)
    letter(a, 'a', x=-0.35)
    bp = fig.add_subplot(gs[1])
    bp.plot(fine, prof, color=COL)
    bp.set_xscale('log'); bp.set_ylim(-30, 1); bp.set_xticks([30, 100, 300, 1000], ['30', '100', '300', '1000'])
    bp.xaxis.set_minor_formatter(plt.NullFormatter())
    bp.set_xlabel(r'$\tau_N$ (ms)'); bp.set_ylabel('Log-likelihood\n(rel. to best)')
    letter(bp, 'b', x=-0.35)
    h = fig.add_subplot(gs[2])
    edges = np.exp(np.linspace(np.log(20), np.log(1500), 41))
    h.hist(t, bins=edges, color=COL, alpha=0.8)
    h.set_xscale('log'); h.xaxis.set_minor_formatter(plt.NullFormatter())
    h.set_xticks([30, 100, 300, 1000], ['30', '100', '300', '1000']); h.set_xlim(20, 1500)
    for x in (lo95, hi95):
        h.axvline(x, color=COL, lw=0.6, ls=(0, (1, 1.5)))
    h.set_xlabel(r'Best $\tau_N$ (ms)'); h.set_ylabel('Bootstrap samples')
    h.axvline(tau, color=INK, lw=0.6, ls=(0, (3, 2)))
    h.set_title(f'{N_BOOT} session bootstraps: {med:.0f} ms\n(95% CI {lo95:.0f}-{hi95:.0f})', fontsize=7, pad=3)
    letter(h, 'c', x=-0.35)
    save(fig, os.path.join(OUT, f'fig_tau_fit_pooled{SUFFIX}.png'))
    with open(os.path.join(OUT, f'fig_tau_fit_pooled{SUFFIX}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample', 'tau_N_ms', 'p0', 'loglik'])
        w.writerow(['data', tau, p0, ll.max()])
        w.writerows([[s, *x] for s, x in enumerate(boot)])
    print(f"best tau_N {tau:.0f} ms, p0 {p0:.3f}, LL gain over constant {ll.max() - ll0:.1f};"
          f" bootstrap median {med:.0f} ms, 95% CI {lo95:.0f}-{hi95:.0f}")
    print('profile: ' + ' '.join(f'{x:.0f}:{y:.1f}' for x, y in zip(fine[::25], prof[::25])))


if __name__ == '__main__':
    main()
