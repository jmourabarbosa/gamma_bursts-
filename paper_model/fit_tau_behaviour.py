"""Fit the slow time constant of each model to the monkeys' performance against time Off, session by session.

Data: per trial fraction of the delay Off (500-1,400 ms) and correct/error, 25 sessions
(data/behaviour_per_trial_correct_and_error.csv, from jmourabarbosa/panichello, branch analysis-replication,
analysis/behaviour_handoff/data/per_trial_correct_and_error.csv).
Model: P(memory held | Off fraction) for each time constant, from the fitted model with its own dips
(tau_sweep.py; NMDA: tau_N, STP: tau_f), in 10 bins of the Off fraction (as Fig. 3b), interpolated in log tau.
Link: P(correct) = p0 x P(held | Off), with p0 free per session (performance when the memory is held; a lost memory
gives an error). Maximum likelihood over tau and p0 per session. Fit quality: log-likelihood gain over a constant
P(correct) (same number of free parameters minus one).
Output: figures/paper_model/fig_tau_fit.{png,svg} and fig_tau_fit.csv (one row per session and model).

usage: python -m paper_model.fit_tau_behaviour
"""
import csv
import glob
import os

import numpy as np

from .make_figures import plt, save, letter, INK, GREY, OUT

DATA = os.path.join(os.path.dirname(__file__), 'data', 'behaviour_per_trial_correct_and_error.csv')
MODELS = (('nmda', 'NMDA model', r'$\tau_N$', '#d7301f'), ('fac', 'STP model', r'$\tau_f$', '#2b83ba'))
EDGES = np.r_[np.arange(0, 1.0, 0.1), 1.0001]
NB = len(EDGES) - 1
P0 = np.linspace(0.3, 1.0, 701)
MONKEY_COL = {'A': '#1b9e77', 'H': '#d95f02', 'J': '#7570b3'}


def bins(frac):
    return np.clip(np.digitize(frac, EDGES) - 1, 0, NB - 1)


def model_curves(v):
    """tau grid, P(held | bin) (n_tau, NB), from the sweep files; fine log-tau grid by linear interpolation."""
    files = glob.glob(os.path.join('figures', 'off_vs_memory', v, 'tau_sweep', 'tau_*.npz'))
    taus = np.array(sorted(int(os.path.basename(f)[4:-4]) for f in files))
    H = []
    for t in taus:
        d = np.load(os.path.join('figures', 'off_vs_memory', v, 'tau_sweep', f'tau_{t}.npz'))
        b, m = bins(d['frac_off']), d['memory_held']
        H.append([(m[b == j].sum() + 0.5) / ((b == j).sum() + 1) for j in range(NB)])
    fine = np.exp(np.linspace(np.log(taus[0]), np.log(taus[-1]), 300))
    Hf = np.array([np.interp(np.log(fine), np.log(taus), np.array(H)[:, j]) for j in range(NB)]).T
    return taus, np.array(H), fine, Hf


def fit_session(k, n, Hf):
    """k, n: correct and trials per bin. Returns log-likelihood over (tau, p0), best indices, and the null LL."""
    p = np.clip(P0[None, :, None] * Hf[:, None, :], 1e-9, 1 - 1e-9)      # tau x p0 x bin
    ll = (k * np.log(p) + (n - k) * np.log(1 - p)).sum(-1)
    q = k.sum() / n.sum()
    ll0 = k.sum() * np.log(q) + (n - k).sum() * np.log(1 - q)
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    return ll, i, j, ll0


def wilson(k, n, z=1.96):
    p = k / np.maximum(n, 1)
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main():
    rows = list(csv.DictReader(open(DATA)))
    ses = sorted({r['session'] for r in rows})
    curves = {v: model_curves(v) for v, *_ in MODELS}
    res, prof = [], {}
    for s in ses:
        rr = [r for r in rows if r['session'] == s]
        b = bins(np.array([float(r['frac_off']) for r in rr]))
        c = np.array([int(r['correct']) for r in rr])
        k = np.array([c[b == j].sum() for j in range(NB)]); n = np.array([(b == j).sum() for j in range(NB)])
        for v, name, _, _ in MODELS:
            fine, Hf = curves[v][2], curves[v][3]
            ll, i, j, ll0 = fit_session(k, n, Hf)
            prof[s, v] = ll.max(1) - ll.max()
            res.append(dict(session=s, monkey=rr[0]['monkey'], model=name, n_trials=len(rr), n_errors=int((c == 0).sum()),
                            tau_ms=float(fine[i]), p0=float(P0[j]), ll=float(ll.max()), ll_null=float(ll0),
                            ll_gain=float(ll.max() - ll0), at_bound=bool(i in (0, len(fine) - 1)), k=k, n=n))
    # pooled over sessions (p0 per session, one tau per model)
    pooled = {}
    for v, name, _, _ in MODELS:
        tot = sum(np.max(fit_session(r['k'], r['n'], curves[v][3])[0], 1) for r in res if r['model'] == name)
        pooled[v] = curves[v][2][np.argmax(tot)]

    # figure: a example session; b its likelihood profile; c, d best tau per session; e fit quality
    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.55, left=0.08, right=0.98, top=0.93, bottom=0.11)
    nm = {r['session']: r for r in res if r['model'] == 'NMDA model'}
    ex = max(nm, key=lambda s: nm[s]['ll_gain'] if nm[s]['monkey'] == 'H' else -1)     # clearest decay, monkey H
    a = fig.add_subplot(gs[0, 0])
    r0 = nm[ex]
    cen = 900 * (EDGES[:-1] + np.minimum(EDGES[1:], 1)) / 2
    ok = r0['n'] >= 5
    lo, hi = wilson(r0['k'], r0['n'])
    a.errorbar(cen[ok], (r0['k'] / np.maximum(r0['n'], 1))[ok], yerr=[(r0['k'] / np.maximum(r0['n'], 1) - lo)[ok],
               (hi - r0['k'] / np.maximum(r0['n'], 1))[ok]], fmt='o', ms=3, color=INK, elinewidth=0.6, capsize=0, label='Session')
    for v, name, sym, col in MODELS:
        r = next(x for x in res if x['session'] == ex and x['model'] == name)
        fine, Hf = curves[v][2], curves[v][3]
        i = np.argmin(np.abs(fine - r['tau_ms']))
        a.plot(cen, r['p0'] * Hf[i], color=col, label=f"{name}, {sym} = {r['tau_ms']:.0f} ms")
    a.set_xlim(0, 900); a.set_xticks([0, 300, 600, 900]); a.set_ylim(0, 1.02)
    a.set_xlabel('Off time in delay (ms)'); a.set_ylabel('Fraction correct')
    a.set_title(f"Session {ex} (monkey {r0['monkey']}, {r0['n_trials']} trials)", fontsize=7.5, pad=3)
    a.legend(fontsize=6, loc='lower left', handlelength=1)
    letter(a, 'a', x=-0.35)
    bp = fig.add_subplot(gs[0, 1])
    for v, name, sym, col in MODELS:
        bp.plot(curves[v][2], prof[ex, v], color=col, label=name)
    bp.set_xscale('log'); bp.set_xticks([30, 100, 300, 1000], ['30', '100', '300', '1000'])
    bp.xaxis.set_minor_formatter(plt.NullFormatter()); bp.set_ylim(-20, 1); bp.set_xlabel('Time constant (ms)'); bp.set_ylabel('Log-likelihood\n(rel. to best NMDA/STP)')
    bp.set_title('Likelihood profile, same session', fontsize=7.5, pad=3)
    letter(bp, 'b', x=-0.35)
    for col_i, (v, name, sym, col) in enumerate(MODELS):
        h = fig.add_subplot(gs[1, col_i])
        rr = [r for r in res if r['model'] == name]
        fine = curves[v][2]
        edges = np.exp(np.linspace(np.log(fine[0]), np.log(fine[-1]), 16))
        h.hist([[r['tau_ms'] for r in rr if r['monkey'] == mk] for mk in 'AHJ'], bins=edges, stacked=True,
               color=[MONKEY_COL[m] for m in 'AHJ'], label=[f'Monkey {m}' for m in 'AHJ'])
        h.axvline(pooled[v], color=col, lw=1)
        h.set_xscale('log'); h.set_xlabel(f'Best {sym} (ms)'); h.set_ylabel('Sessions')
        ticks = [30, 100, 300, 1000]
        h.set_xticks(ticks, [str(t) for t in ticks]); h.xaxis.set_minor_formatter(plt.NullFormatter())
        h.set_ylim(0, h.get_ylim()[1] * 1.25)
        h.text(pooled[v] * 1.07, h.get_ylim()[1] * 0.98, f'pooled\n{pooled[v]:.0f} ms', color=col, fontsize=6.5, va='top', ha='left')
        h.set_title(name, fontsize=7.5, pad=3)
        if col_i == 0:
            h.legend(fontsize=6, loc='upper left', handlelength=0.8)
        letter(h, 'cd'[col_i], x=-0.35)
    e = fig.add_subplot(gs[0, 2])
    for s in ses:
        g = [next(r for r in res if r['session'] == s and r['model'] == name)['ll_gain'] for _, name, _, _ in MODELS]
        e.plot([0, 1], g, color=MONKEY_COL[nm[s]['monkey']], lw=0.6, alpha=0.7, marker='o', ms=2.5)
    e.set_xticks([0, 1], ['NMDA', 'STP']); e.set_xlim(-0.4, 1.4)
    e.set_ylabel('Log-likelihood gain\nover constant performance')
    e.set_title('Fit quality per session', fontsize=7.5, pad=3)
    letter(e, 'e', x=-0.35)
    fsc = fig.add_subplot(gs[1, 2])
    for s_ in ses:
        tn = next(r for r in res if r['session'] == s_ and r['model'] == 'NMDA model')['tau_ms']
        tf = next(r for r in res if r['session'] == s_ and r['model'] == 'STP model')['tau_ms']
        fsc.plot(tn, tf, 'o', ms=3, color=MONKEY_COL[nm[s_]['monkey']])
    fsc.set_xscale('log'); fsc.set_yscale('log')
    fsc.set_xticks([30, 100, 300, 1000], ['30', '100', '300', '1000']); fsc.set_yticks([30, 100, 300, 1000], ['30', '100', '300', '1000'])
    fsc.set_xlim(15, 2000); fsc.set_ylim(15, 2000); fsc.plot([15, 2000], [15, 2000], ':', color=GREY, lw=0.6)
    fsc.xaxis.set_minor_formatter(plt.NullFormatter()); fsc.yaxis.set_minor_formatter(plt.NullFormatter())
    fsc.set_xlabel(r'Best $\tau_N$ (ms)'); fsc.set_ylabel(r'Best $\tau_f$ (ms)')
    fsc.set_title('Same session, both models', fontsize=7.5, pad=3)
    letter(fsc, 'f', x=-0.35)
    save(fig, os.path.join(OUT, 'fig_tau_fit.png'))
    keys = ['session', 'monkey', 'model', 'n_trials', 'n_errors', 'tau_ms', 'p0', 'll', 'll_null', 'll_gain', 'at_bound']
    with open(os.path.join(OUT, 'fig_tau_fit.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(res)
    for v, name, _, _ in MODELS:
        rr = [r for r in res if r['model'] == name]
        t = np.array([r['tau_ms'] for r in rr]); g = np.array([r['ll_gain'] for r in rr])
        print(f"{name}: tau median {np.median(t):.0f} ms (IQR {np.percentile(t, 25):.0f}-{np.percentile(t, 75):.0f}),"
              f" pooled {pooled[v]:.0f} ms, at grid bound {sum(r['at_bound'] for r in rr)}/25;"
              f" LL gain median {np.median(g):.2f}, total {g.sum():.1f}")
        for mk in 'AHJ':
            print(f"   monkey {mk}: tau " + ' '.join(f"{r['tau_ms']:.0f}" for r in rr if r['monkey'] == mk))
    print('example session', ex)


if __name__ == '__main__':
    main()
