"""Per-session time constant from behaviour, with each session's own fitted dynamics (fit_session.py).

sweep: for one session, the NMDA model with that session's fitted parameters and read-out gain, tau_N varied over TAUS
       (N_TRIALS trials each); per trial fraction of the delay Off and memory held.
       Writes figures/off_vs_memory/nmda/session_sweep/<session>.npz.
fit:   for each session, P(correct) = p0 x P(held | Off fraction; tau_N) from its own sweep (10 bins, interpolated in
       log tau), fitted to that session's trials as fit_tau_behaviour.py. Figure figures/paper_model/fig_tau_fit_session.*
       and table fig_tau_fit_session.csv.

usage: python -m paper_model.session_tau sweep <fit dir> <session> ...    then    python -m paper_model.session_tau fit <fit dir>
"""
import csv
import glob
import json
import os
import sys

import numpy as np

from . import model as M, readout as R
from .fit import STATS
from .fit_tau_behaviour import DATA, EDGES, NB, P0, MONKEY_COL, bins, fit_session, wilson
from .make_figures import plt, save, letter, INK, GREY, OUT

TAUS = np.array([20, 30, 40, 60, 80, 100, 150, 200, 300, 500, 1000, 1500])
N_TRIALS, BATCH = 10000, 1000
SWEEP = os.path.join('figures', 'off_vs_memory', 'nmda', 'session_sweep')


def sweep(fit_dir, session):
    f = json.load(open(os.path.join(fit_dir, f'{session}.json')))
    rng = np.random.default_rng(int(session))
    frac, mem = np.zeros((len(TAUS), N_TRIALS)), np.zeros((len(TAUS), N_TRIALS), bool)
    for i, tau in enumerate(TAUS):
        p = M.params(**{**f['theta'], 'tau_N': float(tau)})
        thr, fr, me = None, [], []
        for b in range(N_TRIALS // BATCH):
            sim = M.simulate(p, BATCH, seed=int(session) % 10000 * 100 + b)
            dw, tz = R.rate_diff(sim)
            delay, valid = (tz >= 500) & (tz <= 1400), (tz >= 0) & (tz <= 1400)
            if thr is None:
                thr = R.null_threshold(dw.shape, R.RHO['all'], rng)
            state = R.label(f['gain'] * dw + R.noise(dw.shape, R.RHO['all'], rng), valid, thr)
            fr.append((state[:, delay] == -1).mean(1)); me.append(M.memory_correct(sim))
        frac[i], mem[i] = np.concatenate(fr), np.concatenate(me)
    os.makedirs(SWEEP, exist_ok=True)
    np.savez(os.path.join(SWEEP, f'{session}.npz'), taus=TAUS, frac_off=frac, memory_held=mem)
    print(f"{session}: memory " + ' '.join(f'{t}:{m:.3f}' for t, m in zip(TAUS, mem.mean(1))), flush=True)


def curves(session):
    d = np.load(os.path.join(SWEEP, f'{session}.npz'))
    H = np.array([[(m[bins(fr) == j].sum() + 0.5) / ((bins(fr) == j).sum() + 1) for j in range(NB)]
                  for fr, m in zip(d['frac_off'], d['memory_held'])])
    fine = np.exp(np.linspace(np.log(TAUS[0]), np.log(TAUS[-1]), 300))
    return fine, np.array([np.interp(np.log(fine), np.log(TAUS), H[:, j]) for j in range(NB)]).T


def fit_all(fit_dir):
    rows = list(csv.DictReader(open(DATA)))
    glob_fit = {r['session']: r for r in csv.DictReader(open(os.path.join(OUT, 'fig_tau_fit.csv'))) if r['model'] == 'NMDA model'}
    res, prof = [], {}
    for path in sorted(glob.glob(os.path.join(SWEEP, '*.npz'))):
        s = os.path.basename(path)[:-4]
        f = json.load(open(os.path.join(fit_dir, f'{s}.json')))
        rr = [r for r in rows if r['session'] == s]
        b = bins(np.array([float(r['frac_off']) for r in rr])); c = np.array([int(r['correct']) for r in rr])
        k = np.array([c[b == j].sum() for j in range(NB)]); n = np.array([(b == j).sum() for j in range(NB)])
        fine, Hf = curves(s)
        ll, i, j, ll0 = fit_session(k, n, Hf)
        prof[s] = (fine, ll.max(1) - ll.max())
        res.append(dict(session=s, monkey=rr[0]['monkey'], n_trials=len(rr), n_errors=int((c == 0).sum()),
                        tau_ms=float(fine[i]), p0=float(P0[j]), ll_gain=float(ll.max() - ll0), at_bound=bool(i in (0, len(fine) - 1)),
                        tau_global_dips_ms=float(glob_fit[s]['tau_ms']), fit_loss=f['loss'], memory_at_150=f['mem'],
                        k=k, n=n, Hf=Hf[i], fit=f, **{f'w_{key}': v for key, v in f['theta'].items()}))
    tot = sum(fit_session(r['k'], r['n'], curves(r['session'])[1])[0].max(1) for r in res)
    pooled = curves(res[0]['session'])[0][np.argmax(tot)]

    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.6, left=0.08, right=0.98, top=0.93, bottom=0.11)
    ex = '221024' if any(r['session'] == '221024' for r in res) else res[0]['session']
    r0 = next(r for r in res if r['session'] == ex)
    cen = 900 * (EDGES[:-1] + np.minimum(EDGES[1:], 1)) / 2
    a = fig.add_subplot(gs[0, 0])
    q = r0['k'] / np.maximum(r0['n'], 1); lo, hi = wilson(r0['k'], r0['n']); ok = r0['n'] >= 5
    a.errorbar(cen[ok], q[ok], yerr=[(q - lo)[ok], (hi - q)[ok]], fmt='o', ms=3, color=INK, elinewidth=0.6, capsize=0, label='Session')
    a.plot(cen, r0['p0'] * r0['Hf'], color='#d7301f', label=rf"NMDA model, $\tau_N$ = {r0['tau_ms']:.0f} ms")
    a.set_xlim(0, 900); a.set_xticks([0, 300, 600, 900]); a.set_ylim(0, 1.02)
    a.set_xlabel('Off time in delay (ms)'); a.set_ylabel('Fraction correct'); a.legend(fontsize=6, loc='lower left', handlelength=1)
    a.set_title(f"Session {ex} (monkey {r0['monkey']}), own dips", fontsize=7.5, pad=3)
    letter(a, 'a', x=-0.35)
    bp = fig.add_subplot(gs[0, 1])
    for r in res:
        fine, pr = prof[r['session']]
        bp.plot(fine, pr, color=MONKEY_COL[r['monkey']], lw=1.2 if r['session'] == ex else 0.5, alpha=1 if r['session'] == ex else 0.5)
    bp.set_xscale('log'); bp.set_ylim(-10, 0.5); bp.set_xticks([30, 100, 300, 1000], ['30', '100', '300', '1000'])
    bp.xaxis.set_minor_formatter(plt.NullFormatter())
    bp.set_xlabel(r'$\tau_N$ (ms)'); bp.set_ylabel('Log-likelihood\n(rel. to session best)')
    bp.set_title('Likelihood profiles, all sessions', fontsize=7.5, pad=3)
    letter(bp, 'b', x=-0.35)
    h = fig.add_subplot(gs[0, 2])
    edges = np.exp(np.linspace(np.log(20), np.log(1500), 16))
    h.hist([[r['tau_ms'] for r in res if r['monkey'] == m] for m in 'AHJ'], bins=edges, stacked=True,
           color=[MONKEY_COL[m] for m in 'AHJ'], label=[f'Monkey {m}' for m in 'AHJ'])
    h.axvline(pooled, color='#d7301f', lw=1)
    h.set_ylim(0, h.get_ylim()[1] * 1.25)
    h.text(pooled * 1.07, h.get_ylim()[1] * 0.98, f'pooled\n{pooled:.0f} ms', color='#d7301f', fontsize=6.5, va='top')
    h.set_xscale('log'); h.set_xticks([30, 100, 300, 1000], ['30', '100', '300', '1000']); h.xaxis.set_minor_formatter(plt.NullFormatter())
    h.set_xlabel(r'Best $\tau_N$ (ms)'); h.set_ylabel('Sessions'); h.legend(fontsize=6, loc='upper left', handlelength=0.8)
    h.set_title('Each session with its own dips', fontsize=7.5, pad=3)
    letter(h, 'c', x=-0.35)
    sc = fig.add_subplot(gs[1, 0])
    for r in res:
        sc.plot(r['tau_global_dips_ms'], r['tau_ms'], 'o', ms=3, color=MONKEY_COL[r['monkey']])
    sc.plot([15, 2000], [15, 2000], ':', color=GREY, lw=0.6)
    sc.set_xscale('log'); sc.set_yscale('log'); sc.set_xlim(15, 2000); sc.set_ylim(15, 2000)
    for ax_ in (sc.xaxis, sc.yaxis):
        ax_.set_minor_formatter(plt.NullFormatter())
    sc.set_xticks([30, 100, 300, 1000], ['30', '100', '300', '1000']); sc.set_yticks([30, 100, 300, 1000], ['30', '100', '300', '1000'])
    sc.set_xlabel(r'Best $\tau_N$, global dips (ms)'); sc.set_ylabel(r'Best $\tau_N$, own dips (ms)')
    letter(sc, 'd', x=-0.35)
    g = fig.add_subplot(gs[1, 1])
    for m in 'AHJ':
        rr = [r for r in res if r['monkey'] == m]
        g.plot([r['n_errors'] / r['n_trials'] for r in rr], [r['ll_gain'] for r in rr], 'o', ms=3, color=MONKEY_COL[m])
    g.set_xlabel('Error rate of session'); g.set_ylabel('Log-likelihood gain\nover constant performance')
    letter(g, 'e', x=-0.35)
    st = fig.add_subplot(gs[1, 2])        # per-session On/Off fit quality: model vs data, scaled
    for dk, mk, scale in STATS:
        st.plot([r['fit']['data'][dk] / scale for r in res], [r['fit']['model'][mk] / scale for r in res], '.', ms=3, label=dk)
    lim = st.get_xlim()
    st.plot(lim, lim, ':', color=GREY, lw=0.6)
    st.set_xlabel('Data (statistic / scale)'); st.set_ylabel('Model (statistic / scale)')
    st.set_title('Per-session On/Off fits', fontsize=7.5, pad=3)
    st.legend(fontsize=5, loc='upper left', handlelength=0.6, ncol=2)
    letter(st, 'f', x=-0.35)
    save(fig, os.path.join(OUT, 'fig_tau_fit_session.png'))
    keys = ['session', 'monkey', 'n_trials', 'n_errors', 'tau_ms', 'p0', 'll_gain', 'at_bound', 'tau_global_dips_ms', 'fit_loss',
            'memory_at_150'] + [k for k in res[0] if k.startswith('w_')]
    with open(os.path.join(OUT, 'fig_tau_fit_session.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys, extrasaction='ignore'); w.writeheader(); w.writerows(res)
    t = np.array([r['tau_ms'] for r in res])
    print(f"tau_N median {np.median(t):.0f} ms (IQR {np.percentile(t, 25):.0f}-{np.percentile(t, 75):.0f}), pooled {pooled:.0f} ms,"
          f" at bound {sum(r['at_bound'] for r in res)}; LL gain total {sum(r['ll_gain'] for r in res):.1f}")
    for m in 'AHJ':
        print(f"  monkey {m}: " + ' '.join(f"{r['tau_ms']:.0f}" for r in res if r['monkey'] == m))
    print('  per-session fit loss: ' + ' '.join(f"{r['fit_loss']:.2f}" for r in res))


if __name__ == '__main__':
    if sys.argv[1] == 'sweep':
        for s in sys.argv[3:]:
            sweep(sys.argv[2], s)
    else:
        fit_all(sys.argv[2])
