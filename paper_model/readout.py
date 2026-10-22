"""Read-out of the rate model as in the data analysis (Panichello et al. 2024 On/Off labelling).

The decoder's z score is modelled as z(t) = c * dr(t) + eta(t):
- dr(t): cued minus uncued pool rate in 100 ms windows, 10 ms steps;
- c: read-out gain, not fitted. For each unit set it is set so that the model's mu_on (upper mean of a
  2-component Gaussian mixture fitted to all z values 500-1,400 ms after cue onset) equals the data's;
- eta: unit-variance noise with the autocorrelation of overlapping 100 ms windows plus a slow per-trial
  offset, with the variance share measured in the data's pre-cue z (RHO).
On = runs of z > 1.96 whose summed z exceeds the 95th percentile of the same in pure noise.
Off = >= 5 consecutive points with z < 0.84. States are counted 0-1,400 ms after cue onset.
"""
import csv
import os

import numpy as np
from sklearn.mixture import GaussianMixture

from . import model as M

DATA = os.path.join(os.path.dirname(__file__), 'data')
STEP, WIN = 10, 100
SETS = {'top 5': ('top', '5'), 'top 20': ('top', '20'), 'top 50': ('top', '50'), 'rand 5': ('random', '5'),
        'rand 20': ('random', '20'), 'rand 50': ('random', '50'), 'all': ('all', 'session count')}
RHO = {'top': 0.09, 'random': 0.11, 'all': 0.35}


def load_data():
    """Per-session data values {set: {column: array over sessions}} (random draws averaged within sessions), and
    pooled state-length counts {(set, 'on'/'off'): 20 ms bins from 0 to 600 ms, then >= 600 ms}."""
    rows = list(csv.DictReader(open(os.path.join(DATA, 'readout_per_session_data_lam10000.csv'))))
    ses = {}
    for lab, (s, k) in SETS.items():
        sel = [r for r in rows if r['set'] == s and (s == 'all' or r['k'] == k)]
        ids = sorted({r['session'] for r in sel})
        ses[lab] = {c: np.array([np.mean([float(r[c]) for r in sel if r['session'] == i]) for i in ids])
                    for c in sel[0] if c not in ('session', 'set', 'k', 'draw')}
    hist = {(lab, st): np.zeros(31) for lab in SETS for st in ('on', 'off')}
    for r in csv.DictReader(open(os.path.join(DATA, 'readout_histograms_data_lam10000.csv'))):
        lab = next((lab for lab, sk in SETS.items() if sk == (r['set'], r['k'])), None)
        if lab:
            hist[lab, r['state']][int(r['bin_lo_ms']) // 20] += float(r['count'])
    for h in hist.values():
        h[29] -= h[30]      # the CSV's 580 ms bin also holds the clipped >= 600 ms states
    return ses, hist


def runs(mask):
    a = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(a == 1)[0], np.where(a == -1)[0]))


def rate_diff(sim, t_from=-100):
    """Cued minus uncued pool rate in 100 ms windows every 10 ms; window centres from t_from ms after cue onset."""
    c = np.cumsum(np.c_[np.zeros(len(sim['rc'])), sim['rc'] - sim['ru']], axis=1)
    dw = ((c[:, WIN:] - c[:, :-WIN]) / WIN)[:, ::STEP]
    tz = sim['t'][0] + np.arange(dw.shape[1]) * STEP + WIN / 2
    return dw[:, tz >= t_from], tz[tz >= t_from]


def noise(shape, rho, rng):
    k = WIN // STEP
    c = np.cumsum(np.c_[np.zeros(shape[0]), rng.standard_normal((shape[0], shape[1] + k - 1))], axis=1)
    return np.sqrt(1 - rho) * (c[:, k:] - c[:, :-k]) / np.sqrt(k) + np.sqrt(rho) * rng.standard_normal((shape[0], 1))


def null_threshold(shape, rho, rng, n_rep=3):
    masses = [z[a:b].sum() for _ in range(n_rep) for z in noise(shape, rho, rng) for a, b in runs(z > 1.96)]
    return float(np.percentile(masses, 95))


def label(z, valid, thr):
    """+1 On, -1 Off, 0 unlabelled."""
    state = np.zeros(z.shape, int)
    for k, row in enumerate(z):
        for a, b in runs((row > 1.96) & valid):
            if row[a:b].sum() > thr:
                state[k, a:b] = 1
        for a, b in runs((row < 0.84) & valid):
            if b - a >= 5:
                state[k, a:b] = -1
    return state


def mu_on(z, rng, n_max=40000):
    v = z.ravel()
    v = rng.choice(v, n_max, replace=False) if len(v) > n_max else v
    return float(GaussianMixture(2, random_state=0).fit(v[:, None]).means_.max())


def stats(state, delay):
    """On/Off counts per trial, lengths (mean, median, counts in 20 ms bins) and fractions of the delay."""
    out = {}
    for name, lab in (('on', 1), ('off', -1)):
        d = [[(b - a) * STEP for a, b in runs(s == lab)] for s in state]
        dur = np.concatenate(d + [[]])
        out.update({f'n_{name}': np.mean([len(x) for x in d]), f'frac_{name}': (state[:, delay] == lab).mean(),
                    f'{name}_mean': dur.mean() if len(dur) else 0.0, f'{name}_median': np.median(dur) if len(dur) else 0.0,
                    f'h_{name}': np.bincount(np.minimum(dur.astype(int) // 20, 30), minlength=31)})
    return out


def evaluate(p, ses, n_trials=1056, seed=0, sets=tuple(SETS)):
    """Simulate the model and read it out for each unit set: memory correct, and per set the gain, z, state labels
    and their statistics."""
    sim = M.simulate(p, n_trials, seed=seed * 100 + 1)
    dw, tz = rate_diff(sim)
    rng = np.random.default_rng(seed * 100 + 2)
    delay, valid = (tz >= 500) & (tz <= 1400), (tz >= 0) & (tz <= 1400)
    grid = 10 ** np.linspace(-2.6, -0.3, 10)
    out = dict(mem=float(M.memory_correct(sim).mean()), sim=sim, tz=tz, sets={})
    cache = {}
    for lab in sets:
        rho = RHO[SETS[lab][0]]
        if rho not in cache:     # sets of the same kind share the noise draw
            eta = noise(dw.shape, rho, rng)
            mus = np.maximum.accumulate([mu_on((c * dw + eta)[:, delay], rng) for c in grid])
            cache[rho] = eta, mus, null_threshold(dw.shape, rho, rng)
        eta, mus, thr = cache[rho]
        target = ses[lab]['mu_on'].mean()
        c = float(10 ** np.interp(target, mus, np.log10(grid))) if mus[-1] > target else float(grid[-1])
        z = c * dw + eta
        state = label(z, valid, thr)
        out['sets'][lab] = dict(gain=c, z=z, state=state, **stats(state, delay))
    return out
