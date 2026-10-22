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
from numba import njit
from sklearn.mixture import GaussianMixture

from . import model as M

DATA = os.path.join(os.path.dirname(__file__), 'data')
STEP, WIN = 10, 100
SETS = (('top', '5'), ('top', '20'), ('top', '50'), ('random', '5'), ('random', '20'), ('random', '50'),
        ('all', 'session count'))
LABELS = ('top 5', 'top 20', 'top 50', 'rand 5', 'rand 20', 'rand 50', 'all')
RHO = {'top': 0.09, 'random': 0.11, 'all': 0.35}
# (data column, model key, loss scale)
STATS = (('off_per_trial', 'n_off', 0.3), ('off_mean_ms', 'off_mean', 20.0), ('frac_off', 'frac_off', 0.05),
         ('on_per_trial', 'n_on', 0.3), ('on_mean_ms', 'on_mean', 20.0), ('frac_on', 'frac_on', 0.05))
EDGES = np.arange(40, 620, 20)     # histogram bins (ms), plus a >= 600 ms bin


def load_data():
    """Data summary per unit set (dict of floats, incl. *_sem) and On/Off length histograms (probabilities)."""
    rows = {(r['set'], r['k']): r for r in csv.DictReader(open(os.path.join(DATA, 'readout_summary_data_lam10000.csv')))}
    summ = {s: {k: float(v) for k, v in rows[s].items() if k not in ('set', 'k') and v != ''} for s in SETS}
    H = {}
    for r in csv.DictReader(open(os.path.join(DATA, 'readout_histograms_data_lam10000.csv'))):
        H.setdefault((r['set'], r['k'], r['state']), []).append((float(r['bin_lo_ms']), float(r['count'])))
    hist = {}
    for (s, k, state), v in H.items():
        lo = np.array([a for a, _ in v]); c = np.array([b for _, b in v])
        over = c[lo == 600].sum()
        c = np.where(lo == 580, c - over, c)   # the 580 ms bin of the CSV also holds the clipped >= 600 ms states
        inside = np.array([c[lo == e].sum() for e in EDGES[:-1]])
        hist.setdefault((s, k), {})[state] = np.r_[inside, over] / (inside.sum() + over)
    return summ, hist


def load_sessions():
    """Per-session data values for every unit set: {set: {column: array over sessions}}; random draws are averaged
    within each session, as in the summary."""
    rows = list(csv.DictReader(open(os.path.join(DATA, 'readout_per_session_data_lam10000.csv'))))
    out = {}
    for st in SETS:
        sel = [r for r in rows if r['set'] == st[0] and (st[0] == 'all' or r['k'] == st[1])]
        ses = sorted({r['session'] for r in sel})
        out[st] = {c: np.array([np.mean([float(r[c]) for r in sel if r['session'] == x]) for x in ses])
                   for c in sel[0] if c not in ('session', 'set', 'k', 'draw')}
    return out


def data_counts(st, state):
    """Data state-length counts of one unit set in 20 ms bins from 0 to 600 ms, and the total number of states."""
    v = [(float(r['bin_lo_ms']), float(r['count'])) for r in csv.DictReader(open(os.path.join(DATA, 'readout_histograms_data_lam10000.csv')))
         if (r['set'], r['k'], r['state']) == (*st, state)]
    lo = np.array([a for a, _ in v]); c = np.array([b for _, b in v])
    over = c[lo == 600].sum()
    c = np.where(lo == 580, c - over, c)
    inside = np.array([c[lo == e].sum() for e in np.arange(0, 600, 20)])
    return inside, inside.sum() + over


def rate_diff(sim, t_from=-100):
    """Cued minus uncued pool rate in 100 ms windows every 10 ms, centres from t_from ms relative to cue onset."""
    d = np.where(sim['cued'][:, None] == 0, sim['rA'] - sim['rB'], sim['rB'] - sim['rA'])
    c = np.cumsum(np.c_[np.zeros(len(d)), d], axis=1)
    dw = ((c[:, WIN:] - c[:, :-WIN]) / WIN)[:, ::STEP]
    tc = np.arange(dw.shape[1]) * STEP + WIN / 2 - sim['t_cue']
    keep = tc >= t_from
    return dw[:, keep], tc[keep]


def window_noise(shape, rng):
    k = WIN // STEP
    e = rng.standard_normal((shape[0], shape[1] + k - 1))
    c = np.cumsum(np.c_[np.zeros(shape[0]), e], axis=1)
    return (c[:, k:] - c[:, :-k]) / np.sqrt(k)


def noise(shape, rho, rng):
    return np.sqrt(1 - rho) * window_noise(shape, rng) + np.sqrt(rho) * rng.standard_normal((shape[0], 1))


def runs(mask):
    a = np.r_[False, mask, False].astype(int)
    return list(zip(np.where(np.diff(a) == 1)[0], np.where(np.diff(a) == -1)[0]))


def null_threshold(shape, rho, rng, n_rep=3):
    masses = [z[a:b].sum() for _ in range(n_rep) for z in noise(shape, rho, rng) for a, b in runs(z > 1.96)]
    return float(np.percentile(masses, 95))


@njit(cache=True)
def _label(z, valid, thr, min_off):
    n_tr, n_b = z.shape
    state = np.zeros((n_tr, n_b), np.int64)
    for k in range(n_tr):
        i = 0
        while i < n_b:
            if valid[i] and z[k, i] > 1.96:
                j = i; m = 0.0
                while j < n_b and valid[j] and z[k, j] > 1.96:
                    m += z[k, j]; j += 1
                if m > thr:
                    state[k, i:j] = 1
                i = j
            else:
                i += 1
        i = 0
        while i < n_b:
            if valid[i] and z[k, i] < 0.84:
                j = i
                while j < n_b and valid[j] and z[k, j] < 0.84:
                    j += 1
                if j - i >= min_off:
                    state[k, i:j] = -1
                i = j
            else:
                i += 1
    return state


def label(z, valid, thr, min_off=5):
    """+1 On, -1 Off, 0 unlabelled."""
    return _label(np.ascontiguousarray(z), np.asarray(valid, np.bool_), float(thr), int(min_off))


def mu_on(z, rng, n_max=40000):
    v = z.ravel()
    if len(v) > n_max:
        v = rng.choice(v, n_max, replace=False)
    return float(GaussianMixture(2, random_state=0).fit(v[:, None]).means_.max())


def hist(dur):
    dur = np.asarray(dur, float)
    h, _ = np.histogram(dur, EDGES)
    return np.r_[h, (dur >= EDGES[-1]).sum()] / max(1, (dur >= EDGES[0]).sum())


def evaluate(theta=None, n_trials=1056, seed=0, summ=None, keep=False, sets=SETS):
    """Simulate the model and read it out for every unit set. Returns memory correct and, per set, the gain,
    accuracy, On/Off counts, mean lengths, fractions of time and length histograms."""
    summ = summ or load_data()[0]
    p = M.params(**(theta or {}))
    sim = M.simulate(p, np.tile([0, 1], n_trials // 2), seed=seed * 100 + 1)
    dw, tz = rate_diff(sim)
    rng = np.random.default_rng(seed * 100 + 2)
    delay = (tz >= 500) & (tz <= 1400)
    valid = (tz >= 0) & (tz <= 1400)
    grid = 10 ** np.linspace(-2.6, -0.3, 10)
    out = dict(mem=float(M.memory_correct(sim).mean()), sets={})
    if keep:
        out['sim'], out['tz'], out['dw'] = sim, tz, dw
    cache = {}
    for st in sets:
        rho = RHO[st[0]]
        if rho not in cache:
            eta = noise(dw.shape, rho, rng)
            mus = np.maximum.accumulate([mu_on((c * dw + eta)[:, delay], rng) for c in grid])
            cache[rho] = (eta, mus, null_threshold(dw.shape, rho, rng))
        eta, mus, thr = cache[rho]
        target = summ[st]['mu_on']
        c = float(10 ** np.interp(target, mus, np.log10(grid))) if mus[-1] > target else float(grid[-1])
        z = c * dw + eta
        state = label(z, valid, thr)
        on = [[(b - a) * STEP for a, b in runs(s == 1)] for s in state]
        off = [[(b - a) * STEP for a, b in runs(s == -1)] for s in state]
        on_d, off_d = np.concatenate(on + [[]]), np.concatenate(off + [[]])
        r = dict(gain=c, mu_on=mu_on(z[:, delay], np.random.default_rng(seed)),   # separate rng: check of the anchoring
                 accuracy=float((z[:, delay] > 0).mean()),
                 n_off=float(np.mean([len(x) for x in off])), n_on=float(np.mean([len(x) for x in on])),
                 off_mean=float(off_d.mean()) if len(off_d) else 0.0, on_mean=float(on_d.mean()) if len(on_d) else 0.0,
                 frac_off=float((state[:, delay] == -1).mean()), frac_on=float((state[:, delay] == 1).mean()),
                 h_off=hist(off_d), h_on=hist(on_d))
        if keep:
            r.update(z=z, state=state)
        out['sets'][st] = r
    return out
