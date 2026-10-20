"""Refit of the rate model to the recorded read-out table at lambda = 10,000 (several unit sets at once).

For each unit set (top 5/20/50, random 5/20/50, all units) the read-out gain is not fitted: it is chosen
so that the model's mu_on (upper mean of a 2-component Gaussian mixture fitted, as in the data analysis, to
all z values 500-1,400 ms after cue onset) equals the data's. Only the dynamics are fitted, to every set's
On/Off statistics plus the top-20 On and Off length histograms, with memory >= MEM_MIN.

z noise follows the data's noise check (lambda = 10,000, pre-cue): unit variance, the overlap correlation
of 100 ms windows, plus a slow per-trial offset with the measured plateau of the autocorrelation
(about 0.1 for top and random sets, 0.35 for all units).
"""
import csv
import os

import numpy as np
from sklearn.mixture import GaussianMixture

from . import model as M
from .expected_z import window_noise
from .fit import MEM_MIN, hellinger2
from .rate_states import window_rate_diff
from .recording import state_stats
from .surrogate import label

DATA = os.path.join(os.path.dirname(__file__), 'data')
SETS = (('top', '5'), ('top', '20'), ('top', '50'), ('random', '5'), ('random', '20'), ('random', '50'),
        ('all', 'session count'))
RHO = {'top': 0.09, 'random': 0.11, 'all': 0.35}      # z autocorrelation plateau at 200 ms (data, lambda 1e4)
STATS = (('off_per_trial', 'n_off', 0.3), ('off_mean_ms', 'off_mean', 20.0), ('frac_off', 'frac_off', 0.05),
         ('on_per_trial', 'n_on', 0.3), ('on_mean_ms', 'on_mean', 20.0), ('frac_on', 'frac_on', 0.05))
EDGES = np.arange(40, 620, 20)


def load_targets():
    rows = {(r['set'], r['k']): r for r in csv.DictReader(open(os.path.join(DATA, 'readout_summary_data_lam10000.csv')))}
    tg = {s: {k: float(rows[s][k]) for k in ('mu_on', 'accuracy') + tuple(x[0] for x in STATS)} for s in SETS}
    hist = {st: load_hist(st) for st in SETS}
    return tg, hist


def load_hist(st):
    """Data On/Off length histograms of one unit set, as probabilities in EDGES bins plus a >= 600 ms bin
    (states shorter than 40 ms excluded, as in the model histograms)."""
    H = {}
    for r in csv.DictReader(open(os.path.join(DATA, 'readout_histograms_data_lam10000.csv'))):
        if (r['set'], r['k']) == st:
            H.setdefault(r['state'], []).append((float(r['bin_lo_ms']), float(r['count'])))
    out = {}
    for state, v in H.items():
        lo = np.array([a for a, _ in v]); c = np.array([b for _, b in v])
        # the CSV's 580-600 ms bin also holds every state >= 600 ms (clipped), which the '600' row repeats
        over = c[lo == 600].sum()
        c = np.where(lo == 580, c - over, c)
        inside = np.array([c[lo == e].sum() for e in EDGES[:-1]])
        out[state] = np.r_[inside, over] / (inside.sum() + over)
    return out


def mu_on(z, rng, n_max=40000):
    v = z.ravel()
    if len(v) > n_max:
        v = rng.choice(v, n_max, replace=False)
    g = GaussianMixture(2, random_state=0).fit(v[:, None])
    return float(g.means_.max())


def noise(shape, rho, rng):
    return np.sqrt(1 - rho) * window_noise(shape, rng) + np.sqrt(rho) * rng.standard_normal((shape[0], 1))


def null_thr(shape, rho, rng, n_rep=3):
    masses = []
    for _ in range(n_rep):
        z = noise(shape, rho, rng)
        for k in range(z.shape[0]):
            m = np.r_[False, z[k] > 1.96, False].astype(int)
            masses += [z[k, a:b].sum() for a, b in zip(np.where(np.diff(m) == 1)[0], np.where(np.diff(m) == -1)[0])]
    return float(np.percentile(masses, 95))


def model_hist(dur):
    dur = np.asarray(dur, float)
    if len(dur) == 0:
        return np.r_[np.zeros(len(EDGES) - 1), 1.0]
    h, _ = np.histogram(dur, EDGES)
    return np.r_[h, (dur >= EDGES[-1]).sum()] / max(1, (dur >= EDGES[0]).sum())


def evaluate(theta, tg, n_trials=1056, seed=0, keep=False):
    p = M.params(**theta); p['t_delay'] = 1600.0
    sim = M.simulate(p, np.tile([0, 1], n_trials // 2), seed=seed * 100 + 1)
    dw, tc = window_rate_diff(sim)
    rng = np.random.default_rng(seed * 100 + 2)
    t_rel = tc - sim['t_cue']
    dm = (t_rel >= 500) & (t_rel <= 1400)
    valid = (t_rel >= 0) & (t_rel <= 1400)
    mask = np.broadcast_to(valid, dw.shape)
    grid = 10 ** np.linspace(-2.6, -0.3, 10)
    out = dict(mem=float(M.memory_correct(sim).mean()), sets={})
    cache = {}
    for st in SETS:
        rho = RHO[st[0]]
        if rho not in cache:
            eta = noise(dw.shape, rho, rng)
            mus = np.array([mu_on((c * dw + eta)[:, dm], rng) for c in grid])
            cache[rho] = (eta, np.maximum.accumulate(mus), null_thr(dw.shape, rho, rng))
        eta, mus, thr = cache[rho]
        target = tg[st]['mu_on']
        c = float(10 ** np.interp(target, mus, np.log10(grid))) if mus[-1] > target else float(grid[-1])
        z = c * dw + eta
        state = label(z, valid, thr)
        s = state_stats(state, mask)
        r = dict(gain=c, accuracy=float((z[:, dm] > 0).mean()), n_off=float(s['n_off'].mean()), n_on=float(s['n_on'].mean()),
                 off_mean=float(s['off_dur'].mean()) if len(s['off_dur']) else 0.0,
                 on_mean=float(s['on_dur'].mean()) if len(s['on_dur']) else 0.0,
                 frac_off=float((state[:, dm] == -1).mean()), frac_on=float((state[:, dm] == 1).mean()))
        r['h_off'], r['h_on'] = model_hist(s['off_dur']), model_hist(s['on_dur'])
        if keep:
            r['off_dur'], r['on_dur'] = s['off_dur'].tolist(), s['on_dur'].tolist()
        out['sets'][st] = r
    return out


def loss(out, tg, hist, parts=False):
    terms = {}
    for st in SETS:
        r = out['sets'][st]
        terms['/'.join(st)] = float(np.mean([((r[mk] - tg[st][dk]) / sc) ** 2 for dk, mk, sc in STATS]))
    r20 = out['sets'][('top', '20')]
    h20 = hist[('top', '20')]
    terms['hist_top20'] = 100 * (hellinger2(r20['h_off'], h20['off']) + hellinger2(r20['h_on'], h20['on']))
    terms['mem'] = 2e4 * max(0.0, MEM_MIN - out['mem']) ** 2
    L = float(sum(terms.values()))
    return (L, terms) if parts else L
