"""Targets and loss for fitting the mean-field model to Panichello et al. (2024) Fig. 4c."""
import os

import numpy as np

from . import model as M, recording as R, surrogate as S

DATA = os.path.join(os.path.dirname(__file__), 'data', 'panichello2024_fig4c.csv')
EDGES = np.arange(40, 620, 20)                 # the paper's bars: 40-600 ms in 20 ms bins
TARGET = dict(n_on=2.35, n_off=3.74, on_mean=192.4, off_mean=146.1, on_median=150.0, off_median=100.0, acc=0.70,
              n_on_states=3890, n_off_states=6174, n_trials=1653)

# fitted parameters and bounds (tau_N constrained to the standard 50-150 ms)
BOUNDS = dict(tau_N=(50.0, 150.0), w_s=(1.30, 1.70), dip_rate=(0.5, 8.0), dip_mean=(20.0, 300.0),
              dip_shape=(0.7, 6.0), dip_depth=(0.3, 1.0), sigma=(0.005, 0.04), f_sel=(0.003, 0.08),
              tune_sd=(0.0, 0.05))
# rate read-out fit (meanfield.rate_states): dynamics + one read-out gain, recording parameters dropped
BOUNDS_RATES = dict(tau_N=(50.0, 150.0), w_s=(1.30, 1.70), dip_rate=(0.5, 8.0), dip_mean=(20.0, 300.0),
                    dip_shape=(0.7, 6.0), dip_depth=(0.3, 1.0), sigma=(0.005, 0.04), log10_c=(-2.5, -0.5))
TERMS_STATES = ('hist_on', 'hist_off', 'n_on', 'n_off', 'mem')
MEM_MIN = 0.98      # memory correct at the end of the delay, on the fit's own trials (margin for >= 95% on new ones)
REC_KEYS = ('K', 'f_sel', 'tune_sd', 'base_rate', 'base_cv', 'n_per_class', 'delay_min', 'delay_max')


def target_hist():
    d = np.genfromtxt(DATA, delimiter=',', names=True, skip_header=2)
    keep = d['bin_lo'] >= EDGES[0]
    p_on, p_off = d['p_on'][keep] * 20, d['p_off'][keep] * 20      # probability per bin
    # the remaining mass (> 600 ms) goes into an overflow bin
    return dict(on=np.r_[p_on, max(0.0, 1 - p_on.sum())], off=np.r_[p_off, max(0.0, 1 - p_off.sum())])


def model_hist(dur):
    """Probability per paper bin (40-600 ms) plus overflow (>= 600); states < 40 ms (none in practice
    for Off; rare short On runs) are counted in the first bin's normaliser but not shown, as in the paper."""
    dur = np.asarray(dur, float)
    if len(dur) == 0:
        return np.r_[np.zeros(len(EDGES) - 1), 1.0]
    h, _ = np.histogram(dur, EDGES)
    return np.r_[h, (dur >= EDGES[-1]).sum()] / len(dur)


def hellinger2(p, q):
    return 0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)


def split(theta):
    rec = dict(R.REC_DEFAULTS)
    p = M.params()
    for k, v in theta.items():
        (rec if k in REC_KEYS else p)[k] = v
    return p, rec


def summarize(stats_list, acc_list, mem_list):
    on = np.concatenate([s['on_dur'] for s in stats_list])
    off = np.concatenate([s['off_dur'] for s in stats_list])
    n_on = np.concatenate([s['n_on'] for s in stats_list])
    n_off = np.concatenate([s['n_off'] for s in stats_list])
    return dict(on_dur=on, off_dur=off, n_on=float(n_on.mean()), n_off=float(n_off.mean()),
                on_mean=float(on.mean()) if len(on) else np.nan, off_mean=float(off.mean()) if len(off) else np.nan,
                on_median=float(np.median(on)) if len(on) else np.nan, off_median=float(np.median(off)) if len(off) else np.nan,
                acc=float(np.mean(acc_list)), mem=float(np.mean(mem_list)))


def loss(summ, tgt=None, parts=False, keep_terms=None):
    tgt = tgt or target_hist()
    h_on = hellinger2(model_hist(summ['on_dur']), tgt['on'])
    h_off = hellinger2(model_hist(summ['off_dur']), tgt['off'])
    terms = dict(hist_on=100 * h_on, hist_off=100 * h_off,
                 n_on=((summ['n_on'] - TARGET['n_on']) / 0.5) ** 2,
                 n_off=((summ['n_off'] - TARGET['n_off']) / 0.5) ** 2,
                 acc=((summ['acc'] - TARGET['acc']) / 0.05) ** 2,
                 mem=2e4 * max(0.0, MEM_MIN - summ['mem']) ** 2)       # memory correct (near-hard)
    if keep_terms is not None:
        terms = {k: v for k, v in terms.items() if k in keep_terms}
    L = float(sum(terms.values()))
    return (L, terms) if parts else L


def evaluate(theta, n_sessions=4, seed=0, keep=False):
    """Fast-pipeline evaluation with common random numbers (fixed seeds per session)."""
    p, rec = split(theta)
    stats, acc, mem, sess = [], [], [], []
    for i in range(n_sessions):
        ses = R.simulate_session(p, rec, seed=seed * 100 + 10 * i + 1)
        r = S.fast_pipeline(ses, seed=seed * 100 + i)
        stats.append(r['stats'])
        acc.append(r['acc'])
        mem.append(M.memory_correct(ses['sim']).mean())
        if keep:
            sess.append((ses, r))
    s = summarize(stats, acc, mem)
    return (s, sess) if keep else s
