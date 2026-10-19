"""Turn model rates into a recorded session and label On/Off states as Panichello et al. (2024).

Recorded cells (K per session): a fraction f_sel are memory cells of pool A or B (rate = pool rate);
the rest are background cells with rate b_i * r_bg(t)/r_bg(m=1) + eps_i * (r_A - r_B), clipped at 0,
where b_i is a lognormal baseline (mean 5.8 Hz, the paper's pre-cue rate) and eps_i ~ N(0, tune_sd)
is a weak random tuning. Spikes are Poisson; counts in 100 ms windows stepped by 10 ms.

Session protocol (from the paper): binary decoder of the cue location against the opposite location,
~66 trials per location (Extended Data Fig. 3; n = 1,653 trials over 25 sessions), 329 units per
session; delay 1,400-1,600 ms after a 50 ms cue; states counted from cue onset to the end of the
delay; decoding accuracy averaged over 500-1,400 ms after cue onset.
"""
import numpy as np
from numba import njit

from gamma_wm import panichello as pa

from . import model as M

STEP, WIN = 10, 100
REC_DEFAULTS = dict(K=329, f_sel=0.03, tune_sd=0.01, base_rate=5.8, base_cv=0.8, n_per_class=66,
                    delay_min=1400.0, delay_max=1600.0)


def population(rec, rng):
    K = int(rec['K'])
    n_sel = int(round(rec['f_sel'] * K))
    kind = np.zeros(K, int)
    kind[:n_sel] = 1 + (np.arange(n_sel) % 2)           # 1 = pool A, 2 = pool B
    s2 = np.log(1 + rec['base_cv'] ** 2)
    base = rng.lognormal(np.log(rec['base_rate']) - s2 / 2, np.sqrt(s2), K)
    eps = rng.normal(0, rec['tune_sd'], K)
    eps[kind > 0] = 0
    return dict(kind=kind, base=base, eps=eps)


def binned_rates(sim, pop, bg0):
    """Mean rate (Hz) of each recorded cell in 10 ms bins: (n_tr, K, n_bins)."""
    n_tr, n_t = sim['rA'].shape
    nb = n_t // STEP
    f = lambda x: x[:, :nb * STEP].reshape(n_tr, nb, STEP).mean(2)
    rA, rB, bg = f(sim['rA']), f(sim['rB']), f(sim['bg'])
    lam = pop['base'][None, :, None] * (bg / bg0)[:, None, :] + pop['eps'][None, :, None] * (rA - rB)[:, None, :]
    lam[:, pop['kind'] == 1, :] = rA[:, None, :]
    lam[:, pop['kind'] == 2, :] = rB[:, None, :]
    return np.maximum(lam, 0.0)


def window_counts(binned, rng=None):
    """Counts in 100 ms windows every 10 ms from 10 ms bin rates. Poisson if rng is given, else the
    expectation. Returns (n_tr, K, n_win), window centres (ms)."""
    x = binned * (STEP * 1e-3)
    if rng is not None:
        x = rng.poisson(x).astype(float)
    k = WIN // STEP
    c = np.cumsum(np.concatenate([np.zeros(x.shape[:2] + (1,)), x], axis=2), axis=2)
    out = c[:, :, k:] - c[:, :, :-k]
    centers = np.arange(out.shape[2]) * STEP + WIN / 2
    return out, centers


@njit(cache=True)
def _poisson_windows(rA, rB, bg, kind, base, eps, bg0, k0, seed):
    """Poisson counts of each cell in 100 ms windows every 10 ms, for windows starting at bin k0.
    rA, rB, bg: (n_tr, nb) 10 ms bin rates. Returns (n_tr, K, nb - k0 - 9)."""
    np.random.seed(seed)
    n_tr, nb = rA.shape
    K = kind.shape[0]
    kw = 10
    nw = nb - k0 - kw + 1
    out = np.empty((n_tr, K, nw))
    x = np.empty(nb - k0)
    for k in range(n_tr):
        for i in range(K):
            for j in range(k0, nb):
                if kind[i] == 1:
                    lam = rA[k, j]
                elif kind[i] == 2:
                    lam = rB[k, j]
                else:
                    lam = base[i] * bg[k, j] / bg0 + eps[i] * (rA[k, j] - rB[k, j])
                x[j - k0] = np.random.poisson(max(lam, 0.0) * 0.01)
            c = 0.0
            for j in range(kw):
                c += x[j]
            out[k, i, 0] = c
            for w in range(1, nw):
                c += x[w + kw - 1] - x[w - 1]
                out[k, i, w] = c
    return out


def simulate_session(p, rec, seed, keep_binned=False):
    """Simulate one session: model trials (cue A / B alternating), recorded population, per-trial delay.
    Returns dict with the simulation, labels y (1 = A), Poisson window counts, centres, masks."""
    rng = np.random.default_rng(seed)
    n = 2 * int(rec['n_per_class'])
    y = np.tile([1, 0], n // 2)
    p = dict(p, t_delay=rec['delay_max'])
    sim = M.simulate(p, 1 - y, seed=seed + 1)            # cued 0 = A
    bg0 = M.bg_rate(p)
    pop = population(rec, rng)
    n_tr, n_t = sim['rA'].shape
    nb = n_t // STEP
    f = lambda x: np.ascontiguousarray(x[:, :nb * STEP].reshape(n_tr, nb, STEP).mean(2))
    k0 = max(0, int((sim['t_cue'] - 100 - WIN / 2) // STEP))          # first window centred at t_cue - 100
    C = _poisson_windows(f(sim['rA']), f(sim['rB']), f(sim['bg']), pop['kind'], pop['base'], pop['eps'],
                         float(bg0), k0, int(rng.integers(2 ** 31)))
    tc = (k0 + np.arange(C.shape[2])) * STEP + WIN / 2
    binned = binned_rates(sim, pop, bg0) if keep_binned else None
    delay = rng.uniform(rec['delay_min'], rec['delay_max'], n)
    t_end = sim['t_cue'] + p['cue_dur'] + delay
    count_mask = (tc[None, :] >= sim['t_cue']) & (tc[None, :] < t_end[:, None])
    dec_mask = (tc >= sim['t_cue'] + 500) & (tc < sim['t_cue'] + 1400)
    return dict(sim=sim, y=y, C=C, tc=tc, pop=pop, binned=binned,
                valid=tc >= sim['t_cue'], count_mask=count_mask, dec_mask=dec_mask, bg0=bg0)


def state_stats(state, count_mask, step=STEP):
    """Durations and per-trial counts of On/Off states inside each trial's count window."""
    on_d, off_d, n_on, n_off = [], [], [], []
    for k in range(state.shape[0]):
        s = np.where(count_mask[k], state[k], 0)
        on, off = pa._runs(s == 1), pa._runs(s == -1)
        on_d += [(b - a) * step for a, b in on]
        off_d += [(b - a) * step for a, b in off]
        n_on.append(len(on))
        n_off.append(len(off))
    return dict(on_dur=np.array(on_d, float), off_dur=np.array(off_d, float), n_on=np.array(n_on), n_off=np.array(n_off))


def full_pipeline(ses, n_shuffle=20, lam=1000.0, seed=0, decoder='alo'):
    """Panichello pipeline on a simulated session: decoder confidence, shuffled-label null, z,
    On/Off labels (gamma_wm.panichello.label_states) and state statistics."""
    if decoder == 'alo':
        from .decoder import alo_confidence as dec
    else:
        from gamma_wm.analysis import loo_confidence_dual as dec
    C, y = ses['C'], ses['y']
    from .surrogate import label_states
    conf = dec(C, y, lam)
    nulls = np.array([dec(C, y, lam, shuffle_rng=np.random.default_rng(seed * 1000 + 7 + s)) for s in range(n_shuffle)])
    state, z, thr = label_states(conf, nulls, ses['valid'])     # compiled gamma_wm.panichello.label_states
    st = state_stats(state, ses['count_mask'])
    return dict(conf=conf, z=z, state=state, thr=thr, stats=st, null_mu=nulls.mean(0), null_sd=nulls.std(0),
                acc=float((conf[:, ses['dec_mask']] > 0.5).mean()))
