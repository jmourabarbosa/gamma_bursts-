"""On/Off states read directly from the pool rates (no recorded cells, no spikes, no decoder fit).

The decoder's z-score is modelled as   z(t) = c * dr(t) + eta(t),
  dr  = cued - uncued pool rate averaged in 100 ms windows every 10 ms (the paper's windows),
  eta = the z noise: unit variance (z is normalised by the shuffled null) and correlated over 100 ms
        like the overlapping windows,
  c   = the only read-out parameter (z per Hz): it stands for everything about the recording.
The paper's rules are then applied unchanged: On = runs of z > 1.96 whose summed z exceeds the 95th
percentile of the same runs in pure noise (cluster mass); Off = >= 5 consecutive bins with z < 0.84.
States are counted from cue onset to the end of a 1,400-1,600 ms delay.
"""
import numpy as np

from . import model as M, recording as R
from .expected_z import window_noise
from .surrogate import label


def window_rate_diff(sim, step=R.STEP, win=R.WIN):
    cued = sim['cued'][:, None] == 0
    d = np.where(cued, sim['rA'] - sim['rB'], sim['rB'] - sim['rA'])
    c = np.cumsum(np.c_[np.zeros(len(d)), d], axis=1)
    dw = (c[:, win:] - c[:, :-win]) / win                 # 1 ms resolution, window starting at each ms
    dw = dw[:, ::step]
    tc = np.arange(dw.shape[1]) * step + win / 2
    keep = tc >= sim['t_cue'] - 100
    return dw[:, keep], tc[keep]


def null_threshold(shape, rng, n_rep=4):
    """95th percentile of summed z over runs of z > 1.96 in pure noise."""
    masses = []
    for _ in range(n_rep):
        z = window_noise(shape, rng)
        for k in range(z.shape[0]):
            m = np.r_[False, z[k] > 1.96, False].astype(int)
            on, off = np.where(np.diff(m) == 1)[0], np.where(np.diff(m) == -1)[0]
            masses += [z[k, a:b].sum() for a, b in zip(on, off)]
    return float(np.percentile(masses, 95))


def states(sim, c, rng, delay=(1400.0, 1600.0), cue_dur=50.0):
    dw, tc = window_rate_diff(sim)
    z = c * dw + window_noise(dw.shape, rng)
    thr = null_threshold(dw.shape, rng)
    valid = tc >= sim['t_cue']
    state = label(z, valid, thr)
    t_end = sim['t_cue'] + cue_dur + rng.uniform(*delay, len(dw))
    mask = (tc[None] >= sim['t_cue']) & (tc[None] < t_end[:, None])
    return R.state_stats(state, mask), z, state, tc, thr


def evaluate(theta, n_trials=528, seed=0):
    """theta: model parameters plus 'log10_c'. Returns the same summary as meanfield.fit.evaluate."""
    from .fit import summarize
    p = M.params(**{k: v for k, v in theta.items() if k != 'log10_c'})
    p['t_delay'] = 1600.0
    sim = M.simulate(p, np.tile([0, 1], n_trials // 2), seed=seed * 100 + 1)
    st, z, state, tc, thr = states(sim, 10 ** theta['log10_c'], np.random.default_rng(seed * 100 + 2))
    dm = (tc >= sim['t_cue'] + 500) & (tc < sim['t_cue'] + 1400)
    return summarize([st], [float((z[:, dm] > 0).mean())], [float(M.memory_correct(sim).mean())])


# ---------------------------------------------------------------------------------------------
# Read-out-free states: the network's own attractor states (perfect read-out limit)
# ---------------------------------------------------------------------------------------------

def saddle_rate(p, r_low=1.0):
    """Rate of the unstable fixed point between a pool's silent and active states at m = 1 (other pool
    silent, s at its steady state): the boundary used to call a pool active."""
    r = np.linspace(0.05, p['r_max'] - 1e-3, 20000)
    s = p['beta'] * r / (1 + p['beta'] * r)
    J = p['w_I'] * p['w_EI']
    x = p['w_s'] * s + p['w_a'] * r - J * (r + r_low) + p['I_0'] - p['w_ff']
    phi_inv = p['theta'] + p['kappa'] * np.log(r / (p['r_max'] - r))
    g = x - phi_inv
    up = np.where((np.diff(np.sign(g)) > 0) & (r[:-1] > 1.2 * r_low))[0]
    return float(r[up[0]]) if len(up) else float(r[np.argmax(g)])


def silence_states(sim, r_sad, delay=(1400.0, 1600.0), cue_dur=50.0, rng=None, step=R.STEP, win=R.WIN, min_off=5):
    """On: cued pool (100 ms mean) above the saddle rate and above the uncued pool. Off: both pools below
    it for >= min_off windows. Else unlabelled."""
    cued = sim['cued'][:, None] == 0
    rc = np.where(cued, sim['rA'], sim['rB']); ru = np.where(cued, sim['rB'], sim['rA'])

    def wmean(x):
        c = np.cumsum(np.c_[np.zeros(len(x)), x], axis=1)
        return ((c[:, win:] - c[:, :-win]) / win)[:, ::step]
    wc, wu = wmean(rc), wmean(ru)
    tc = np.arange(wc.shape[1]) * step + win / 2
    keep = tc >= sim['t_cue'] - 100
    wc, wu, tc = wc[:, keep], wu[:, keep], tc[keep]
    valid = tc >= sim['t_cue']
    on = (wc > r_sad) & (wc > wu) & valid[None]
    silent = (wc < r_sad) & (wu < r_sad) & valid[None]
    state = np.zeros(wc.shape, int)
    state[on] = 1
    for k in range(len(state)):
        m = np.r_[False, silent[k], False].astype(int)
        for a, b in zip(np.where(np.diff(m) == 1)[0], np.where(np.diff(m) == -1)[0]):
            if b - a >= min_off:
                state[k, a:b] = -1
    rng = rng or np.random.default_rng(0)
    t_end = sim['t_cue'] + cue_dur + rng.uniform(*delay, len(wc))
    mask = (tc[None] >= sim['t_cue']) & (tc[None] < t_end[:, None])
    return R.state_stats(state, mask), state, tc


def evaluate_silence(theta, n_trials=1056, seed=0):
    from .fit import summarize
    p = M.params(**theta)
    p['t_delay'] = 1600.0
    sim = M.simulate(p, np.tile([0, 1], n_trials // 2), seed=seed * 100 + 1)
    st, state, tc = silence_states(sim, saddle_rate(p), rng=np.random.default_rng(seed * 100 + 2))
    dm = (tc >= sim['t_cue'] + 500) & (tc < sim['t_cue'] + 1400)
    return summarize([st], [float((state[:, dm] == 1).mean())], [float(M.memory_correct(sim).mean())])
