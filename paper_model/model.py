"""Rate model of the memory network: two memory pools (cued c, uncued u) with shared inhibition I.

    tau_r  dr_X/dt = -r_X + Phi( w_s*s_X + w_a*r_X - w_I*r_I + I_0*m(t) - w_ff*q + cue_X + sigma*xi_X )
    tau_N  ds_X/dt = -s_X + beta*r_X*(1 - s_X)          NMDA gating
    tau_I  dr_I/dt = -r_I + w_EI*(r_c + r_u)            shared inhibition (competition)
    tau_ff dq/dt   = -q + m(t)                          feed-forward inhibition, lags the input

Rates are nominally in Hz (sigmoid ceiling r_max; the figures show them as r / r_max, arbitrary units),
time in ms. Phi(x) = r_max / (1 + exp(-(x - 1)/kappa)); xi are OU noises (tau_xi). m(t) is the
common input: 1, except during dips (Poisson onsets, gamma-distributed lengths) where it drops to 1 - depth,
with 10 ms ramps. When a dip ends, the lagging feed-forward inhibition gives both pools a brief kick and the
pool with more residual NMDA wins the competition.
"""
import numpy as np
from numba import njit

FIXED = dict(r_max=150.0, kappa=0.06, w_a=0.0141, beta=0.02, w_I=0.04, w_EI=0.5, w_ff=0.8, tau_ff=15.0, I_0=1.496,
             tau_r=10.0, tau_I=1.0, tau_xi=10.0, cue_amp=1.0, cue_dur=50.0, t_pre=300.0, t_delay=1600.0, dt=0.1)

# fitted (fit.py) to the On/Off statistics and length histograms of the all-units read-out only
FIT = dict(tau_N=150.0,       # ms, NMDA decay
           w_s=1.30,          # NMDA recurrent strength
           dip_rate=10.207,   # dips per s
           dip_mean=90.12,    # ms, mean dip length
           dip_shape=8.70,    # gamma shape of dip lengths
           dip_depth=0.5987,  # fractional drop of the input in a dip
           sigma=0.0058)      # fast noise on the pools
KEYS = ('r_max', 'kappa', 'w_a', 'beta', 'w_I', 'w_EI', 'w_ff', 'tau_ff', 'I_0', 'tau_r', 'tau_I', 'cue_amp',
        'dt', 'tau_N', 'w_s', 'sigma')


def params(**over):
    return {**FIXED, **FIT, **over}


def common_input(n_tr, n, p, rng, ramp=10.0):
    """(n_tr, n) input m(t) with dips, at resolution dt."""
    m = np.ones((n_tr, n))
    dt, T = p['dt'], n * p['dt']
    for k in range(n_tr):
        t = rng.exponential(1000 / p['dip_rate']) - 500 * rng.random()   # stationary at t = 0
        while t < T:
            d = rng.gamma(p['dip_shape'], p['dip_mean'] / p['dip_shape'])
            a, b = int(max(t, 0) / dt), int(min(T, t + d) / dt)
            if b > a:
                m[k, a:b] = 1 - p['dip_depth']
            t += d + rng.exponential(1000 / p['dip_rate'])
    w = int(ramp / dt)
    pad = np.c_[np.ones((n_tr, w)), m, np.ones((n_tr, w))]
    return np.array([np.convolve(r, np.ones(w) / w, mode='same') for r in pad])[:, w:-w]


@njit(cache=True)
def _sim(m, cue, xi, p, rec):
    r_max, kappa, w_a, beta, w_I, w_EI, w_ff, tau_ff, I_0, tau_r, tau_I, cue_amp, dt, tau_N, w_s, sigma = p
    n_tr, n = m.shape
    out = np.zeros((n_tr, n // rec, 4))    # rc, ru, sc, su every rec steps (1 ms)
    for k in range(n_tr):
        rc = 1.0; ru = 1.0; rI = 1.0; sc = 0.0; su = 0.0; q = m[k, 0]
        for i in range(n):
            x = -w_I * rI + I_0 * m[k, i] - w_ff * q
            fc = r_max / (1 + np.exp(-(w_s * sc + w_a * rc + x + cue_amp * cue[i] + sigma * xi[k, i, 0] - 1) / kappa))
            fu = r_max / (1 + np.exp(-(w_s * su + w_a * ru + x + sigma * xi[k, i, 1] - 1) / kappa))
            rI += dt / tau_I * (w_EI * (rc + ru) - rI)
            rc += dt / tau_r * (fc - rc)
            ru += dt / tau_r * (fu - ru)
            q += dt / tau_ff * (m[k, i] - q)
            sc += dt / tau_N * (-sc + beta * rc * (1 - sc))
            su += dt / tau_N * (-su + beta * ru * (1 - su))
            if i % rec == rec - 1:
                j = i // rec
                out[k, j, 0] = rc; out[k, j, 1] = ru; out[k, j, 2] = sc; out[k, j, 3] = su
    return out


@njit(cache=True)
def _ou(white, a):
    x = white.copy()
    for i in range(1, x.shape[1]):
        x[:, i] = a * x[:, i - 1] + np.sqrt(1 - a * a) * white[:, i]
    return x


def simulate(p, n_tr, seed=0, quiet=0.0):
    """n_tr trials with pool c cued. Returns 1 ms traces (n_tr, n_t) rc, ru, sc, su, m and the time t from cue
    onset. quiet: no dips during the first `quiet` ms after cue onset."""
    rng = np.random.default_rng(seed)
    dt = p['dt']
    n = int(round((p['t_pre'] + p['cue_dur'] + p['t_delay']) / dt))
    m = common_input(n_tr, n, p, rng)
    m[:, :int((p['t_pre'] + quiet) / dt)] = 1.0
    cue = ((np.arange(n) * dt >= p['t_pre']) & (np.arange(n) * dt < p['t_pre'] + p['cue_dur'])).astype(float)
    xi = _ou(rng.standard_normal((n_tr, n, 2)), np.exp(-dt / p['tau_xi']))
    rec = int(round(1 / dt))
    out = _sim(m, cue, xi, np.array([float(p[k]) for k in KEYS]), rec)
    return dict(rc=out[..., 0], ru=out[..., 1], sc=out[..., 2], su=out[..., 3], m=m[:, rec - 1::rec],
                t=np.arange(n // rec) - p['t_pre'])


def memory_correct(sim):
    """Memory held at the end of the delay: NMDA gating of the cued pool exceeds the uncued pool's by > 0.05."""
    return (sim['sc'][:, -100:] - sim['su'][:, -100:]).mean(1) > 0.05
