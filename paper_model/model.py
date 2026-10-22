"""Reduced two-pool model of Wong et al. (2007) [Wong & Wang (2006)], with two additions.

    r_X     = f(I_X),  f(I) = (a I - b) / (1 - exp(-d (a I - b)))                       firing rate (Hz)
    dS_X/dt = -S_X / tau_S + (1 - S_X) gamma r_X                                         NMDA gating
    I_X     = J_s S_X - J_c S_Y + I_0 m(t) + w_ff (m(t) - q) + I_cue,X + I_noise,X       input current (nA)
    tau_ff dq/dt = -q + m(t)                                                             lagging feed-forward inhibition
    tau_n dI_noise,X/dt = -I_noise,X + eta(t) sqrt(tau_n) sigma                         OU noise

X, Y in {cued c, uncued u}, X != Y. Published values (Wong et al. 2007, p. 2): a = 270 Hz/nA, b = 108 Hz,
d = 0.154 s, gamma = 0.641, I_0 = 0.3297 nA, sigma = 0.009 nA, tau_n = 2 ms, cue J_A,ext mu_0 (1 + f) with
J_A,ext = 1.1e-3 nA/Hz, mu_0 = 30 Hz, f = 0.45; couplings J_s = 0.3725 nA, J_c = 0.1137 nA at tau_S = 60 ms.
Here tau_S = 150 ms (NMDA receptors with GluN2B) and both couplings are scaled by ALPHA = 0.56, the factor at which the
resting state (~2 Hz) survives a whole trial and a 50 ms cue loads the memory (~17 Hz; scripts in the README).

Additions: (1) the common input m(t) scales the background current: 1, except during dips (Poisson onsets,
gamma-distributed lengths) where it drops to 1 - depth, with 10 ms ramps. (2) Feed-forward inhibition driven by
m(t) with a lag tau_ff; it cancels at steady state (m = q), so at m = 1 the network is the published one. When a dip
ends, excitation returns ~tau_ff before inhibition: both pools get a brief kick of ~w_ff * depth, and the pool with
more NMDA gating wins.
"""
import numpy as np
from numba import njit

ALPHA = 0.56
FIXED = dict(a=270.0, b=108.0, d=0.154, gamma=0.641, I_0=0.3297, sigma=0.009, tau_n=2.0,
             J_s=0.3725 * ALPHA, J_c=0.1137 * ALPHA, tau_S=150.0,
             cue_amp=1.1e-3 * 30.0 * 1.45, tau_ff=15.0,
             cue_dur=50.0, t_pre=300.0, t_delay=1600.0, t_resp=300.0, go_rel=3.1, dt=0.5)

# fitted (fit.py) to the On/Off statistics and length distributions of the all-units read-out only
FIT = dict(dip_rate=3.390,    # 1/s, rate parameter of the gaps between dips
           dip_mean=20.0,     # ms, mean dip length (at the lower bound)
           dip_shape=10.00,    # gamma shape of dip lengths (at the upper bound)
           dip_depth=0.1109,  # fractional drop of the background current in a dip
           w_ff=0.5856)       # nA, feed-forward inhibition
KEYS = ('a', 'b', 'd', 'gamma', 'J_s', 'J_c', 'I_0', 'w_ff', 'tau_ff', 'tau_S', 'cue_amp', 'dt')


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
def _f(I, a, b, d):
    u = a * I - b
    return u / (1 - np.exp(-d * u)) if abs(u) > 1e-9 else 1 / d


@njit(cache=True)
def _sim(m, cue, go, noise, p, rec):
    a, b, d, gamma, J_s, J_c, I_0, w_ff, tau_ff, tau_S, cue_amp, dt = p
    n_tr, n = m.shape
    out = np.zeros((n_tr, n // rec, 4))    # rc, ru, sc, su every rec steps (1 ms)
    for k in range(n_tr):
        sc = 0.1; su = 0.1; q = m[k, 0]
        for i in range(n):
            bg = I_0 * m[k, i] + w_ff * (m[k, i] - q) + go[i]
            rc = _f(J_s * sc - J_c * su + bg + cue_amp * cue[i] + noise[k, i, 0], a, b, d)
            ru = _f(J_s * su - J_c * sc + bg + noise[k, i, 1], a, b, d)
            sc += dt * (-sc / tau_S + (1 - sc) * gamma * rc / 1000)
            su += dt * (-su / tau_S + (1 - su) * gamma * ru / 1000)
            q += dt / tau_ff * (m[k, i] - q)
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
    """n_tr trials with pool c cued. Returns 1 ms traces (n_tr, n_t) rc, ru, sc, su, m over the pre-cue, cue and
    delay periods, the time t from cue onset, and the rates in the response period (rc_resp, ru_resp: t_resp ms
    after the delay, attention at 1, starting with a re-engagement transient go_rel times that at the end of a dip).
    quiet: no dips during the first `quiet` ms after cue onset."""
    rng = np.random.default_rng(seed)
    dt = p['dt']
    n = int(round((p['t_pre'] + p['cue_dur'] + p['t_delay']) / dt))
    n_resp = int(round(p['t_resp'] / dt))
    m = common_input(n_tr, n, p, rng)
    m[:, :int((p['t_pre'] + quiet) / dt)] = 1.0
    m = np.c_[m, np.ones((n_tr, n_resp))]
    t = np.arange(n + n_resp) * dt
    cue = ((t >= p['t_pre']) & (t < p['t_pre'] + p['cue_dur'])).astype(float)
    white = rng.standard_normal((n_tr, n, 2))
    white = np.concatenate([white, np.random.default_rng(seed + 10**6).standard_normal((n_tr, n_resp, 2))], axis=1)
    noise = p['sigma'] * _ou(white, np.exp(-dt / p['tau_n']))      # stationary OU, SD sigma
    tr = t - n * dt
    go = np.where(tr >= 0, p['go_rel'] * p['w_ff'] * p['dip_depth'] * np.exp(-np.maximum(tr, 0) / p['tau_ff']), 0.0)
    rec = int(round(1 / dt))
    out = _sim(m, cue, go, noise, np.array([float(p[k]) for k in KEYS]), rec)
    k = n // rec
    return dict(rc=out[:, :k, 0], ru=out[:, :k, 1], sc=out[:, :k, 2], su=out[:, :k, 3], m=m[:, rec - 1:n:rec],
                t=np.arange(k) - p['t_pre'], rc_resp=out[:, k:, 0], ru_resp=out[:, k:, 1])


def memory_correct(sim):
    """Memory correct: in the last 100 ms of the response period, the cued pool fires above 8 Hz (memory state ~17 Hz,
    rest ~2 Hz) and above the uncued pool."""
    rc, ru = sim['rc_resp'][:, -100:].mean(1), sim['ru_resp'][:, -100:].mean(1)
    return (rc > 8) & (rc > ru)
