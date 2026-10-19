"""Rate model: two memory pools A, B, background E and a shared inhibitory population I.

    tau_r dr_X/dt = -r_X + Phi( w_s*s_X + w_a*r_X - w_I*r_I + I_0*m(t) - w_ff*q + cue + sigma*xi_X )
    tau_N ds_X/dt = -s_X + beta*r_X*(1 - s_X)                     (NMDA gating, saturating)
    tau_I dr_I/dt = -r_I + w_EI*(r_A + r_B)                       (shared inhibition: competition)
    tau_ff dq/dt  = -q + m(t)                                     (feed-forward inhibition, lags the drive)
    tau_r dr_bg/dt = -r_bg + Phi_bg( I_bg*m(t) + sigma_bg*xi_bg ) (background E, read out only)

X in {A, B}. Rates in Hz, time in ms. Phi is a sigmoid with threshold 1 (the pools are bistable
at m = 1: ~1 Hz and ~100 Hz). xi are OU noises (tau_xi). m(t) is the shared input multiplier:
Poisson dips (rate, gamma lengths, depth, 10 ms ramps) or a clipped OU process. Same generators
as gamma_wm.task (shared_dips, slow_input_noise), re-implemented to be vectorised over trials.
"""
import numpy as np
from numba import njit

# structural constants, chosen once so that the rates match the spiking network (see
# MEANFIELD_REPORT.md); the fitted parameters are in FIT_DEFAULTS
STRUCT = dict(
    r_max=150.0,     # Hz, pool sigmoid ceiling
    theta=1.0,       # pool threshold (input units)
    kappa=0.06,      # pool sigmoid slope
    w_a=0.0141,      # fast (AMPA) self-excitation per Hz: 60% of the recurrent excitation at 110 Hz
    beta=0.02,       # NMDA gating: half-saturation at 1/beta = 50 Hz. Normalised so that the steady
                     # state does not depend on tau_N: tau_N only sets how long the trace lasts
    w_I=0.04,        # inhibition onto pools per Hz of r_I
    w_EI=0.5,        # r_I = w_EI (r_A + r_B): shared, winner-take-all (w_I*w_EI = 0.02 per Hz)
    w_ff=0.8,        # feed-forward inhibition driven by the same external input m(t) ...
    tau_ff=15.0,     # ... but lagging it: when a dip ends, both pools get a brief rebound kick and the
                     # one with more residual NMDA wins the competition
    I_0=1.496,       # external excitation onto the pools at m = 1; net of feed-forward inhibition it is
                     # 0.696: low state 1 Hz, cued pool 110 Hz, saddle 7 Hz; the high state needs m > 0.75
    tau_r=10.0, tau_I=1.0,   # fast inhibition (slower values give a fast E-I oscillation of the plateau)
    bg_max=40.0, bg_theta=1.0, bg_kappa=0.15,   # background E: ~5 Hz at m = 1, silent in dips
    I_bg=0.72,
    tau_xi=10.0,
    cue_amp=1.0, cue_dur=50.0,
    t_pre=300.0, t_delay=1650.0,
    dt=0.5,
)

FIT_DEFAULTS = dict(
    tau_N=100.0,     # ms, NMDA decay
    w_s=1.50,        # NMDA recurrent strength (the alpha of the spiking model)
    dip_rate=3.0,    # 1/s
    dip_mean=100.0,  # ms
    dip_shape=2.0,
    dip_depth=0.8,
    sigma=0.05,      # fast noise on the pools (input units)
    sigma_bg=0.1,
    ou_sd=0.0, ou_tau=500.0,   # optional OU drive (used when ou_sd > 0, instead of dips)
)


def params(**over):
    p = dict(STRUCT)
    p.update(FIT_DEFAULTS)
    p.update(over)
    return p


def shared_dips(n_tr, n, dt, rate, mean_dur, depth, shape, rng, ramp=10.0):
    """(n_tr, n) multiplier: Poisson dip onsets (rate per s), gamma(shape) lengths with mean
    mean_dur (ms), drive x (1 - depth) inside a dip, edges smoothed with `ramp` ms ramps."""
    m = np.ones((n_tr, n))
    T = n * dt
    if rate <= 0 or depth <= 0:
        return m
    for k in range(n_tr):
        t = rng.exponential(1000.0 / rate) - 500.0 * rng.random()   # stationary at t = 0
        while t < T:
            d = rng.gamma(shape, mean_dur / shape)
            a, b = int(max(t, 0) / dt), int(min(T, t + d) / dt)
            if b > a:
                m[k, a:b] = 1.0 - depth
            t += d + rng.exponential(1000.0 / rate)
    w = max(1, int(ramp / dt))
    if w > 1:
        ker = np.ones(w) / w
        pad = np.c_[np.ones((n_tr, w)), m, np.ones((n_tr, w))]
        m = np.array([np.convolve(r, ker, mode='same') for r in pad])[:, w:-w]
    return m


def ou_drive(n_tr, n, dt, sd, tau, rng, floor=0.0):
    a = np.exp(-dt / tau)
    x = np.empty((n_tr, n))
    x[:, 0] = rng.standard_normal(n_tr)
    e = rng.standard_normal((n_tr, n)) * np.sqrt(1 - a * a)
    for i in range(1, n):
        x[:, i] = a * x[:, i - 1] + e[:, i]
    return np.clip(1.0 + sd * x, floor, None)


@njit(cache=True)
def _sim(m, cue_on, cued, xi, p_arr, rec_every):
    (r_max, theta, kappa, w_a, beta, w_I, I_0, tau_r, tau_I, w_EI, w_ff, tau_ff,
     bg_max, bg_theta, bg_kappa, I_bg, cue_amp, dt, tau_N, w_s, sigma, sigma_bg) = p_arr
    n_tr, n = m.shape
    n_rec = n // rec_every
    out = np.zeros((n_tr, n_rec, 6))    # rA, rB, r_bg, r_I, sA, sB
    for k in range(n_tr):
        rA = 1.0; rB = 1.0; rI = 1.0; rbg = 3.0; sA = 0.0; sB = 0.0; q = m[k, 0]
        for i in range(n):
            mm = m[k, i]
            cA = cue_amp if (cue_on[i] and cued[k] == 0) else 0.0
            cB = cue_amp if (cue_on[i] and cued[k] == 1) else 0.0
            common = -w_I * rI + I_0 * mm - w_ff * q
            xA = w_s * sA + w_a * rA + common + cA + sigma * xi[k, i, 0]
            xB = w_s * sB + w_a * rB + common + cB + sigma * xi[k, i, 1]
            fA = r_max / (1.0 + np.exp(-(xA - theta) / kappa))
            fB = r_max / (1.0 + np.exp(-(xB - theta) / kappa))
            xbg = I_bg * mm + sigma_bg * xi[k, i, 2]
            fbg = bg_max / (1.0 + np.exp(-(xbg - bg_theta) / bg_kappa))
            fI = w_EI * (rA + rB)
            rA += dt / tau_r * (fA - rA)
            rB += dt / tau_r * (fB - rB)
            rbg += dt / tau_r * (fbg - rbg)
            rI += dt / tau_I * (fI - rI)
            q += dt / tau_ff * (mm - q)
            sA += dt / tau_N * (-sA + beta * rA * (1.0 - sA))
            sB += dt / tau_N * (-sB + beta * rB * (1.0 - sB))
            if (i + 1) % rec_every == 0:
                j = (i + 1) // rec_every - 1
                out[k, j, 0] = rA; out[k, j, 1] = rB; out[k, j, 2] = rbg
                out[k, j, 3] = rI; out[k, j, 4] = sA; out[k, j, 5] = sB
    return out


@njit(cache=True)
def _ou_noise(n_tr, n, dt, tau, white):
    a = np.exp(-dt / tau)
    b = np.sqrt(1 - a * a)
    x = np.empty_like(white)
    for k in range(n_tr):
        for c in range(white.shape[2]):
            x[k, 0, c] = white[k, 0, c]
            for i in range(1, n):
                x[k, i, c] = a * x[k, i - 1, c] + b * white[k, i, c]
    return x


def simulate(p, cued, seed=0, m=None, rec_ms=1.0):
    """Simulate trials. cued: (n_tr,) 0 = A, 1 = B. Returns dict with rates sampled every
    rec_ms (arrays (n_tr, n_t)): rA, rB, bg, I, sA, sB; plus the drive m and timing."""
    rng = np.random.default_rng(seed)
    cued = np.asarray(cued, np.int64)
    n_tr = len(cued)
    dt = p['dt']
    T = p['t_pre'] + p['cue_dur'] + p['t_delay']
    n = int(round(T / dt))
    if m is None:
        if p.get('ou_sd', 0) > 0:
            m = ou_drive(n_tr, n, dt, p['ou_sd'], p['ou_tau'], rng)
        else:
            m = shared_dips(n_tr, n, dt, p['dip_rate'], p['dip_mean'], p['dip_depth'], p['dip_shape'], rng)
    t = np.arange(n) * dt
    cue_on = (t >= p['t_pre']) & (t < p['t_pre'] + p['cue_dur'])
    xi = _ou_noise(n_tr, n, dt, p['tau_xi'], rng.standard_normal((n_tr, n, 3)))
    keys = ('r_max', 'theta', 'kappa', 'w_a', 'beta', 'w_I', 'I_0', 'tau_r', 'tau_I', 'w_EI', 'w_ff', 'tau_ff',
            'bg_max', 'bg_theta', 'bg_kappa', 'I_bg', 'cue_amp', 'dt', 'tau_N', 'w_s', 'sigma', 'sigma_bg')
    p_arr = np.array([float(p[k]) for k in keys])
    rec = int(round(rec_ms / dt))
    out = _sim(m, cue_on, cued, xi, p_arr, rec)
    return dict(rA=out[..., 0], rB=out[..., 1], bg=out[..., 2], I=out[..., 3], sA=out[..., 4], sB=out[..., 5],
                m=m[:, rec - 1::rec][:, :out.shape[1]], cued=cued, rec_ms=rec_ms,
                t_cue=p['t_pre'], t_don=p['t_pre'] + p['cue_dur'], T=T)


def bg_rate(p, m=1.0):
    """Steady background rate at drive m (no noise)."""
    return p['bg_max'] / (1.0 + np.exp(-(p['I_bg'] * m - p['bg_theta']) / p['bg_kappa']))


def memory_correct(sim, last_ms=100.0):
    """Memory held at the end of the delay: NMDA gating of the cued pool exceeds the uncued pool's
    by > 0.05 over the last `last_ms` (a pause is fine; a lost or flipped memory is not)."""
    k = int(last_ms / sim['rec_ms'])
    sA, sB = sim['sA'][:, -k:].mean(1), sim['sB'][:, -k:].mean(1)
    d = np.where(sim['cued'] == 0, sA - sB, sB - sA)
    return d > 0.05
