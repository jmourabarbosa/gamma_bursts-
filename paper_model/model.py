"""Rate model of the memory network: two memory pools A, B, background E, shared inhibition I.

    tau_r  dr_X/dt  = -r_X + Phi( w_s*s_X + w_a*r_X - w_I*r_I + I_0*m(t) - w_ff*q + cue_X + sigma*xi_X )
    tau_N  ds_X/dt  = -s_X + beta*r_X*(1 - s_X)              NMDA gating
    tau_I  dr_I/dt  = -r_I + w_EI*(r_A + r_B)                shared inhibition (competition)
    tau_ff dq/dt    = -q + m(t)                              feed-forward inhibition, lags the input
    tau_r  dr_bg/dt = -r_bg + Phi_bg( I_bg*m(t) + sigma_bg*xi_bg )

X in {A, B}; rates in Hz, time in ms. Phi(x) = r_max / (1 + exp(-(x - 1)/kappa)); xi are OU noises (tau_xi).
m(t) is the common input: 1, except during dips (Poisson onsets, gamma-distributed lengths) where it drops
to 1 - depth, with 10 ms ramps. When a dip ends, the lagging feed-forward inhibition gives both pools a brief
kick and the pool with more residual NMDA wins the competition.
"""
import numpy as np
from numba import njit

# fixed (not fitted) parameters
FIXED = dict(
    r_max=150.0, theta=1.0, kappa=0.06,   # pool sigmoid: ceiling (Hz), threshold, slope
    w_a=0.0141,                           # fast (AMPA) self-excitation per Hz
    beta=0.02,                            # NMDA gating, half-saturation at 50 Hz
    w_I=0.04, w_EI=0.5,                   # shared inhibition
    w_ff=0.8, tau_ff=15.0,                # lagging feed-forward inhibition
    I_0=1.496,                            # input onto the pools at m = 1
    tau_r=10.0, tau_I=1.0,
    bg_max=40.0, bg_theta=1.0, bg_kappa=0.15, I_bg=0.72, sigma_bg=0.1,   # background E
    tau_xi=10.0,
    cue_amp=1.0, cue_dur=50.0,
    t_pre=300.0, t_delay=1600.0,
    dt=0.5,
)

# best fit (CMA-ES, fit.py) to the lambda = 10,000 read-out table of 25 sessions: On/Off statistics and length
# histograms of all 7 unit sets
FIT = dict(
    tau_N=131.65,      # ms, NMDA decay
    w_s=1.6095,         # NMDA recurrent strength
    dip_rate=10.249,    # dips per s
    dip_mean=118.56,   # ms, mean dip length
    dip_shape=5.243,   # gamma shape of dip lengths
    dip_depth=0.8426,  # fractional drop of the input in a dip
    sigma=0.0115,      # fast noise on the pools
)

BOUNDS = dict(tau_N=(50.0, 150.0), w_s=(1.30, 1.70), dip_rate=(0.5, 12.0), dip_mean=(20.0, 300.0),
              dip_shape=(0.7, 10.0), dip_depth=(0.3, 1.0), sigma=(0.005, 0.04))


def params(**over):
    return {**FIXED, **FIT, **over}


def common_input(n_tr, n, dt, rate, mean_dur, depth, shape, rng, ramp=10.0):
    """(n_tr, n) input multiplier m(t) with Poisson dips."""
    m = np.ones((n_tr, n))
    T = n * dt
    for k in range(n_tr):
        t = rng.exponential(1000.0 / rate) - 500.0 * rng.random()   # stationary at t = 0
        while t < T:
            d = rng.gamma(shape, mean_dur / shape)
            a, b = int(max(t, 0) / dt), int(min(T, t + d) / dt)
            if b > a:
                m[k, a:b] = 1.0 - depth
            t += d + rng.exponential(1000.0 / rate)
    w = int(ramp / dt)
    pad = np.c_[np.ones((n_tr, w)), m, np.ones((n_tr, w))]
    return np.array([np.convolve(r, np.ones(w) / w, mode='same') for r in pad])[:, w:-w]


@njit(cache=True)
def _ou(white, a):
    x = np.empty_like(white)
    b = np.sqrt(1 - a * a)
    for k in range(white.shape[0]):
        for c in range(white.shape[2]):
            x[k, 0, c] = white[k, 0, c]
            for i in range(1, white.shape[1]):
                x[k, i, c] = a * x[k, i - 1, c] + b * white[k, i, c]
    return x


@njit(cache=True)
def _sim(m, cue_on, cued, xi, p, rec_every):
    (r_max, theta, kappa, w_a, beta, w_I, I_0, tau_r, tau_I, w_EI, w_ff, tau_ff,
     bg_max, bg_theta, bg_kappa, I_bg, cue_amp, dt, tau_N, w_s, sigma, sigma_bg) = p
    n_tr, n = m.shape
    out = np.zeros((n_tr, n // rec_every, 5))    # rA, rB, r_bg, sA, sB
    for k in range(n_tr):
        rA = 1.0; rB = 1.0; rI = 1.0; rbg = 3.0; sA = 0.0; sB = 0.0; q = m[k, 0]
        for i in range(n):
            mm = m[k, i]
            cA = cue_amp if (cue_on[i] and cued[k] == 0) else 0.0
            cB = cue_amp if (cue_on[i] and cued[k] == 1) else 0.0
            common = -w_I * rI + I_0 * mm - w_ff * q
            fA = r_max / (1.0 + np.exp(-(w_s * sA + w_a * rA + common + cA + sigma * xi[k, i, 0] - theta) / kappa))
            fB = r_max / (1.0 + np.exp(-(w_s * sB + w_a * rB + common + cB + sigma * xi[k, i, 1] - theta) / kappa))
            fbg = bg_max / (1.0 + np.exp(-(I_bg * mm + sigma_bg * xi[k, i, 2] - bg_theta) / bg_kappa))
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
                out[k, j, 0] = rA; out[k, j, 1] = rB; out[k, j, 2] = rbg; out[k, j, 3] = sA; out[k, j, 4] = sB
    return out


KEYS = ('r_max', 'theta', 'kappa', 'w_a', 'beta', 'w_I', 'I_0', 'tau_r', 'tau_I', 'w_EI', 'w_ff', 'tau_ff',
        'bg_max', 'bg_theta', 'bg_kappa', 'I_bg', 'cue_amp', 'dt', 'tau_N', 'w_s', 'sigma', 'sigma_bg')


def simulate(p, cued, seed=0, m=None):
    """Simulate trials; cued: (n_tr,) 0 = A, 1 = B. Returns 1 ms traces (n_tr, n_t): rA, rB, bg, sA, sB, m;
    time 0 is the start of the pre-cue period, the cue starts at t_cue. m: optional common input (n_tr, n) at
    resolution dt; by default it is drawn with dips."""
    rng = np.random.default_rng(seed)
    cued = np.asarray(cued, np.int64)
    dt = p['dt']
    n = int(round((p['t_pre'] + p['cue_dur'] + p['t_delay']) / dt))
    if m is None:
        m = common_input(len(cued), n, dt, p['dip_rate'], p['dip_mean'], p['dip_depth'], p['dip_shape'], rng)
    t = np.arange(n) * dt
    cue_on = (t >= p['t_pre']) & (t < p['t_pre'] + p['cue_dur'])
    xi = _ou(rng.standard_normal((len(cued), n, 3)), np.exp(-dt / p['tau_xi']))
    rec = int(round(1.0 / dt))
    out = _sim(m, cue_on, cued, xi, np.array([float(p[k]) for k in KEYS]), rec)
    return dict(rA=out[..., 0], rB=out[..., 1], bg=out[..., 2], sA=out[..., 3], sB=out[..., 4],
                m=m[:, rec - 1::rec][:, :out.shape[1]], cued=cued, t_cue=p['t_pre'])


def memory_correct(sim, last_ms=100):
    """Memory held at the end of the delay: NMDA gating of the cued pool exceeds the uncued pool's by > 0.05."""
    sA, sB = sim['sA'][:, -last_ms:].mean(1), sim['sB'][:, -last_ms:].mean(1)
    return np.where(sim['cued'] == 0, sA - sB, sB - sA) > 0.05
