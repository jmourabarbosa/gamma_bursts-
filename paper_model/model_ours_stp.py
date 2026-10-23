"""Our rate model (model.py) with short-term synaptic plasticity in place of the NMDA gating.

Everything is as in model.py except the slow recurrent term w_s*s_X: the NMDA gating variable s_X (tau_N) is replaced
by the same saturating synaptic activation without its own time constant, weighted by the Tsodyks-Markram efficacy
u_X x_X of the recurrent synapses (Mi, Katkov & Tsodyks 2017, Eqs. 2-3):

    tau_r  dr_X/dt = -r_X + Phi( w_s*S_X + w_a*r_X - w_I*r_I + I_0*m(t) - w_ff*q + cue_X + sigma*xi_X )
    S_X            = g / (1 + g),   g = beta * r_X * u_X x_X / U    (= the NMDA steady state when u x = U)
    du_X/dt        = (U - u_X)/tau_f + U (1 - u_X) r_X
    dx_X/dt        = (1 - x_X)/tau_d - u_X x_X r_X
    tau_I  dr_I/dt = -r_I + w_EI*(r_c + r_u)
    tau_ff dq/dt   = -q + m(t)

The slow variable is now facilitation u (tau_f) instead of NMDA (tau_N): tau_f takes the place of tau_N among the fitted
parameters. U = 0.3 and tau_d = 300 ms from Mi et al. (2017); all other fixed parameters as in model.py.
"""
import numpy as np
from numba import njit

from .model import FIXED as FIXED_NMDA, FIT as FIT_NMDA, common_input, _ou

FIXED = {**FIXED_NMDA, 'U': 0.3, 'tau_d': 300.0, 'depression': 1.0}   # depression 0: facilitation only (x = 1)
FIT = {**{k: v for k, v in FIT_NMDA.items() if k != 'tau_N'}, 'tau_f': 1500.0}
KEYS = ('r_max', 'kappa', 'w_a', 'beta', 'w_I', 'w_EI', 'w_ff', 'tau_ff', 'I_0', 'tau_r', 'tau_I', 'cue_amp',
        'dt', 'tau_f', 'tau_d', 'U', 'w_s', 'sigma', 'depression')

# facilitation only (depression 0), tau_f fixed at 1.5 s; fit_ours_stp --no-depression --tau-f 1500, seed 0
FIT_FAC = dict(depression=0.0, tau_f=1500.0, w_s=0.6919, dip_rate=5.486, dip_mean=211.73, dip_shape=0.70, dip_depth=0.3936,
               sigma=0.0050)

# with depression (tau_d 300 ms), tau_f fitted; fit_ours_stp seed 1, the best of 3 (all-units loss 37, poor fit)
FIT_DEP = dict(depression=1.0, tau_f=1295.7, w_s=0.5747, dip_rate=12.000, dip_mean=171.06, dip_shape=1.281, dip_depth=0.4277,
               sigma=0.0050)


def params(**over):
    return {**FIXED, **FIT, **over}


@njit(cache=True)
def _sim(m, cue, xi, p, rec):
    r_max, kappa, w_a, beta, w_I, w_EI, w_ff, tau_ff, I_0, tau_r, tau_I, cue_amp, dt, tau_f, tau_d, U, w_s, sigma, depr = p
    n_tr, n = m.shape
    out = np.zeros((n_tr, n // rec, 8))    # rc, ru, Sc, Su, uc, uu, xc, xu every rec steps (1 ms)
    for k in range(n_tr):
        rc = 1.0; ru = 1.0; rI = 1.0; q = m[k, 0]
        uc = U * (1 + tau_f * 1e-3) / (1 + U * tau_f * 1e-3); xc = 1 / (1 + depr * uc * tau_d * 1e-3)   # STP at rest (1 Hz)
        uu = uc; xu = xc
        for i in range(n):
            gc = beta * rc * uc * xc / U; gu = beta * ru * uu * xu / U
            Sc = gc / (1 + gc); Su = gu / (1 + gu)
            x = -w_I * rI + I_0 * m[k, i] - w_ff * q
            fc = r_max / (1 + np.exp(-(w_s * Sc + w_a * rc + x + cue_amp * cue[i] + sigma * xi[k, i, 0] - 1) / kappa))
            fu = r_max / (1 + np.exp(-(w_s * Su + w_a * ru + x + sigma * xi[k, i, 1] - 1) / kappa))
            rI += dt / tau_I * (w_EI * (rc + ru) - rI)
            ruc = 1e-3 * rc; ruu = 1e-3 * ru      # Hz -> per ms
            duc = (U - uc) / tau_f + U * (1 - uc) * ruc; dxc = (1 - xc) / tau_d - depr * uc * xc * ruc
            duu = (U - uu) / tau_f + U * (1 - uu) * ruu; dxu = (1 - xu) / tau_d - depr * uu * xu * ruu
            uc += dt * duc; xc += dt * dxc; uu += dt * duu; xu += dt * dxu
            rc += dt / tau_r * (fc - rc)
            ru += dt / tau_r * (fu - ru)
            q += dt / tau_ff * (m[k, i] - q)
            if i % rec == rec - 1:
                j = i // rec
                out[k, j, 0] = rc; out[k, j, 1] = ru; out[k, j, 2] = Sc; out[k, j, 3] = Su
                out[k, j, 4] = uc; out[k, j, 5] = uu; out[k, j, 6] = xc; out[k, j, 7] = xu
    return out


def simulate(p, n_tr, seed=0, quiet=0.0):
    """As model.simulate; returns rc, ru, sc, su (= S, the STP-weighted synaptic activation), uc, uu, xc, xu, m, t."""
    rng = np.random.default_rng(seed)
    dt = p['dt']
    n = int(round((p['t_pre'] + p['cue_dur'] + p['t_delay']) / dt))
    m = common_input(n_tr, n, p, rng)
    m[:, :int((p['t_pre'] + quiet) / dt)] = 1.0
    cue = ((np.arange(n) * dt >= p['t_pre']) & (np.arange(n) * dt < p['t_pre'] + p['cue_dur'])).astype(float)
    xi = _ou(rng.standard_normal((n_tr, n, 2)), np.exp(-dt / p['tau_xi']))
    rec = int(round(1 / dt))
    o = _sim(m, cue, xi, np.array([float(p[k]) for k in KEYS]), rec)
    return dict(rc=o[..., 0], ru=o[..., 1], sc=o[..., 2], su=o[..., 3], uc=o[..., 4], uu=o[..., 5], xc=o[..., 6],
                xu=o[..., 7], m=m[:, rec - 1::rec], t=np.arange(n // rec) - p['t_pre'])


def memory_correct(sim):
    """Memory held at the end of the delay: facilitation of the cued pool exceeds the uncued pool's by > 0.05."""
    return (sim['uc'][:, -100:] - sim['uu'][:, -100:]).mean(1) > 0.05
