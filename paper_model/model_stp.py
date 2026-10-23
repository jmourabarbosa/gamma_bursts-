"""Binary version of the Mi, Katkov & Tsodyks (2017, Neuron 93:323) short-term-plasticity working-memory model:
two excitatory clusters (cued c, uncued u) and one shared inhibitory pool, Equations 1-4 of the paper with P = 2.

    tau dh_X/dt = -h_X + J_EE u_X x_X R_X - J_EI R_I + I_b m(t) + I_e,X(t) + sigma xi_X     (Eq. 1)
    du_X/dt     = (U - u_X)/tau_f + U (1 - u_X) R_X                                         (Eq. 2)
    dx_X/dt     = (1 - x_X)/tau_d - u_X x_X R_X                                             (Eq. 3)
    tau dh_I/dt = -h_I + J_IE (R_c + n_other R_u)                                           (Eq. 4)

R(h) = alpha ln(1 + exp(h/alpha)) (Hz), the same for excitation and inhibition. Parameters from the Figure 1 legend:
J_EE 8, tau_f 1.5 s, tau_d 0.3 s, U 0.3, tau 8 ms, J_IE 1.75, J_EI 1.1, alpha 1.5. I_b is the background input that
the paper takes as the attentional state; here it is multiplied by the same common input m(t) with dips as in
model.py. Added to the paper's model: the OU noise sigma xi (tau_xi), as in model.py. The cue I_e (amplitude, duration)
is not given in the main text; cue_amp is our choice.
Time in ms; rates in Hz (multiplied by 1e-3 in Eqs. 2-3).
"""
import numpy as np
from numba import njit

from .model import common_input, _ou

FIXED = dict(J_EE=8.0, tau_f=1500.0, tau_d=300.0, U=0.3, tau=8.0, J_IE=1.75, J_EI=1.1, alpha=1.5,
             n_other=15.0, tau_xi=10.0, cue_dur=50.0, t_pre=300.0, t_delay=1600.0, dt=0.1)
FIT = dict(I_b=3.0, cue_amp=10.0, sigma=0.0, dip_rate=10.207, dip_mean=90.12, dip_shape=8.70, dip_depth=0.0)
KEYS = ('J_EE', 'tau_f', 'tau_d', 'U', 'tau', 'J_IE', 'J_EI', 'alpha', 'I_b', 'cue_amp', 'sigma', 'dt', 'n_other')


def params(**over):
    return {**FIXED, **FIT, **over}


@njit(cache=True)
def _R(h, a):
    return a * np.log1p(np.exp(h / a)) if h < 30 * a else h


@njit(cache=True)
def _sim(m, cue, xi, p, rec):
    J_EE, tau_f, tau_d, U, tau, J_IE, J_EI, a, I_b, cue_amp, sigma, dt, n_other = p
    n_tr, n = m.shape
    out = np.zeros((n_tr, n // rec, 6))    # Rc, Ru, uc, uu, xc, xu every rec steps (1 ms)
    for k in range(n_tr):
        hc = 0.0; hu = 0.0; hI = 0.0; uc = U; uu = U; xc = 1.0; xu = 1.0
        for i in range(n):
            Rc = _R(hc, a); Ru = _R(hu, a); RI = _R(hI, a)
            inp = I_b * m[k, i] - J_EI * RI
            dhc = (-hc + J_EE * uc * xc * Rc + inp + cue_amp * cue[i] + sigma * xi[k, i, 0]) / tau
            dhu = (-hu + J_EE * uu * xu * Ru + inp + sigma * xi[k, i, 1]) / tau
            hI += dt * (-hI + J_IE * (Rc + n_other * Ru)) / tau
            hc += dt * dhc
            hu += dt * dhu
            ruc = 1e-3 * Rc; ruu = 1e-3 * Ru
            duc = (U - uc) / tau_f + U * (1 - uc) * ruc; dxc = (1 - xc) / tau_d - uc * xc * ruc
            duu = (U - uu) / tau_f + U * (1 - uu) * ruu; dxu = (1 - xu) / tau_d - uu * xu * ruu
            uc += dt * duc; xc += dt * dxc; uu += dt * duu; xu += dt * dxu
            if i % rec == rec - 1:
                j = i // rec
                out[k, j, 0] = Rc; out[k, j, 1] = Ru; out[k, j, 2] = uc; out[k, j, 3] = uu
                out[k, j, 4] = xc; out[k, j, 5] = xu
    return out


def simulate(p, n_tr, seed=0, quiet=0.0):
    """Same interface as model.simulate: rc, ru (rates, Hz), uc, uu, xc, xu, m and t from cue onset (1 ms)."""
    rng = np.random.default_rng(seed)
    dt = p['dt']
    n = int(round((p['t_pre'] + p['cue_dur'] + p['t_delay']) / dt))
    m = common_input(n_tr, n, p, rng)
    m[:, :int((p['t_pre'] + quiet) / dt)] = 1.0
    cue = ((np.arange(n) * dt >= p['t_pre']) & (np.arange(n) * dt < p['t_pre'] + p['cue_dur'])).astype(float)
    xi = _ou(rng.standard_normal((n_tr, n, 2)), np.exp(-dt / p['tau_xi']))
    rec = int(round(1 / dt))
    o = _sim(m, cue, xi, np.array([float(p[k]) for k in KEYS]), rec)
    return dict(rc=o[..., 0], ru=o[..., 1], uc=o[..., 2], uu=o[..., 3], xc=o[..., 4], xu=o[..., 5],
                m=m[:, rec - 1::rec], t=np.arange(n // rec) - p['t_pre'])


def memory_correct(sim):
    """Memory held at the end of the delay: facilitation of the cued cluster exceeds the uncued one's by > 0.05."""
    return (sim['uc'][:, -100:] - sim['uu'][:, -100:]).mean(1) > 0.05
