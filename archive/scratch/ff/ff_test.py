"""Which edge of the feed-forward lag matters: silencing at dip onset or the kick at dip end?
q is computed outside the simulation with different lags at the two edges, then the model runs with that q."""
import numpy as np
from numba import njit
from paper_model import model as M, readout as R
from paper_model.off_vs_memory import RHO, reference

@njit(cache=True)
def sim_q(m, q_all, cue, xi, p, rec):
    r_max, kappa, w_a, beta, w_I, w_EI, w_ff, tau_ff, I_0, tau_r, tau_I, cue_amp, dt, tau_N, w_s, sigma = p
    n_tr, n = m.shape
    out = np.zeros((n_tr, n // rec, 4))
    for k in range(n_tr):
        rc = 1.0; ru = 1.0; rI = 1.0; sc = 0.0; su = 0.0
        for i in range(n):
            x = -w_I * rI + I_0 * m[k, i] - w_ff * q_all[k, i]
            fc = r_max / (1 + np.exp(-(w_s * sc + w_a * rc + x + cue_amp * cue[i] + sigma * xi[k, i, 0] - 1) / kappa))
            fu = r_max / (1 + np.exp(-(w_s * su + w_a * ru + x + sigma * xi[k, i, 1] - 1) / kappa))
            rI += dt / tau_I * (w_EI * (rc + ru) - rI)
            rc += dt / tau_r * (fc - rc); ru += dt / tau_r * (fu - ru)
            sc += dt / tau_N * (-sc + beta * rc * (1 - sc)); su += dt / tau_N * (-su + beta * ru * (1 - su))
            if i % rec == rec - 1:
                j = i // rec
                out[k, j, 0] = rc; out[k, j, 1] = ru; out[k, j, 2] = sc; out[k, j, 3] = su
    return out

@njit(cache=True)
def make_q(m, dt, tau_down, tau_up):
    """q follows m with time constant tau_down when m < q (dip onset) and tau_up when m > q (dip end); 0 = instant."""
    q = np.empty_like(m)
    for k in range(m.shape[0]):
        v = m[k, 0]
        for i in range(m.shape[1]):
            tau = tau_down if m[k, i] < v else tau_up
            v = m[k, i] if tau == 0 else v + dt / tau * (m[k, i] - v)
            q[k, i] = v
    return q

def run(tau_down, tau_up, gain, n_tr=2112, seed=1):
    p = M.params()
    rng = np.random.default_rng(seed)
    dt = p['dt']; n = int(round((p['t_pre'] + p['cue_dur'] + p['t_delay']) / dt))
    m = M.common_input(n_tr, n, p, rng); m[:, :int(p['t_pre'] / dt)] = 1.0
    cue = ((np.arange(n) * dt >= p['t_pre']) & (np.arange(n) * dt < p['t_pre'] + p['cue_dur'])).astype(float)
    xi = M._ou(rng.standard_normal((n_tr, n, 2)), np.exp(-dt / p['tau_xi']))
    q = make_q(m, dt, tau_down, tau_up)
    rec = 10
    o = sim_q(m, q, cue, xi, np.array([float(p[k]) for k in M.KEYS]), rec)
    sim = dict(rc=o[..., 0], ru=o[..., 1], sc=o[..., 2], su=o[..., 3], m=m[:, rec - 1::rec], t=np.arange(n // rec) - p['t_pre'])
    dw, tz = R.rate_diff(sim)
    delay, valid = (tz >= 500) & (tz <= 1400), (tz >= 0) & (tz <= 1400)
    r2 = np.random.default_rng(seed + 7)
    st = R.label(gain * dw + R.noise(dw.shape, RHO, r2), valid, R.null_threshold(dw.shape, RHO, r2))
    s = R.stats(st, delay)
    return M.memory_correct(sim).mean(), s



def run_scaled(alpha, gain, n_tr=2112, seed=1, tau=15):
    """Kick only, scaled: q' = m - alpha (m - q_kick); the positive pulse at dip end is alpha x the fitted one."""
    p = M.params()
    rng = np.random.default_rng(seed)
    dt = p['dt']; n = int(round((p['t_pre'] + p['cue_dur'] + p['t_delay']) / dt))
    m = M.common_input(n_tr, n, p, rng); m[:, :int(p['t_pre'] / dt)] = 1.0
    cue = ((np.arange(n) * dt >= p['t_pre']) & (np.arange(n) * dt < p['t_pre'] + p['cue_dur'])).astype(float)
    xi = M._ou(rng.standard_normal((n_tr, n, 2)), np.exp(-dt / p['tau_xi']))
    q = m - alpha * (m - make_q(m, dt, 0, tau))
    o = sim_q(m, q, cue, xi, np.array([float(p[k]) for k in M.KEYS]), 10)
    sim = dict(rc=o[..., 0], ru=o[..., 1], sc=o[..., 2], su=o[..., 3], m=m[:, 9::10], t=np.arange(n // 10) - p['t_pre'])
    dw, tz = R.rate_diff(sim)
    delay, valid = (tz >= 500) & (tz <= 1400), (tz >= 0) & (tz <= 1400)
    r2 = np.random.default_rng(seed + 7)
    st = R.label(gain * dw + R.noise(dw.shape, RHO, r2), valid, R.null_threshold(dw.shape, RHO, r2))
    return M.memory_correct(sim).mean(), R.stats(st, delay)

if __name__ == '__main__':
    import sys
    R.M = M
    ses, _ = R.load_data()
    gain = reference(ses)
    if sys.argv[1:] == ['scale']:
        for a in (0.25, 0.5, 0.75, 1.0, 1.5):
            mem, s = run_scaled(a, gain)
            print(f"kick x {a:.2f} (peak {0.8 * 0.5987 * a:.2f}): memory {mem:.3f} | Off {s['n_off']:.2f}/trial {s['off_mean']:.0f} ms frac {s['frac_off']:.2f} | On {s['n_on']:.2f}/trial {s['on_mean']:.0f} ms frac {s['frac_on']:.2f}", flush=True)
