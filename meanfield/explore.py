"""Deterministic checks of the rate model: steady rates, and whether the cued pool restarts after a
single dip of length d (the memory-survival curve)."""
import numpy as np

from . import model as M


def one_dip(p, d_list, depth=1.0, t_dip=900.0, sigma=0.0, n_rep=1, seed=0):
    """Cue A, then one dip [t_dip, t_dip + d) of the given depth. Returns per d: fraction of trials
    where A is high / B is high 600 ms after the dip, and the rates before the dip."""
    p = dict(p, sigma=sigma, sigma_bg=sigma, t_delay=t_dip - p['t_pre'] - p['cue_dur'] + max(d_list) + 700)
    dt = p['dt']
    T = p['t_pre'] + p['cue_dur'] + p['t_delay']
    n = int(round(T / dt))
    rows = []
    for d in d_list:
        m = np.ones((n_rep, n))
        m[:, int(t_dip / dt):int((t_dip + d) / dt)] = 1 - depth
        k = np.ones(int(10 / dt)) / int(10 / dt)
        m = np.array([np.convolve(np.r_[np.ones(len(k)), r, np.ones(len(k))], k, 'same')[len(k):-len(k)] for r in m])
        s = M.simulate(p, np.zeros(n_rep, int), seed=seed, m=m)
        e = int(t_dip + d + 600)
        rows.append(dict(d=d, A_high=float((s['rA'][:, e] > 20).mean()), B_high=float((s['rB'][:, e] > 20).mean()),
                         rA_pre=float(s['rA'][:, int(t_dip) - 5].mean()), rB_pre=float(s['rB'][:, int(t_dip) - 5].mean()),
                         bg_pre=float(s['bg'][:, int(t_dip) - 5].mean()), I_pre=float(s['I'][:, int(t_dip) - 5].mean()),
                         sA_pre=float(s['sA'][:, int(t_dip) - 5].mean()),
                         rA_dip=float(s['rA'][:, int(t_dip + d / 2)].mean()), sA_end=float(s['sA'][:, int(t_dip + d) - 1].mean())))
    return rows


def spontaneous(p, T=3000.0, n_rep=20, seed=0):
    """No cue: fraction of trials in which a pool ignites (should be ~0)."""
    p = dict(p, cue_amp=0.0, t_delay=T)
    s = M.simulate(p, np.zeros(n_rep, int), seed=seed)
    return float(((s['rA'] > 20) | (s['rB'] > 20)).any(1).mean()), float(s['rA'][:, -1].mean()), float(s['bg'][:, -1].mean())
