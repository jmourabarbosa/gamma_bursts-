"""Binary delayed-response task: a brief cue to pool A or B, followed by a delay."""
import numpy as np

from .network import A, B, NGROUPS, simulate


def cue_profile(p, T, cue=None, t_cue=1000.0, cue_dur=150.0, cue_strength=0.1):
    """External-rate table (1 ms rows, columns NS/A/B/INH/IA/IB). The cue multiplies the
    external rate onto the cued pool by (1 + cue_strength) during [t_cue, t_cue+cue_dur)."""
    ext = np.full((int(np.ceil(T)) + 1, NGROUPS), p.nu_ext)
    if cue is not None:
        col = {'A': A, 'B': B}[cue]
        ext[int(t_cue):int(t_cue + cue_dur), col] *= 1.0 + cue_strength
    return ext


def slow_input_noise(n, sd, tau, seed, floor=0.0):
    """Multiplicative slow fluctuation of the external drive shared by all cells:
    1 + sd * OU(tau), clipped below at `floor` (1 ms rows)."""
    rng = np.random.default_rng(seed)
    a = np.exp(-1.0 / tau)
    x = np.empty(n)
    x[0] = rng.standard_normal()
    eps = rng.standard_normal(n) * np.sqrt(1 - a * a)
    for i in range(1, n):
        x[i] = a * x[i - 1] + eps[i]
    return np.clip(1.0 + sd * x, floor, None)


def run_trial(net, cue='A', t_pre=1000.0, cue_dur=150.0, t_delay=2500.0,
              cue_strength=0.1, seed=0, V0=None, ext_noise_sd=0.0, ext_noise_tau=200.0, ext_noise_floor=0.0, **kw):
    T = t_pre + cue_dur + t_delay
    ext = cue_profile(net.p, T, cue, t_pre, cue_dur, cue_strength)
    if ext_noise_sd > 0:
        ext *= slow_input_noise(len(ext), ext_noise_sd, ext_noise_tau, seed + 7919, ext_noise_floor)[:, None]
    res = simulate(net, T, ext_profile=ext, seed=seed, V0=V0, **kw)
    res.update(cue=cue, t_cue=t_pre, cue_dur=cue_dur, t_delay_on=t_pre + cue_dur, T=T)
    return res
