"""Spiking ring attractor (Compte, Brunel, Goldman-Rakic & Wang 2000) with NMDA decay 150 ms, a common input
m(t) to the excitatory cells that briefly drops ("attention dips"), and an overshoot of that input when it recovers.

Neurons: leaky integrate-and-fire, N_E pyramidal cells and N_I interneurons, each on a ring at angle theta.

    C dV/dt = -g_L (V - E_L) - I_ext - I_NMDA - I_GABA + I_cue

    I_ext  = g_ext s_ext (V - E_E)                       external Poisson AMPA drive (rate nu_ext, x m_eff(t) on E)
    I_NMDA = g_N[post] (V - E_E) B(V) sum_j W_ij s_j     recurrent NMDA from E cells, B(V) the Mg2+ block
    I_GABA = g_G[post] (V - E_I) sum_j s_GABA_j          recurrent GABA_A from I cells (all-to-all, untuned)

    NMDA gating of each E cell j:  dx/dt = -x/tau_x + sum_k delta(t - t_k),  ds/dt = -s/tau_N + alpha x (1 - s)
    AMPA / GABA_A gating:          ds/dt = -s/tau + sum_k delta(t - t_k)

E->E weights are tuned in angle, W(dtheta) = J_minus + (J_plus - J_minus) exp(-dtheta^2 / 2 sigma^2), normalised to
a mean of 1; E->I, I->E and I->I are uniform (Compte et al. 2000). The E->E sum is applied through the Fourier
series of W (exact up to K_FOURIER harmonics), which lets many trials run in parallel on a few cores.

Common input (section 4 of AGENT_INSTRUCTIONS.md): m(t) = 1 except during dips (Poisson onsets, gamma lengths) where
it drops to 1 - depth, with 10 ms ramps. The E cells receive external Poisson input at nu_ext * m_eff(t), with

    m_eff = m + A (m - m_bar),   m_bar follows m instantly into a dip and recovers with tau_kick

so the drive overshoots briefly after each dip (option A, "the kick"). Interneurons get a constant nu_ext.

Heterogeneity: firing thresholds are drawn per cell (Gaussian, SD sd_vth), fixed for the network.

Units: ms, mV, nS, pF, pA, Hz.
"""
import numpy as np
from numba import njit, prange

# all parameters in one place; sources and notes in PARAMETERS.md
P = dict(
    # populations (Compte et al. 2000)
    N_E=2048, N_I=512,
    # pyramidal cells
    C_E=500.0, gL_E=25.0, EL=-70.0, Vth=-50.0, Vreset=-60.0, ref_E=2.0,
    # interneurons
    C_I=200.0, gL_I=20.0, ref_I=1.0,
    # reversal potentials, Mg2+ block (Jahr & Stevens 1990 as used by Wang 1999)
    E_E=0.0, E_I=-70.0, Mg=1.0,
    # synaptic kinetics
    tau_AMPA=2.0, tau_x=2.0, alpha=0.5, tau_N=150.0, tau_GABA=10.0,
    # conductances per synapse (nS); g_EE retuned for tau_N = 150 ms (Compte et al. 2000: 0.381 nS at tau_N = 100 ms)
    g_EE=0.381, g_EI=0.292, g_IE=1.336, g_II=1.024,
    # external drive: Poisson AMPA at nu_ext (one lumped train of 1,000 inputs x 1.8 Hz)
    nu_ext=1800.0, g_ext_E=3.1, g_ext_I=2.38,
    # E->E footprint
    J_plus=1.62, sigma_deg=18.0,
    # heterogeneity
    sd_vth=1.0,
    # cue: current to E cells, Gaussian in angle
    cue_amp=200.0, cue_sigma_deg=18.0, cue_dur=50.0,
    # trial timing (ms); cue onset at t_pre
    t_pre=500.0, t_delay=1600.0, dt=0.1,
    # response probe after the delay: the common input back at 1 (no dips), to test whether the bump restarts
    t_probe=0.0,
    # common input: dips and kick
    dip_rate=10.0, dip_mean=90.0, dip_shape=8.7, dip_depth=0.6, ramp=10.0, kick_A=1.0, tau_kick=15.0,
    # truncation of the Fourier series of W
    K_FOURIER=24,
    net_seed=1,
)


def params(**over):
    p = {**P, **over}
    unknown = set(over) - set(P)
    if unknown:
        raise KeyError(f'unknown parameters {unknown}')
    return p


class Network:
    """Fixed structure of one network (angles, thresholds, Fourier tables of W)."""

    def __init__(self, p):
        self.p = p
        NE, NI = p['N_E'], p['N_I']
        self.theta_E = 2 * np.pi * np.arange(NE) / NE
        self.theta_I = 2 * np.pi * np.arange(NI) / NI
        rng = np.random.default_rng(p['net_seed'])
        self.vth = np.r_[p['Vth'] + p['sd_vth'] * rng.standard_normal(NE),
                         p['Vth'] + p['sd_vth'] * rng.standard_normal(NI)]
        # W(dtheta) sampled on the grid, normalised to mean 1
        sig = np.deg2rad(p['sigma_deg'])
        d = np.angle(np.exp(1j * self.theta_E))
        g = np.exp(-d ** 2 / (2 * sig ** 2))
        J_minus = (1 - p['J_plus'] * g.mean()) / (1 - g.mean())
        self.J_minus = J_minus
        self.W = J_minus + (p['J_plus'] - J_minus) * g        # W[k] = weight at angular distance theta_E[k]
        what = np.fft.rfft(self.W).real / NE                    # W even -> real coefficients
        K = p['K_FOURIER']
        # W(th_i - th_j) = sum_k wk[k] cos k(th_i - th_j), so sum_j W_ij s_j = sum_k wk[k] (cos k th_i C_k + sin k th_i S_k)
        # with C_k = sum_j cos(k th_j) s_j and S_k = sum_j sin(k th_j) s_j
        self.wk = np.r_[what[0], 2 * what[1:K + 1]]
        k = np.arange(K + 1)[:, None]
        self.cos_tab = np.cos(k * self.theta_E[None])
        self.sin_tab = np.sin(k * self.theta_E[None])
        self.trunc_err = np.abs(self.W - (self.wk[:, None] * self.cos_tab).sum(0)).max()

    def cue_current(self, cue_deg):
        """(n_trials, N_E) cue current."""
        d = np.angle(np.exp(1j * (self.theta_E[None] - np.deg2rad(np.asarray(cue_deg, float))[:, None])))
        return self.p['cue_amp'] * np.exp(-d ** 2 / (2 * np.deg2rad(self.p['cue_sigma_deg']) ** 2))


def dips(n_tr, T, p, rng, t_free=0.0, fixed=None):
    """(n_tr, n) common input m(t) at resolution dt over T ms, 1 before t_free.
    fixed: optional list of (onset, length) in ms per trial (single-dip tests) instead of random dips.
    Returns m and the list of dips [(onset, length)] per trial."""
    dt = p['dt']
    n = int(round(T / dt))
    m = np.ones((n_tr, n))
    events = []
    for k in range(n_tr):
        ev = []
        if fixed is not None:
            ev = [fixed[k]] if fixed[k] is not None and fixed[k][1] > 0 else []
        elif p['dip_rate'] > 0:
            # dip process started 500 ms before t_free so that it is near-stationary there; clipped at t_free
            t = t_free - rng.random() * 500 + rng.exponential(1000 / p['dip_rate'])
            while t < T:
                d = rng.gamma(p['dip_shape'], p['dip_mean'] / p['dip_shape'])
                if t + d > t_free:
                    ev.append((max(t, t_free), t + d - max(t, t_free)))
                t += d + rng.exponential(1000 / p['dip_rate'])
        for a, d in ev:
            i0, i1 = int(a / dt), int(min(T, a + d) / dt)
            m[k, i0:i1] = 1 - p['dip_depth']
        events.append(ev)
    w = int(round(p['ramp'] / dt))
    if w > 1:
        pad = np.c_[np.ones((n_tr, w)), m, np.ones((n_tr, w))]
        m = np.array([np.convolve(r, np.ones(w) / w, mode='same') for r in pad])[:, w:-w]
    return m, events


def kick(m, p):
    """m_eff = m + A (m - m_bar): m_bar follows m instantly down and recovers with tau_kick."""
    return _kick(m, p['kick_A'], np.exp(-p['dt'] / p['tau_kick']))


@njit(cache=True)
def _kick(m, A, a):
    out = np.empty_like(m)
    for k in range(m.shape[0]):
        mb = m[k, 0]
        for i in range(m.shape[1]):
            if m[k, i] < mb:
                mb = m[k, i]
            else:
                mb = m[k, i] + (mb - m[k, i]) * a
            out[k, i] = m[k, i] + A * (m[k, i] - mb)
    return out


@njit(parallel=True, cache=True)
def _run(m_eff, cue, cue_on, cue_off, pf, vth, wk, cos_tab, sin_tab, rec_idx, seeds, raster_on, s_from, s_to):
    """Simulate trials in parallel. Returns spike counts of the recorded cells in 10 ms bins, E population vector of
    spikes and of NMDA gating every 10 ms, mean NMDA gating and spike count of each E cell from step s_from to
    s_to, and
    (if raster_on) the E raster in 1 ms bins."""
    (NE, NI, C_E, gL_E, EL, Vreset, ref_E, C_I, gL_I, ref_I, E_E, E_I, Mg, tau_AMPA, tau_x, alpha, tau_N, tau_GABA,
     g_EE, g_EI, g_IE, g_II, nu_ext, g_ext_E, g_ext_I, dt) = pf
    NE = int(NE); NI = int(NI)
    n_tr, n = m_eff.shape
    K1 = wk.shape[0]
    b10 = int(round(10.0 / dt)); b1 = int(round(1.0 / dt))
    nb = n // b10
    counts = np.zeros((n_tr, nb, rec_idx.shape[0]), np.uint8)
    popv = np.zeros((n_tr, nb, 6))        # spikes: sum cos, sum sin, count; NMDA gating: sum cos, sum sin, sum
    s_end = np.zeros((n_tr, NE))
    c_end = np.zeros((n_tr, NE), np.int32)
    n_ms = n // b1 if raster_on else 0
    raster = np.zeros((n_tr if raster_on else 0, n_ms, NE), np.uint8)
    rec_map = -np.ones(NE + NI, np.int64)
    for r in range(rec_idx.shape[0]):
        rec_map[rec_idx[r]] = r
    dA = np.exp(-dt / tau_AMPA); dX = np.exp(-dt / tau_x); dG = np.exp(-dt / tau_GABA)
    nE_ref = int(round(ref_E / dt)); nI_ref = int(round(ref_I / dt))
    lam_I = nu_ext * dt / 1000.0
    for b in prange(n_tr):
        np.random.seed(seeds[b])
        N = NE + NI
        V = EL + (vth - EL) * np.random.random(N)
        ref = np.zeros(N, np.int64)
        sA = np.zeros(N)
        x = np.zeros(NE); s = np.zeros(NE)
        sG = np.zeros(NI)
        # external Poisson spikes by time rescaling: each cell waits until its integrated rate passes an Exp(1) draw
        nxt = np.random.exponential(1.0, N)
        Ck = np.zeros(K1); Sk = np.zeros(K1)
        spk = np.zeros(N, np.uint8)
        for i in range(n):
            lam_E = nu_ext * m_eff[b, i] * dt / 1000.0
            # Fourier projections of the NMDA gating
            for k in range(K1):
                c = 0.0; sn = 0.0
                for j in range(NE):
                    c += cos_tab[k, j] * s[j]
                    sn += sin_tab[k, j] * s[j]
                Ck[k] = c * wk[k]; Sk[k] = sn * wk[k]
            sum_s = Ck[0] / wk[0]
            sum_g = 0.0
            for j in range(NI):
                sum_g += sG[j]
            cue_now = cue_on <= i < cue_off
            for j in range(N):
                spk[j] = 0
                lam = lam_E if j < NE else lam_I
                nxt[j] -= lam
                while nxt[j] <= 0:
                    sA[j] += 1.0
                    nxt[j] += np.random.exponential(1.0)
                if ref[j] > 0:
                    ref[j] -= 1
                    continue
                v = V[j]
                Bm = 1.0 / (1.0 + Mg * np.exp(-0.062 * v) / 3.57)
                if j < NE:
                    rec = 0.0
                    for k in range(K1):
                        rec += Ck[k] * cos_tab[k, j] + Sk[k] * sin_tab[k, j]
                    I = (gL_E * (v - EL) + g_ext_E * sA[j] * (v - E_E) + g_EE * rec * (v - E_E) * Bm
                         + g_IE * sum_g * (v - E_I))
                    if cue_now:
                        I -= cue[b, j]
                    v += -dt * I / C_E
                else:
                    I = (gL_I * (v - EL) + g_ext_I * sA[j] * (v - E_E) + g_EI * sum_s * (v - E_E) * Bm
                         + g_II * sum_g * (v - E_I))
                    v += -dt * I / C_I
                if v >= vth[j]:
                    v = Vreset
                    spk[j] = 1
                    ref[j] = nE_ref if j < NE else nI_ref
                V[j] = v
            # gating updates
            for j in range(N):
                sA[j] *= dA
            ib = i // b10
            for j in range(NE):
                x[j] = x[j] * dX + spk[j]
                s[j] += dt * (-s[j] / tau_N + alpha * x[j] * (1.0 - s[j]))
                if spk[j]:
                    popv[b, ib, 0] += cos_tab[1, j]; popv[b, ib, 1] += sin_tab[1, j]; popv[b, ib, 2] += 1
                    if raster_on:
                        raster[b, i // b1, j] += 1
            for j in range(NI):
                sG[j] = sG[j] * dG + spk[NE + j]
            for j in range(N):
                if spk[j] and rec_map[j] >= 0:
                    counts[b, ib, rec_map[j]] += 1
            if i % b10 == b10 - 1:
                cs = 0.0; ss = 0.0; tot = 0.0
                for j in range(NE):
                    cs += cos_tab[1, j] * s[j]; ss += sin_tab[1, j] * s[j]; tot += s[j]
                popv[b, ib, 3] = cs; popv[b, ib, 4] = ss; popv[b, ib, 5] = tot
            if s_from <= i < s_to:
                for j in range(NE):
                    s_end[b, j] += s[j]
                    c_end[b, j] += spk[j]
        for j in range(NE):
            s_end[b, j] /= max(s_to - s_from, 1)
    return counts, popv, s_end, c_end, raster


PF_KEYS = ('N_E', 'N_I', 'C_E', 'gL_E', 'EL', 'Vreset', 'ref_E', 'C_I', 'gL_I', 'ref_I', 'E_E', 'E_I', 'Mg',
           'tau_AMPA', 'tau_x', 'alpha', 'tau_N', 'tau_GABA', 'g_EE', 'g_EI', 'g_IE', 'g_II', 'nu_ext', 'g_ext_E',
           'g_ext_I', 'dt')


def simulate(net, cue_deg, seed=0, rec_idx=None, m=None, raster=False, dip_free=0.0, fixed_dips=None, m_eff=None):
    """Trials with cues at cue_deg (one per trial). Time axis: cue onset at t_pre; dips only after cue onset +
    dip_free (default: from cue onset). Returns a dict with 10 ms spike counts of rec_idx (cells 0..N_E-1 are E,
    N_E.. are I), population vectors, end-of-delay NMDA profile and rates, the common input (1 ms) and the dips.
    The delay is followed by t_probe ms with the input at 1 (response probe).
    m, m_eff: optional common input without / with the kick at resolution dt (n_trials, n_steps)."""
    p = net.p
    cue_deg = np.atleast_1d(np.asarray(cue_deg, float))
    n_tr = len(cue_deg)
    rng = np.random.default_rng(seed)
    T = p['t_pre'] + p['t_delay']
    if m is None:
        m, ev = dips(n_tr, T, p, rng, t_free=p['t_pre'] + dip_free, fixed=fixed_dips)
        m = np.c_[m, np.ones((n_tr, int(round(p['t_probe'] / p['dt']))))]
    else:
        ev = None
    if m_eff is None:
        m_eff = kick(m, p)
    dt = p['dt']
    rec_idx = np.arange(0) if rec_idx is None else np.asarray(rec_idx, np.int64)
    seeds = rng.integers(0, 2 ** 31 - 1, n_tr)
    cue_on, cue_off = int(round(p['t_pre'] / dt)), int(round((p['t_pre'] + p['cue_dur']) / dt))
    n = m.shape[1]
    s_to = int(round(T / dt))                      # end of the delay
    s_from = s_to - int(round(100 / dt))
    pf = np.array([float(p[k]) for k in PF_KEYS])
    counts, popv, s_end, c_end, ras = _run(m_eff, net.cue_current(cue_deg), cue_on, cue_off, pf, net.vth, net.wk,
                                    net.cos_tab, net.sin_tab, rec_idx, seeds, raster, s_from, s_to)
    b1 = int(round(1 / dt))
    return dict(counts=counts, popv=popv, s_end=s_end, c_end=c_end, raster=ras if raster else None,
                m=m[:, b1 - 1::b1], m_eff=m_eff[:, b1 - 1::b1], dips=ev, cue_deg=cue_deg,
                t10=np.arange(counts.shape[1]) * 10.0 - p['t_pre'],     # left edge of each 10 ms bin, from cue onset
                t1=np.arange(m.shape[1] // b1) * 1.0 - p['t_pre'])


def angdiff(a, b):
    """Signed angular difference a - b in degrees, in (-180, 180]."""
    return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180


def _smooth(x, w):
    """Circular running mean over 2w + 1 cells."""
    return np.array([np.convolve(np.r_[r[-w:], r, r[:w]], np.ones(2 * w + 1) / (2 * w + 1), 'valid') for r in x])


def memory(sim, net, min_rate=15.0, min_s=0.2, probe_amp=2.0):
    """Memory held at the end of the delay (last 100 ms), read from the whole E population.
    rate: bump active if the E rate profile (last 100 ms, smoothed over +-10 deg) peaks more than min_rate Hz above
          its median; held if its peak (circular mean of the profile above half its maximum) is within 22.5 deg
          of the cue.
    nmda: the same on the mean NMDA gating profile, which must peak more than min_s above its median (spontaneous
          activity alone gives a gating of about 0.1-0.2); when the bump is not active (e.g. the trial ends in a
          dip) this is the only criterion that can hold.
    probe: after the delay the input returns to 1 for t_probe ms; held if a bump is then active at the cue.
    Returns boolean / float arrays."""
    NE = net.p['N_E']
    w = int(round(NE * 10 / 360))
    out = {}
    for nm, prof, thr in (('rate', _smooth(sim['c_end'] / 0.1, w), min_rate), ('nmda', _smooth(sim['s_end'], w), min_s)):
        base = np.median(prof, 1, keepdims=True)
        top = np.maximum(prof - base - 0.5 * (prof.max(1, keepdims=True) - base), 0)
        ang = np.rad2deg(np.arctan2(top @ np.sin(net.theta_E), top @ np.cos(net.theta_E)))
        active = (prof.max(1) - base[:, 0]) > thr
        err = angdiff(ang, sim['cue_deg'])
        out[f'active_{nm}'] = active
        out[f'err_{nm}'] = err
        out[f'peak_{nm}'] = prof.max(1)
        out[f'held_{nm}'] = active & (np.abs(err) <= 22.5)
        out[f'wrong_{nm}'] = active & (np.abs(err) > 22.5)
    # combined criterion: the rate bump if active, else the NMDA trace
    out['held'] = np.where(out['active_rate'], out['held_rate'], out['held_nmda'])
    # response probe: bump in the last 100 ms of the probe (first spatial harmonic of the E rate above probe_amp Hz)
    if net.p['t_probe'] > 0:
        pv = sim['popv'][:, -10:]
        amp = np.hypot(pv[..., 0].sum(1), pv[..., 1].sum(1)) / NE / 0.1
        err = angdiff(np.rad2deg(np.arctan2(pv[..., 1].sum(1), pv[..., 0].sum(1))), sim['cue_deg'])
        out['active_probe'] = amp > probe_amp
        out['err_probe'] = err
        out['held_probe'] = out['active_probe'] & (np.abs(err) <= 22.5)
        out['wrong_probe'] = out['active_probe'] & (np.abs(err) > 22.5)
    return out
