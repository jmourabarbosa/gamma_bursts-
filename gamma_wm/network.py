"""Sparse E/I network of leaky integrate-and-fire neurons (Brunel 2000, model A)
with a rank-one perturbation of the E->E weights that stores one binary memory.

Connectivity
------------
Every neuron receives exactly C_E = eps*N_E excitatory and C_I = eps*N_I inhibitory
recurrent synapses, plus C_ext = C_E external Poisson synapses of rate nu_ext.
Synapses are delta currents: a presynaptic spike makes the postsynaptic potential jump
by J (excitatory) or -g*J (inhibitory) after a delay D (fixed, or spread uniformly
over [D - D_spread, D + D_spread]).

Memory structure
----------------
Excitatory neurons carry a pattern xi_i in {+1 (pop A), -1 (pop B), 0 (non-selective)}.
(Optionally, inhibitory partner pools I_A / I_B carry +1 / -1 and the E->I and I->E
blocks get their own low-rank terms; see Params.)
Existing E->E synapses are rescaled as

    J_ij = J * (1 + beta * xi_i * xi_j),

so the mean E->E connectivity is eps*J*(1 1^T + beta * xi xi^T): the random sparse
matrix plus a rank-one perturbation. Within-pool synapses are potentiated (1+beta),
A<->B synapses depressed (1-beta), all others unchanged. No synaptic plasticity of any
kind is present during the simulation: weights are fixed.
"""
from dataclasses import dataclass, field

import numpy as np
from numba import njit

# group ids
NS, A, B, INH, IA, IB = 0, 1, 2, 3, 4, 5   # IA/IB: inhibitory partners of pools A/B
NGROUPS = 6


@dataclass
class Params:
    N_E: int = 10000
    N_I: int = 2500
    eps: float = 0.1          # connection probability (fixed in-degree)
    J: float = 0.1            # EPSP amplitude, mV
    g: float = 5.0            # relative inhibition
    nu_ext_ratio: float = 2.0  # nu_ext / nu_thr
    D: float = 1.5            # synaptic delay, ms
    D_spread: float = 0.0     # delays uniform in [D - D_spread, D + D_spread] (0: all equal)
    tau: float = 20.0         # membrane time constant, ms
    theta: float = 20.0       # threshold, mV
    V_r: float = 10.0         # reset, mV
    t_ref: float = 2.0        # refractory period, ms
    dt: float = 0.1           # integration step, ms
    # synaptic kinetics (exponential currents; tau = 0 means Brunel's delta synapse).
    # Each spike delivers the same total charge as the delta synapse (PSP integral J),
    # E charge split into AMPA (1 - q_nmda) and NMDA (q_nmda). NMDA is linear (no Mg block).
    tau_ampa: float = 0.0     # ms
    tau_nmda: float = 0.0     # ms
    tau_gaba: float = 0.0     # ms
    q_nmda: float = 0.0       # fraction of recurrent E charge carried by NMDA
    # memory structure
    f: float = 0.0            # fraction of E neurons in each selective pool (A and B)
    beta: float = 0.0         # strength of the rank-one perturbation (E->E block)
    # pool-specific inhibition: fraction of I neurons in each partner pool (I_A, I_B) and
    # strength of the rank-one perturbation of the E->I and I->E blocks
    f_I: float = 0.0
    beta_ie: float = 0.0      # E -> I
    beta_ei: float = 0.0      # I -> E
    # optional uniform boost of all pool-to-pool synapses in each block (second low-rank
    # term |xi||xi|^T): within-pool factor 1 + alpha + beta, between-pool 1 + alpha - beta
    alpha: float = 0.0        # E -> E
    alpha_ie: float = 0.0     # E -> I
    alpha_ei: float = 0.0     # I -> E
    seed: int = 0

    @property
    def C_E(self):
        return int(round(self.eps * self.N_E))

    @property
    def C_I(self):
        return int(round(self.eps * self.N_I))

    @property
    def nu_thr(self):
        """External rate (Hz) for which the mean external input reaches threshold."""
        return self.theta / (self.J * self.C_E * self.tau * 1e-3)

    @property
    def nu_ext(self):
        return self.nu_ext_ratio * self.nu_thr

    @property
    def has_kinetics(self):
        return self.tau_ampa > 0 or self.tau_gaba > 0 or (self.tau_nmda > 0 and self.q_nmda > 0)


@dataclass
class Network:
    p: Params
    group: np.ndarray        # (N,) group id per neuron
    xi: np.ndarray           # (N,) pattern (+1, -1, 0; 0 for inhibitory)
    indptr: np.ndarray       # CSR over presynaptic neurons -> outgoing synapses
    targets: np.ndarray
    weights: np.ndarray
    delays: np.ndarray       # per-synapse delay in integration steps
    extra: dict = field(default_factory=dict)


def build_network(p: Params) -> Network:
    rng = np.random.default_rng(p.seed)
    N = p.N_E + p.N_I
    C_E, C_I = p.C_E, p.C_I

    group = np.full(N, NS, dtype=np.int64)
    group[p.N_E:] = INH
    n_sel = int(round(p.f * p.N_E))
    perm = rng.permutation(p.N_E)
    group[perm[:n_sel]] = A
    group[perm[n_sel:2 * n_sel]] = B
    n_sel_I = int(round(p.f_I * p.N_I))
    perm_I = p.N_E + rng.permutation(p.N_I)
    group[perm_I[:n_sel_I]] = IA
    group[perm_I[n_sel_I:2 * n_sel_I]] = IB
    xi = np.zeros(N)
    xi[(group == A) | (group == IA)] = 1.0
    xi[(group == B) | (group == IB)] = -1.0

    # fixed in-degree: for each postsynaptic neuron choose C_E E and C_I I presynaptic cells
    pre = _fixed_indegree(N, p.N_E, p.N_I, C_E, C_I, p.seed)
    post = np.repeat(np.arange(N), C_E + C_I)

    w = np.where(pre < p.N_E, p.J, -p.g * p.J)
    # low-rank perturbation of each block:
    # w_ij *= 1 + alpha_block * |xi_i||xi_j| + beta_block * xi_i * xi_j
    pre_E, post_E = pre < p.N_E, post < p.N_E
    for blk, a, b in (((pre_E & post_E), p.alpha, p.beta),
                      ((pre_E & ~post_E), p.alpha_ie, p.beta_ie),
                      ((~pre_E & post_E), p.alpha_ei, p.beta_ei)):
        if a != 0 or b != 0:
            xpo, xpr = xi[post[blk]], xi[pre[blk]]
            fac = 1.0 + a * np.abs(xpo * xpr) + b * xpo * xpr
            assert fac.min() >= 0, 'Dale violation: between-pool factor 1 + alpha - beta < 0'
            w[blk] *= fac

    order = np.argsort(pre, kind="stable")
    targets = post[order].astype(np.int32)
    weights = w[order].astype(np.float32)
    counts = np.bincount(pre, minlength=N)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    d_steps = int(round(p.D / p.dt))
    if p.D_spread > 0:
        lo = max(1, int(round((p.D - p.D_spread) / p.dt)))
        hi = int(round((p.D + p.D_spread) / p.dt))
        delays = rng.integers(lo, hi + 1, len(targets)).astype(np.int16)
    else:
        delays = np.full(len(targets), d_steps, dtype=np.int16)
    return Network(p, group, xi, indptr, targets, weights, delays)


@njit(cache=True)
def _fixed_indegree(N, N_E, N_I, C_E, C_I, seed):
    np.random.seed(seed)
    pre = np.empty(N * (C_E + C_I), dtype=np.int64)
    poolE = np.arange(N_E)
    poolI = np.arange(N_E, N_E + N_I)
    k = 0
    for i in range(N):
        for j in range(C_E):  # partial Fisher-Yates
            r = j + np.random.randint(N_E - j)
            poolE[j], poolE[r] = poolE[r], poolE[j]
            pre[k] = poolE[j]
            k += 1
        for j in range(C_I):
            r = j + np.random.randint(N_I - j)
            poolI[j], poolI[r] = poolI[r], poolI[j]
            pre[k] = poolI[j]
            k += 1
    return pre


@njit(cache=True)
def _simulate(n_steps, dt, tau, theta, V_r, ref_steps, max_delay,
              indptr, targets, weights, delays, group, J_ext, ext_lambda, ext_res,
              V0, seed, max_spikes, rec_V_idx):
    np.random.seed(seed)
    N = V0.shape[0]
    L = max_delay + 1
    buf = np.zeros((L, N), dtype=np.float32)
    V = V0.copy()
    ref = np.zeros(N, dtype=np.int64)
    decay = np.exp(-dt / tau)
    spk_t = np.empty(max_spikes, dtype=np.int32)
    spk_i = np.empty(max_spikes, dtype=np.int32)
    n_spk = 0
    pop_count = np.zeros((n_steps, ext_lambda.shape[1]), dtype=np.int32)
    n_rec = rec_V_idx.shape[0]
    V_rec = np.zeros((n_steps, n_rec), dtype=np.float32)
    for t in range(n_steps):
        slot = t % L
        lam_row = (t * 1) // ext_res
        for i in range(N):
            inp = buf[slot, i]
            buf[slot, i] = 0.0
            if ref[i] > 0:
                ref[i] -= 1
                continue
            lam = ext_lambda[lam_row, group[i]]
            n_ext = np.random.poisson(lam) if lam > 0 else 0
            V[i] = V[i] * decay + inp + J_ext * n_ext
            if V[i] >= theta:
                V[i] = V_r
                ref[i] = ref_steps
                if n_spk < max_spikes:
                    spk_t[n_spk] = t
                    spk_i[n_spk] = i
                    n_spk += 1
                pop_count[t, group[i]] += 1
                for k in range(indptr[i], indptr[i + 1]):
                    buf[(t + delays[k]) % L, targets[k]] += weights[k]
        for r in range(n_rec):
            V_rec[t, r] = V[rec_V_idx[r]]
    return spk_t[:n_spk], spk_i[:n_spk], pop_count, V, V_rec


@njit(cache=True)
def _simulate_syn(n_steps, dt, tau, theta, V_r, ref_steps, max_delay,
                  indptr, targets, weights, delays, group, N_E, J_ext, ext_lambda, ext_res,
                  V0, seed, max_spikes, rec_V_idx, tau_a, tau_n, tau_g, q):
    """As _simulate, with exponentially decaying synaptic currents. x_c (mV/ms) obeys
    dx_c/dt = -x_c/tau_c; a spike of efficacy w adds w*(1 - exp(-dt/tau_c))/dt (= w/tau_c
    for tau_c >> dt), so the summed input of one spike is exactly w, as for the delta
    synapse. Channels with tau_c = 0
    act as delta synapses. Synaptic currents keep evolving during refractoriness."""
    np.random.seed(seed)
    N = V0.shape[0]
    L = max_delay + 1
    bufE = np.zeros((L, N), dtype=np.float32)
    bufI = np.zeros((L, N), dtype=np.float32)
    xa = np.zeros(N)
    xn = np.zeros(N)
    xg = np.zeros(N)
    V = V0.copy()
    ref = np.zeros(N, dtype=np.int64)
    decay = np.exp(-dt / tau)
    da = np.exp(-dt / tau_a) if tau_a > 0 else 0.0
    dn = np.exp(-dt / tau_n) if tau_n > 0 else 0.0
    dg = np.exp(-dt / tau_g) if tau_g > 0 else 0.0
    # increment per unit efficacy such that sum_k dt * x_k = w exactly in discrete time
    ka = (1.0 - da) / dt
    kn = (1.0 - dn) / dt
    kg = (1.0 - dg) / dt
    spk_t = np.empty(max_spikes, dtype=np.int32)
    spk_i = np.empty(max_spikes, dtype=np.int32)
    n_spk = 0
    pop_count = np.zeros((n_steps, ext_lambda.shape[1]), dtype=np.int32)
    n_rec = rec_V_idx.shape[0]
    V_rec = np.zeros((n_steps, n_rec), dtype=np.float32)
    for t in range(n_steps):
        slot = t % L
        lam_row = t // ext_res
        for i in range(N):
            inE = bufE[slot, i]
            inI = bufI[slot, i]
            bufE[slot, i] = 0.0
            bufI[slot, i] = 0.0
            jump = 0.0
            # AMPA
            if tau_a > 0:
                xa[i] = xa[i] * da + (1.0 - q) * inE * ka
            else:
                jump += (1.0 - q) * inE
            # NMDA
            if tau_n > 0:
                xn[i] = xn[i] * dn + q * inE * kn
            else:
                jump += q * inE
            # GABA
            if tau_g > 0:
                xg[i] = xg[i] * dg + inI * kg
            else:
                jump += inI
            if ref[i] > 0:
                ref[i] -= 1
                continue
            lam = ext_lambda[lam_row, group[i]]
            n_ext = np.random.poisson(lam) if lam > 0 else 0
            V[i] = V[i] * decay + dt * (xa[i] + xn[i] + xg[i]) + jump + J_ext * n_ext
            if V[i] >= theta:
                V[i] = V_r
                ref[i] = ref_steps
                if n_spk < max_spikes:
                    spk_t[n_spk] = t
                    spk_i[n_spk] = i
                    n_spk += 1
                pop_count[t, group[i]] += 1
                if i < N_E:
                    for k in range(indptr[i], indptr[i + 1]):
                        bufE[(t + delays[k]) % L, targets[k]] += weights[k]
                else:
                    for k in range(indptr[i], indptr[i + 1]):
                        bufI[(t + delays[k]) % L, targets[k]] += weights[k]
        for r in range(n_rec):
            V_rec[t, r] = V[rec_V_idx[r]]
    return spk_t[:n_spk], spk_i[:n_spk], pop_count, V, V_rec


def simulate(net: Network, T: float, ext_profile=None, seed=None, V0=None,
             rec_V_idx=None, ext_res_ms=1.0, max_spikes=None):
    """Simulate for T ms.

    ext_profile: optional array (n_bins, NGROUPS) of external rates (Hz per external
    synapse) for groups [NS, A, B, INH, IA, IB], one row per `ext_res_ms` ms. Defaults to the
    constant p.nu_ext for all groups.
    Returns dict with spike times (ms), spike indices, population counts per step,
    final V and recorded membrane potentials.
    """
    p = net.p
    n_steps = int(round(T / p.dt))
    ext_res = int(round(ext_res_ms / p.dt))
    if ext_profile is None:
        ext_profile = np.full((n_steps // ext_res + 1, NGROUPS), p.nu_ext)
    ext_profile = np.asarray(ext_profile, dtype=np.float64)
    assert ext_profile.shape[0] * ext_res >= n_steps and ext_profile.shape[1] == NGROUPS
    ext_lambda = ext_profile * p.C_E * p.dt * 1e-3  # expected external spikes per step
    N = p.N_E + p.N_I
    if V0 is None:
        rng = np.random.default_rng(p.seed + 12345 if seed is None else seed + 12345)
        V0 = rng.uniform(p.V_r, p.theta, N)
    if rec_V_idx is None:
        rec_V_idx = np.zeros(0, dtype=np.int64)
    if max_spikes is None:
        max_spikes = int(N * T * 1e-3 * 200)
    common = (n_steps, p.dt, p.tau, p.theta, p.V_r, int(round(p.t_ref / p.dt)),
              int(net.delays.max()), net.indptr, net.targets, net.weights, net.delays, net.group)
    tail = (V0.astype(np.float64), np.random.randint(2**31 - 1) if seed is None else seed,
            max_spikes, np.asarray(rec_V_idx, dtype=np.int64))
    if p.has_kinetics:
        st, si, pop, Vf, Vrec = _simulate_syn(*common, p.N_E, p.J, ext_lambda, ext_res, *tail,
                                              p.tau_ampa, p.tau_nmda, p.tau_gaba, p.q_nmda)
    else:
        st, si, pop, Vf, Vrec = _simulate(*common, p.J, ext_lambda, ext_res, *tail)
    return dict(t=st * p.dt, i=si, pop=pop, V=Vf, V_rec=Vrec, T=T, dt=p.dt)
