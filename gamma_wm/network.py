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
NS, A, B, INH = 0, 1, 2, 3


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
    # memory structure
    f: float = 0.0            # fraction of E neurons in each selective pool (A and B)
    beta: float = 0.0         # strength of the rank-one perturbation
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
    xi = np.zeros(N)
    xi[group == A] = 1.0
    xi[group == B] = -1.0

    # fixed in-degree: for each postsynaptic neuron choose C_E E and C_I I presynaptic cells
    pre = _fixed_indegree(N, p.N_E, p.N_I, C_E, C_I, p.seed)
    post = np.repeat(np.arange(N), C_E + C_I)

    w = np.where(pre < p.N_E, p.J, -p.g * p.J)
    # rank-one perturbation on E->E synapses
    ee = (pre < p.N_E) & (post < p.N_E)
    w[ee] *= 1.0 + p.beta * xi[post[ee]] * xi[pre[ee]]

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
    n_groups = 4
    pop_count = np.zeros((n_steps, n_groups), dtype=np.int32)
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


def simulate(net: Network, T: float, ext_profile=None, seed=None, V0=None,
             rec_V_idx=None, ext_res_ms=1.0, max_spikes=None):
    """Simulate for T ms.

    ext_profile: optional array (n_bins, 4) of external rates (Hz per external
    synapse) for groups [NS, A, B, INH], one row per `ext_res_ms` ms. Defaults to the
    constant p.nu_ext for all groups.
    Returns dict with spike times (ms), spike indices, population counts per step,
    final V and recorded membrane potentials.
    """
    p = net.p
    n_steps = int(round(T / p.dt))
    ext_res = int(round(ext_res_ms / p.dt))
    if ext_profile is None:
        ext_profile = np.full((n_steps // ext_res + 1, 4), p.nu_ext)
    ext_profile = np.asarray(ext_profile, dtype=np.float64)
    assert ext_profile.shape[0] * ext_res >= n_steps
    ext_lambda = ext_profile * p.C_E * p.dt * 1e-3  # expected external spikes per step
    N = p.N_E + p.N_I
    if V0 is None:
        rng = np.random.default_rng(p.seed + 12345 if seed is None else seed + 12345)
        V0 = rng.uniform(p.V_r, p.theta, N)
    if rec_V_idx is None:
        rec_V_idx = np.zeros(0, dtype=np.int64)
    if max_spikes is None:
        max_spikes = int(N * T * 1e-3 * 200)
    st, si, pop, Vf, Vrec = _simulate(
        n_steps, p.dt, p.tau, p.theta, p.V_r, int(round(p.t_ref / p.dt)),
        int(net.delays.max()), net.indptr, net.targets, net.weights, net.delays, net.group,
        p.J, ext_lambda, ext_res, V0.astype(np.float64),
        np.random.randint(2**31 - 1) if seed is None else seed, max_spikes,
        np.asarray(rec_V_idx, dtype=np.int64))
    return dict(t=st * p.dt, i=si, pop=pop, V=Vf, V_rec=Vrec, T=T, dt=p.dt)
