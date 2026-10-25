"""Cross-check of the numba simulator (network.py) against Brian2 on the same equations.

A reduced ring (512 E + 128 I cells, recurrent conductances x 4 to keep the summed input) without heterogeneity
and without dips; E->E through explicit all-to-all synapses with weights W(theta_i - theta_j) in Brian2, through the
Fourier series of W in network.py. Compared: spontaneous E and I rates before the cue, delay rates, the bump's peak
rate and its position at the end of the delay. Writes spiking/results/validate_brian2.json.

    python -m spiking.validate_brian2
"""
import json
import os

import numpy as np

from . import network as N, run as RU

SCALE = 4
SMALL = dict(N_E=512, N_I=128, g_EE=RU.TUNED['g_EE'] * SCALE, g_EI=N.P['g_EI'] * SCALE, g_IE=N.P['g_IE'] * SCALE,
             g_II=N.P['g_II'] * SCALE, J_plus=RU.TUNED['J_plus'], sd_vth=0.0, h_max=0.0, dip_rate=0.0,
             cue_amp=RU.TUNED['cue_amp'], t_pre=1000.0, t_delay=1500.0, t_probe=0.0)
CUE = 180.0


def summary(rE, rI, t, cue_deg, theta):
    """rE (n_tr, n_t, N_E) and rI rates in 10 ms bins."""
    pre, dl, end = (t >= -800) & (t < 0), (t >= 500) & (t < 1500), (t >= 1300) & (t < 1500)
    prof = rE[:, dl].mean(1)
    w = 7
    sm = np.array([np.convolve(np.r_[r[-w:], r, r[:w]], np.ones(2 * w + 1) / (2 * w + 1), 'valid') for r in prof])
    pe = rE[:, end].mean(1)
    ang = np.rad2deg(np.arctan2(pe @ np.sin(theta), pe @ np.cos(theta)))
    return dict(E_spont=float(rE[:, pre].mean()), I_spont=float(rI[:, pre].mean()), E_delay=float(rE[:, dl].mean()),
                I_delay=float(rI[:, dl].mean()), bump_peak=float(sm.max(1).mean()),
                end_error_deg=[float(x) for x in N.angdiff(ang, cue_deg)])


def run_numba(n_tr=8, seed=1):
    net = N.Network(N.params(**SMALL))
    sim = N.simulate(net, [CUE] * n_tr, seed=seed, rec_idx=np.arange(640))
    c = sim['counts'].astype(float) * 100
    return summary(c[..., :512], c[..., 512:], sim['t10'], CUE, net.theta_E)


def run_brian2(n_tr=3, seed=1):
    import brian2 as b2
    from brian2 import ms, mV, nS, pF, pA, Hz
    b2.prefs.codegen.target = 'cython'
    p = N.params(**SMALL)
    net0 = N.Network(p)
    NE, NI = p['N_E'], p['N_I']
    out = []
    for k in range(n_tr):
        b2.start_scope()
        b2.seed(seed + k)
        b2.defaultclock.dt = p['dt'] * ms
        eqs = '''
        dv/dt = (-gL*(v - EL) - g_ext*s_ext*(v - E_E) - g_N*sN_tot*(v - E_E)/(1 + Mg*exp(-0.062*v/mV)/3.57)
                 - g_G*sG_tot*(v - E_I) + I_cue)/C : volt (unless refractory)
        ds_ext/dt = -s_ext/tau_AMPA : 1
        dx/dt = -x/tau_x : 1
        ds/dt = -s/tau_N + alpha*x*(1 - s) : 1
        dsg/dt = -sg/tau_GABA : 1
        sN_tot : 1
        sG_tot : 1
        I_cue : amp
        gL : siemens
        C : farad
        g_ext : siemens
        g_N : siemens
        g_G : siemens
        '''
        ns = dict(EL=p['EL'] * mV, E_E=p['E_E'] * mV, E_I=p['E_I'] * mV, Mg=p['Mg'], tau_AMPA=p['tau_AMPA'] * ms,
                  tau_x=p['tau_x'] * ms, tau_N=p['tau_N'] * ms, alpha=p['alpha'] / ms, tau_GABA=p['tau_GABA'] * ms)
        # on a spike: reset, and the cell's own NMDA (E) or GABA (I) gating jumps
        G = b2.NeuronGroup(NE + NI, eqs, threshold='v >= -50*mV',
                           reset='v = -60*mV\nx += 1.0*(i < %d)\nsg += 1.0*(i >= %d)' % (NE, NE),
                           refractory='(i < %d)*%g*ms + (i >= %d)*%g*ms' % (NE, p['ref_E'], NE, p['ref_I']),
                           method='euler', namespace=ns)
        E, I = G[:NE], G[NE:]
        E.gL, E.C, E.g_ext, E.g_N, E.g_G = p['gL_E'] * nS, p['C_E'] * pF, p['g_ext_E'] * nS, p['g_EE'] * nS, p['g_IE'] * nS
        I.gL, I.C, I.g_ext, I.g_N, I.g_G = p['gL_I'] * nS, p['C_I'] * pF, p['g_ext_I'] * nS, p['g_EI'] * nS, p['g_II'] * nS
        rng = np.random.default_rng(seed + k)
        G.v = (p['EL'] + (p['Vth'] - p['EL']) * rng.random(NE + NI)) * mV
        ext = b2.PoissonInput(G, 's_ext', 1, p['nu_ext'] * Hz, weight=1.0)
        # E->E NMDA with weights W, E->I NMDA uniform; I->all GABA uniform
        S_EE = b2.Synapses(E, E, 'w : 1\nsN_tot_post = w*s_pre : 1 (summed)')
        S_EE.connect()
        dth = (net0.theta_E[S_EE.i[:]] - net0.theta_E[S_EE.j[:]]) % (2 * np.pi)
        S_EE.w = net0.W[np.round(dth / (2 * np.pi) * NE).astype(int) % NE]
        S_EI = b2.Synapses(E, I, 'sN_tot_post = s_pre : 1 (summed)')
        S_EI.connect()
        S_GA = b2.Synapses(I, G, 'sG_tot_post = sg_pre : 1 (summed)')
        S_GA.connect()
        mon = b2.SpikeMonitor(G)
        T0 = p['t_pre']
        b2.run(T0 * ms)
        E.I_cue = net0.cue_current([CUE])[0] * pA
        b2.run(p['cue_dur'] * ms)
        E.I_cue = 0 * pA
        b2.run((p['t_delay'] - p['cue_dur']) * ms)
        t, i = np.asarray(mon.t / ms), np.asarray(mon.i)
        nb = int((T0 + p['t_delay']) / 10)
        cnt = np.zeros((nb, NE + NI))
        np.add.at(cnt, (np.minimum((t / 10).astype(int), nb - 1), i), 1)
        out.append(cnt * 100)
        print(f'brian2 trial {k} done', flush=True)
    c = np.array(out)
    t10 = np.arange(c.shape[1]) * 10.0 - p['t_pre']
    return summary(c[..., :NE], c[..., NE:], t10, CUE, net0.theta_E)


if __name__ == '__main__':
    os.makedirs(RU.OUT, exist_ok=True)
    res = dict(params=SMALL, numba=run_numba(), brian2=run_brian2())
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(RU.OUT, 'validate_brian2.json'), 'w'), indent=1)
