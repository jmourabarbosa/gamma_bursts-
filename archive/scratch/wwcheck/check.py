"""First check of the Wong & Wang (2006) reduced model (main-text parameters), constant input: does a brief cue
switch the network into a persistent state that lasts 1.6 s with the uncued pool low?"""
import numpy as np
from numba import njit

WW = dict(c=310.0, I_E=125.0, g=0.16,                 # phi (Eq. 2), Hz/nA, Hz, s
          J_N11=0.1561, J_N12=0.0264,                 # nA
          J_A11=9.9026e-4, J_A12=6.5177e-5,           # nC, nA/Hz
          I_0=0.2346, gamma=0.641, tau_S=150.0,       # nA, -, ms (paper: 100 ms)
          tau_A=2.0, sigma=0.007, tau_r=2.0,          # ms, nA, ms
          J_ext=0.2243e-3, mu_0=30.0, dt=0.1)         # nA/Hz, Hz, ms

@njit(cache=True)
def run(n_tr, T, cue_on, cue_off, cue_amp, p, seed, noise):
    c, I_E, g, J11, J12, JA11, JA12, I0, gam, tauS, tauA, sig, taur, dt = p
    np.random.seed(seed)
    n = int(T / dt); rec = int(1 / dt)
    out = np.zeros((n_tr, n // rec, 4))
    for k in range(n_tr):
        r1 = 1.0; r2 = 1.0; S1 = 0.05; S2 = 0.05; e1 = 0.0; e2 = 0.0
        for i in range(n):
            t = i * dt
            I1 = cue_amp if (t >= cue_on and t < cue_off) else 0.0
            x1 = J11 * S1 - J12 * S2 + JA11 * r1 - JA12 * r2 + I0 + I1 + e1
            x2 = J11 * S2 - J12 * S1 + JA11 * r2 - JA12 * r1 + I0 + e2
            u1 = c * x1 - I_E; u2 = c * x2 - I_E
            f1 = u1 / (1 - np.exp(-g * u1)) if abs(u1) > 1e-9 else 1 / g
            f2 = u2 / (1 - np.exp(-g * u2)) if abs(u2) > 1e-9 else 1 / g
            r1 += dt / taur * (f1 - r1); r2 += dt / taur * (f2 - r2)
            S1 += dt * (-S1 / tauS + (1 - S1) * gam * r1 / 1000.0)
            S2 += dt * (-S2 / tauS + (1 - S2) * gam * r2 / 1000.0)
            if noise:
                e1 += dt / tauA * (-e1) + np.sqrt(dt / tauA) * sig * np.random.randn()
                e2 += dt / tauA * (-e2) + np.sqrt(dt / tauA) * sig * np.random.randn()
            if i % rec == rec - 1:
                j = i // rec
                out[k, j, 0] = r1; out[k, j, 1] = r2; out[k, j, 2] = S1; out[k, j, 3] = S2
    return out

def pv(p):
    return np.array([p[k] for k in ('c', 'I_E', 'g', 'J_N11', 'J_N12', 'J_A11', 'J_A12', 'I_0', 'gamma', 'tau_S', 'tau_A', 'sigma', 'tau_r', 'dt')])

if __name__ == '__main__':
    import sys
    for tauS in (100.0, 150.0):
        p = dict(WW, tau_S=tauS)
        o = run(1, 2000.0, 300.0, 300.0, 0.0, pv(p), 0, False)   # no cue, no noise: spontaneous state
        print(f'tau_S {tauS:.0f}: spontaneous r = {o[0,-1,0]:.2f} Hz, S = {o[0,-1,2]:.3f}')
        for cue_dur in (50, 100, 200, 500, 1000):
            for amp_label, amp in (('W&W stimulus, c=100% (2 J_ext mu0 = %.4f nA)' % (2 * WW['J_ext'] * WW['mu_0']), 2 * WW['J_ext'] * WW['mu_0']),):
                o = run(200, 300 + cue_dur + 1600, 300.0, 300.0 + cue_dur, amp, pv(p), 1, True)
                rc, ru = o[:, -100:, 0].mean(1), o[:, -100:, 1].mean(1)
                ok = (rc > 10) & (ru < 5)
                print(f'   cue {cue_dur:4d} ms {amp_label}: persistent (cued >10 Hz, uncued <5 Hz at 1.6 s) {ok.mean():.2f}; '
                      f'cued {np.median(rc):.1f} Hz, uncued {np.median(ru):.2f} Hz')
