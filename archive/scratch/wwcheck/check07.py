"""First check, Wong et al. (2007) reduced model (values from p. 2 of the paper), constant input, no dips:
does a brief cue start a persistent state lasting 1.6 s with the uncued pool low?"""
import numpy as np
from numba import njit

P = dict(a=270.0, b=108.0, d=0.154, J_s=0.3725, J_c=0.1137, I_0=0.3297, gamma=0.641, tau_S=60.0,
         sigma=0.009, tau_n=2.0, J_ext=1.1e-3, mu_0=30.0, f=0.45, dt=0.1)

@njit(cache=True)
def run(n_tr, T, t_on, t_off, Ic, Iu, a, b, d, Js, Jc, I0, gam, tauS, sig, taun, dt, seed, noisy):
    np.random.seed(seed)
    n = int(T / dt); rec = int(1 / dt)
    out = np.zeros((n_tr, n // rec, 4))
    for k in range(n_tr):
        S1 = 0.0; S2 = 0.0; e1 = I0; e2 = I0
        for i in range(n):
            t = i * dt
            on = t >= t_on and t < t_off
            x1 = Js * S1 - Jc * S2 + e1 + (Ic if on else 0.0)
            x2 = Js * S2 - Jc * S1 + e2 + (Iu if on else 0.0)
            u1 = a * x1 - b; u2 = a * x2 - b
            r1 = u1 / (1 - np.exp(-d * u1)) if abs(u1) > 1e-9 else 1 / d
            r2 = u2 / (1 - np.exp(-d * u2)) if abs(u2) > 1e-9 else 1 / d
            S1 += dt * (-S1 / tauS + (1 - S1) * gam * r1 / 1000.0)
            S2 += dt * (-S2 / tauS + (1 - S2) * gam * r2 / 1000.0)
            if noisy:
                e1 += dt / taun * (I0 - e1) + np.sqrt(dt / taun) * sig * np.random.randn()
                e2 += dt / taun * (I0 - e2) + np.sqrt(dt / taun) * sig * np.random.randn()
            if i % rec == rec - 1:
                j = i // rec
                out[k, j, 0] = r1; out[k, j, 1] = r2; out[k, j, 2] = S1; out[k, j, 3] = S2
    return out

def go(p, n_tr, T, t_on, t_off, Ic, Iu, seed=1, noisy=True):
    return run(n_tr, T, t_on, t_off, Ic, Iu, p['a'], p['b'], p['d'], p['J_s'], p['J_c'], p['I_0'], p['gamma'],
               p['tau_S'], p['sigma'], p['tau_n'], p['dt'], seed, noisy)

if __name__ == '__main__':
    Jm = P['J_ext'] * P['mu_0']
    stims = {f"c'=100pct (cued {Jm*1.45:.4f}, uncued {Jm*0.55:.4f} nA)": (Jm * 1.45, Jm * 0.55),
             f'cued pool only ({Jm*1.45:.4f} nA)': (Jm * 1.45, 0.0)}
    for tauS in (60.0, 100.0, 150.0):
        p = dict(P, tau_S=tauS)
        o = go(p, 1, 3000, 0, 0, 0, 0, noisy=False)
        o2 = go(p, 1, 3000, 0, 1e9, 0.2, 0.0, noisy=False)[:, :, :]   # strong drive to pool 1: is there a persistent state?
        p2 = dict(p); 
        print(f'tau_S {tauS:.0f} ms: spontaneous (no noise) r = {o[0,-1,0]:.2f} Hz, S = {o[0,-1,2]:.3f}')
        # does a high state persist after a strong 500 ms push? (noise-free)
        o3 = go(p, 1, 3000, 300, 800, 0.1, 0.0, noisy=False)
        print(f'   after a strong 500 ms push (0.1 nA): cued {o3[0,-1,0]:.1f} Hz, uncued {o3[0,-1,1]:.2f} Hz (noise-free, 2.2 s later)')
        for name, (Ic, Iu) in stims.items():
            for dur in (50, 100, 200, 500, 1000):
                o = go(p, 300, 300 + dur + 1600, 300, 300 + dur, Ic, Iu)
                rc, ru = o[:, -100:, 0].mean(1), o[:, -100:, 1].mean(1)
                ok = (rc > 10) & (ru < 5)
                spont = ((rc < 5) & (ru < 5)).mean(); wrong = ((ru > 10) & (rc < 5)).mean()
                print(f'   {name}, {dur:4d} ms: persistent & correct at 1.6 s {ok.mean():.2f} | spontaneous {spont:.2f} | wrong pool {wrong:.2f} | cued {np.median(rc):5.1f} Hz, uncued {np.median(ru):4.1f} Hz')
