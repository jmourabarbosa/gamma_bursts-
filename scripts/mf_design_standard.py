"""Design the 'standard' variant of the rate model: choose the structural constants so that, with fast-rise
NMDA (Wang 1999), 80% NMDA recurrence, active state ~40 Hz, low state 1 Hz, tau_I = 5 ms and no feed-forward
inhibition, the pools are bistable, winner-take-all, and silenced by deep dips. Prints candidate designs and
checks them by simulation (stable plateau, no oscillation, no spontaneous ignition).

usage: python scripts/mf_design_standard.py
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanfield import model as M  # noqa: E402

G, TAU = 0.641e-3, 100.0          # NMDA rise per Hz per ms; design at tau_N = 100 ms


def s_ss(r):
    x = G * TAU * r
    return x / (1 + x)


def design(kappa, rmax, J, f=0.8, rh=40.0, rl=1.0):
    Phi_inv = lambda x: 1 + kappa * np.log(x / (rmax - x))
    u = lambda x: (1 - f) * x / rh + f * s_ss(x) / s_ss(rh)
    R = (Phi_inv(rh) - Phi_inv(rl) + J * (rh - rl)) / (u(rh) - u(rl))
    E = Phi_inv(rl) - R * u(rl) + 2 * J * rl
    rr = np.linspace(0.05, rmax - 1e-3, 30000)
    g1 = E + R * u(rr) - J * (rr + rl) - Phi_inv(rr)
    g2 = E + R * u(rr) - 2 * J * rr - Phi_inv(rr)
    up = rr[(rr > 1.2 * rl)][np.where(np.diff(np.sign(g1[rr > 1.2 * rl])) > 0)[0]]
    if len(up) == 0 or R <= 0 or E <= 0:
        return None
    m_c = -(g1 - E)[rr > up[0]].max() / E
    return dict(kappa=kappa, rmax=rmax, J=J, w_s=R * f / s_ss(rh), w_a=R * (1 - f) / rh, E=E, r_sad=float(up[0]),
                sym=float(g2[rr > 3].max()), m_c=float(m_c))


def check(d, sigma=0.02):
    p = M.params(nmda_wang=1.0, gamma=G, tau_N=TAU, w_s=d['w_s'], w_a=d['w_a'], kappa=d['kappa'], r_max=d['rmax'],
                 w_I=d['J'] / 0.5, w_EI=0.5, w_ff=0.0, I_0=d['E'], tau_I=5.0, dip_rate=0.0, sigma=0.0, sigma_bg=0.0,
                 cue_amp=1.0)
    s = M.simulate(p, [0], seed=0)
    x = s['rA'][0, 800:1900]
    q = M.simulate(dict(p, cue_amp=0.0, sigma=sigma, sigma_bg=sigma), [0] * 10, seed=1)
    return dict(high=float(x.mean()), osc=float(x.std()), B=float(s['rB'][0, 800:1900].mean()),
                spont=float(((q['rA'] > 20) | (q['rB'] > 20)).any(1).mean()))


def main():
    out = []
    for kappa, rmax, J in itertools.product([0.04, 0.06, 0.1, 0.15], [60, 80, 120], [0.01, 0.02, 0.03, 0.05, 0.08]):
        d = design(kappa, rmax, J)
        if d and d['sym'] < -0.02 and 3 < d['r_sad'] < 20 and 0.3 < d['m_c'] < 0.8:
            out.append(d)
    print(len(out), 'analytic candidates')
    for d in sorted(out, key=lambda d: d['r_sad'])[:12]:
        c = check(d)
        print({k: round(v, 4) for k, v in d.items()}, '|', {k: round(v, 2) for k, v in c.items()})


if __name__ == '__main__':
    main()
