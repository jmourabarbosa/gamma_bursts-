"""Check the simulator against Brunel (2000), Table 1 / Fig. 8
(N_E = 10,000, N_I = 2,500, C_E = 1,000, J = 0.1 mV, D = 1.5 ms, no memory structure)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_wm.network import Params, build_network, simulate  # noqa: E402

CASES = [  # g, nu_ext/nu_thr, label, paper rate, paper global frequency
    (3.0, 2.0, 'A  SR', '-', '-'),
    (6.0, 4.0, 'B  SI fast', '60.7 Hz', '180 Hz'),
    (5.0, 2.0, 'C  AI', '37.7 Hz', '-'),
    (4.5, 0.9, 'D  SI slow', '5.5 Hz', '22 Hz'),
]

for g, nu, lab, r_paper, f_paper in CASES:
    p = Params(g=g, nu_ext_ratio=nu)
    res = simulate(build_network(p), 1200.0, seed=1)
    pop = res['pop'][2000:].sum(1).astype(float)          # drop first 200 ms
    rate = pop.sum() / ((p.N_E + p.N_I) * (len(pop) * p.dt * 1e-3))
    x = pop - pop.mean()
    f = np.fft.rfftfreq(len(x), p.dt * 1e-3)
    P = np.abs(np.fft.rfft(x)) ** 2
    m = (f > 5) & (f < 1000)
    print(f'{lab:11s} rate {rate:6.1f} Hz (paper {r_paper:>7s})   '
          f'spectral peak {f[m][np.argmax(P[m])]:5.0f} Hz (paper {f_paper})')
