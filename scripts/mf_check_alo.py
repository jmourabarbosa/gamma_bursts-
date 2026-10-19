"""Check the approximate leave-one-out decoder (meanfield.decoder) against the exact one
(gamma_wm.analysis.loo_confidence_dual) on synthetic Poisson data."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_wm.analysis import loo_confidence_dual  # noqa: E402
from meanfield.decoder import alo_confidence  # noqa: E402

rng = np.random.default_rng(0)
for n_tr, K, sig in ((40, 100, 0.3), (132, 329, 0.15), (132, 329, 0.0)):
    y = np.repeat([0, 1], n_tr // 2)
    base = rng.lognormal(np.log(0.5), 0.8, K)
    tune = rng.normal(0, sig, K)
    lam = base[None, :, None] * np.exp(tune[None, :, None] * (2 * y[:, None, None] - 1)) * np.ones((1, 1, 30))
    C = rng.poisson(lam).astype(float)
    t0 = time.time(); ex = loo_confidence_dual(C, y, 10.0); t1 = time.time()
    ap = alo_confidence(C, y, 10.0); t2 = time.time()
    exs = loo_confidence_dual(C, y, 10.0, shuffle_rng=np.random.default_rng(1))
    aps = alo_confidence(C, y, 10.0, shuffle_rng=np.random.default_rng(1))
    print(f'n_tr={n_tr} K={K} tuning={sig}: exact {t1 - t0:.1f}s, ALO {t2 - t1:.2f}s | '
          f'r={np.corrcoef(ex.ravel(), ap.ravel())[0, 1]:.4f} max|d|={np.abs(ex - ap).max():.3f} '
          f'mean conf {ex.mean():.3f}/{ap.mean():.3f} | null r={np.corrcoef(exs.ravel(), aps.ravel())[0, 1]:.4f} '
          f'null sd {exs.std():.4f}/{aps.std():.4f}')
