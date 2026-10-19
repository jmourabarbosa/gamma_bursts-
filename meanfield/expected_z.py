"""Expected-z On/Off states: the decoder's z-score computed from the rates, without spikes.

For strong regularisation (lam = 1000) the L2 logistic decoder is linear in the data: the leave-one-out
output for test trial k is f_k = N_k / (2 lam), N_k = sum_{j != k} tau_j (x_k . x_j), with x the z-scored
window counts and tau the training labels (+-1; true or shuffled). Writing x = mu + eps, where mu is the
expected count (from the rates) and eps Poisson noise (variance = mean), both in z-scored units:

    E[N_k]   = sum_{j != k} tau_j (mu_k . mu_j)
    Var[N_k] = sum_i v_ki (sum_{j != k} tau_j mu_ji)^2 + sum_{j != k} [ mu_k^2 . v_j + v_k . v_j ]

N_k is drawn as Gaussian with that mean and variance, with the autocorrelation that overlapping 100 ms
windows stepped by 10 ms give to Poisson noise (triangular, zero beyond 100 ms). Confidence =
sigmoid(tau_k-signed f_k); the true labels and 20 shuffled label vectors go through the unchanged
pipeline (z against the shuffled null, cluster-mass On states, Off = >= 5 bins with z < 0.84).
Nothing is fitted: the recording (cells, tuning, trials) enters only through mu and v.
"""
import numpy as np
from numba import njit, prange

from . import recording as R
from .surrogate import LAM, label_states


@njit(cache=True, parallel=True)
def _moments(lamw, T):
    """lamw: expected counts (n_tr, K, n_w); T: (n_lab, n_tr) labels +-1.
    Returns mean and SD of N for each label vector, trial and window: (n_lab, n_tr, n_w) x 2."""
    n_tr, K, n_w = lamw.shape
    n_lab = T.shape[0]
    mean = np.zeros((n_lab, n_tr, n_w))
    sd = np.zeros((n_lab, n_tr, n_w))
    for w in prange(n_w):
        L = lamw[:, :, w]
        m = np.zeros(K)
        for i in range(n_tr):
            m += L[i]
        m /= n_tr
        var = np.zeros(K)
        for i in range(n_tr):
            var += (L[i] - m) ** 2
        var = var / n_tr + m                      # latent variance across trials + Poisson
        sdf = np.sqrt(var) + 1e-6
        mu = (L - m.reshape(1, K)) / sdf.reshape(1, K)
        v = L / (sdf ** 2).reshape(1, K)
        G = mu @ mu.T
        mu2 = mu ** 2
        vsum = np.zeros(K)
        for i in range(n_tr):
            vsum += v[i]
        for l in range(n_lab):
            s = T[l] @ mu                          # sum_j tau_j mu_j  (K,)
            for k in range(n_tr):
                sk = s - T[l, k] * mu[k]           # exclude j = k
                nm = 0.0
                for j in range(n_tr):
                    if j != k:
                        nm += T[l, j] * G[k, j]
                vo = vsum - v[k]
                var_n = 0.0
                for i in range(K):
                    var_n += v[k, i] * sk[i] ** 2 + mu2[k, i] * vo[i] + v[k, i] * vo[i]
                mean[l, k, w] = nm
                sd[l, k, w] = np.sqrt(var_n)
    return mean, sd


def window_noise(shape, rng, step=R.STEP, win=R.WIN):
    """Unit-variance Gaussian noise along the last axis with the autocorrelation of overlapping
    windows: a moving sum of win/step independent increments."""
    k = win // step
    e = rng.standard_normal(shape[:-1] + (shape[-1] + k - 1,))
    c = np.cumsum(np.concatenate([np.zeros(shape[:-1] + (1,)), e], axis=-1), axis=-1)
    return (c[..., k:] - c[..., :-k]) / np.sqrt(k)


def expected_counts(ses):
    """Expected counts in the decoder's windows, (n_tr, K, n_w), from the 10 ms bin rates."""
    lam, tc = R.window_counts(ses['binned'])
    keep = np.isin(tc, ses['tc'])
    return np.ascontiguousarray(lam[:, :, keep])


def expected_pipeline(ses, lam=LAM, n_shuffle=20, seed=0):
    y = np.asarray(ses['y'])
    t = 2.0 * y - 1.0
    cols = [t] + [2.0 * np.random.default_rng(seed * 1000 + 7 + s).permutation(y) - 1.0 for s in range(n_shuffle)]
    T = np.ascontiguousarray(np.array(cols))
    mean, sd = _moments(expected_counts(ses), T)
    rng = np.random.default_rng(seed + 12345)
    N = mean + sd * window_noise(mean.shape, rng)
    f = np.clip(t[None, :, None] * N / (2.0 * lam), -30, 30)    # signed toward the test trial's true label
    conf = 1.0 / (1.0 + np.exp(-f))
    state, z, thr = label_states(conf[0], conf[1:], ses['valid'])
    return dict(conf=conf[0], z=z, state=state, thr=thr, stats=R.state_stats(state, ses['count_mask']),
                acc=float((conf[0][:, ses['dec_mask']] > 0.5).mean()), zmean=mean[0] * t[:, None] / np.maximum(sd[0], 1e-9))
