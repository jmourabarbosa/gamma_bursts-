"""Panichello decoder at the paper's session size (~130 trials x ~330 units).

Same L2 logistic decoder as gamma_wm.analysis.loo_confidence_dual (one per time bin, z-scored counts,
sum loss + lam/2 |w|^2, unpenalised intercept), but the leave-one-trial-out prediction is the
approximate leave-one-out of Rad & Maleki (2020, JRSS-B): fit once on all trials, then
    z_{-i} = z_i + H_ii / (1 - H_ii W_ii) * (p_i - y_i),   H = X (X^T W X + Lambda)^-1 X^T,
which is accurate to O(1/n) for ridge-type GLMs and avoids n refits. Checked against the exact
LOO in scripts/mf_check_alo.py. Features are z-scored over all trials (exact LOO: training trials).
"""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _fit_alo_bin(X, yy, lam, eps_c, n_iter):
    n, d = X.shape
    K = X @ X.T
    a = np.zeros(n)
    c = 0.0
    H = np.zeros((n + 1, n + 1))
    g = np.zeros(n + 1)
    p = np.empty(n)
    for _ in range(n_iter):
        z = K @ a + c
        for i in range(n):
            zi = min(30.0, max(-30.0, z[i]))
            p[i] = 1.0 / (1.0 + np.exp(-zi))
        W = p * (1 - p)
        r = p - yy
        ga = K @ r + lam * (K @ a)
        Haa = (K * W.reshape(1, n)) @ K + lam * K
        jit = 1e-8
        for i in range(n):
            jit = max(jit, 1e-10 * Haa[i, i])
        for i in range(n):
            Haa[i, i] += jit
        Hac = K @ W
        H[:n, :n] = Haa
        H[:n, n] = Hac
        H[n, :n] = Hac
        H[n, n] = W.sum() + 1e-8
        H[n, n] += eps_c
        g[:n] = ga
        g[n] = r.sum() + eps_c * c
        st = np.linalg.solve(H, g)
        a -= st[:n]
        c -= st[n]
        if np.max(np.abs(st)) < 1e-6:
            break
    z = K @ a + c
    for i in range(n):
        zi = min(30.0, max(-30.0, z[i]))
        p[i] = 1.0 / (1.0 + np.exp(-zi))
    W = p * (1 - p)
    # H = Xt (Xt^T W Xt + Lam)^-1 Xt^T with Xt = [X, 1], Lam = diag(lam.., eps_c) = G (W G + I)^-1
    G = K / lam + 1.0 / eps_c
    M = W.reshape(n, 1) * G
    for i in range(n):
        M[i, i] += 1.0
    # H = G M^-1 ; need diag only: H_ii = sum_j G_ij (M^-1)_ji  ->  solve M^T Y = G^T  (Y = M^-T G^T, H = Y^T)
    Y = np.linalg.solve(M.T, G.T)
    out = np.empty(n)
    for i in range(n):
        hii = Y[i, i]
        out[i] = z[i] + hii / (1.0 - hii * W[i]) * (p[i] - yy[i])
    return out


@njit(cache=True, parallel=True)
def _alo_all(counts, y, yy, lam, eps_c, n_iter):
    n_tr, n_neu, n_b = counts.shape
    conf = np.empty((n_tr, n_b))
    for b in prange(n_b):
        X = counts[:, :, b].copy()
        for j in range(n_neu):
            mu = X[:, j].mean()
            sd = X[:, j].std() + 1e-6
            X[:, j] = (X[:, j] - mu) / sd
        zl = _fit_alo_bin(X, yy, lam, eps_c, n_iter)
        for k in range(n_tr):
            zk = min(30.0, max(-30.0, zl[k]))
            pk = 1.0 / (1.0 + np.exp(-zk))
            conf[k, b] = pk if y[k] == 1 else 1 - pk
    return conf


def alo_confidence(counts, y, lam=10.0, shuffle_rng=None, n_iter=30, eps_c=None):
    """Approximate leave-one-trial-out confidence (n_trials, n_bins); same interface as
    gamma_wm.analysis.loo_confidence_dual. With shuffle_rng, the decoder is trained on permuted
    labels and confidence is read against the true label (the null).
    eps_c: L2 penalty on the intercept. Default lam/100, as the data analysis (a constant feature of
    value 10 penalised like the weights); a near-zero value leaves the intercept free, which biases
    leave-one-out predictions below chance when there is little signal."""
    if eps_c is None:
        eps_c = lam / 100.0
    yy = y if shuffle_rng is None else shuffle_rng.permutation(y)
    return _alo_all(np.ascontiguousarray(counts, dtype=np.float64), np.asarray(y, np.float64),
                    np.asarray(yy, np.float64), float(lam), float(eps_c), int(n_iter))


@njit(cache=True, parallel=True)
def _linear_all(counts, y, T, lam):
    """One-Newton-step (from w = 0) L2 logistic decoder = kernel ridge regression of t = 4(y - 1/2)
    with ridge 4*lam; exact closed-form leave-one-out. T: (n_tr, n_lab) label columns (true + shuffled).
    Returns confidence for each label column, (n_lab, n_tr, n_b)."""
    n_tr, n_neu, n_b = counts.shape
    n_lab = T.shape[1]
    out = np.empty((n_lab, n_tr, n_b))
    for b in prange(n_b):
        X = counts[:, :, b].copy()
        for j in range(n_neu):
            mu = X[:, j].mean()
            sd = X[:, j].std() + 1e-6
            X[:, j] = (X[:, j] - mu) / sd
        K = X @ X.T
        ev, U = np.linalg.eigh(K)
        g = ev / (ev + 4.0 * lam)
        S = (U * g.reshape(1, n_tr)) @ U.T
        F = S @ T
        for l in range(n_lab):
            for i in range(n_tr):
                sii = S[i, i]
                f = (F[i, l] - sii * T[i, l]) / (1.0 - sii)
                f = min(30.0, max(-30.0, f))
                pk = 1.0 / (1.0 + np.exp(-f))
                out[l, i, b] = pk if y[i] == 1 else 1 - pk
    return out


def linear_confidence(counts, y, lam=100.0, n_shuffle=20, seed=0):
    """Confidence of the linearised decoder and its shuffled-label nulls: (conf (n_tr, n_b),
    nulls (n_shuffle, n_tr, n_b)). Labels shuffled with np.random.default_rng(seed*1000+7+s), as
    gamma_wm.panichello / meanfield.recording.full_pipeline do."""
    y = np.asarray(y)
    cols = [y] + [np.random.default_rng(seed * 1000 + 7 + s).permutation(y) for s in range(n_shuffle)]
    T = 4.0 * (np.array(cols, float).T - 0.5)
    out = _linear_all(np.ascontiguousarray(counts, dtype=np.float64), y.astype(np.float64), np.ascontiguousarray(T), float(lam))
    return out[0], out[1:]
