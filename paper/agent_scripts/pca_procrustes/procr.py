"""Procrustes shape similarity between per-cue centroids of On, Off and pre-cue activity and the ring of cue means,
computed in the full space of the held-out units and in the space of the cue-mean PCs that hold most variance.
Cross-validated: centroids of the trials of one half are compared with the cue means of the other half."""
import numpy as np
from scipy.spatial import procrustes as sp_procrustes

PERIODS = ('on', 'off', 'pre')


def halves(cue, rng):
    half = np.zeros(len(cue), int)
    for c in range(8):
        idx = rng.permutation(np.flatnonzero(cue == c))
        half[idx[len(idx) // 2:]] = 1
    return half


def trial_vectors(Zd, Zpre, state):
    """Per-trial mean activity (units) over On points, Off points and pre-cue points; NaN when a trial has none."""
    out = {}
    for p in PERIODS:
        if p == 'pre':
            out[p] = Zpre.mean(1)
            continue
        m = state == (1 if p == 'on' else -1)
        n = m.sum(1)
        v = np.einsum('tk,tku->tu', m.astype(float), Zd) / np.maximum(n, 1)[:, None]
        v[n == 0] = np.nan
        out[p] = v
    return out


def shape_similarity(Zd, Zpre, cue, state, half, var_keep=0.9, n_shuf=50, rng=None):
    """Returns {space: {period: (sim, shuffle_mean)}} for space in ('full', 'pcs'), plus the number of PCs kept."""
    rng = rng or np.random.default_rng(0)
    V = trial_vectors(Zd, Zpre, state)
    res = {'full': {}, 'pcs': {}}
    ks = []
    ref = {}
    for h in (0, 1):
        C = np.stack([Zd[(half != h) & (cue == c)].mean((0, 1)) for c in range(8)])
        mu = C.mean(0)
        _, sv, vt = np.linalg.svd(C - mu, full_matrices=False)
        k = int(np.searchsorted(np.cumsum(sv ** 2) / np.sum(sv ** 2), var_keep) + 1)
        ks.append(k)
        ref[h] = (C - mu, mu, vt[:k].T)

    def sim(P, cu, hf, space):
        v = []
        for h in (0, 1):
            Cm, mu, W = ref[h]
            m = hf == h
            cen = np.array([P[m & (cu == c)].mean(0) - mu if (m & (cu == c)).any() else np.full(P.shape[1], np.nan)
                            for c in range(8)])
            ok = ~np.isnan(cen).any(1)
            if ok.sum() < 4:
                continue
            A, B = (Cm[ok], cen[ok]) if space == 'full' else (Cm[ok] @ W, cen[ok] @ W)
            v.append(1 - sp_procrustes(A, B)[2])
        return np.mean(v) if v else np.nan

    for p in PERIODS:
        ok = ~np.isnan(V[p]).any(1)
        P, cu, hf = V[p][ok], cue[ok], half[ok]
        for space in res:
            sh = []
            for _ in range(n_shuf):
                c2 = cu.copy()
                for hh in (0, 1):
                    mm = hf == hh
                    c2[mm] = rng.permutation(c2[mm])
                sh.append(sim(P, c2, hf, space))
            res[space][p] = (float(sim(P, cu, hf, space)), float(np.nanmean(sh)), int(ok.sum()))
    return res, ks
