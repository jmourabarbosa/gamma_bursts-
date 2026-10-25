"""Read-out of the spiking network like the data (Panichello et al. 2024 On/Off labelling;
paper_model/data/readout_note_data_lam10000.md).

- "Recorded" cells: spike counts in 100 ms windows every 10 ms (window centres from -50 to 1,450 ms after cue onset).
- For each cue and its opposite cue (4 pairs), at every time point, an L2 logistic regression (lambda = 10,000 on spike
  counts, plus a constant feature of value 1 penalised like a unit weight) trained on correct trials, with
  leave-one-out predictions.
- At lambda = 10,000 the fitted logits are tiny (the data note: confidences within ~0.004 of 0.5), so the logistic
  loss is replaced by its second-order expansion at w = 0 (one Newton step from zero). For that quadratic loss the
  leave-one-out predictions are exact (Sherman-Morrison), and the true labels and all 30 label shuffles share one
  eigendecomposition. check_newton() compares this with the full Newton solution and the approximate leave-one-out
  of Rad & Maleki used for the data.
- Decoder output for the cued location (confidence of the true cue) is z-scored against the shuffle null at each
  time point (mean and SD over trials and shuffles of the pair).
- On = runs of z > 1.96 whose summed z exceeds the 95th percentile of the same quantity in the null z;
  Off = >= 5 consecutive points with z < 0.84; 0-1,400 ms, fractions over 500-1,400 ms (paper_model.readout).
- Decoded populations: all recorded cells; top 5/20/50 ranked by one-way ANOVA across the 8 cues on the other half
  of the trials (split-half, so a test trial never ranks its own units); random 5/20/50 (10 draws).
"""
import numpy as np

from paper_model import readout as PR

LAM = 1e4
STEP, WIN = 10, 100
N_SHUF = 30
SETS = ('all', 'top 5', 'top 20', 'top 50', 'rand 5', 'rand 20', 'rand 50')


def windows(counts, t10):
    """(n_tr, n_win, n_cells) counts in 100 ms windows every 10 ms, and window centres (ms from cue onset)."""
    k = WIN // STEP
    c = np.cumsum(np.concatenate([np.zeros_like(counts[:, :1], dtype=np.int32), counts.astype(np.int32)], 1), 1)
    X = (c[:, k:] - c[:, :-k]).astype(np.float64)
    tz = t10[:X.shape[1]] + WIN / 2
    keep = (tz >= -50) & (tz <= 1450)
    return X[:, keep], tz[keep]


def _decode_pair(X, y, shuf):
    """X (n, n_t, d) counts, y (n,) labels 0/1, shuf (n_shuf, n) shuffled labels.
    Returns the leave-one-out confidence of each trial's true label, with the true labels (n, n_t) and with each
    shuffle (n_shuf, n, n_t)."""
    n, n_t, _ = X.shape
    Y = np.vstack([y[None], shuf]).astype(float)           # (1 + n_shuf, n)
    R = Y - 0.5
    o = np.empty((len(Y), n, n_t))
    for t in range(n_t):
        Z = np.c_[X[:, t], np.ones(n)]
        e, U = np.linalg.eigh(Z @ Z.T)
        e = np.maximum(e, 0)
        g = e / (e / 4 + LAM)                               # H = Z (Z'Z/4 + lam I)^-1 Z' = U diag(g) U'
        H_diag = (U ** 2) @ g
        F = ((R @ U) * g) @ U.T                             # fitted logits (1 + n_shuf, n)
        F_loo = F + H_diag / (1 - H_diag / 4) * (F / 4 - R)
        p1 = 1 / (1 + np.exp(-F_loo))                      # confidence of class 1
        o[:, :, t] = np.where(y[None] == 1, p1, 1 - p1)     # confidence of the trial's true cue
    return o[0], o[1:]


def decode(X, cue, rng, n_shuf=N_SHUF):
    """z of the decoder for the cued location, (n_tr, n_t), and the null z (n_shuf, n_tr, n_t).
    X: (n_tr, n_t, d) window counts of the decoded cells; cue: (n_tr,) cue index 0..7 (correct trials only)."""
    n_tr, n_t, _ = X.shape
    z = np.full((n_tr, n_t), np.nan)
    zn = np.full((n_shuf, n_tr, n_t), np.nan)
    for c in range(4):
        idx = np.where((cue == c) | (cue == c + 4))[0]
        y = (cue[idx] == c).astype(int)
        shuf = np.array([rng.permutation(y) for _ in range(n_shuf)])
        o, on = _decode_pair(X[idx], y, shuf)
        mu, sd = on.mean((0, 1)), on.std((0, 1))
        sd = np.maximum(sd, 1e-300)
        z[idx] = (o - mu) / sd
        zn[:, idx] = (on - mu) / sd
    return z, zn


def null_threshold(zn, valid):
    masses = [row[a:b].sum() for row in zn.reshape(-1, zn.shape[-1]) for a, b in PR.runs((row > 1.96) & valid)]
    return float(np.percentile(masses, 95)) if masses else np.inf


def anova_f(X, cue):
    """One-way ANOVA F across cues of each cell's count summed over the given windows; X (n_tr, d)."""
    groups = [X[cue == c] for c in np.unique(cue)]
    grand = X.mean(0)
    k, n = len(groups), len(X)
    ssb = sum(len(g) * (g.mean(0) - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean(0)) ** 2).sum(0) for g in groups)
    return (ssb / (k - 1)) / np.maximum(ssw / (n - k), 1e-12)


def read_session(counts, t10, cue, correct, seed=0, sets=SETS, n_draws=10, n_shuf=N_SHUF, keep_z=False):
    """Decode one simulated session. counts: (n_tr, n_bins10, n_cells) uint8 of the recorded cells; cue: (n_tr,) cue
    index; correct: (n_tr,) bool, trials used (memory held). Returns {set: stats} (random sets averaged over draws as
    in the data) and optionally z / states of each set (first draw)."""
    rng = np.random.default_rng(seed)
    Xall, tz = windows(counts[correct], t10)
    cue = np.asarray(cue)[correct]
    n_tr, n_t, d = Xall.shape
    valid, delay = (tz >= 0) & (tz <= 1400), (tz >= 500) & (tz <= 1400)
    # split halves balanced over cues, for the ANOVA ranking
    half = np.zeros(n_tr, int)
    for c in range(8):
        i = rng.permutation(np.where(cue == c)[0])
        half[i[len(i) // 2:]] = 1
    tot = Xall[:, valid].sum(1)
    rank = [np.argsort(-anova_f(tot[half == h], cue[half == h])) for h in (0, 1)]
    active = Xall.sum((0, 1)) > 0
    out, zs = {}, {}
    for lab in sets:
        if lab == 'all':
            draws = [[np.where(active)[0]] * 2]
        elif lab.startswith('top'):
            k = int(lab.split()[1])
            draws = [[rank[1][:k], rank[0][:k]]]           # trials of half h use units ranked on the other half
        else:
            k = int(lab.split()[1])
            draws = [[s, s] for s in (rng.choice(d, k, replace=False) for _ in range(n_draws))]
        res = []
        for units in draws:
            z = np.full((n_tr, n_t), np.nan)
            zn_all = []
            if units[0] is units[1] or np.array_equal(units[0], units[1]):
                z, zn = decode(Xall[:, :, units[0]], cue, rng, n_shuf)
                zn_all.append(zn)
            else:
                for h in (0, 1):
                    zh, zn = decode(Xall[:, :, units[h]], cue, rng, n_shuf)
                    z[half == h] = zh[half == h]
                    zn_all.append(zn[:, half == h])
            thr = null_threshold(np.concatenate(zn_all, 1), valid)
            zz = np.nan_to_num(z)
            state = PR.label(zz, valid, thr)
            st = PR.stats(state, delay)
            st['mu_on'] = PR.mu_on(zz[:, delay], rng)
            st['precue_frac_on'] = np.nan
            res.append(st)
            if keep_z and lab not in zs:
                zs[lab] = dict(z=zz, state=state, thr=thr)
        out[lab] = _average_draws(res)
    return (out, zs, tz) if keep_z else out


def _average_draws(res):
    """Mean over draws; medians and means of lengths skip draws without states of that kind (as in the data)."""
    o = {}
    for k in res[0]:
        v = [r[k] for r in res]
        if k.startswith('h_'):
            o[k] = np.sum(v, 0)
        elif k in ('on_mean', 'on_median', 'off_mean', 'off_median'):
            st = k.split('_')[0]
            v = [r[k] for r in res if r[f'n_{st}'] > 0]
            o[k] = float(np.mean(v)) if v else np.nan
        else:
            o[k] = float(np.nanmean(v)) if not all(np.isnan(np.atleast_1d(v))) else np.nan
    return o


def check_newton(X, y, n_iter=25):
    """Max |confidence difference| between the quadratic leave-one-out used here and (a) the full L2-logistic Newton
    fit with the approximate leave-one-out of Rad & Maleki, (b) exact refits, for one time point. X: (n, d)."""
    n = len(X)
    Z = np.c_[X, np.ones(n)]
    _, o_quad = None, None
    o_quad, _ = _decode_pair(X[:, None], y, np.zeros((0, n), int))
    o_quad = o_quad[:, 0]

    def fit(Zf, yf):
        w = np.zeros(Zf.shape[1])
        for _ in range(n_iter):
            p = 1 / (1 + np.exp(-Zf @ w))
            g = Zf.T @ (p - yf) + LAM * w
            Hm = (Zf * (p * (1 - p))[:, None]).T @ Zf + LAM * np.eye(Zf.shape[1])
            w -= np.linalg.solve(Hm, g)
        return w
    w = fit(Z, y)
    f = Z @ w
    p = 1 / (1 + np.exp(-f))
    D = p * (1 - p)
    A = (Z * D[:, None]).T @ Z + LAM * np.eye(Z.shape[1])
    h = np.einsum('ij,jk,ik->i', Z, np.linalg.inv(A), Z)
    f_alo = f + h / (1 - D * h) * (p - y)
    p_alo = 1 / (1 + np.exp(-f_alo))
    o_alo = np.where(y == 1, p_alo, 1 - p_alo)
    f_ex = np.array([Z[i] @ fit(np.delete(Z, i, 0), np.delete(y, i)) for i in range(n)])
    p_ex = 1 / (1 + np.exp(-f_ex))
    o_ex = np.where(y == 1, p_ex, 1 - p_ex)
    return dict(max_diff_alo=float(np.abs(o_quad - o_alo).max()), max_diff_exact=float(np.abs(o_quad - o_ex).max()),
                typical_dev=float(np.abs(o_ex - 0.5).mean()), sign_agree=float(np.mean(np.sign(o_quad - .5) == np.sign(o_ex - .5))))
