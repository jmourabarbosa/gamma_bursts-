"""Analysis tools: population rates, LFP proxy, single-neuron irregularity, gamma
burst detection (Lundqvist et al. 2016) and memory read-out."""
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from .network import NGROUPS


def pop_rates(res, net, bin_ms=1.0):
    """Population rates (Hz) per group [NS, A, B, INH, IA, IB] in bins of bin_ms."""
    k = int(round(bin_ms / res['dt']))
    pop = res['pop']
    n = pop.shape[0] // k * k
    ng = pop.shape[1]
    sizes = np.maximum(np.bincount(net.group, minlength=ng), 1)
    return pop[:n].reshape(-1, k, ng).sum(1) / sizes / (bin_ms * 1e-3)


def lfp_proxy(res, net, bin_ms=1.0):
    """LFP proxy: total recurrent synaptic input received by E cells, i.e. the
    spike count of each population weighted by |efficacy| and out-degree
    (E spikes weighted by J, I spikes by g*J). Equivalent to summed |PSP| input."""
    p = net.p
    r = pop_rates(res, net, bin_ms)
    sizes = np.bincount(net.group, minlength=r.shape[1])
    e = r[:, :3] @ sizes[:3]
    i = r[:, 3:] @ sizes[3:]
    return p.J * e + p.g * p.J * i


def bandpass(x, lo, hi, fs=1000.0, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], 'band')
    return filtfilt(b, a, x)


def morlet_power(x, freqs, fs=1000.0, n_cycles=6):
    """Time-frequency power with complex Morlet wavelets (same length as x)."""
    out = np.empty((len(freqs), len(x)))
    for k, f in enumerate(freqs):
        s = n_cycles / (2 * np.pi * f)
        t = np.arange(-4 * s, 4 * s, 1 / fs)
        w = np.exp(2j * np.pi * f * t) * np.exp(-t**2 / (2 * s**2))
        w /= np.sqrt(np.sum(np.abs(w)**2))
        out[k] = np.abs(np.convolve(x - x.mean(), w, mode='same'))**2
    return out


def detect_bursts(power, freqs, t_ms, n_sd=2.0, min_cycles=3, mean=None, sd=None):
    """Lundqvist et al. (2016) criterion applied to a single-trial spectrogram:
    a burst is a contiguous epoch in which power at some frequency exceeds
    mean + n_sd*SD (statistics per frequency) for at least `min_cycles` cycles.
    Overlapping supra-threshold epochs at neighbouring frequencies are merged into
    one event. Returns list of dicts with onset, offset, duration (ms), peak freq."""
    if mean is None:
        mean = power.mean(1, keepdims=True)
    if sd is None:
        sd = power.std(1, keepdims=True)
    above = power > mean + n_sd * sd
    # keep only runs at each frequency long enough
    keep = np.zeros_like(above)
    for k, f in enumerate(freqs):
        min_len = int(np.ceil(min_cycles / f * 1000.0 / (t_ms[1] - t_ms[0])))
        a = np.r_[False, above[k], False].astype(int)
        on = np.where(np.diff(a) == 1)[0]
        off = np.where(np.diff(a) == -1)[0]
        for s, e in zip(on, off):
            if e - s >= min_len:
                keep[k, s:e] = True
    any_t = keep.any(0)
    a = np.r_[False, any_t, False].astype(int)
    on = np.where(np.diff(a) == 1)[0]
    off = np.where(np.diff(a) == -1)[0]
    bursts = []
    z = (power - mean) / sd
    for s, e in zip(on, off):
        seg = np.where(keep[:, s:e], z[:, s:e], -np.inf)
        kf, kt = np.unravel_index(np.argmax(seg), seg.shape)
        # frequency span at half power around the peak (Lundqvist's 3 dB width)
        prof = power[:, s:e].mean(1)
        half = prof >= prof[kf] / 2
        lo = kf
        while lo > 0 and half[lo - 1]:
            lo -= 1
        hi = kf
        while hi < len(freqs) - 1 and half[hi + 1]:
            hi += 1
        bursts.append(dict(on=t_ms[s], off=t_ms[e - 1], dur=t_ms[e - 1] - t_ms[s] + (t_ms[1] - t_ms[0]),
                           f_peak=freqs[kf], t_peak=t_ms[s + kt], f_lo=freqs[lo], f_hi=freqs[hi]))
    return bursts, keep


def burst_mask(bursts, t_ms):
    m = np.zeros(len(t_ms), bool)
    for b in bursts:
        m |= (t_ms >= b['on']) & (t_ms <= b['off'])
    return m


def isi_cv(res, idx, t0, t1, min_spikes=5):
    """Coefficient of variation of ISIs for neurons `idx` in window [t0, t1)."""
    t, i = res['t'], res['i']
    m = (t >= t0) & (t < t1)
    t, i = t[m], i[m]
    order = np.lexsort((t, i))
    t, i = t[order], i[order]
    cvs = []
    sel = set(np.asarray(idx).tolist())
    bounds = np.r_[0, np.where(np.diff(i) != 0)[0] + 1, len(i)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s >= min_spikes and i[s] in sel:
            d = np.diff(t[s:e])
            cvs.append(d.std() / d.mean())
    return np.array(cvs)


def cv2_intervals(onsets):
    """CV2 (Holt et al. 1996) of a sequence of event times."""
    I = np.diff(np.sort(onsets))
    if len(I) < 2:
        return np.nan
    return np.mean(2 * np.abs(I[1:] - I[:-1]) / (I[1:] + I[:-1]))


def spike_counts(res, idx, t_edges):
    """Spike count matrix (len(idx), n_bins) for neurons idx."""
    idx = np.asarray(idx)
    lut = -np.ones(int(res['i'].max()) + 2 if len(res['i']) else 1, dtype=np.int64)
    lut[idx[idx < len(lut)]] = np.arange(len(idx))[idx < len(lut)]
    m = res['i'] < len(lut)
    row = lut[res['i'][m]]
    tt = res['t'][m]
    ok = row >= 0
    H, _, _ = np.histogram2d(row[ok], tt[ok], bins=[np.arange(len(idx) + 1) - 0.5, t_edges])
    return H


# ---------------------------------------------------------------------------
# decoding (Panichello et al. 2024 style)
# ---------------------------------------------------------------------------

def fit_logistic(X, y, lam=1.0, n_iter=25):
    """L2-regularised logistic regression by Newton's method. X (n, d), y in {0,1}."""
    n, d = X.shape
    Xb = np.c_[X, np.ones(n)]
    w = np.zeros(d + 1)
    R = lam * np.eye(d + 1)
    R[-1, -1] = 0.0
    for _ in range(n_iter):
        z = np.clip(Xb @ w, -30, 30)
        pr = 1 / (1 + np.exp(-z))
        grad = Xb.T @ (pr - y) + R @ w
        H = (Xb * (pr * (1 - pr))[:, None]).T @ Xb + R + 1e-8 * np.eye(d + 1)
        step = np.linalg.solve(H, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w


def predict_logistic(w, X):
    z = np.clip(np.c_[X, np.ones(len(X))] @ w, -30, 30)
    return 1 / (1 + np.exp(-z))


def loo_confidence(counts, y, lam=1.0, shuffle_rng=None):
    """Leave-one-trial-out decoder, one classifier per time bin.
    counts: (n_trials, n_neurons, n_bins); y: (n_trials,) in {0,1}.
    Returns confidence = probability assigned to the correct label, (n_trials, n_bins).
    Features are z-scored per neuron and time bin using training trials only."""
    n_tr, n_neu, n_b = counts.shape
    conf = np.empty((n_tr, n_b))
    yy = y.copy()
    if shuffle_rng is not None:
        yy = shuffle_rng.permutation(y)
    for b in range(n_b):
        X = counts[:, :, b]
        for k in range(n_tr):
            tr = np.ones(n_tr, bool)
            tr[k] = False
            mu = X[tr].mean(0)
            sd = X[tr].std(0) + 1e-6
            w = fit_logistic((X[tr] - mu) / sd, yy[tr], lam)
            pk = predict_logistic(w, ((X[k] - mu) / sd)[None])[0]
            conf[k, b] = pk if y[k] == 1 else 1 - pk
    return conf


def beta_mixture_cv(c, n_fold=4, seed=0):
    """Cross-validated log-likelihood (bits per sample) of a 1- vs 2-component beta
    mixture for confidence values c in (0,1). Returns (ll1, ll2)."""
    from scipy import optimize, stats
    c = np.clip(np.asarray(c), 1e-3, 1 - 1e-3)
    rng = np.random.default_rng(seed)
    folds = rng.integers(0, n_fold, len(c))

    def nll1(th, x):
        a, b = np.exp(th)
        return -stats.beta.logpdf(x, a, b).sum()

    def nll2(th, x):
        a1, b1, a2, b2 = np.exp(th[:4])
        w = 1 / (1 + np.exp(-th[4]))
        return -np.log(w * stats.beta.pdf(x, a1, b1) + (1 - w) * stats.beta.pdf(x, a2, b2) + 1e-300).sum()

    ll1 = ll2 = 0.0
    for k in range(n_fold):
        tr, te = c[folds != k], c[folds == k]
        r1 = optimize.minimize(nll1, np.zeros(2), args=(tr,), method='Nelder-Mead')
        best = None
        for init in ([1, 0, 0, 1, 0], [2, 0, 0, 0, 0], [0.5, 0.5, 1.5, -0.5, 0.0]):
            r2 = optimize.minimize(nll2, np.array(init, float), args=(tr,), method='Nelder-Mead',
                                   options=dict(maxiter=4000))
            if best is None or r2.fun < best.fun:
                best = r2
        ll1 += -nll1(r1.x, te)
        ll2 += -nll2(best.x, te)
    return ll1 / len(c) / np.log(2), ll2 / len(c) / np.log(2)
