"""Panichello et al. (2024) style single-trial decoding analysis.

Pipeline (following their Methods as closely as a binary task allows):
- spike counts of a recorded sample of neurons in 100 ms boxcar windows, 10 ms steps;
- for each time point, a leave-one-trial-out L2 logistic classifier of the cue;
  'confidence' = probability assigned to the correct cue;
- a null from classifiers trained on shuffled cue labels;
- On states: contiguous runs of z > 1.96 whose cluster mass exceeds the 95th
  percentile of null cluster masses; Off states: >= 5 consecutive points with
  z < 0.84 (one-sided P > 0.20);
- split-half test: states labelled with one half of the sample; tuning (cue-explained
  variance) and firing of the held-out half measured in On vs Off states.
"""
import numpy as np

from .analysis import beta_mixture_cv, loo_confidence_dual, spike_counts


def counts_matrix(trials, sample, n_t, win=100, step=10):
    """(n_trials, n_neurons, n_bins) spike counts in `win` ms windows every `step` ms."""
    edges = np.arange(0, n_t + 1, step)
    k = win // step
    out = []
    for d in trials:
        H = spike_counts(dict(t=d['spk_t'], i=d['spk_i']), sample, edges)
        c = np.cumsum(np.c_[np.zeros(len(sample)), H], axis=1)
        out.append(c[:, k:] - c[:, :-k])
    centers = np.arange(out[0].shape[1]) * step + win / 2
    return np.array(out), centers


def _runs(mask):
    a = np.r_[False, mask, False].astype(int)
    return list(zip(np.where(np.diff(a) == 1)[0], np.where(np.diff(a) == -1)[0]))


def label_states(conf, nulls, valid, min_off=5, rng=None):
    """On/Off labels per trial and time point. conf (n_tr, n_b); nulls (n_sh, n_tr, n_b);
    valid (n_b,) bool mask of time points that may be labelled."""
    mu, sd = nulls.mean(0), nulls.std(0) + 1e-9
    z = (conf - mu) / sd
    # null cluster masses from the shuffled runs themselves (leave-one-out z)
    null_mass = []
    for s in range(len(nulls)):
        others = np.delete(nulls, s, axis=0)
        zs = (nulls[s] - others.mean(0)) / (others.std(0) + 1e-9)
        for k in range(zs.shape[0]):
            for a, b in _runs((zs[k] > 1.96) & valid):
                null_mass.append(zs[k, a:b].sum())
    thr = np.percentile(null_mass, 95) if null_mass else 0.0
    state = np.zeros(conf.shape, int)
    for k in range(conf.shape[0]):
        for a, b in _runs((z[k] > 1.96) & valid):
            if z[k, a:b].sum() > thr:
                state[k, a:b] = 1
        for a, b in _runs((z[k] < 0.84) & valid):
            if b - a >= min_off:
                state[k, a:b] = -1
    return state, z, thr


def state_stats(state, step, count_mask):
    """Durations and per-trial counts of On/Off states inside count_mask."""
    on_d, off_d, n_on, n_off = [], [], [], []
    for k in range(state.shape[0]):
        s = np.where(count_mask, state[k], 0)
        on = _runs(s == 1)
        off = _runs(s == -1)
        on_d += [(b - a) * step for a, b in on]
        off_d += [(b - a) * step for a, b in off]
        n_on.append(len(on))
        n_off.append(len(off))
    return dict(on_dur=np.array(on_d), off_dur=np.array(off_d), n_on=np.array(n_on), n_off=np.array(n_off))


def explained_variance(X, y):
    """Fraction of variance of each column of X explained by the binary label y."""
    tot = X.var(0) + 1e-12
    m = np.array([X[y == c].mean(0) for c in (0, 1)])
    between = sum((y == c).mean() * (m[c] - X.mean(0)) ** 2 for c in (0, 1))
    return between / tot


def run(trials, sample, y, n_t, t_cue, delay_start, count_window, n_shuffle=20, win=100, step=10,
        lam=1.0, seed=0, split_half=True):
    """Full analysis for one recorded sample. Returns a dict of results."""
    rng = np.random.default_rng(seed)
    C, tc = counts_matrix(trials, sample, n_t, win, step)
    keep = tc >= t_cue - 100          # states are only labelled from the cue onward
    C, tc = np.ascontiguousarray(C[:, :, keep]), tc[keep]
    conf = loo_confidence_dual(C, y, lam)
    nulls = np.array([loo_confidence_dual(C, y, lam, shuffle_rng=np.random.default_rng(seed * 1000 + 7 + s))
                      for s in range(n_shuffle)])
    valid = tc >= t_cue
    state, z, thr = label_states(conf, nulls, valid)
    cmask = (tc >= count_window[0]) & (tc < count_window[1])
    st = state_stats(state, step, cmask)
    dmask = tc >= delay_start
    res = dict(conf=conf, z=z, state=state, t=tc, stats=st,
               acc_delay=float((conf[:, dmask] > 0.5).mean()),
               conf_delay=float(conf[:, dmask].mean()),
               acc_on=float((conf[state == 1] > 0.5).mean()) if (state == 1).any() else np.nan,
               acc_off=float((conf[state == -1] > 0.5).mean()) if (state == -1).any() else np.nan,
               frac_on_delay=float((state[:, dmask] == 1).mean()), frac_off_delay=float((state[:, dmask] == -1).mean()))
    ll1, ll2 = beta_mixture_cv(conf[:, dmask].ravel())
    res['mixture_mbit'] = float((ll2 - ll1) * 1000)
    if split_half and len(sample) >= 20:
        perm = rng.permutation(len(sample))
        h1, h2 = perm[: len(sample) // 2], perm[len(sample) // 2:]
        ev_on, ev_off, r_on, r_off = [], [], [], []
        for lab_idx, test_idx in ((h1, h2), (h2, h1)):
            c1 = loo_confidence_dual(C[:, lab_idx], y, lam)
            n1 = np.array([loo_confidence_dual(C[:, lab_idx], y, lam, shuffle_rng=np.random.default_rng(seed * 1000 + 99 + s))
                           for s in range(n_shuffle)])
            s1, _, _ = label_states(c1, n1, valid)
            Xt = C[:, test_idx]
            for lab, ev, rr in ((1, ev_on, r_on), (-1, ev_off, r_off)):
                # per trial: mean count of each held-out neuron over the trial's delay bins in this state
                rows, ys = [], []
                for k in range(len(y)):
                    m = (s1[k] == lab) & dmask
                    if m.sum() >= 3:
                        rows.append(Xt[k][:, m].mean(1))
                        ys.append(y[k])
                if len(rows) >= 6 and len(set(ys)) == 2:
                    R = np.array(rows)
                    ev.append(np.nanmean(explained_variance(R, np.array(ys))))
                    rr.append(R.mean() / (win * 1e-3))
        res['split_half'] = dict(ev_on=float(np.mean(ev_on)) if ev_on else np.nan,
                                 ev_off=float(np.mean(ev_off)) if ev_off else np.nan,
                                 rate_on=float(np.mean(r_on)) if r_on else np.nan,
                                 rate_off=float(np.mean(r_off)) if r_off else np.nan)
    return res
