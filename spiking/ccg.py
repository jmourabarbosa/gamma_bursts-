"""Cross-correlograms (CCGs) and cue-specific functional connectivity in the spiking ring, as in Panichello et al.
(2024), Fig. 5 and Methods "CCG analysis" / "Manhattan distance", following their code
(github.com/panichem/SingleTrialDynamics, ccgs/analyze_ccgs.m, xcorr_gm.m, jitter.m).

Per simulated session (the same trials and recorded cells as run.sessions, re-simulated with 1 ms spikes):
- binary spike trains in 1 ms bins ("at least one spike"), 500-1,400 ms after cue onset, correct trials;
- cells with mean rate > 1 Hz in that window;
- for each cue condition and pair (j, k), at lags -100..100 ms (xcorr_gm.m):
      CCG(L) = sum_trials sum_t x_j(t) x_k(t + L) / sqrt(sum x_j * sum x_k), sums over the overlapping bins;
- jitter correction: subtract the CCG of the expected jittered trains (25 ms windows). Here each spike is jittered
  uniformly within its window (interval jitter, Amarasingham et al. 2012), so the expected train is the window count
  spread evenly over the window and the jitter CCG follows exactly from window counts. The authors' jitter.m weights
  the spread by the condition PSTH instead; both keep the count per window and trial;
- significant CCG: the peak of the corrected CCG (zero lag excluded) lies within 10 ms of zero and exceeds the mean
  of the baseline (50 < |L| < 100 ms) by more than 7 SD;
- one graph per cue (edge = significant CCG); Manhattan distance = number of differing edges, for each of the 28 cue
  pairs, averaged;
- null: for each cue pair, the cue labels of its trials are shuffled and the whole pipeline repeated; 50 shuffles (25
  for On / Off); z = (observed - null mean) / null SD;
- On / Off: spikes set to zero outside On (Off) time points of the all-units decoder (run.sessions read-out; each
  1 ms bin takes the state of the nearest 10 ms decoder time point), then the same analysis.

    python -m spiking.ccg --sessions 0 1 2 3 4 5        # writes spiking/results/ccg/session_<s>.json
"""
import argparse
import json
import os
import time

import numpy as np
from numba import njit, prange

from . import network as N, readout as R, run as RU

LAG, JIT, T0, T1 = 100, 25, 500, 1400
OUT = os.path.join(RU.OUT, 'ccg')


def session_spikes(s, n_trials=RU.N_TRIALS, seed0=0):
    """Re-simulate session s (same seeds as run.sessions) with 1 ms spikes of the recorded cells in [T0, T1).
    Returns spikes (n_trials, T1 - T0, n_rec) uint8, cue index, recorded cell ids."""
    f = os.path.join(OUT, f'spikes_s{s}.npz')
    if os.path.exists(f):
        d = np.load(f)
        X = np.zeros((d['n_trials'], T1 - T0, len(d['rec'])), np.uint8)
        X[d['tr'], d['t'], d['c']] = 1
        return X, d['cue'], d['rec']
    net = RU.make_net()
    rng = np.random.default_rng(seed0 + 100 + s)
    rec = RU.record(net, rng)
    cue = rng.permutation(np.repeat(np.arange(8), n_trials // 8))
    parts, counts = [], []
    for i in range(0, n_trials, RU.BATCH):
        sim = N.simulate(net, 45.0 * cue[i:i + RU.BATCH], seed=(seed0 + s) * 1000 + i // RU.BATCH, rec_idx=rec,
                         spikes_1ms=(T0, T1))
        parts.append(sim['spk1'])
        counts.append(sim['counts'])
    X = np.concatenate(parts)
    saved = np.load(os.path.join(RU.OUT, 'sessions_dips_trials.npz'))
    if f's{s}_counts' in saved:     # only session 0 keeps all counts: check that the trials are the same
        assert np.array_equal(np.concatenate(counts), saved[f's{s}_counts'])
    tr, t, c = np.nonzero(X)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f, tr=tr.astype(np.int16), t=t.astype(np.int16), c=c.astype(np.int16), cue=cue, rec=rec,
                        n_trials=n_trials)
    return X, cue, rec


@njit(parallel=True, cache=True)
def _pair_counts(st, sc, starts, n_cells, lag, n_chunks=4):
    """Raw CCG counts C[j, k, lag + L] = sum_trials sum_t x_j(t) x_k(t + L) from spike lists sorted by time within
    each trial (st times, sc cells; trial r spans starts[r]:starts[r + 1]). Trials are split into chunks counted in
    parallel and summed."""
    n_tr = len(starts) - 1
    parts = np.zeros((n_chunks, n_cells, n_cells, 2 * lag + 1), np.float32)
    for ch in prange(n_chunks):
        _count_trials(st, sc, starts, lag, parts[ch], ch * n_tr // n_chunks, (ch + 1) * n_tr // n_chunks)
    return parts.sum(0)


@njit(cache=True)
def _count_trials(st, sc, starts, lag, C, r0, r1):
    for r in range(r0, r1):
        for a in range(starts[r], starts[r + 1]):
            j = sc[a]
            for b in range(a + 1, starts[r + 1]):
                L = st[b] - st[a]
                if L > lag:
                    break
                k = sc[b]
                if j == k:
                    continue
                C[j, k, lag + L] += 1
                C[k, j, lag - L] += 1


def _spike_lists(X):
    tr, t, c = np.nonzero(X)                     # sorted by trial, then time
    starts = np.r_[0, np.cumsum(np.bincount(tr, minlength=X.shape[0]))]
    return t.astype(np.int64), c.astype(np.int64), starts.astype(np.int64)


def ccg_corrected(X):
    """Jitter-corrected, normalised CCG for all ordered pairs: (n_cells, n_cells, 2 LAG + 1). X: (trials, T, cells)."""
    n, T, nc = X.shape
    lags = np.arange(-LAG, LAG + 1)
    raw = _pair_counts(*_spike_lists(X), nc, LAG)
    # overlap sums: pre j over t in [max(0, -L), T - max(0, L)), post k over [max(0, L), T + min(0, L))
    cum = np.r_[np.zeros((1, nc)), np.cumsum(X.sum(0), 0)]          # (T + 1, cells), summed over trials
    a = np.maximum(0, -lags); b = T - np.maximum(0, lags)
    S_pre = cum[b] - cum[a]                                          # (lags, cells)
    S_post = cum[T + np.minimum(0, lags)] - cum[np.maximum(0, lags)]
    norm = np.sqrt(S_pre.T[:, None, :] * S_post.T[None, :, :])      # (j, k, lags)
    obs = np.where(norm > 0, raw / np.where(norm > 0, norm, 1), 0)
    # expected CCG under interval jitter: window counts spread evenly over each 25 ms window
    W = T // JIT
    A = X.reshape(n, W, JIT, nc).sum(2).astype(np.float32)           # (trials, windows, cells)
    jit = np.zeros_like(obs)
    for d in range(-(LAG // JIT) - 1, LAG // JIT + 2):
        tri = np.maximum(0, JIT - np.abs(lags - JIT * d)) / JIT ** 2   # overlap of window w and window w + d at lag L
        if not tri.any():
            continue
        if d >= 0:
            G = np.einsum('iwj,iwk->jk', A[:, :W - d], A[:, d:])
        else:
            G = np.einsum('iwj,iwk->jk', A[:, -d:], A[:, :W + d])
        jit += G[:, :, None] * tri[None, None, :].astype(np.float32)
    cumj = np.r_[np.zeros((1, nc)), np.cumsum(np.repeat(A.sum(0) / JIT, JIT, 0), 0)]
    Sj_pre = cumj[b] - cumj[a]
    Sj_post = cumj[T + np.minimum(0, lags)] - cumj[np.maximum(0, lags)]
    normj = np.sqrt(Sj_pre.T[:, None, :] * Sj_post.T[None, :, :])
    jit = np.where(normj > 0, jit / np.where(normj > 0, normj, 1), 0)
    return (obs - jit).astype(np.float32)


@njit(parallel=True, cache=True)
def _sig_kernel(raw, S_pre, S_post, Sj_pre, Sj_post, G, tri, lag):
    """Fused normalisation, jitter subtraction and significance test (as ccg_corrected + significant)."""
    nc = raw.shape[0]
    nL = 2 * lag + 1
    sig = np.zeros((nc, nc), np.bool_)
    for j in prange(nc):
        cc = np.empty(nL)
        for k in range(j + 1, nc):
            for li in range(nL):
                n1 = np.sqrt(S_pre[li, j] * S_post[li, k])
                o = raw[j, k, li] / n1 if n1 > 0 else 0.0
                e = 0.0
                for d in range(G.shape[0]):
                    if tri[d, li] > 0:
                        e += G[d, j, k] * tri[d, li]
                n2 = np.sqrt(Sj_pre[li, j] * Sj_post[li, k])
                cc[li] = o - (e / n2 if n2 > 0 else 0.0)
            m = 0.0; m2 = 0.0; nb = 0
            for li in range(nL):
                L = abs(li - lag)
                if 50 < L < 100:
                    m += cc[li]; m2 += cc[li] * cc[li]; nb += 1
            m /= nb
            sd = np.sqrt(max(m2 / nb - m * m, 0.0))
            best = -1e30; bl = 0
            for li in range(nL):
                if li != lag and cc[li] > best:
                    best = cc[li]; bl = li - lag
            sig[j, k] = (abs(bl) <= 10) and (best > m + 7 * sd) and (sd > 0)
    return sig


def graph(X):
    """Significant-CCG graph (upper triangular) of one condition; same result as significant(ccg_corrected(X))."""
    n, T, nc = X.shape
    lags = np.arange(-LAG, LAG + 1)
    raw = _pair_counts(*_spike_lists(X), nc, LAG)
    cum = np.r_[np.zeros((1, nc)), np.cumsum(X.sum(0), 0)]
    a = np.maximum(0, -lags); b = T - np.maximum(0, lags)
    S_pre, S_post = cum[b] - cum[a], cum[T + np.minimum(0, lags)] - cum[np.maximum(0, lags)]
    W = T // JIT
    A = X.reshape(n, W, JIT, nc).sum(2).astype(np.float32)
    ds = range(-(LAG // JIT), LAG // JIT + 1)
    G = np.zeros((len(ds), nc, nc))
    tri = np.zeros((len(ds), len(lags)))
    for i, d in enumerate(ds):
        tri[i] = np.maximum(0, JIT - np.abs(lags - JIT * d)) / JIT ** 2
        A1 = A[:, :W - d] if d >= 0 else A[:, -d:]
        A2 = A[:, d:] if d >= 0 else A[:, :W + d]
        G[i] = A1.reshape(-1, nc).T @ A2.reshape(-1, nc)
    cumj = np.r_[np.zeros((1, nc)), np.cumsum(np.repeat(A.sum(0) / JIT, JIT, 0), 0)]
    Sj_pre, Sj_post = cumj[b] - cumj[a], cumj[T + np.minimum(0, lags)] - cumj[np.maximum(0, lags)]
    return _sig_kernel(raw, S_pre, S_post, Sj_pre, Sj_post, G, tri, LAG)


def significant(cc):
    """Upper-triangular boolean graph of significant CCGs, and the peak lag of each pair."""
    lags = np.arange(-LAG, LAG + 1)
    base = (np.abs(lags) > 50) & (np.abs(lags) < 100)
    nz = lags != 0
    mu, sd = cc[:, :, base].mean(-1), cc[:, :, base].std(-1)
    sub = cc[:, :, nz]
    pk = sub.argmax(-1)
    peak_lag = lags[nz][pk]
    peak = np.take_along_axis(sub, pk[..., None], -1)[..., 0]
    sig = (np.abs(peak_lag) <= 10) & (peak > mu + 7 * sd) & (sd > 0)
    return np.triu(sig, 1), peak_lag


def graphs(X, cue):
    return {c: graph(X[cue == c]) for c in range(8)}


def manhattan(g):
    pairs = [(a, b) for a in range(8) for b in range(a + 1, 8)]
    return float(np.mean([(g[a] != g[b]).sum() for a, b in pairs]))


def null(X, cue, n_shuf, rng):
    """Mean Manhattan distance with the cue labels shuffled within each pair of conditions."""
    out = []
    for _ in range(n_shuf):
        d = []
        for a in range(8):
            for b in range(a + 1, 8):
                idx = np.where((cue == a) | (cue == b))[0]
                lab = rng.permutation(cue[idx])
                ga = graph(X[idx[lab == a]])
                gb = graph(X[idx[lab == b]])
                d.append((ga != gb).sum())
        out.append(float(np.mean(d)))
    return out


def state_mask(state, tz):
    """(trials, T1 - T0) mask in 1 ms bins: the decoder state of the nearest 10 ms time point."""
    t = np.arange(T0, T1) + 0.5
    idx = np.clip(np.round((t - tz[0]) / (tz[1] - tz[0])).astype(int), 0, len(tz) - 1)
    return state[:, idx]


def run_session(s, n_shuf=50, n_shuf_state=25, seed=0):
    os.makedirs(OUT, exist_ok=True)
    X, cue, rec = session_spikes(s)
    d = np.load(os.path.join(RU.OUT, 'sessions_dips_trials.npz'))
    held, tz, state = d[f's{s}_held'], d['tz'], d[f's{s}_state_all']
    if len(state) != held.sum():
        # session 0 stores only a 40-trial preview of the states: recompute the all-units read-out from its counts
        # (same seed and order of draws as run.sessions, so the same states)
        _, zs, _ = R.read_session(d[f's{s}_counts'], np.arange(d[f's{s}_counts'].shape[1]) * 10.0 - RU.TUNED['t_pre'],
                                  cue, held, seed=s, sets=('all',), keep_z=True)
        assert np.array_equal(zs['all']['state'][:len(state)], state)
        state = zs['all']['state']
    X, cue = X[held], cue[held]
    rate = X.mean((0, 1)) * 1000
    keep = rate > 1.0
    X, cells = X[:, :, keep], rec[keep]
    st = state_mask(state, tz)
    rng = np.random.default_rng(seed + s)
    res = dict(session=s, n_trials=int(len(cue)), n_cells=int(keep.sum()), n_E=int((cells < N.P['N_E']).sum()),
               n_pairs=int(keep.sum() * (keep.sum() - 1) // 2), variants={})
    for name, Xv, ns in (('all', X, n_shuf), ('on', X * (st == 1)[:, :, None], n_shuf_state),
                         ('off', X * (st == -1)[:, :, None], n_shuf_state)):
        t0 = time.time()
        g = graphs(Xv, cue)
        obs = manhattan(g)
        nl = null(Xv, cue, ns, rng)
        z = (obs - np.mean(nl)) / np.std(nl, ddof=1)
        n_sig = [int(g[c].sum()) for c in range(8)]
        res['variants'][name] = dict(manhattan=obs, null=nl, z=float(z), n_sig=n_sig,
                                     frac_sig=float(np.mean(n_sig) / res['n_pairs']),
                                     spikes_per_cell_trial=float(Xv.sum() / Xv.shape[0] / Xv.shape[2]))
        if name == 'all':      # graphs (bit-packed) and 40 example significant CCGs (cue 0, lags -25..25) for the figure
            cc0 = ccg_corrected(Xv[cue == 0])
            jj, kk = np.nonzero(g[0])
            pick = np.random.default_rng(0).choice(len(jj), min(40, len(jj)), replace=False)
            np.savez_compressed(os.path.join(OUT, f'ccg_figure_s{s}.npz'), cells=cells, n=len(cells),
                                examples=np.array([cc0[jj[i], kk[i], LAG - 25:LAG + 26] for i in pick]),
                                **{f'g{c}': np.packbits(g[c]) for c in range(8)})
        print(f'session {s} {name}: Manhattan {obs:.1f}, null {np.mean(nl):.1f} +- {np.std(nl, ddof=1):.1f}, '
              f'z {z:.2f}, significant CCGs per cue {np.mean(n_sig):.0f} '
              f'({100 * np.mean(n_sig) / res["n_pairs"]:.2f}% of pairs) ({time.time() - t0:.0f} s)', flush=True)
        json.dump(res, open(os.path.join(OUT, f'session_{s}.json'), 'w'), indent=1)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', type=int, nargs='+', default=[0])
    ap.add_argument('--n_shuf', type=int, default=50)
    ap.add_argument('--n_shuf_state', type=int, default=25)
    a = ap.parse_args()
    for s in a.sessions:
        run_session(s, a.n_shuf, a.n_shuf_state)
