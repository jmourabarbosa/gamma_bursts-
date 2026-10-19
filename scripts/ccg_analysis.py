"""Panichello et al. 2024 Fig. 5 on simulated data: cue-specific functional connectivity from
cross-correlograms (CCGs), for the whole delay and separately for On and Off states.

Follows the paper's Methods and their code (github.com/panichem/SingleTrialDynamics, ccgs/):
- 1 ms bins, binary spike trains ("at least one spike"), units with mean rate > 1 Hz,
  delay window 500-1400 ms after cue onset;
- CCG per pair and cue condition, normalised by the geometric mean spike count
  (sqrt(sum pre * sum post) at each lag), lags -100..100 ms;
- jitter correction: subtract the CCG of the expected jittered trains (25 ms window,
  preserves PSTH and per-trial counts per window);
- significant CCG: peak (lag != 0) within 10 ms of zero and > 7 SD above the mean of the
  baseline 50 < |lag| < 100;
- graph per condition (edge = significant CCG); Manhattan distance between the two graphs;
  null from shuffling cue labels across trials; z-score;
- On / Off variants: spikes set to zero outside On (Off) time points (decoder states from
  gamma_wm.panichello on the same recorded population).

usage: python scripts/ccg_analysis.py --data data/m1_r4_s4_d85_session --K 300 --out figures/ccg_m1
"""
import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_wm import panichello as pa  # noqa: E402
from gamma_wm.network import A, B, NS, INH  # noqa: E402
from heldout_tuning import load  # noqa: E402

C_ON, C_OFF, C_A, C_B, INK, INK2 = '#2a78d6', '#eb6834', '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
plt.rcParams.update({'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 8,
                     'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'legend.frameon': False})
LAG = 100


def spike_array(trials, sample, t0, t1):
    """(trials, units, T) binary 1 ms spike trains in [t0, t1)."""
    T = int(t1 - t0)
    lut = np.full(max(sample.max() + 1, 1), -1)
    lut[sample] = np.arange(len(sample))
    X = np.zeros((len(trials), len(sample), T), np.float32)
    for k, d in enumerate(trials):
        st, si = d['spk_t'], d['spk_i']
        m = (st >= t0) & (st < t1) & (si < len(lut))
        st, si = st[m], si[m]
        u = lut[si]
        ok = u >= 0
        X[k, u[ok], (st[ok] - t0).astype(int)] = 1.0
    return X


def jitter_expect(X, w=25):
    """Expected spike train under jitter within w ms windows (per trial counts, PSTH shape)."""
    n, N, T = X.shape
    pad = (-T) % w
    Xp = np.concatenate([X, np.zeros((n, N, pad), X.dtype)], 2) if pad else X
    W = Xp.shape[2] // w
    psth = Xp.mean(0).reshape(N, W, w)
    psum = psth.sum(-1, keepdims=True)
    shape = np.where(psum > 0, psth / np.where(psum > 0, psum, 1), 0)
    cnt = Xp.reshape(n, N, W, w).sum(-1, keepdims=True)
    return (cnt * shape[None]).reshape(n, N, W * w)[:, :, :T].astype(np.float32)


def ccg(X):
    """Normalised CCG for all ordered pairs: (N, N, 2*LAG+1); entry [j, k, LAG + l] =
    sum_t x_j(t) x_k(t + l) / sqrt(sum x_j * sum x_k) over the overlapping bins."""
    n, N, T = X.shape
    Xn = np.ascontiguousarray(X.transpose(1, 0, 2))
    out = np.zeros((N, N, 2 * LAG + 1), np.float32)
    for l in range(LAG + 1):
        a = Xn[:, :, :T - l].reshape(N, -1)
        b = Xn[:, :, l:].reshape(N, -1)
        c = a @ b.T
        norm = np.sqrt(np.outer(a.sum(1), b.sum(1)))
        c = np.where(norm > 0, c / np.where(norm > 0, norm, 1), 0)
        out[:, :, LAG + l] = c
        out[:, :, LAG - l] = c.T
    return out


def significant(X):
    """Boolean (N, N) upper-triangular matrix of significant CCGs, and the corrected CCGs."""
    cc = ccg(X) - ccg(jitter_expect(X))
    lags = np.arange(-LAG, LAG + 1)
    base = (np.abs(lags) > 50) & (np.abs(lags) < 100)
    nz = lags != 0
    mu = cc[:, :, base].mean(-1)
    sd = cc[:, :, base].std(-1)
    sub = cc[:, :, nz]
    pk = sub.argmax(-1)
    peak_lag = lags[nz][pk]
    peak = sub.max(-1)
    sig = (np.abs(peak_lag) <= 10) & (peak > mu + 7 * sd) & (sd > 0)
    return np.triu(sig, 1), cc


def manhattan(X, y):
    gA, _ = significant(X[y == 1])
    gB, _ = significant(X[y == 0])
    return int((gA != gB).sum()), gA, gB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--K', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-shuffle', type=int, default=25)
    ap.add_argument('--lam', type=float, default=10.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meta, group, trials = load(args.data)
    T_ = meta['task']
    t_cue, t_don = T_['t_pre'], T_['t_pre'] + T_['cue_dur']
    n_t = int(t_don + T_['t_delay'])
    t0, t1 = t_cue + 500, min(t_cue + 1400, n_t)
    y = np.array([d['cue'] == 'A' for d in trials], int)
    rng = np.random.default_rng(args.seed)
    sample = np.sort(rng.choice(len(group), args.K, replace=False))
    # decoder states on the recorded population (as in the paper)
    r = pa.run(trials, sample, y, n_t, t_cue, delay_start=t_don + 500, count_window=(t_don, n_t),
               n_shuffle=10, seed=args.seed + 1, lam=args.lam, split_half=False)
    state, tc = r['state'], r['t']
    X = spike_array(trials, sample, t0, t1)
    rate = X.sum((0, 2)) / (len(trials) * (t1 - t0) / 1000)
    keep = rate > 1.0
    X, units = X[:, keep], sample[keep]
    g = group[units]
    # per-ms state masks
    tt = np.arange(t0, t1)
    idx = np.clip(np.searchsorted(tc, tt), 0, len(tc) - 1)
    smask = state[:, idx]                               # (trials, T)
    S = dict(data=args.data, K=args.K, n_units=int(keep.sum()), n_pairs=int(keep.sum() * (keep.sum() - 1) // 2),
             n_pool_units=int(np.isin(g, [A, B]).sum()), window=[t0, t1],
             frac_on=float((smask == 1).mean()), frac_off=float((smask == -1).mean()))
    print(json.dumps(S), flush=True)
    variants = {'all': np.ones_like(smask, bool), 'On': smask == 1, 'Off': smask == -1}
    res = {}
    keep_graphs = {}
    for name, m in variants.items():
        t_ = time.time()
        Xm = X * m[:, None, :].astype(np.float32)
        md, gA, gB = manhattan(Xm, y)
        null = []
        for s in range(args.n_shuffle):
            ys = np.random.default_rng(1000 + s).permutation(y)
            null.append(manhattan(Xm, ys)[0])
        null = np.array(null, float)
        z = (md - null.mean()) / null.std() if null.std() > 0 else float('nan')
        # which pairs carry the cue-specific edges
        diff = np.argwhere(gA != gB)
        kinds = {}
        for j, k in diff:
            a, b = g[j], g[k]
            if a in (A, B) and b == a:
                key = 'same memory pool'
            elif a in (A, B) and b in (A, B):
                key = 'different pools'
            elif a in (A, B) or b in (A, B):
                key = 'pool-background'
            else:
                key = 'background-background'
            kinds[key] = kinds.get(key, 0) + 1
        res[name] = dict(manhattan=md, null_mean=float(null.mean()), null_sd=float(null.std()), z=float(z),
                         n_sig_A=int(gA.sum()), n_sig_B=int(gB.sum()),
                         frac_sig=float((gA.sum() + gB.sum()) / 2 / S['n_pairs']),
                         cue_specific_edges_by_type=kinds, null=[float(v) for v in null])
        keep_graphs[name] = (gA, gB, null, md)
        print(name, json.dumps({k: v for k, v in res[name].items() if k != 'null'}), f'({time.time() - t_:.0f}s)', flush=True)
        S['results'] = res
        json.dump(S, open(os.path.join(args.out, 'ccg_summary.json'), 'w'), indent=1)

    # ---------------- figure (Panichello Fig. 5 b-f analogue)
    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.4)
    # b: example CCGs of a same-pool pair under the two cues
    gA, gB, _, _ = keep_graphs['all']
    _, ccA = significant(X[y == 1])
    _, ccB = significant(X[y == 0])
    pairs = np.argwhere(gA ^ gB)
    lags = np.arange(-LAG, LAG + 1)
    if len(pairs):
        j, k = pairs[np.argmax([max(ccA[j, k].max(), ccB[j, k].max()) for j, k in pairs])]
        a = fig.add_subplot(gs[0, 0])
        a.plot(lags, ccA[j, k], color=C_A, lw=1, label='cue A')
        a.plot(lags, ccB[j, k], color=C_B, lw=1, label='cue B')
        a.set_xlabel('lag (ms)')
        a.set_ylabel('jitter-corrected CCG')
        a.set_title(f'Example pair ({["NS", "A", "B", "I", "IA", "IB"][g[j]]}–{["NS", "A", "B", "I", "IA", "IB"][g[k]]})',
                    fontsize=8)
        a.legend(fontsize=7)
    # c: connectivity maps
    N = len(units)
    order = np.argsort(g * 10000 + np.arange(N))
    pos = np.zeros(N, int)
    pos[order] = np.arange(N)
    ang = 2 * np.pi * pos / N
    xy = np.c_[np.cos(ang), np.sin(ang)]
    cols = np.array([INK2 if gg in (NS,) else ('#1baf7a' if gg >= INH else (C_A if gg == A else C_B)) for gg in g])
    for i, (gm, lab) in enumerate(((gA, 'cue A'), (gB, 'cue B'))):
        a = fig.add_subplot(gs[0, 1 + i])
        for j, k in np.argwhere(gm):
            a.plot(xy[[j, k], 0], xy[[j, k], 1], color=INK, lw=0.4, alpha=0.6)
        a.scatter(xy[:, 0], xy[:, 1], s=6, c=cols, zorder=3)
        a.set_aspect('equal')
        a.axis('off')
        a.set_title(f'Significant CCGs, {lab} (n={int(gm.sum())})', fontsize=8)
    # d: null vs real (all delay)
    a = fig.add_subplot(gs[0, 3])
    _, _, null, md = keep_graphs['all']
    a.hist(null, bins=12, color='#e4e3df', edgecolor=INK2)
    a.axvline(md, color=INK, ls='--')
    a.set_xlabel('Manhattan distance (A vs B)')
    a.set_ylabel('count (shuffled cues)')
    a.set_title(f'Whole delay: z = {res["all"]["z"]:.2f}', fontsize=8)
    # f: z by state
    a = fig.add_subplot(gs[1, 0:2])
    names = ['all', 'On', 'Off']
    zs = [res[n]['z'] for n in names]
    a.bar(range(3), zs, color=['#9a9893', C_ON, C_OFF], width=0.6)
    a.axhline(0, color=INK2, lw=0.8)
    a.axhline(1.96, color=INK2, ls=':', lw=0.8)
    a.set_xticks(range(3), ['whole delay', 'On states', 'Off states'])
    a.set_ylabel('normalised Manhattan distance (z)')
    a.set_title('Cue-specific connectivity (one simulated session)', fontsize=9)
    # edge types
    a = fig.add_subplot(gs[1, 2:])
    types = ['same memory pool', 'different pools', 'pool-background', 'background-background']
    w = 0.25
    for i, n in enumerate(names):
        v = [res[n]['cue_specific_edges_by_type'].get(t, 0) for t in types]
        a.bar(np.arange(4) + (i - 1) * w, v, w, color=['#9a9893', C_ON, C_OFF][i], label=n)
    a.set_xticks(range(4), types, fontsize=7)
    a.set_ylabel('cue-specific edges')
    a.set_title('Which pairs differ between cues', fontsize=9)
    a.legend(fontsize=7)
    fig.suptitle(f'{args.data}: {N} units > 1 Hz from {args.K} random cells '
                 f'({S["n_pool_units"]} memory cells), {len(trials)} trials', x=0.01, ha='left', fontsize=9)
    fig.savefig(os.path.join(args.out, 'fig_ccg.png'), dpi=150, bbox_inches='tight')
    print('wrote', os.path.join(args.out, 'fig_ccg.png'))


if __name__ == '__main__':
    main()
