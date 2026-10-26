"""On/Off states in the plane of the cue-condition means (PC1-PC2), on the spiking-network sessions.

Re-implementation of the data analysis (jmourabarbosa/panichello, branch analysis-replication,
analysis/pca_onoff.py and pca_onoff_handoff/README.md), applied to the simulated sessions of run.sessions:
- activity: 100 ms rates every 10 ms, z-scored across trials at each time point and cell; correct trials;
- On/Off labels from the data-matched decoder (readout.decode: L2 logistic, lambda = 10,000, cue vs opposite cue,
  leave-one-out, 30 shuffles; On/Off labelling as the data). Two versions:
    'same'  states and PCA from all recorded cells (partly circular);
    'split' states from the top 20 cells of a random half of the cells (8-cue ANOVA of the delay rate, ranked on
            the other trial half), PCA and projection on the other half of the cells (independent);
- PCA of the 8 cue-condition means of delay activity (500-1,400 ms), cross-validated over trial halves (axes from one
  half, applied to the other; the second half's plane aligned to the first's by orthogonal Procrustes);
- per trial, the mean PC1-PC2 position over its delay time points labelled On, labelled Off and over the whole delay;
  measures: cosine to the cue's direction, nearest-centroid cue decoding (centroids = condition means of the other
  trial half, chance 1/8) and projection on the cue's direction.
Differences from the data code: the decoder features are spike counts (as in the main read-out) rather than z-scored
rates; the main simulation has 480 trials per session (about 410 correct) and 330 cells.

    python -m spiking.pca_onoff --sessions 0 1 2 ...     # re-simulates sessions' counts if needed (~7 min each)
    python -m spiking.pca_onoff --figure
"""
import argparse
import csv
import json
import os

import numpy as np

from paper_model import readout as PR
from . import network as N, readout as R, run as RU

OUT = os.path.join(RU.OUT, 'pca_onoff')
FIG = os.path.join(os.path.dirname(__file__), 'figures')
DEL = (500, 1400)
REF = '/home/user/panichello/analysis/pca_onoff_handoff/reference/pca_onoff_stats.csv'


def session_counts(s, n_trials=RU.N_TRIALS, seed0=0):
    """10 ms counts, cue and memory of session s of run.sessions (re-simulated with the same seeds if not stored)."""
    d = np.load(os.path.join(RU.OUT, 'sessions_dips_trials.npz'))
    if f's{s}_counts' in d and len(d[f's{s}_counts']) == n_trials:
        return d[f's{s}_counts'], d[f's{s}_cue'], d[f's{s}_held']
    f = os.path.join(OUT, f'counts_s{s}.npz')
    if os.path.exists(f):
        c = np.load(f)
        return c['counts'], c['cue'], c['held']
    net = RU.make_net()
    rng = np.random.default_rng(seed0 + 100 + s)
    rec = RU.record(net, rng)
    cue = rng.permutation(np.repeat(np.arange(8), n_trials // 8))
    sim = RU.run_trials(net, cue, seed0 + s, rec)
    assert np.array_equal(sim['mem']['held'], d[f's{s}_held']), 'not the same trials as run.sessions'
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f, counts=sim['counts'], cue=cue, held=sim['mem']['held'])
    return sim['counts'], cue, sim['mem']['held']


def label_states(X, cue, units_by_half, half, rng, tz):
    """On/Off states (trials x time) with the decoder on the given units; units_by_half: (units for trials of half 0,
    units for trials of half 1), or one array for all trials."""
    valid = (tz >= 0) & (tz <= 1400)
    if not isinstance(units_by_half, tuple):
        z, zn = R.decode(X, cue, rng, units_by_half)
    else:
        z = np.zeros(X.shape[:2]); zns = []
        for h in (0, 1):
            zh, zn = R.decode(X, cue, rng, units_by_half[h])
            z[half == h] = zh[half == h]
            zns.append(zn[:, half == h])
        zn = np.concatenate(zns, 1)
    return PR.label(np.nan_to_num(z), valid, R.null_threshold(zn, valid))


def procrustes(A, B):
    u, _, vt = np.linalg.svd(B.T @ A)
    return u @ vt


def project(Zd, cue, rng):
    """Cross-validated PCA of the 8 condition means; per-trial projections (trials x T x 2), trial halves, aligned
    condition means per half (2 x 8 x 2) and variance explained."""
    half = np.zeros(len(cue), int)
    for c in range(8):
        idx = rng.permutation(np.flatnonzero(cue == c))
        half[idx[len(idx) // 2:]] = 1
    proj = np.zeros(Zd.shape[:2] + (2,))
    cms, evr = [], []
    for h in (0, 1):
        tr = half != h
        C = np.stack([Zd[tr & (cue == c)].mean((0, 1)) for c in range(8)])
        mu = C.mean(0)
        _, sv, vt = np.linalg.svd(C - mu, full_matrices=False)
        W = vt[:2].T
        evr.append(sv[:2] ** 2 / np.sum(sv ** 2))
        cm, P = (C - mu) @ W, (Zd[half == h] - mu) @ W
        if h == 1:
            Rm = procrustes(cms[0], cm)
            cm, P = cm @ Rm, P @ Rm
        cms.append(cm)
        proj[half == h] = P
    return proj, half, np.stack(cms), np.array(evr)


def run_session(s):
    f = os.path.join(OUT, f'session_{s}.npz')
    if os.path.exists(f):
        return f
    counts, cue, held = session_counts(s)
    X, tz = R.windows(counts[held], np.arange(counts.shape[1]) * 10.0 - RU.TUNED['t_pre'])
    cue = cue[held]
    Zt = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    dl = (tz >= DEL[0]) & (tz <= DEL[1])
    n_cells = X.shape[2]
    res = dict(cue=cue, n_cells=n_cells, t=tz[dl])
    # trial halves for the ANOVA ranking (stratified by cue)
    rng = np.random.default_rng(0)
    half = np.zeros(len(cue), int)
    for c in range(8):
        idx = rng.permutation(np.flatnonzero(cue == c))
        half[idx[len(idx) // 2:]] = 1
    dr = Zt[:, dl].mean(1)
    F = [R.anova_f(dr[half == h], cue[half == h]) for h in (0, 1)]
    # same: states and PCA from all cells
    st = label_states(X, cue, np.arange(n_cells), half, np.random.default_rng(s), tz)
    out = project(Zt[:, dl], cue, np.random.default_rng(0))
    for k, v in zip(('proj', 'half', 'cmeans', 'evr'), out):
        res['same_' + k] = v
    res['same_state'] = st[:, dl]
    # split: states from the top 20 of half H1 of the cells, PCA on H2
    pu = np.random.default_rng(1).permutation(n_cells)
    H1, H2 = np.sort(pu[:n_cells // 2]), np.sort(pu[n_cells // 2:])
    uA, uB = H1[np.argsort(-F[1][H1])[:20]], H1[np.argsort(-F[0][H1])[:20]]
    st2 = label_states(X, cue, (uA, uB), half, np.random.default_rng(s + 1), tz)
    out = project(Zt[:, dl][:, :, H2], cue, np.random.default_rng(0))
    for k, v in zip(('proj', 'half', 'cmeans', 'evr'), out):
        res['split_' + k] = v
    res['split_state'] = st2[:, dl]
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f, **res)
    return f


def summarise(sessions):
    """Per session and version: mean over trials of the three measures for On, Off and the whole delay."""
    rows = []
    for s in sessions:
        d = np.load(os.path.join(OUT, f'session_{s}.npz'))
        for v in ('same', 'split'):
            proj, half, cms, st = d[v + '_proj'], d[v + '_half'], d[v + '_cmeans'], d[v + '_state']
            acc = {p: dict(cos=[], dec=[], along=[], n=[]) for p in ('on', 'off', 'all')}
            for i, c in enumerate(d['cue']):
                cen = cms[half[i]]
                for p, m in (('on', st[i] == 1), ('off', st[i] == -1), ('all', np.ones(st.shape[1], bool))):
                    if not m.any():
                        continue
                    x = proj[i][m].mean(0)
                    u = cen[c] / np.linalg.norm(cen[c])
                    acc[p]['cos'].append(np.dot(x, u) / (np.linalg.norm(x) + 1e-12))
                    acc[p]['dec'].append(int(np.argmin(((cen - x) ** 2).sum(1)) == c))
                    acc[p]['along'].append(np.dot(x, u))
                    acc[p]['n'].append(m.sum())
            for p, a in acc.items():
                rows.append(dict(version=v, session=s, period=p, cos_to_cue=float(np.mean(a['cos'])),
                                 correct_decode=float(np.mean(a['dec'])), along_cue=float(np.mean(a['along'])),
                                 n_trials=len(a['cos']), n_points=float(np.mean(a['n'])),
                                 evr_pc1=float(d[v + '_evr'][:, 0].mean()), evr_pc2=float(d[v + '_evr'][:, 1].mean())))
    with open(os.path.join(OUT, 'pca_sessions.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return rows


def data_means():
    """Data means per version and period (25 sessions) from the hand-off reference, if available."""
    out = {}
    if not os.path.exists(REF):
        return out
    for r in csv.DictReader(open(REF)):           # rows: measure, version; columns on, off, all
        for p in ('on', 'off', 'all'):
            out.setdefault((r['version'], p), {})[r['measure']] = r[p]
    return out


def figure(sessions, ex=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_rel, ttest_1samp
    rows = summarise(sessions)
    INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#d8d7d2'
    plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6, 'axes.edgecolor': MUTED, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    cmap = plt.get_cmap('hsv')
    ang = np.deg2rad(np.arange(8) * 45.0)
    col = lambda c: cmap(((ang[c] + np.pi) / (2 * np.pi)) % 1)
    d = np.load(os.path.join(OUT, f'session_{ex}.npz'))
    proj, half, cms, st, cue = d['split_proj'], d['split_half'], d['split_cmeans'], d['split_state'], d['cue']
    pos = {p: [] for p in ('on', 'off')}
    for i, c in enumerate(cue):
        for p, m in (('on', st[i] == 1), ('off', st[i] == -1)):
            if m.any():
                pos[p].append((c, *proj[i][m].mean(0)))
    pos = {p: np.array(v) for p, v in pos.items()}
    ring = cms.mean(0)
    lims = np.percentile(np.abs(np.r_[pos['on'][:, 1:], pos['off'][:, 1:]]), 98) * 1.1
    fig = plt.figure(figsize=(9.5, 6.4))
    gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.45, left=0.07, right=0.98, top=0.9, bottom=0.08)
    fig.suptitle('Spiking network: delay activity in the plane of the cue-condition means (PC1-PC2), during On and '
                 'Off states', fontsize=9)

    def draw_ring(a):
        a.plot(np.r_[ring[:, 0], ring[:1, 0]], np.r_[ring[:, 1], ring[:1, 1]], color=GRID, lw=1.2, zorder=1)
        a.axhline(0, color=GRID, lw=0.4, zorder=0); a.axvline(0, color=GRID, lw=0.4, zorder=0)
        a.set_xlim(-lims, lims); a.set_ylim(-lims, lims); a.set_aspect('equal')
        a.set_xlabel('PC1'); a.set_ylabel('PC2')
    cents = {}
    for k, (p, title) in enumerate((('on', 'On moments'), ('off', 'Off moments'))):
        a = fig.add_subplot(gs[0, k]); draw_ring(a)
        P = pos[p]
        a.scatter(P[:, 1], P[:, 2], s=3, c=[col(int(c)) for c in P[:, 0]], alpha=0.5, lw=0)
        cents[p] = np.array([P[P[:, 0] == c, 1:].mean(0) for c in range(8)])
        for c in range(8):
            a.scatter(*cents[p][c], s=60, color=col(c), edgecolor=INK, lw=0.8, zorder=3)
        a.set_title(f'{"ab"[k]}  {title}\none dot per trial (n = {len(P)}); large:\naverage per cue', loc='left',
                    fontsize=8, fontweight='bold')
    a = fig.add_subplot(gs[0, 2]); draw_ring(a)
    for c in range(8):
        a.annotate('', xy=cents['off'][c], xytext=cents['on'][c], arrowprops=dict(arrowstyle='->', color=col(c), lw=1))
        a.scatter(*cents['on'][c], s=60, color=col(c), edgecolor=INK, lw=0.8, zorder=3)
        a.scatter(*cents['off'][c], s=60, facecolor='white', edgecolor=col(c), lw=1.5, zorder=3)
    a.set_title('c  From On to Off\nfilled: On average; open: Off\naverage, per cue', loc='left', fontsize=8,
                fontweight='bold')
    a = fig.add_subplot(gs[0, 3]); a.axis('off')
    for c in range(8):
        a.scatter(np.cos(ang[c]), np.sin(ang[c]), s=80, color=col(c), edgecolor=INK, lw=0.6)
    a.plot(0, 0, '+', color=INK); a.set_xlim(-1.6, 1.6); a.set_ylim(-1.6, 1.6); a.set_aspect('equal')
    evr = d['split_evr'].mean(0)
    a.set_title(f'Colour = cue location. Session {ex},\nindependent units. Grey ring (a-c):\naverage activity per '
                f'cue over the\ndelay (PC1 {100 * evr[0]:.0f}%, PC2 {100 * evr[1]:.0f}% of its variance)',
                loc='left', fontsize=7)
    ref = data_means()
    for k, (m, title, ylab, chance) in enumerate((('cos_to_cue', 'd  Alignment with the cue', 'cosine to cue direction', 0),
                                                  ('correct_decode', 'e  Cue decoding (chance 1/8)', 'cue decoding (nearest centroid)', 1 / 8),
                                                  ('along_cue', 'f  Distance along the cue', 'projection on cue direction', 0))):
        a = fig.add_subplot(gs[1, k])
        for g, v in enumerate(('split', 'same')):
            vals = np.array([[next(r[m] for r in rows if r['version'] == v and r['session'] == s and r['period'] == p)
                              for p in ('on', 'off', 'all')] for s in sessions])
            x = np.arange(3) + 4 * g
            for row in vals:
                a.plot(x, row, color=GRID, lw=0.6, zorder=1)
            mu = vals.mean(0)
            se = vals.std(0, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.zeros(3)
            a.errorbar(x, mu, yerr=2 * se, fmt='o', color=INK, ms=4, elinewidth=1.5, capsize=0, zorder=3)
            dv = 'split' if v == 'split' else 'same'
            dm = [ref.get((dv, p), {}).get(m) for p in ('on', 'off', 'all')]
            if all(x_ is not None for x_ in dm):
                a.plot(x + 0.25, [float(x_) for x_ in dm], 'x', color='#d62728', ms=5, mew=1, zorder=4)
            if len(vals) > 1:
                p_on_off = ttest_rel(vals[:, 0], vals[:, 1]).pvalue
                p_off = ttest_1samp(vals[:, 1], chance).pvalue
                a.text(x[1], 1.0, f'On vs Off p = {p_on_off:.1g}\nOff vs {"chance" if chance else "0"}: p = {p_off:.1g}',
                       transform=a.get_xaxis_transform(), ha='center', va='top', fontsize=5.5, color=MUTED)
        a.axhline(chance, color=MUTED, ls='--', lw=0.6)
        a.set_xticks([0, 1, 2, 4, 5, 6], ['On', 'Off', 'all'] * 2)
        a.set_title(title, loc='left', fontsize=8, fontweight='bold'); a.set_ylabel(ylab)
        a.text(1, -0.16, 'independent units', transform=a.get_xaxis_transform(), ha='center', va='top')
        a.text(5, -0.16, 'same units', transform=a.get_xaxis_transform(), ha='center', va='top')
    a = fig.add_subplot(gs[1, 3]); a.axis('off')
    a.text(0, 1, f'{len(sessions)} simulated sessions.\nGrey lines: sessions; black: mean\n± 2 SEM; red x: data mean '
                 '(25\nsessions); dashed: chance (1/8) or 0.\n\nIndependent units: On/Off labelled\nfrom the top '
                 '20 cells of one random\nhalf of the cells; PCA and projection\non the other half.\n\nSame units: '
                 'labels and PCA from\nall cells (partly circular).\n\nPCA axes cross-validated across\ntrial halves.',
           va='top', fontsize=7)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, 'fig_pca_onoff.png'), dpi=200)
    fig.savefig(os.path.join(FIG, 'fig_pca_onoff.svg'))
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', type=int, nargs='+', default=[0])
    ap.add_argument('--figure', action='store_true')
    a = ap.parse_args()
    for s in a.sessions:
        run_session(s)
        print(f'session {s} done', flush=True)
    if a.figure:
        figure(a.sessions)
        for r in summarise(a.sessions):
            if r['session'] == a.sessions[0] or len(a.sessions) == 1:
                print(r)
