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
PRE = (-400, -50)     # window centres: 100 ms windows that end before cue onset (model spikes follow the cue within ms)
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


def project(Zd, cue, rng, Zpre=None):
    """Cross-validated PCA of the 8 condition means; per-trial projections (trials x T x 2), trial halves, aligned
    condition means per half (2 x 8 x 2) and variance explained."""
    half = np.zeros(len(cue), int)
    for c in range(8):
        idx = rng.permutation(np.flatnonzero(cue == c))
        half[idx[len(idx) // 2:]] = 1
    proj = np.zeros(Zd.shape[:2] + (2,))
    pre = None if Zpre is None else np.zeros(Zpre.shape[:2] + (2,))
    cms, evr = [], []
    for h in (0, 1):
        tr = half != h
        C = np.stack([Zd[tr & (cue == c)].mean((0, 1)) for c in range(8)])
        mu = C.mean(0)
        _, sv, vt = np.linalg.svd(C - mu, full_matrices=False)
        W = vt[:2].T
        evr.append(sv[:2] ** 2 / np.sum(sv ** 2))
        cm, P = (C - mu) @ W, (Zd[half == h] - mu) @ W
        Pp = None if Zpre is None else (Zpre[half == h] - mu) @ W
        if h == 1:
            Rm = procrustes(cms[0], cm)
            cm, P = cm @ Rm, P @ Rm
            Pp = None if Pp is None else Pp @ Rm
        cms.append(cm)
        proj[half == h] = P
        if Pp is not None:
            pre[half == h] = Pp
    return proj, half, np.stack(cms), np.array(evr), pre


def run_session(s):
    f = os.path.join(OUT, f'session_{s}.npz')
    if os.path.exists(f):
        return f
    counts, cue, held = session_counts(s)
    t10 = np.arange(counts.shape[1]) * 10.0 - RU.TUNED['t_pre']
    X, tz = R.windows(counts[held], t10)
    cue = cue[held]
    Zt = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    dl = (tz >= DEL[0]) & (tz <= DEL[1])
    # pre-cue activity (window centres PRE), z-scored the same way: null control
    k = R.WIN // R.STEP
    c = np.cumsum(np.concatenate([np.zeros_like(counts[held][:, :1], dtype=np.int32), counts[held].astype(np.int32)], 1), 1)
    Xa = (c[:, k:] - c[:, :-k]).astype(float)
    ta = t10[:Xa.shape[1]] + R.WIN / 2
    Xp = Xa[:, (ta >= PRE[0]) & (ta <= PRE[1])]
    Zpre = (Xp - Xp.mean(0)) / np.where(Xp.std(0) > 0, Xp.std(0), 1)
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
    out = project(Zt[:, dl], cue, np.random.default_rng(0), Zpre)
    for k, v in zip(('proj', 'half', 'cmeans', 'evr', 'preproj'), out):
        res['same_' + k] = v
    res['same_state'] = st[:, dl]
    # split: states from the top 20 of half H1 of the cells, PCA on H2
    pu = np.random.default_rng(1).permutation(n_cells)
    H1, H2 = np.sort(pu[:n_cells // 2]), np.sort(pu[n_cells // 2:])
    uA, uB = H1[np.argsort(-F[1][H1])[:20]], H1[np.argsort(-F[0][H1])[:20]]
    st2 = label_states(X, cue, (uA, uB), half, np.random.default_rng(s + 1), tz)
    out = project(Zt[:, dl][:, :, H2], cue, np.random.default_rng(0), Zpre[:, :, H2])
    for k, v in zip(('proj', 'half', 'cmeans', 'evr', 'preproj'), out):
        res['split_' + k] = v
    res['split_state'] = st2[:, dl]
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f, **res)
    return f


def redo_pre(s):
    """Recompute only the pre-cue projections of a stored session (same axes, no re-decoding)."""
    f = os.path.join(OUT, f'session_{s}.npz')
    d = dict(np.load(f))
    counts, cue, held = session_counts(s)
    t10 = np.arange(counts.shape[1]) * 10.0 - RU.TUNED['t_pre']
    X, tz = R.windows(counts[held], t10)
    Zt = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    dl = (tz >= DEL[0]) & (tz <= DEL[1])
    k = R.WIN // R.STEP
    c = np.cumsum(np.concatenate([np.zeros_like(counts[held][:, :1], dtype=np.int32), counts[held].astype(np.int32)], 1), 1)
    Xa = (c[:, k:] - c[:, :-k]).astype(float)
    ta = t10[:Xa.shape[1]] + R.WIN / 2
    Xp = Xa[:, (ta >= PRE[0]) & (ta <= PRE[1])]
    Zpre = (Xp - Xp.mean(0)) / np.where(Xp.std(0) > 0, Xp.std(0), 1)
    n = X.shape[2]
    pu = np.random.default_rng(1).permutation(n)
    H2 = np.sort(pu[n // 2:])
    out_same = project(Zt[:, dl], cue[held], np.random.default_rng(0), Zpre)
    out_split = project(Zt[:, dl][:, :, H2], cue[held], np.random.default_rng(0), Zpre[:, :, H2])
    assert np.allclose(out_split[0], d['split_proj']) and np.allclose(out_same[0], d['same_proj'])
    d['same_preproj'], d['split_preproj'] = out_same[4], out_split[4]
    np.savez_compressed(f, **d)


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


def paper_panels(sessions, ex=None, n_shuf=50):
    """Two-panel figure as the data's fig_pca_onoff_paper (independent units): c, example session, per-cue mean
    positions during On (filled) and Off (open) moments; d, alignment with the cue (cosine to the cue's mean, minus
    the same with cue labels shuffled within trial half, n_shuf shuffles) for On, Off and pre-cue activity."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_rel, ttest_1samp
    INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#d8d7d2'
    rng = np.random.default_rng(0)
    rows, pos_ex, best = [], {}, (-1, None)
    for s in sessions:
        d = np.load(os.path.join(OUT, f'session_{s}.npz'))
        proj, half, cms, st, cue, pre = (d['split_proj'], d['split_half'], d['split_cmeans'], d['split_state'], d['cue'],
                                         d['split_preproj'])
        per = {}
        for p in ('on', 'off', 'pre', 'all'):
            tri, P = [], []
            for i in range(len(cue)):
                m = (st[i] == 1) if p == 'on' else (st[i] == -1) if p == 'off' else np.ones(st.shape[1], bool)
                if p == 'pre':
                    tri.append(i); P.append(pre[i].mean(0))
                elif m.any():
                    tri.append(i); P.append(proj[i][m].mean(0))
            per[p] = (np.array(tri), np.array(P))
        dec_all = np.mean([np.argmin(((cms[half[i]] - x) ** 2).sum(1)) == cue[i] for i, x in zip(*per['all'])])
        if dec_all > best[0]:
            best = (dec_all, s)
        pos_ex[s] = (per, cms, cue)
        for p in ('on', 'off', 'pre'):
            tri, P = per[p]
            h = half[tri]

            def mcos(c):
                cen = cms[h, c]
                return np.mean(np.sum(P * cen, 1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(cen, axis=1) + 1e-12))
            raw = mcos(cue[tri])
            sh = []
            for _ in range(n_shuf):
                c2 = cue[tri].copy()
                for hh in (0, 1):
                    mm = h == hh
                    c2[mm] = rng.permutation(c2[mm])
                sh.append(mcos(c2))
            rows.append(dict(session=s, period=p, raw=float(raw), shuffle=float(np.mean(sh)),
                             corrected=float(raw - np.mean(sh)), n_trials=len(tri)))
    ex = best[1] if ex is None else ex
    with open(os.path.join(OUT, 'panel_d_raw_and_shuffle.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    D = {p: np.array([r['corrected'] for r in rows if r['period'] == p]) for p in ('on', 'off', 'pre')}
    stats = dict(n_sessions=len(sessions), example_session=ex,
                 **{f'{p}_mean': float(D[p].mean()) for p in D}, **{f'{p}_sem': float(D[p].std(ddof=1) / np.sqrt(len(D[p]))) for p in D},
                 p_on_vs_off_paired_t=float(ttest_rel(D['on'], D['off']).pvalue),
                 p_off_vs_0_t=float(ttest_1samp(D['off'], 0).pvalue), p_pre_vs_0_t=float(ttest_1samp(D['pre'], 0).pvalue),
                 p_off_vs_pre_paired_t=float(ttest_rel(D['off'], D['pre']).pvalue),
                 n_off_above_0=int((D['off'] > 0).sum()))
    json.dump(stats, open(os.path.join(OUT, 'panel_d_stats.json'), 'w'), indent=1)
    plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                         'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
                         'axes.labelcolor': INK, 'axes.spines.top': False, 'axes.spines.right': False})
    cmap = plt.get_cmap('hsv')
    ang = np.deg2rad(np.arange(8) * 45.0)
    ang = np.angle(np.exp(1j * ang))
    col = [cmap(((a_ + np.pi) / (2 * np.pi)) % 1) for a_ in ang]
    per, cms, cue = pos_ex[ex]
    ring = cms.mean(0)
    on = np.array([per['on'][1][cue[per['on'][0]] == c].mean(0) for c in range(8)])
    off = np.array([per['off'][1][cue[per['off'][0]] == c].mean(0) for c in range(8)])
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.6, 2.3), gridspec_kw=dict(width_ratios=[1.15, 1]), constrained_layout=True)
    o = np.argsort(ang)
    r = ring[o]
    a.plot(np.r_[r[:, 0], r[:1, 0]], np.r_[r[:, 1], r[:1, 1]], color=GRID, lw=1.2, zorder=1)
    for i in range(8):
        a.annotate('', xy=off[i], xytext=on[i], zorder=3,
                   arrowprops=dict(arrowstyle='-|>', color=col[i], lw=1.0, shrinkA=3.5, shrinkB=3.5, mutation_scale=7))
    a.scatter(on[:, 0], on[:, 1], s=28, c=col, edgecolors=INK, linewidths=0.6, zorder=4, label='On')
    a.scatter(off[:, 0], off[:, 1], s=28, facecolors='white', edgecolors=col, linewidths=1.2, zorder=4, label='Off')
    a.axhline(0, color=GRID, lw=0.4, zorder=0); a.axvline(0, color=GRID, lw=0.4, zorder=0)
    lim = np.abs(np.r_[ring.ravel(), on.ravel()]).max() * 1.15
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect('equal')
    a.set_xlabel('PC1'); a.set_ylabel('PC2')
    a.legend(frameon=False, fontsize=6, loc='lower right', handletextpad=0.2, borderpad=0.1)
    k = a.inset_axes([0.0, 0.76, 0.24, 0.24])
    k.scatter(np.cos(ang), np.sin(ang), s=7, c=col, lw=0)
    k.set_xlim(-1.4, 1.4); k.set_ylim(-1.4, 1.4); k.set_aspect('equal'); k.axis('off')
    for s in sessions:
        v = {r_['period']: r_['corrected'] for r_ in rows if r_['session'] == s}
        b.plot([0, 1], [v['on'], v['off']], color=MUTED, lw=0.6, alpha=0.55, marker='o', ms=2, mew=0)
        b.plot([2.3], [v['pre']], color=MUTED, alpha=0.55, marker='o', ms=2, mew=0, ls='')
    m = [D[p].mean() for p in ('on', 'off', 'pre')]
    se = [D[p].std(ddof=1) / np.sqrt(len(D[p])) for p in ('on', 'off', 'pre')]
    b.errorbar([0, 1], m[:2], se[:2], color=INK, lw=1.8, marker='o', ms=4, capsize=0, zorder=5, label='Model')
    b.errorbar([2.3], [m[2]], [se[2]], color=INK, marker='o', ms=4, capsize=0, zorder=5)
    ref = os.path.join(os.path.dirname(REF), '..', 'paper', 'panel_d_stats.csv')
    if os.path.exists(ref):
        r_ = {row[0]: row[1] for row in csv.reader(open(ref)) if len(row) == 2 and row[0]}   # key,value rows
        b.plot([0.15, 1.15, 2.45], [float(r_['on_mean']), float(r_['off_mean']), float(r_['pre_mean'])], 'x',
               color='#d62728', ms=5, mew=1, zorder=6, label='Data mean')
    b.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    b.set_xticks([0, 1, 2.3], ['On', 'Off', 'pre-cue\n(null)']); b.set_xlim(-0.4, 2.7)
    b.set_ylabel('alignment with cue\n(cosine, minus label-shuffle)')
    b.legend(frameon=False, fontsize=5.5, loc='upper right')
    b.set_title(f'Spiking network, {len(sessions)} sessions', fontsize=7, color=MUTED)
    os.makedirs(FIG, exist_ok=True)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(os.path.join(FIG, f'fig_pca_onoff_paper.{ext}'), dpi=300)
    plt.close(fig)
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', type=int, nargs='+', default=[0])
    ap.add_argument('--figure', action='store_true')
    ap.add_argument('--paper', action='store_true')
    a = ap.parse_args()
    for s in a.sessions:
        run_session(s)
        print(f'session {s} done', flush=True)
    if a.paper:
        paper_panels(a.sessions)
    if a.figure:
        figure(a.sessions)
        for r in summarise(a.sessions):
            if r['session'] == a.sessions[0] or len(a.sessions) == 1:
                print(r)
