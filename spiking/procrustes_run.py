"""Procrustes shape similarity of On, Off and pre-cue activity to the ring of cue means (paper Fig. 4e), spiking
network, following the common spec (paper/handoffs/procrustes_common_spec.md and handoff_spiking_procrustes.md on
branch claude/intermittent-rate-coding-paper-ced984).

- 10 sessions (run.sessions seeds 0-9; the same trials as results/pca_onoff/session_s.npz); correct trials.
- Cells split H1/H2 with default_rng(1); trial halves stratified by cue with default_rng(0).
- On/Off labelled from ALL cells of H1 (pca_onoff.label_states with H1 as one set, rng default_rng(s + 1)).
- Activity of H2: z-scored 100 ms window counts (step 10 ms); delay window centres 500-1,400 ms; pre-cue centres
  -400 to -50 ms.
- procrustes_shape.shape_similarity: full H2 space and first k PCs (>= 90% of cue-mean variance); 50 shuffles.

    python -m spiking.procrustes_run      # writes figures/spiking/procrustes_sessions.csv, procrustes_stats.json,
                                          # net_panel_c.csv
"""
import csv
import json
import os

import numpy as np
from scipy.stats import ttest_rel

from . import pca_onoff as P, procrustes_shape as PS, readout as R, run as RU

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures', 'spiking')


def session_data(s):
    counts, cue, held = P.session_counts(s)
    f = os.path.join(P.OUT, f'counts_s{s}.npz')
    if not os.path.exists(f):
        np.savez_compressed(f, counts=counts, cue=cue, held=held)
    t10 = np.arange(counts.shape[1]) * 10.0 - RU.TUNED['t_pre']
    X, tz = R.windows(counts[held], t10)
    Zt = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    k = R.WIN // R.STEP
    c = np.cumsum(np.concatenate([np.zeros_like(counts[held][:, :1], dtype=np.int32),
                                  counts[held].astype(np.int32)], 1), 1)
    Xa = (c[:, k:] - c[:, :-k]).astype(float)
    ta = t10[:Xa.shape[1]] + R.WIN / 2
    Xp = Xa[:, (ta >= P.PRE[0]) & (ta <= P.PRE[1])]
    Zpre = (Xp - Xp.mean(0)) / np.where(Xp.std(0) > 0, Xp.std(0), 1)
    return X, Zt, Zpre, tz, cue[held]


def run_session(s):
    X, Zt, Zpre, tz, cue = session_data(s)
    n = X.shape[2]
    pu = np.random.default_rng(1).permutation(n)
    H1, H2 = np.sort(pu[:n // 2]), np.sort(pu[n // 2:])
    half = PS.halves(cue, np.random.default_rng(0))
    st = P.label_states(X, cue, H1, half, np.random.default_rng(s + 1), tz)
    dl = (tz >= P.DEL[0]) & (tz <= P.DEL[1])
    state = st[:, dl]
    res, ks = PS.shape_similarity(Zt[:, dl][:, :, H2], Zpre[:, :, H2], cue, state, half)
    np.savez_compressed(os.path.join(P.OUT, f'states_allH1_s{s}.npz'), state=state.astype(np.int8), half=half, cue=cue)
    return res, ks, state


def panel_c(state, s=0):
    """Example session: per-cue mean On / Off positions in the existing 2-PC plane (split_proj), with the all-H1 labels."""
    d = np.load(os.path.join(P.OUT, f'session_{s}.npz'))
    proj, cms, cue = d['split_proj'], d['split_cmeans'], d['cue']
    ring = cms.mean(0)
    with open(os.path.join(OUT, 'net_panel_c.csv'), 'w') as fh:
        fh.write('cue,angle_deg,ring_pc1,ring_pc2,on_pc1,on_pc2,n_on,off_pc1,off_pc2,n_off\n')
        for c in range(8):
            row = [c + 1, 45 * c, *ring[c]]
            for v in (1, -1):
                pts = [proj[i][state[i] == v].mean(0) for i in range(len(cue)) if cue[i] == c and (state[i] == v).any()]
                row += list(np.mean(pts, 0)) if pts else [np.nan, np.nan]
                row.append(len(pts))
            fh.write(','.join(f'{x:.5g}' if isinstance(x, float) else str(x) for x in row) + '\n')


def main(sessions=range(10)):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for s in sessions:
        res, ks, state = run_session(s)
        for space in ('full', 'pcs'):
            for p in PS.PERIODS:
                sim, sh, nt, p95 = res[space][p]
                rows.append(dict(session=s, period=p, space=space, k_pcs=f'{ks[0]};{ks[1]}' if space == 'pcs' else '',
                                 similarity=sim, shuffle_mean=sh, shuffle_p95=p95, n_trials=nt))
        if s == 0:
            panel_c(state)
        print(f'session {s}: k = {ks}; ' + '; '.join(f'{sp} ' + ' '.join(f'{p} {res[sp][p][0]:.2f}/{res[sp][p][1]:.2f}'
                                                                      for p in PS.PERIODS) for sp in res), flush=True)
    with open(os.path.join(OUT, 'procrustes_sessions.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    stats = {}
    for space in ('full', 'pcs'):
        g = {p: [r for r in rows if r['space'] == space and r['period'] == p] for p in PS.PERIODS}
        v = {p: np.array([r['similarity'] for r in g[p]]) for p in g}
        sh = {p: np.array([r['shuffle_mean'] for r in g[p]]) for p in g}
        p95 = {p: np.array([r['shuffle_p95'] for r in g[p]]) for p in g}
        stats[space] = dict(n_sessions=len(v['on']),
                            **{f'{p}_mean': float(v[p].mean()) for p in v},
                            **{f'{p}_sem': float(v[p].std(ddof=1) / np.sqrt(len(v[p]))) for p in v},
                            **{f'{p}_shuffle_mean': float(sh[p].mean()) for p in v},
                            p_on_vs_off=float(ttest_rel(v['on'], v['off']).pvalue),
                            p_off_vs_pre=float(ttest_rel(v['off'], v['pre']).pvalue),
                            p_off_vs_shuffle=float(ttest_rel(v['off'], sh['off']).pvalue),
                            p_pre_vs_shuffle=float(ttest_rel(v['pre'], sh['pre']).pvalue),
                            n_off_above_shuffle_p95=int((v['off'] > p95['off']).sum()),
                            n_pre_above_shuffle_p95=int((v['pre'] > p95['pre']).sum()),
                            k_pcs=sorted({r['k_pcs'] for r in g['on']}) if space == 'pcs' else None)
    json.dump(stats, open(os.path.join(OUT, 'procrustes_stats.json'), 'w'), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
