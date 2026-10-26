"""Network PCA panels with On/Off labelled from ALL cells of the labelling half (instead of its top 20).
Re-simulates the sessions of run.sessions (same seeds), labels states with the paper's decoder on all cells of
half H1, and reuses the stored PCA projections of half H2 (spiking/results/pca_onoff/session_s.npz, same split).
Writes per-session states and the panel c / d tables in the format of the agent's hand-off."""
import csv, json, os, sys
import numpy as np
from spiking import run as RU, readout as R, pca_onoff as P
from paper_model import readout as PR

D = os.path.dirname(os.path.abspath(__file__))


def states_half_all(s):
    f = os.path.join(D, f'states_s{s}.npz')
    if os.path.exists(f):
        return np.load(f)['state']
    net = RU.make_net()
    rng = np.random.default_rng(100 + s)
    rec = RU.record(net, rng)
    cue = rng.permutation(np.repeat(np.arange(8), RU.N_TRIALS // 8))
    sim = RU.run_trials(net, cue, s, rec)
    counts, held = sim['counts'], sim['mem']['held']
    stored = np.load(os.path.join(P.OUT, f'session_{s}.npz'))
    assert np.array_equal(cue[held], stored['cue']), 'not the same trials as the stored session'
    t10 = np.arange(counts.shape[1]) * 10.0 - RU.TUNED['t_pre']
    X, tz = R.windows(counts[held], t10)
    cueh = cue[held]
    n_cells = X.shape[2]
    pu = np.random.default_rng(1).permutation(n_cells)          # same cell split as run_session
    H1 = np.sort(pu[:n_cells // 2])
    rng0 = np.random.default_rng(0)
    half = np.zeros(len(cueh), int)
    for c in range(8):
        idx = rng0.permutation(np.flatnonzero(cueh == c)); half[idx[len(idx) // 2:]] = 1
    st = P.label_states(X, cueh, H1, half, np.random.default_rng(s + 1), tz)
    dl = (tz >= P.DEL[0]) & (tz <= P.DEL[1])
    st = st[:, dl]
    np.savez_compressed(f, state=st)
    return st


def main(sessions, n_shuf=50, ex=0):
    from scipy.stats import ttest_rel, ttest_1samp
    rng = np.random.default_rng(0)
    rows, pc = [], []
    for s in sessions:
        st = states_half_all(s)
        d = np.load(os.path.join(P.OUT, f'session_{s}.npz'))
        proj, half, cms, cue, pre = d['split_proj'], d['split_half'], d['split_cmeans'], d['cue'], d['split_preproj']
        per = {}
        for p in ('on', 'off', 'pre'):
            tri, Q = [], []
            for i in range(len(cue)):
                if p == 'pre':
                    tri.append(i); Q.append(pre[i].mean(0)); continue
                m = st[i] == (1 if p == 'on' else -1)
                if m.any():
                    tri.append(i); Q.append(proj[i][m].mean(0))
            per[p] = (np.array(tri), np.array(Q))
        for p in ('on', 'off', 'pre'):
            tri, Q = per[p]; h = half[tri]
            def mcos(c):
                cen = cms[h, c]
                return np.mean(np.sum(Q * cen, 1) / (np.linalg.norm(Q, axis=1) * np.linalg.norm(cen, axis=1) + 1e-12))
            raw = mcos(cue[tri]); sh = []
            for _ in range(n_shuf):
                c2 = cue[tri].copy()
                for hh in (0, 1):
                    mm = h == hh; c2[mm] = rng.permutation(c2[mm])
                sh.append(mcos(c2))
            rows.append(dict(session=s, period=p, raw=float(raw), shuffle=float(np.mean(sh)), corrected=float(raw - np.mean(sh)), n_trials=len(tri)))
        if s == ex:
            ring = cms.mean(0)
            for c in range(8):
                r = dict(cue=c, ring_pc1=ring[c, 0], ring_pc2=ring[c, 1])
                for p in ('on', 'off'):
                    tri, Q = per[p]; m = cue[tri] == c
                    r[f'{p}_pc1'], r[f'{p}_pc2'], r[f'n_{p}'] = Q[m, 0].mean(), Q[m, 1].mean(), int(m.sum())
                pc.append(r)
        print('session', s, {p: round(r['corrected'], 3) for r in rows[-3:] for p in [r['period']]}, flush=True)
    with open(os.path.join(D, 'panel_d_halfall.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(D, 'panel_c_halfall.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(pc[0])); w.writeheader(); w.writerows(pc)
    Dd = {p: np.array([r['corrected'] for r in rows if r['period'] == p]) for p in ('on', 'off', 'pre')}
    stats = dict(n_sessions=len(sessions), example_session=ex, **{f'{p}_mean': float(Dd[p].mean()) for p in Dd},
                 **{f'{p}_sem': float(Dd[p].std(ddof=1) / np.sqrt(len(Dd[p]))) for p in Dd},
                 p_on_vs_off=float(ttest_rel(Dd['on'], Dd['off']).pvalue), p_off_vs_0=float(ttest_1samp(Dd['off'], 0).pvalue),
                 p_pre_vs_0=float(ttest_1samp(Dd['pre'], 0).pvalue), p_off_vs_pre=float(ttest_rel(Dd['off'], Dd['pre']).pvalue),
                 n_off_above_0=int((Dd['off'] > 0).sum()))
    json.dump(stats, open(os.path.join(D, 'panel_d_halfall_stats.json'), 'w'), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main(list(range(10)))
