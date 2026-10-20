"""Panichello-style On/Off analysis of spiking sessions, matched to the data analysis
(meanfield/data/readout_note_data.md): 329 random units, top-20 (split-half, two-sample test on
500-1,400 ms, never using the test trial) and random-20 sets; L2 logistic decoder with approximate
leave-one-out; 30 shuffled-label nulls; per-trial cluster-mass threshold (95th percentile over shuffles of
each shuffled series' largest cluster); states labelled and counted 0-1,400 ms after cue onset.

usage: python scripts/sp_decode.py --data data/sp_start_s --out data/sp_start_decode --lams 30 1000
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_wm import panichello as pa  # noqa: E402
from meanfield.decoder import alo_confidence  # noqa: E402
from meanfield.fit import EDGES  # noqa: E402

STEP, WIN = 10, 100


def load(sd):
    meta = json.load(open(os.path.join(sd, 'meta.json')))
    group = np.load(os.path.join(sd, 'net.npz'))['group']
    trials = []
    for f in sorted(glob.glob(os.path.join(sd, 'trial*.npz'))):
        d = np.load(f)
        trials.append(dict(cue=str(d['cue']), spk_t=d['spk_t'], spk_i=d['spk_i'], rates=d['rates']))
    return meta, group, trials


def cluster_label(conf, nulls, t_rel):
    """z vs the shuffled null; On = runs z > 1.96 with mass above the trial's 95th percentile of shuffled
    largest-cluster masses; Off = >= 5 points z < 0.8416. Only t_rel in [0, 1400] is labelled."""
    lab = (t_rel >= 0) & (t_rel <= 1400)
    allc = np.concatenate([conf[None], nulls])                 # real + shuffles
    mu, sd = nulls.mean(0), nulls.std(0) + 1e-9
    z = (conf - mu) / sd
    n_tr = conf.shape[0]
    state = np.zeros(conf.shape, int)
    for k in range(n_tr):
        maxes = []
        for s in range(1, len(allc)):
            others = np.delete(allc[:, k], s, axis=0)
            zs = (allc[s, k] - others.mean(0)) / (others.std(0) + 1e-9)
            ms = [zs[a:b].sum() for a, b in pa._runs((zs > 1.96) & lab)]
            maxes.append(max(ms) if ms else 0.0)
        thr = np.percentile(maxes, 95)
        for a, b in pa._runs((z[k] > 1.96) & lab):
            if z[k, a:b].sum() > thr:
                state[k, a:b] = 1
        for a, b in pa._runs((z[k] < 0.8416) & lab):
            if b - a >= 5:
                state[k, a:b] = -1
    return state, z


def stats(state, conf, z, t_rel):
    cm = (t_rel >= 0) & (t_rel <= 1400)
    dm = (t_rel >= 500) & (t_rel <= 1400)
    on, off, n_on, n_off = [], [], [], []
    for k in range(state.shape[0]):
        s = np.where(cm, state[k], 0)
        r1, r0 = pa._runs(s == 1), pa._runs(s == -1)
        on += [(b - a) * STEP for a, b in r1]; off += [(b - a) * STEP for a, b in r0]
        n_on.append(len(r1)); n_off.append(len(r0))
    pre = (t_rel >= -400) & (t_rel < 0)
    return dict(accuracy=float((conf[:, dm] > 0.5).mean()), on_per_trial=float(np.mean(n_on)), off_per_trial=float(np.mean(n_off)),
                on_mean_ms=float(np.mean(on)) if on else np.nan, off_mean_ms=float(np.mean(off)) if off else np.nan,
                on_median_ms=float(np.median(on)) if on else np.nan, off_median_ms=float(np.median(off)) if off else np.nan,
                frac_on=float((state[:, dm] == 1).mean()), frac_off=float((state[:, dm] == -1).mean()),
                on_dur=[float(v) for v in on], off_dur=[float(v) for v in off], z_pre_mean=float(z[:, pre].mean()) if pre.any() else np.nan,
                z_pre_sd=float(z[:, pre].std()) if pre.any() else np.nan)


def decode(C, y, lam, n_sh, seed):
    conf = alo_confidence(C, y, lam)
    nulls = np.array([alo_confidence(C, y, lam, shuffle_rng=np.random.default_rng(seed * 100 + s)) for s in range(n_sh)])
    return conf, nulls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--lams', nargs='+', type=float, default=[30.0, 1000.0])
    ap.add_argument('--K', type=int, default=329)
    ap.add_argument('--n-shuffle', type=int, default=30)
    ap.add_argument('--n-random', type=int, default=5)
    ap.add_argument('--save-traces', action='store_true', help='also save per-trial confidence, z and states '
                    '(all units and top 20, first lambda) and pool rates, for plotting')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for si, sd in enumerate(sorted(glob.glob(a.data + '*'))):
        f = os.path.join(a.out, f'{os.path.basename(sd)}.json')
        if os.path.exists(f):
            continue
        meta, group, trials = load(sd)
        T = meta['task']
        t_cue = T['t_pre']
        n_t = int(T['t_pre'] + T['cue_dur'] + T['t_delay'])
        y = np.array([t['cue'] == 'A' for t in trials], int)
        rng = np.random.default_rng(500 + si)
        sample = np.sort(rng.choice(len(group), a.K, replace=False))
        C_s, tc = pa.counts_matrix(trials, sample, n_t, WIN, STEP)          # only the recorded sample
        pos = {u: i for i, u in enumerate(sample)}
        t_rel = tc - t_cue
        keep = (t_rel >= -400) & (t_rel <= 1400)
        C_s, t_rel = C_s[:, :, keep], t_rel[keep]
        cols = lambda units: np.ascontiguousarray(C_s[:, [pos[u] for u in units]])
        # top-20 within the 329 sample, split-half by trials (stratified)
        halves = np.zeros(len(y), int)
        for c in (0, 1):
            idx = rng.permutation(np.where(y == c)[0]); halves[idx[: len(idx) // 2]] = 1
        dm = (t_rel >= 500) & (t_rel <= 1400)
        def rank(trials_mask):
            X = C_s[trials_mask][:, :, dm].mean(2)
            yy = y[trials_mask]
            m1, m0 = X[yy == 1].mean(0), X[yy == 0].mean(0)
            v = X[yy == 1].var(0) / max(1, (yy == 1).sum()) + X[yy == 0].var(0) / max(1, (yy == 0).sum()) + 1e-9
            return sample[np.argsort(-np.abs(m1 - m0) / np.sqrt(v))[:20]]
        top = {0: rank(halves == 1), 1: rank(halves == 0)}       # test half h uses units ranked on the other half
        out = dict(session=sd, n_trials=len(y), K=a.K, results=[])
        for lam in a.lams:
            sets = [('all', a.K, [sample]), ('random', 20, [np.sort(rng.choice(sample, 20, replace=False)) for _ in range(a.n_random)])]
            for name, k, unit_sets in sets:
                rows = []
                for j, units in enumerate(unit_sets):
                    conf, nulls = decode(cols(units), y, lam, a.n_shuffle, si * 10 + j)
                    st, z = cluster_label(conf, nulls, t_rel)
                    rows.append(stats(st, conf, z, t_rel))
                    if name == 'all' and lam == a.lams[0]:
                        traces = dict(all_conf=conf, all_state=st, all_z=z)
                out['results'].append(dict(set=name, k=k, lam=lam, draws=rows))
            # top 20: decode with each half's units, keep the test trials of that half
            conf = np.zeros((len(y), len(t_rel))); nulls = np.zeros((a.n_shuffle, len(y), len(t_rel)))
            for h in (0, 1):
                c, n = decode(cols(top[h]), y, lam, a.n_shuffle, si * 10 + 7 + h)
                conf[halves == h], nulls[:, halves == h] = c[halves == h], n[:, halves == h]
            st, z = cluster_label(conf, nulls, t_rel)
            out['results'].append(dict(set='top', k=20, lam=lam, draws=[stats(st, conf, z, t_rel)]))
            if lam == a.lams[0]:
                traces.update(top_conf=conf, top_state=st, top_z=z)
            print(sd, lam, {r['set']: {k: round(v, 3) for k, v in r['draws'][0].items() if not isinstance(v, list)}
                            for r in out['results'] if r['lam'] == lam}, flush=True)
        json.dump(out, open(f, 'w'))
        if a.save_traces:
            from gamma_wm.network import A, B
            rates = np.array([t['rates'][:n_t][:, [A, B]] for t in trials], np.float32)
            np.savez_compressed(f.replace('.json', '_traces.npz'), t_rel=t_rel, y=y, rates=rates, t_cue=t_cue,
                                **{k: v.astype(np.float32) if v.dtype.kind == 'f' else v.astype(np.int8) for k, v in traces.items()})


if __name__ == '__main__':
    main()
