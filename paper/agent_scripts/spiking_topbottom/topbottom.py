"""Top vs bottom vs random decoders in the spiking network (diagnostic, not in the paper).
Simulates sessions as run.sessions and decodes each with: all units; top k ranked per cue pair (as in run.sessions);
top k balanced (per pair, alternately the best cell preferring each cue); bottom k (least selective active cells,
per pair); random k (10 draws). Saves stats and example z traces to spiking/results/topbottom.npz/.json."""
import json, os, sys, time
import numpy as np
from spiking import readout as R, run as RUN
from paper_model import readout as PR

K = (5, 20, 50)


def read(counts, t10, cue, ok, seed):
    rng = np.random.default_rng(seed)
    Xall, tz = R.windows(counts[ok], t10)
    cue = np.asarray(cue)[ok]
    n_tr, n_t, d = Xall.shape
    valid, delay = (tz >= 0) & (tz <= 1400), (tz >= 500) & (tz <= 1400)
    half = np.zeros(n_tr, int)
    for c in range(8):
        i = rng.permutation(np.where(cue == c)[0]); half[i[len(i) // 2:]] = 1
    tot = Xall[:, valid].sum(1)
    active = np.where(Xall.sum((0, 1)) > 0)[0]
    rank, bal = [], []
    for h in (0, 1):
        rh, bh = [], []
        for c in range(4):
            m = (half == h) & (cue % 4 == c)
            f = np.nan_to_num(R.anova_f(tot[m], cue[m]), nan=0.0)
            o = np.array([u for u in np.argsort(-f) if u in set(active)])
            rh.append(o)
            pref = tot[m & (cue == c)].mean(0) > tot[m & (cue == c + 4)].mean(0)
            a_, b_ = o[pref[o]], o[~pref[o]]
            n = min(len(a_), len(b_)); bh.append(np.ravel(np.c_[a_[:n], b_[:n]]))
        rank.append(rh); bal.append(bh)
    sets = {'all': [[active] * 2]}
    for k in K:
        sets[f'top {k}'] = [[[r[:k] for r in rank[1]], [r[:k] for r in rank[0]]]]
        sets[f'bot {k}'] = [[[r[-k:] for r in rank[1]], [r[-k:] for r in rank[0]]]]
        if k == 20:
            sets[f'rand {k}'] = [[s, s] for s in (rng.choice(active, k, replace=False) for _ in range(10))]
    out, ex = {}, {}
    for lab, draws in sets.items():
        res = []
        for units in draws:
            z = np.full((n_tr, n_t), np.nan); zn_all = []
            if units[0] is units[1]:
                z, zn = R.decode(Xall, cue, rng, units[0]); zn_all.append(zn)
            else:
                for h in (0, 1):
                    zh, zn = R.decode(Xall, cue, rng, units[h]); z[half == h] = zh[half == h]; zn_all.append(zn[:, half == h])
            thr = R.null_threshold(np.concatenate(zn_all, 1), valid)
            zz = np.nan_to_num(z); state = PR.label(zz, valid, thr)
            st = PR.stats(state, delay); st['mu_on'] = PR.mu_on(zz[:, delay], rng)
            res.append(st)
            if lab not in ex: ex[lab] = (zz[:12].astype(np.float32), state[:12].astype(np.int8))
        out[lab] = {k: (np.sum([r[k] for r in res], 0).tolist() if k.startswith('h_') else float(np.mean([r[k] for r in res]))) for k in res[0]}
    return out, ex, tz, cue[:12]


def main(n_sessions=4):
    RUN.OUT = os.path.dirname(os.path.abspath(__file__))
    net = RUN.make_net()
    allres, exs = [], {}
    for s in range(n_sessions):
        t0 = time.time()
        rng = np.random.default_rng(100 + s)
        rec = RUN.record(net, rng)
        cue = rng.permutation(np.repeat(np.arange(8), RUN.N_TRIALS // 8))
        sim = RUN.run_trials(net, cue, s, rec)
        res, ex, tz, c12 = read(sim['counts'], sim['t10'], cue, sim['mem']['held'], seed=s)
        allres.append(res)
        if s == 0:
            exs = {f'z_{k}': v[0] for k, v in ex.items()} | {f'state_{k}': v[1] for k, v in ex.items()}
            exs['tz'] = tz; exs['cue'] = c12
        print(f'session {s} ({time.time() - t0:.0f} s): ' + '  '.join(f"{k} off {v['frac_off']:.2f} on {v['frac_on']:.2f} mu {v['mu_on']:.1f}" for k, v in res.items()), flush=True)
        json.dump(allres, open(os.path.join(RUN.OUT, 'topbottom.json'), 'w'), indent=1)
        np.savez_compressed(os.path.join(RUN.OUT, 'topbottom.npz'), **exs)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
