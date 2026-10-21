"""Model predictions for DATA_REQUEST_EXCITATORY.md: excitatory rates around Off states (top-20 read-out)
and recovery after Off by Off length. Preferred-cue tuned cells = cued pool, opposite-cue tuned cells = uncued
pool, untuned cells = background population. Rates: 20 ms boxcar, sampled at the read-out window centres.

usage: python scripts/mf_predict_exc.py --res data/mf_refit_best_fresh.json --label default --out data/pred_exc_default.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanfield import model as M, refit as RF  # noqa: E402
from meanfield.rate_states import window_rate_diff  # noqa: E402
from meanfield.surrogate import label  # noqa: E402

BINS = ((50, 100), (100, 200), (200, 400), (400, 2000))
LAGS = np.arange(-20, 41)          # -200 .. +400 ms in 10 ms steps


def runs(mask):
    a = np.r_[False, mask, False].astype(int)
    return list(zip(np.where(np.diff(a) == 1)[0], np.where(np.diff(a) == -1)[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', required=True)
    ap.add_argument('--base', default='default', choices=('default', 'standard'))
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-trials', type=int, default=2112)
    a = ap.parse_args()
    r = json.load(open(a.res))
    th, gain = r['theta'], r['sets']['top/20']['gain']
    base = M.STANDARD if a.base == 'standard' else {}
    p = M.params(**dict(base, **th)); p['t_delay'] = 1600.0
    cued = np.tile([0, 1], a.n_trials // 2)
    sim = M.simulate(p, cued, seed=21)
    rng = np.random.default_rng(22)
    dw, tc = window_rate_diff(sim)
    z = gain * dw + RF.noise(dw.shape, RF.RHO['top'], rng)
    thr = RF.null_thr(dw.shape, RF.RHO['top'], rng)
    tz = tc - sim['t_cue']
    valid = (tz >= 0) & (tz <= 1400)
    state = label(z, valid, thr)
    ci = label(-z, valid, thr) == 1
    idx = tc.astype(int)
    sm = lambda x: uniform_filter1d(x, 20, axis=1)[:, idx]
    pref = sm(np.where(cued[:, None] == 0, sim['rA'], sim['rB']))
    opp = sm(np.where(cued[:, None] == 0, sim['rB'], sim['rA']))
    bg = sm(sim['bg'])
    pre = (tz >= -250) & (tz < -50)
    groups = dict(pref=pref, opp=opp, untuned=bg)
    pre_rate = {g: float(v[:, pre].mean()) for g, v in groups.items()}
    # mid-state means
    mid = {g: {'on': [], 'off': []} for g in groups}
    on_al = {g: [] for g in groups}; off_al = {g: [] for g in groups}
    rec = {b: [] for b in BINS}; follow = {b: {'on': 0, 'ci': 0, 'n': 0} for b in BINS}
    for k in range(len(cued)):
        for lab, key in ((1, 'on'), (-1, 'off')):
            for i0, i1 in runs(state[k] == lab):
                if (i1 - i0) * 10 > 100 and 300 <= tz[i0] <= 1300:
                    for g, v in groups.items():
                        mid[g][key].append(v[k, i0 + 5:i1 - 5].mean())
        for i0, i1 in runs(state[k] == -1):
            L = (i1 - i0) * 10
            if 300 <= tz[i0] <= 1000 and L >= 100 and i0 + LAGS[0] >= 0 and i0 + LAGS[-1] < len(tz):
                for g, v in groups.items():
                    on_al[g].append(v[k, i0 + LAGS])
            if 300 <= tz[i0] <= 1100 and L >= 50:
                for b in BINS:
                    if b[0] <= L < b[1]:
                        nxt_on = (state[k, i1:i1 + 20] == 1).any()
                        nxt_ci = ci[k, i1:i1 + 20].any()
                        follow[b]['n'] += 1; follow[b]['on'] += int(nxt_on); follow[b]['ci'] += int(nxt_ci)
                        if nxt_on and i1 + 30 < len(tz):
                            rec[b].append(pref[k, i1:i1 + 31])
            if i1 + LAGS[-1] < len(tz) and i1 + LAGS[0] >= 0 and L >= 100:
                for g, v in groups.items():
                    off_al[g].append(v[k, i1 + LAGS])
    on_level = float(np.mean(mid['pref']['on']))
    off_level = float(np.mean(mid['pref']['off']))
    rec_out = {}
    for b in BINS:
        if rec[b]:
            c = np.mean(rec[b], 0)
            lo = c[0]
            t50 = int(np.argmax(c >= lo + 0.5 * (on_level - lo))) * 10 if (c >= lo + 0.5 * (on_level - lo)).any() else None
            t90 = int(np.argmax(c >= lo + 0.9 * (on_level - lo))) * 10 if (c >= lo + 0.9 * (on_level - lo)).any() else None
            rec_out[f'{b[0]}-{b[1]}'] = dict(n=len(rec[b]), t50_ms=t50, t90_ms=t90, curve=(c / pre_rate['pref']).tolist(),
                                            frac_followed_on=follow[b]['on'] / max(1, follow[b]['n']),
                                            frac_followed_ci=follow[b]['ci'] / max(1, follow[b]['n']), n_off=follow[b]['n'])
    out = dict(label=a.label, theta=th, base=a.base, pre_rate=pre_rate,
               mid={g: {k: float(np.mean(v)) for k, v in d.items()} for g, d in mid.items()},
               mid_ratio_to_pre={g: {k: float(np.mean(v)) / pre_rate[g] for k, v in d.items()} for g, d in mid.items()},
               off_over_on={g: float(np.mean(d['off']) / np.mean(d['on'])) for g, d in mid.items()},
               onset_aligned={g: (np.mean(v, 0) / pre_rate[g]).tolist() for g, v in on_al.items()},
               offset_aligned={g: (np.mean(v, 0) / pre_rate[g]).tolist() for g, v in off_al.items()},
               lags_ms=(LAGS * 10).tolist(), recovery=rec_out, pref_on_level=on_level, pref_off_level=off_level)
    json.dump(out, open(a.out, 'w'))
    print(a.label, '| Off/On ratio:', {g: round(v, 2) for g, v in out['off_over_on'].items()})
    print('  mid ratio to pre-cue:', {g: {k: round(v, 2) for k, v in d.items()} for g, d in out['mid_ratio_to_pre'].items()})
    for b, v in rec_out.items():
        print(f'  Off {b} ms: n {v["n_off"]}, followed by On {v["frac_followed_on"]:.2f}, by wrong cue {v["frac_followed_ci"]:.3f} | '
              f'recovery t50 {v["t50_ms"]} ms, t90 {v["t90_ms"]} ms')


if __name__ == '__main__':
    main()
