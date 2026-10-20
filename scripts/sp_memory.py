"""Quick network check of spiking sessions (no decoding): memory at the end of the delay and pool pauses.

Memory correct: cued pool rate > uncued pool rate over the last 300 ms of the delay (as scripts/nmda_sweep.py);
'active' additionally requires the cued pool > 10 Hz there. Pool pauses: both pools < 2 Hz (20 ms smoothing)
for >= 50 ms after the cue, and whether the cued pool leads in the 100 ms after each pause.

usage: python scripts/sp_memory.py data/sp_start_s
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_wm.network import A, B  # noqa: E402


def main(prefix):
    rows = []
    for sd in sorted(glob.glob(prefix + '*')):
        meta = json.load(open(os.path.join(sd, 'meta.json')))
        T = meta['task']
        t_cue = int(T['t_pre'])
        t_end = int(T['t_pre'] + T['cue_dur'] + T['t_delay'])
        for f in sorted(glob.glob(os.path.join(sd, 'trial*.npz'))):
            d = np.load(f)
            r = d['rates'].astype(float)
            C, U = (A, B) if str(d['cue']) == 'A' else (B, A)
            last = slice(t_end - 300, t_end)
            cp, up = uniform_filter1d(r[:, C], 20), uniform_filter1d(r[:, U], 20)
            m = np.r_[0, ((cp[t_cue:t_end] < 2) & (up[t_cue:t_end] < 2)).astype(int), 0]
            pauses = []
            for a, b in zip(np.where(np.diff(m) == 1)[0], np.where(np.diff(m) == -1)[0]):
                if b - a >= 50:
                    a2, b2 = a + t_cue, b + t_cue
                    lead = cp[b2:b2 + 100].sum() > up[b2:b2 + 100].sum() if b2 + 100 <= t_end else None
                    pauses.append((b - a, lead, a2 - t_cue))
            rows.append(dict(ok=bool(r[last, C].mean() > r[last, U].mean()), active=bool(r[last, C].mean() > 10),
                             pauses=pauses, ns=float(r[t_cue + 500:t_end, 0].mean())))
    n = len(rows)
    pa = [p for r in rows for p in r['pauses']]
    pa14 = [p for r in rows for p in r['pauses'] if p[2] < 1400]
    led = [p[1] for p in pa if p[1] is not None]
    print(f'{prefix}: {n} trials | memory correct {np.mean([r["ok"] for r in rows]):.1%}, cued pool active at end '
          f'{np.mean([r["active"] for r in rows]):.1%} | pool pauses starting 0-1,400 ms: {len(pa14) / n:.2f}/trial, '
          f'mean {np.mean([p[0] for p in pa14]) if pa14 else 0:.0f} ms, median {np.median([p[0] for p in pa14]) if pa14 else 0:.0f} ms '
          f'| cued restarts first {np.mean(led) if led else float("nan"):.0%} of {len(led)} | background {np.mean([r["ns"] for r in rows]):.1f} Hz')
    for lo, hi in ((50, 150), (150, 300), (300, 600), (600, 5000)):
        s = [p[1] for p in pa if p[1] is not None and lo <= p[0] < hi]
        if s:
            print(f'   pauses {lo}-{hi} ms: cued restarts first {np.mean(s):.0%} (n={len(s)})')


if __name__ == '__main__':
    for pfx in sys.argv[1:]:
        main(pfx)
