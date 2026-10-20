"""Keep trials in which the memory was not lost (as the paper kept correct trials), up to n per cue.

Memory not lost: in the last 300 ms of the delay the cued pool's rate (50 ms smoothing) exceeds 20 Hz at
some point and its mean exceeds the uncued pool's. Trials of sessions sharing a network seed are pooled,
in order. Writes data/<out>_sNN/ with meta.json, net.npz and links to the kept trials (renumbered).

usage: python scripts/sp_select.py --data data/sp_start_s --out data/sp_start_sel --per-cue 66
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_wm.network import A, B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--per-cue', type=int, default=66)
    a = ap.parse_args()
    by_net = {}
    for sd in sorted(glob.glob(a.data + '[0-9]*')):
        meta = json.load(open(os.path.join(sd, 'meta.json')))
        by_net.setdefault(meta['params']['seed'], []).append(sd)
    report = {}
    for net, sds in sorted(by_net.items()):
        meta = json.load(open(os.path.join(sds[0], 'meta.json')))
        T = meta['task']
        t_end = int(T['t_pre'] + T['cue_dur'] + T['t_delay'])
        kept = {'A': [], 'B': []}
        n_all = n_ok = 0
        for sd in sds:
            for f in sorted(glob.glob(os.path.join(sd, 'trial*.npz'))):
                d = np.load(f)
                cue = str(d['cue'])
                r = d['rates'].astype(float)
                C, U = (A, B) if cue == 'A' else (B, A)
                sm = uniform_filter1d(r[:, C], 50)[t_end - 300:t_end]
                ok = sm.max() > 20 and r[t_end - 300:t_end, C].mean() > r[t_end - 300:t_end, U].mean()
                n_all += 1; n_ok += int(ok)
                if ok and len(kept[cue]) < a.per_cue:
                    kept[cue].append(f)
        od = f'{a.out}_s{net:02d}'
        os.makedirs(od, exist_ok=True)
        shutil.copy(os.path.join(sds[0], 'meta.json'), od); shutil.copy(os.path.join(sds[0], 'net.npz'), od)
        files = sorted(kept['A'] + kept['B'], key=lambda f: (os.path.dirname(f), f))
        for i, f in enumerate(files):
            dst = os.path.join(od, f'trial{i:03d}.npz')
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(f), dst)
        report[net] = dict(trials=n_all, memory_kept=n_ok, frac=n_ok / n_all, kept_A=len(kept['A']), kept_B=len(kept['B']))
        print(f'network {net}: {n_ok}/{n_all} trials keep the memory ({n_ok / n_all:.0%}); selected A {len(kept["A"])}, B {len(kept["B"])}')
    json.dump(report, open(a.out + '_report.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
