"""Run a fitted parameter set through the full pipeline (L2 logistic decoder, approximate LOO,
20 shuffled-label nulls) on paper-sized sessions, and save state lengths and summary statistics.

usage: python scripts/mf_validate.py --best data/mf_fit/best.json --sessions 25 --out data/mf_validate
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanfield import fit as F, model as M, recording as R, surrogate as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--best', required=True)
    ap.add_argument('--sessions', type=int, default=25)
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--keep-example', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    th = json.load(open(a.best))['theta']
    p, rec = F.split(th)
    for i in range(a.seed0, a.seed0 + a.sessions):
        f = os.path.join(a.out, f'ses{i:03d}.json')
        if os.path.exists(f):
            continue
        ses = R.simulate_session(p, rec, seed=7000 + 10 * i)
        full = R.full_pipeline(ses, n_shuffle=20, lam=S.LAM, seed=i)
        fast = S.fast_pipeline(ses, n_shuffle=20, seed=i)
        row = dict(theta=th, mem=float(M.memory_correct(ses['sim']).mean()))
        for name, r in (('full', full), ('fast', fast)):
            st = r['stats']
            row[name] = dict(acc=r['acc'], n_on=st['n_on'].tolist(), n_off=st['n_off'].tolist(),
                             on_dur=st['on_dur'].tolist(), off_dur=st['off_dur'].tolist())
        json.dump(row, open(f, 'w'))
        if a.keep_example and i == a.seed0:
            sim = ses['sim']
            np.savez_compressed(os.path.join(a.out, 'example.npz'), conf=full['conf'], z=full['z'], state=full['state'],
                                tc=ses['tc'], y=ses['y'], count_mask=ses['count_mask'], rA=sim['rA'][:12], rB=sim['rB'][:12],
                                bg=sim['bg'][:12], m=sim['m'][:12], sA=sim['sA'][:12], sB=sim['sB'][:12], t_cue=sim['t_cue'])
        print(i, f'acc {full["acc"]:.2f} mem {row["mem"]:.2f} On {np.mean(full["stats"]["n_on"]):.2f} '
              f'Off {np.mean(full["stats"]["n_off"]):.2f}', flush=True)


if __name__ == '__main__':
    main()
