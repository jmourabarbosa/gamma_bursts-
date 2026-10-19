"""Compare expected-z states (meanfield.expected_z, no spikes) with the full pipeline on the validation
sessions of scripts/mf_calibrate.py (same parameters, same recorded population, same seeds).

usage: python scripts/mf_check_expected_z.py --calib data/mf_calib
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanfield import expected_z as EZ, model as M, recording as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib', default='data/mf_calib')
    a = ap.parse_args()
    for f in sorted(glob.glob(os.path.join(a.calib, 'val*.json'))):
        row = json.load(open(f))
        if 'expected' in row:
            continue
        i = int(os.path.basename(f)[3:6])
        th = row['theta']
        rec = dict(R.REC_DEFAULTS, f_sel=th['f_sel'], tune_sd=th['tune_sd'])
        p = M.params(**{k: v for k, v in th.items() if k not in rec})
        ses = R.simulate_session(p, rec, seed=10 * i + 1, keep_binned=True)
        res = EZ.expected_pipeline(ses, seed=i)
        st = res['stats']
        row['expected'] = dict(acc=res['acc'], thr=float(res['thr']), n_on=float(st['n_on'].mean()), n_off=float(st['n_off'].mean()),
                               on_dur=st['on_dur'].tolist(), off_dur=st['off_dur'].tolist())
        json.dump(row, open(f, 'w'))
        fu, ex = row['full'], row['expected']
        print(i, ' | '.join(f'{k}: acc {d["acc"]:.2f} On {d["n_on"]:.2f}/{np.mean(d["on_dur"] or [0]):.0f} '
                            f'Off {d["n_off"]:.2f}/{np.mean(d["off_dur"] or [0]):.0f} thr {d["thr"]:.0f}'
                            for k, d in (('full', fu), ('expected', ex))), flush=True)


if __name__ == '__main__':
    main()
