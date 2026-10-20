"""Summarise scripts/sp_decode.py output: mean and SEM over sessions per (set, lambda), next to the data.

usage: python scripts/sp_summary.py data/sp_start_decode
"""
import csv
import glob
import json
import os
import sys

import numpy as np

KEYS = ('accuracy', 'off_per_trial', 'off_mean_ms', 'off_median_ms', 'frac_off', 'on_per_trial', 'on_mean_ms', 'frac_on')
DATA = os.path.join(os.path.dirname(__file__), '..', 'meanfield', 'data', 'readout_summary_data.csv')


def summary(d):
    ses = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, '*.json')))]
    out = {}
    for r0 in ses[0]['results']:
        key = (r0['set'], r0['k'], r0['lam'])
        per = []
        for s in ses:
            r = [x for x in s['results'] if (x['set'], x['k'], x['lam']) == key][0]
            per.append({k: np.nanmean([dr[k] for dr in r['draws']]) for k in KEYS})
        out[key] = {k: (float(np.mean([p[k] for p in per])), float(np.std([p[k] for p in per]) / np.sqrt(len(per)))) for k in KEYS}
        out[key]['n_sessions'] = len(per)
    return out


def main(d):
    S = summary(d)
    data = {(r['set'], r['k']): r for r in csv.DictReader(open(DATA))}
    dkey = {'all': ('all', 'session count'), 'top': ('top', '20'), 'random': ('random', '20')}
    hdr = ['set', 'λ', 'source'] + list(KEYS)
    print(' | '.join(hdr))
    for (st, k, lam), v in S.items():
        print(' | '.join([st if st != 'all' else 'all (329)', f'{lam:g}', 'spiking'] + [f'{v[x][0]:.2f} ± {v[x][1]:.2f}' if x in ('accuracy', 'frac_off', 'frac_on', 'off_per_trial', 'on_per_trial') else f'{v[x][0]:.0f} ± {v[x][1]:.0f}' for x in KEYS]))
        if lam == 30.0:
            r = data[dkey[st]]
            print(' | '.join([st if st != 'all' else 'all (329)', '30', 'data'] + [f'{float(r[x]):.2f}' if x in ('accuracy', 'frac_off', 'frac_on', 'off_per_trial', 'on_per_trial') else f'{float(r[x]):.0f}' for x in KEYS]))
    json.dump({f'{a}|{b}|{c}': v for (a, b, c), v in S.items()}, open(os.path.join(d, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
