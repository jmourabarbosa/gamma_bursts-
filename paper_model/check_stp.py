"""First check of the binary STP model (model_stp.py), before any fitting: one example trial for each setting, and
the all-units read-out statistics next to the data's. Writes figures/mongillo/check_stp.png.

usage: python -m paper_model.check_stp
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import model_stp as S, readout as R

CASES = (('P = 2 (literal), I_b 3', dict(I_b=3.0, n_other=1.0)),
         ('P = 16 inhibition, I_b 8, no noise', dict(I_b=8.0)),
         ('P = 16 inhibition, I_b 8, noise', dict(I_b=8.0, sigma=1.0)),
         ('P = 16 inhibition, I_b 8, noise + dips', dict(I_b=8.0, sigma=1.0, dip_depth=0.6)))


def main():
    R.M = S      # read out the STP model exactly like the paper model
    ses, _ = R.load_data()
    d = ses['all']
    print(f"data: Off {d['off_per_trial'].mean():.2f}/trial {d['off_mean_ms'].mean():.0f} ms frac {d['frac_off'].mean():.2f}"
          f" | On {d['on_per_trial'].mean():.2f}/trial {d['on_mean_ms'].mean():.0f} ms frac {d['frac_on'].mean():.2f}")
    fig, ax = plt.subplots(len(CASES), 1, figsize=(7, 2.0 * len(CASES)), sharex=True)
    for a, (name, over) in zip(ax, CASES):
        p = S.params(cue_amp=20.0, **over)
        out = R.evaluate(p, ses, n_trials=264, seed=3, sets=('all',))
        r = out['sets']['all']
        print(f"{name}: memory {out['mem']:.2f} | Off {r['n_off']:.2f}/trial {r['off_mean']:.0f} ms frac {r['frac_off']:.2f}"
              f" | On {r['n_on']:.2f}/trial {r['on_mean']:.0f} ms frac {r['frac_on']:.2f}")
        s = S.simulate(p, 1, seed=5)
        a.plot(s['t'], s['rc'][0], color='C3', lw=0.8, label='cued')
        a.plot(s['t'], s['ru'][0], color='C0', lw=0.8, label='uncued')
        a2 = a.twinx()
        a2.plot(s['t'], s['uc'][0] * s['xc'][0], color='C3', ls='--', lw=0.8)
        a2.plot(s['t'], s['m'][0], color='0.6', lw=0.8)
        a2.set_ylim(0, 1.05)
        a2.set_ylabel('ux (dashed), m (grey)', fontsize=7)
        a.set_title(name, fontsize=9)
        a.set_ylabel('rate (Hz)')
        a.set_yscale('symlog', linthresh=1)
    ax[0].legend(fontsize=7, loc='upper left')
    ax[-1].set_xlabel('time from cue onset (ms)')
    fig.tight_layout()
    os.makedirs('figures/mongillo', exist_ok=True)
    fig.savefig('figures/mongillo/check_stp.png', dpi=200)


if __name__ == '__main__':
    main()
