"""Memory vs fraction of the delay Off, with the Off fraction controlled through the common input.

The model fitted to the all-units read-out (model.FIT) is run with the dip statistics changed across trials. Each
trial is read out with the fitted model's all-units gain (not re-calibrated) and the all-units noise, and labelled as
in readout.py. Per trial: fraction of the delay (500-1,400 ms after cue onset, as frac_off) labelled Off, and memory
held at the end of the delay (NMDA criterion, model.memory_correct). Definitions as the data's behaviour analysis
(panichello repo, analysis/behaviour_handoff): per-trial fraction Off / On, number of Off periods (runs clipped to the
delay) and mean Off length over the delay's 91 time points; memory vs Off fraction in windows [lo, lo + 0.025),
lo = 0.05, 0.0525, ..., 0.975 (the last window includes 1.0), 95% Wilson intervals.

usage: python -m paper_model.off_vs_memory
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import model as M, readout as R

OUT = os.path.join('figures', 'off_vs_memory')
RHO = R.RHO['all']
N_PER_LEVEL = 300
SWEEPS = {'fitted model (no manipulation)': ('dip_rate', [M.FIT['dip_rate']] * 20),
          'dip rate varied (0-30 per s)': ('dip_rate', list(np.linspace(0.0, 30.0, 31))),
          'dip depth varied (0.3-1.0)': ('dip_depth', list(np.linspace(0.3, 1.0, 29)))}


def reference(ses):
    """All-units gain of the fitted model and the On cluster-mass threshold."""
    out = R.evaluate(M.params(), ses, n_trials=1056, seed=0, sets=('all',))
    return out['sets']['all']['gain']


def run(key, levels, gain, seed):
    rng = np.random.default_rng(seed)
    frac, mem, lev = [], [], []
    thr = None
    for i, v in enumerate(levels):
        sim = M.simulate(M.params(**{key: v}), N_PER_LEVEL, seed=seed * 1000 + i)
        dw, tz = R.rate_diff(sim)
        delay, valid = (tz >= 500) & (tz <= 1400), (tz >= 0) & (tz <= 1400)
        if thr is None:
            thr = R.null_threshold(dw.shape, RHO, rng)
        z = gain * dw + R.noise(dw.shape, RHO, rng)
        state = R.label(z, valid, thr)
        sd = state[:, delay]
        offs = [[(b - a) * R.STEP for a, b in R.runs(row == -1)] for row in sd]
        frac.append(np.c_[(sd == -1).mean(1), [len(o) for o in offs], [np.mean(o) if o else np.nan for o in offs],
                          (sd == 1).mean(1)])
        mem.append(M.memory_correct(sim))
        lev.append(np.full(N_PER_LEVEL, v))
    return np.concatenate(frac), np.concatenate(mem), np.concatenate(lev)


def sliding(frac, ok, width=0.025, step=0.0025, start=0.05, n_min=30):
    los = np.round(np.arange(start, 1 - width + 1e-9, step), 6)
    c = los + width / 2
    p, lo, hi, n = [], [], [], []
    for x0 in los:
        x1 = x0 + width
        sel = (frac >= x0 - 1e-9) & ((frac <= x1 + 1e-9) if x1 >= 1 - 1e-9 else (frac < x1 - 1e-9))
        k, m = ok[sel].sum(), sel.sum()
        n.append(m)
        if m < n_min:
            p.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
        q = k / m; zz = 1.96   # Wilson interval
        d = 1 + zz ** 2 / m; ce = (q + zz ** 2 / (2 * m)) / d
        h = zz * np.sqrt(q * (1 - q) / m + zz ** 2 / (4 * m * m)) / d
        p.append(q); lo.append(ce - h); hi.append(ce + h)
    return c, np.array(p), np.array(lo), np.array(hi), np.array(n)


def main():
    ses, _ = R.load_data()
    gain = reference(ses)
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(2, len(SWEEPS), figsize=(4.2 * len(SWEEPS), 6.4), gridspec_kw=dict(height_ratios=(3, 1.4)))
    summary = dict(gain_all=gain)
    for j, (name, (key, levels)) in enumerate(SWEEPS.items()):
        st, mem, lev = run(key, levels, gain, seed=j + 1)
        frac = st[:, 0]
        c, p, lo, hi, n = sliding(frac, mem)
        d_off = frac[~mem].mean() - frac[mem].mean() if (~mem).any() else np.nan
        summary[name] = dict(n_trials=int(len(mem)), memory=float(mem.mean()), errors=int((~mem).sum()))
        for i, sn in enumerate(('frac_off', 'n_off', 'off_len_ms', 'frac_on')):
            summary[name][sn] = dict(correct=float(np.nanmean(st[mem, i])), error=float(np.nanmean(st[~mem, i])))
        a = ax[0, j]
        a.fill_between(c, lo, hi, color='0.85', lw=0)
        a.plot(c, p, 'k', lw=1)
        a.set_ylim(0.3, 1.02); a.set_xlim(0.05, 1.0)
        a.set_title(f'{name}\n{len(mem)} trials, {(~mem).sum()} errors; Off fraction {d_off:+.3f} on errors', fontsize=8)
        a.set_xlabel('fraction of delay Off (window centre)')
        a.set_ylabel('memory held (fraction of trials)' if j == 0 else '')
        b = ax[1, j]
        if len(set(levels)) > 1:
            u = np.unique(lev)
            b.plot(u, [frac[lev == v].mean() for v in u], 'k.-', ms=3, lw=0.8, label='Off fraction')
            b.plot(u, [mem[lev == v].mean() for v in u], '.-', color='C3', ms=3, lw=0.8, label='memory held')
            b.set_xlabel(key.replace('_', ' ')); b.set_ylim(0, 1.02); b.legend(fontsize=7)
        else:
            b.hist(frac, bins=np.linspace(0, 1, 41), color='0.5'); b.set_xlabel('fraction of delay Off'); b.set_ylabel('trials')
    fig.suptitle('Model fitted to all units: memory (NMDA, end of delay) vs fraction of delay Off (all-units read-out, fixed gain)',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'memory_vs_off_fraction.png'), dpi=200)
    json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
