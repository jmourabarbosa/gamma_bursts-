"""Dips of the common input vs decoded Off states in the best-fit NMDA model (model.FIT, Figure 2).

N_TRIALS trials with the fitted dips, read out with the all-units read-out (fitted gain, all-units noise) and labelled
as in readout.py; states and dips 0-1,400 ms after cue onset. A dip = common input below the middle of the dip
(1 - depth / 2), at 1 ms. A dip is detected if an Off state overlaps it; an Off state is explained if a dip overlaps it.
a, b: dip lengths and gaps between dips, from the fitted distributions (gamma lengths, Poisson onsets: exponential
gaps), not simulated; c: dip vs Off-state lengths; d: P(dip detected as Off) vs
dip length; e: Off length vs summed length of the dips it overlaps; f: per trial, total dip time vs total Off time.
Output: figures/paper_model/figS_dips_vs_states.{png,svg} and figS_dips_vs_states.json.

usage: python -m paper_model.dips_vs_states
"""
import json
import os

import numpy as np
from scipy import stats

from . import model as M, readout as R
from .off_vs_memory import RHO, reference
from .make_figures import plt, save, letter, INK, GREY, C_OFF, OUT

N_TRIALS, T0, T1 = 2000, 0, 1400


def main():
    R.M = M
    ses, _ = R.load_data()
    gain = reference(ses)
    p = M.params()
    sim = M.simulate(p, N_TRIALS, seed=123)
    rng = np.random.default_rng(5)
    dw, tz = R.rate_diff(sim)
    valid = (tz >= T0) & (tz <= T1)
    state = R.label(gain * dw + R.noise(dw.shape, RHO, rng), valid, R.null_threshold(dw.shape, RHO, rng))
    t = sim['t']
    in_dip = sim['m'] < 1 - p['dip_depth'] / 2
    dips, offs = [], []          # (trial, start ms, end ms)
    for k in range(N_TRIALS):
        dips += [(k, t[a], t[b - 1] + 1) for a, b in R.runs(in_dip[k]) if t[b - 1] >= T0 and t[a] <= T1]
        offs += [(k, tz[a] - R.STEP / 2, tz[b - 1] + R.STEP / 2) for a, b in R.runs(state[k] == -1)]
    dips, offs = np.array(dips), np.array(offs)
    by_trial = lambda x, k: x[x[:, 0] == k]
    det, expl, off_len, dip_in_off = [], [], [], []
    for k in range(N_TRIALS):
        D, O = by_trial(dips, k), by_trial(offs, k)
        for d in D:
            det.append(any((o[1] < d[2]) and (d[1] < o[2]) for o in O))
        for o in O:
            ov = [d for d in D if (o[1] < d[2]) and (d[1] < o[2])]
            expl.append(len(ov) > 0)
            off_len.append(o[2] - o[1]); dip_in_off.append(sum(min(d[2], T1) - max(d[1], T0) for d in ov))
    det, expl, off_len, dip_in_off = map(np.array, (det, expl, off_len, dip_in_off))
    dlen = dips[:, 2] - dips[:, 1]
    gaps = np.concatenate([by_trial(dips, k)[1:, 1] - by_trial(dips, k)[:-1, 2] for k in range(N_TRIALS)])
    tot_dip = np.array([sum(min(d[2], T1) - max(d[1], T0) for d in by_trial(dips, k)) for k in range(N_TRIALS)])
    tot_off = (state[:, valid] == -1).sum(1) * R.STEP
    n_dip = np.array([len(by_trial(dips, k)) for k in range(N_TRIALS)]); n_off = np.array([len(by_trial(offs, k)) for k in range(N_TRIALS)])
    has = dip_in_off > 0
    r_len = np.corrcoef(off_len[has], dip_in_off[has])[0, 1]
    r_tot = np.corrcoef(tot_dip, tot_off)[0, 1]; r_n = np.corrcoef(n_dip, n_off)[0, 1]

    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.4), gridspec_kw=dict(hspace=0.7, wspace=0.55, left=0.08, right=0.98, top=0.93, bottom=0.11))
    k_, mu, rate = p['dip_shape'], p['dip_mean'], p['dip_rate']
    a = axs[0, 0]
    x = np.linspace(0, 300, 601)
    a.fill_between(x, stats.gamma.pdf(x, k_, scale=mu / k_), color=GREY, lw=0)
    a.set_xlabel('Dip length (ms)'); a.set_ylabel('Density'); a.set_ylim(bottom=0)
    a.set_title(f"Dip length: gamma, mean {mu:.0f} ms,\nshape {k_:.1f} (SD {mu / np.sqrt(k_):.0f} ms)", fontsize=7.5, pad=3)
    b = axs[0, 1]
    x = np.linspace(0, 600, 601)
    b.fill_between(x, rate / 1000 * np.exp(-rate / 1000 * x), color=GREY, lw=0)
    b.set_xlabel('Gap between dips (ms)'); b.set_ylabel('Density'); b.set_ylim(bottom=0)
    b.set_title(f"Gaps: exponential, {rate:.1f} dips/s\n(mean {1000 / rate:.0f} ms)", fontsize=7.5, pad=3)
    c = axs[0, 2]
    e_ = np.arange(0, 501, 20)
    xx = np.linspace(0, 500, 501)
    c.fill_between(xx, stats.gamma.pdf(xx, k_, scale=mu / k_), color=GREY, alpha=0.6, lw=0, label='Dips (fitted)')
    c.hist(off_len, bins=e_, color=C_OFF, alpha=0.6, density=True, label='Off states (decoded)')
    c.set_xlabel('Length (ms)'); c.set_ylabel('Density'); c.legend(fontsize=6.5)
    c.set_title(f"Off: mean {off_len.mean():.0f} ms, {n_off.mean():.1f} per trial", fontsize=7.5, pad=3)
    d = axs[1, 0]
    eb = np.arange(0, 251, 25)
    i = np.digitize(dlen, eb) - 1
    pr = [det[i == j].mean() if (i == j).sum() > 20 else np.nan for j in range(len(eb) - 1)]
    d.plot(eb[:-1] + 12.5, pr, 'o-', color=INK, ms=3)
    d.set_ylim(0, 1.02); d.set_xlabel('Dip length (ms)'); d.set_ylabel('P(dip seen as Off state)')
    d.set_title(f"{100 * det.mean():.0f}% of dips seen; {100 * expl.mean():.0f}% of Off states\noverlap a dip", fontsize=7.5, pad=3)
    e = axs[1, 1]
    e.hist2d(dip_in_off[has], off_len[has], bins=[np.arange(0, 401, 20), np.arange(0, 601, 20)], cmap='Greys')
    e.plot([0, 400], [0, 400], ':', color=C_OFF, lw=0.8)
    e.set_xlabel('Dip time inside the Off state (ms)'); e.set_ylabel('Off-state length (ms)')
    e.set_title(f"Off states with a dip: r = {r_len:.2f}", fontsize=7.5, pad=3)
    f = axs[1, 2]
    f.hist2d(tot_dip, tot_off, bins=[np.arange(0, 1001, 40), np.arange(0, 1401, 40)], cmap='Greys')
    f.set_xlabel('Total dip time per trial (ms)'); f.set_ylabel('Total Off time per trial (ms)')
    f.set_title(f"Per trial: r = {r_tot:.2f} (counts r = {r_n:.2f})", fontsize=7.5, pad=3)
    for ax, l in zip(axs.ravel(), 'abcdef'):
        letter(ax, l, x=-0.35)
    save(fig, os.path.join(OUT, 'figS_dips_vs_states.png'))
    summ = dict(n_trials=N_TRIALS, window_ms=[T0, T1], dip_mean_ms=float(dlen.mean()), dip_median_ms=float(np.median(dlen)),
                dips_per_trial=float(n_dip.mean()), gap_mean_ms=float(gaps.mean()), off_mean_ms=float(off_len.mean()),
                off_median_ms=float(np.median(off_len)), offs_per_trial=float(n_off.mean()),
                frac_dips_detected=float(det.mean()), frac_offs_with_dip=float(expl.mean()),
                p_detect_by_length={f'{eb[j]}-{eb[j + 1]}': (None if np.isnan(pr[j]) else float(pr[j])) for j in range(len(pr))},
                r_off_length_vs_dip_time=float(r_len), r_total_dip_vs_total_off=float(r_tot), r_n_dips_vs_n_offs=float(r_n))
    json.dump(summ, open(os.path.join(OUT, 'figS_dips_vs_states.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
