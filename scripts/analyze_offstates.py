"""Panichello-style decoding of the bistable-pool model: single-trial decoder
confidence from a Neuropixels-like sample (all pool cells + non-selective E + I),
On/Off states against a shuffled-label null, decoding and rates in On vs Off states,
and whether the memory comes back after each Off state.

usage: python scripts/analyze_offstates.py --data data/updown_q06_session --fig figures/offstates_q06
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_wm import analysis as an  # noqa: E402
from gamma_wm.network import A, B, INH, NS  # noqa: E402

C_ON, C_OFF, C_CUED, C_UNCUED, INK, INK2, GRID = '#2a78d6', '#eb6834', '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'lines.linewidth': 1.2, 'legend.frameon': False})


def runs(mask, min_len):
    a = np.r_[False, mask, False].astype(int)
    return [(s, e) for s, e in zip(np.where(np.diff(a) == 1)[0], np.where(np.diff(a) == -1)[0]) if e - s >= min_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--fig', required=True)
    ap.add_argument('--n-ns', type=int, default=200)
    ap.add_argument('--n-inh', type=int, default=50)
    ap.add_argument('--n-shuffle', type=int, default=10)
    ap.add_argument('--win', type=int, default=3, help='window in 20 ms bins (3 = 60 ms)')
    args = ap.parse_args()
    os.makedirs(args.fig, exist_ok=True)
    sdirs = sorted(glob.glob(args.data + '*'))
    meta = json.load(open(os.path.join(sdirs[0], 'meta.json')))
    group = np.load(os.path.join(sdirs[0], 'net.npz'))['group']
    T = meta['task']
    t_cue, t_don = T['t_pre'], T['t_pre'] + T['cue_dur']
    n_t = int(t_don + T['t_delay'])
    trials = []
    for sd in sdirs:
        for f in sorted(glob.glob(os.path.join(sd, 'trial*.npz'))):
            d = dict(np.load(f))
            d['cue'] = str(d['cue'])
            trials.append(d)
    y = np.array([d['cue'] == 'A' for d in trials], int)
    rng = np.random.default_rng(0)
    sample = np.r_[np.where(group == A)[0], np.where(group == B)[0],
                   rng.choice(np.where(group == NS)[0], args.n_ns, replace=False),
                   rng.choice(np.where(group == INH)[0], args.n_inh, replace=False)]
    n_sel = int((group == A).sum() + (group == B).sum())
    step = 20
    edges = np.arange(0, n_t + 1, step)
    counts = []
    for d in trials:
        H = an.spike_counts(dict(t=d['spk_t'], i=d['spk_i']), sample, edges)
        C = sum(H[:, j:H.shape[1] - args.win + 1 + j] for j in range(args.win))
        counts.append(C)
    counts = np.array(counts)
    t_dec = (np.arange(counts.shape[2]) + args.win / 2) * step
    print(f'{len(trials)} trials, sample {len(sample)} cells ({n_sel} pool cells), {counts.shape[2]} bins', flush=True)
    conf = an.loo_confidence_dual(counts, y)
    nulls = np.array([an.loo_confidence_dual(counts, y, shuffle_rng=np.random.default_rng(100 + k))
                      for k in range(args.n_shuffle)])
    z = (conf - nulls.mean(0)) / (nulls.std(0) + 1e-9)
    dmask = t_dec >= t_don + 500
    state = np.zeros_like(conf, dtype=int)
    on_d, off_d = [], []
    after_off = []   # (Off duration, decoder correct in the next On state)
    for k in range(len(trials)):
        zk = np.where(dmask, z[k], np.nan)
        for s, e in runs(zk > 1.96, 3):
            state[k, s:e] = 1
            on_d.append((e - s) * step)
        for s, e in runs(zk < 0.84, 3):
            state[k, s:e] = -1
            off_d.append((e - s) * step)
        # memory recovery: for each Off state followed later by an On state
        offs = runs(state[k] == -1, 1)
        ons = runs(state[k] == 1, 1)
        for s, e in offs:
            nxt = [o for o in ons if o[0] >= e]
            if nxt:
                o = nxt[0]
                after_off.append(((e - s) * step, bool(conf[k, o[0]:o[1]].mean() > 0.5)))
    # rates by decoder state
    pool_c, pool_u, ns_r = {1: [], -1: []}, {1: [], -1: []}, {1: [], -1: []}
    for k, d in enumerate(trials):
        c_, u_ = (A, B) if d['cue'] == 'A' else (B, A)
        for b, tb in enumerate(t_dec):
            st = state[k, b]
            if st:
                sl = slice(int(tb - 30), int(tb + 30))
                pool_c[st].append(d['rates'][sl, c_].mean())
                pool_u[st].append(d['rates'][sl, u_].mean())
                ns_r[st].append(d['rates'][sl, NS].mean())
    # memory at end of delay from the whole pools
    end_ok = np.mean([(d['rates'][n_t - 500:n_t, A].mean() > d['rates'][n_t - 500:n_t, B].mean()) == (d['cue'] == 'A')
                      for d in trials])
    acc = (conf > 0.5)
    ll1, ll2 = an.beta_mixture_cv(conf[:, dmask].ravel())
    S = dict(
        n_trials=len(trials), sample=dict(n=int(len(sample)), pool_cells=n_sel, ns=args.n_ns, inh=args.n_inh),
        window_ms=args.win * step, mean_conf_delay=float(conf[:, dmask].mean()), acc_delay=float(acc[:, dmask].mean()),
        acc_on=float(acc[state == 1].mean()) if (state == 1).any() else None,
        acc_off=float(acc[state == -1].mean()) if (state == -1).any() else None,
        conf_on=float(conf[state == 1].mean()) if (state == 1).any() else None,
        conf_off=float(conf[state == -1].mean()) if (state == -1).any() else None,
        frac_on=float((state[:, dmask] == 1).mean()), frac_off=float((state[:, dmask] == -1).mean()),
        n_on=len(on_d), n_off=len(off_d), on_per_trial=len(on_d) / len(trials), off_per_trial=len(off_d) / len(trials),
        on_dur_mean=float(np.mean(on_d)) if on_d else None, on_dur_median=float(np.median(on_d)) if on_d else None,
        off_dur_mean=float(np.mean(off_d)) if off_d else None, off_dur_median=float(np.median(off_d)) if off_d else None,
        rates_on=dict(cued=float(np.mean(pool_c[1])), uncued=float(np.mean(pool_u[1])), NS=float(np.mean(ns_r[1]))) if pool_c[1] else None,
        rates_off=dict(cued=float(np.mean(pool_c[-1])), uncued=float(np.mean(pool_u[-1])), NS=float(np.mean(ns_r[-1]))) if pool_c[-1] else None,
        recovery_after_off=float(np.mean([c for _, c in after_off])) if after_off else None, n_recovery=len(after_off),
        recovery_by_off_duration={f'{lo}-{hi}': (float(np.mean([c for du, c in after_off if lo <= du < hi])),
                                                  int(sum(lo <= du < hi for du, _ in after_off)))
                                  for lo, hi in [(0, 100), (100, 200), (200, 400), (400, 800), (800, 100000)]
                                  if any(lo <= du < hi for du, _ in after_off)},
        memory_correct_end=float(end_ok), ll_1beta=ll1, ll_2beta=ll2)
    json.dump(S, open(os.path.join(args.fig, 'summary.json'), 'w'), indent=1)
    print(json.dumps(S, indent=1))

    # ---------------------------------------------------------------- figures
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.6, wspace=0.4)
    a = fig.add_subplot(gs[0, :2])
    im = a.imshow(conf, aspect='auto', cmap='Blues', vmin=0, vmax=1, origin='lower',
                  extent=[t_dec[0] - 10, t_dec[-1] + 10, -0.5, len(conf) - 0.5], interpolation='nearest')
    a.axvline(t_cue, color=C_OFF, lw=1)
    a.set_ylabel('trial')
    a.set_xlabel('time (ms)')
    a.set_title(f'Decoder confidence ({len(sample)} cells, {args.win * step} ms windows)')
    fig.colorbar(im, ax=a, fraction=0.025).set_label('P(correct cue)')
    a = fig.add_subplot(gs[0, 2])
    a.hist(conf[:, dmask].ravel(), np.linspace(0, 1, 26), color=C_ON, density=True)
    a.set_xlabel('confidence (delay)')
    a.set_title(f'2- vs 1-beta: {(ll2 - ll1) * 1000:.0f} mbit/sample')
    a = fig.add_subplot(gs[1, :2])
    for j in range(min(5, len(conf))):
        a.plot(t_dec, conf[j] + 1.2 * j, color=INK, lw=0.8)
        a.fill_between(t_dec, 1.2 * j, 1.2 * j + 1, where=state[j] == 1, color=C_ON, alpha=0.25, lw=0, step='mid')
        a.fill_between(t_dec, 1.2 * j, 1.2 * j + 1, where=state[j] == -1, color=C_OFF, alpha=0.25, lw=0, step='mid')
        a.axhline(1.2 * j + 0.5, color=INK2, lw=0.5, ls=':')
    a.axvline(t_cue, color=C_OFF, lw=1)
    a.set_yticks([])
    a.set_xlabel('time (ms)')
    a.set_title('Example trials: On (blue) / Off (orange)')
    a = fig.add_subplot(gs[1, 2])
    bins = np.arange(0, 2001, 80)
    if on_d:
        a.hist(on_d, bins, color=C_ON, alpha=0.85, label=f'On (median {np.median(on_d):.0f} ms)')
    if off_d:
        a.hist(off_d, bins, histtype='step', color=C_OFF, lw=1.6, label=f'Off (median {np.median(off_d):.0f} ms)')
    a.set_xlabel('state duration (ms)')
    a.set_ylabel('count')
    a.legend(fontsize=7.5)
    a = fig.add_subplot(gs[2, 0])
    v = [S['acc_on'] or 0, S['acc_off'] or 0]
    a.bar([0, 1], v, width=0.55, color=[C_ON, C_OFF])
    for xi, vi in enumerate(v):
        a.text(xi, vi, f'{vi:.2f}', ha='center', va='bottom')
    a.axhline(0.5, color=INK2, ls=':', lw=0.8)
    a.set_xticks([0, 1], ['On', 'Off'])
    a.set_ylim(0, 1.05)
    a.set_ylabel('decoding accuracy')
    a.set_title('Decoding by state')
    a = fig.add_subplot(gs[2, 1])
    if S['rates_on'] and S['rates_off']:
        x = np.arange(3)
        a.bar(x - 0.18, [S['rates_on'][k] for k in ('cued', 'uncued', 'NS')], 0.34, color=C_ON, label='On')
        a.bar(x + 0.18, [S['rates_off'][k] for k in ('cued', 'uncued', 'NS')], 0.34, color=C_OFF, label='Off')
        a.set_xticks(x, ['cued pool', 'uncued pool', 'non-selective'])
        a.legend(fontsize=7.5)
    a.set_ylabel('rate (Hz)')
    a.set_title('Rates by state')
    a = fig.add_subplot(gs[2, 2])
    rb = S['recovery_by_off_duration']
    if rb:
        ks = list(rb)
        a.bar(range(len(ks)), [rb[k][0] for k in ks], color=C_ON)
        for i, k in enumerate(ks):
            a.text(i, rb[k][0], f'n={rb[k][1]}', ha='center', va='bottom', fontsize=7)
        a.set_xticks(range(len(ks)), ks, rotation=30)
    a.axhline(0.5, color=INK2, ls=':', lw=0.8)
    a.set_ylim(0, 1.1)
    a.set_xlabel('Off duration (ms)')
    a.set_ylabel('next On state correct')
    a.set_title('Memory after Off states')
    fig.savefig(os.path.join(args.fig, 'decoding_on_off.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # example trial with pool rates
    d = trials[0]
    c_, u_ = (A, B) if d['cue'] == 'A' else (B, A)
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    tt = np.arange(n_t)
    ax[0].plot(tt, uniform_filter1d(d['rates'][:n_t, c_], 50), color=C_CUED, label='cued pool')
    ax[0].plot(tt, uniform_filter1d(d['rates'][:n_t, u_], 50), color=C_UNCUED, label='uncued pool')
    ax[0].plot(tt, uniform_filter1d(d['rates'][:n_t, NS], 50), color=INK2, label='non-selective', lw=0.9)
    ax[0].set_ylabel('rate (Hz, 50 ms)')
    ax[0].legend(fontsize=8, ncol=3)
    ax[0].axvspan(t_cue, t_don, color=GRID, lw=0)
    ax[1].plot(t_dec, conf[0], color=INK, lw=0.9)
    ax[1].fill_between(t_dec, 0, 1, where=state[0] == 1, color=C_ON, alpha=0.25, lw=0, step='mid')
    ax[1].fill_between(t_dec, 0, 1, where=state[0] == -1, color=C_OFF, alpha=0.25, lw=0, step='mid')
    ax[1].axhline(0.5, color=INK2, ls=':', lw=0.8)
    ax[1].set_ylabel('decoder confidence')
    ax[1].set_xlabel('time (ms)')
    fig.savefig(os.path.join(args.fig, 'example_trial.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
