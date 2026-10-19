"""Short-report figures for one model / recorded-sample size:
  fig_silences.png  network-wide silences and memory survival across them
  fig_decoding.png  trial-averaged accuracy, single-trial confidence, On/Off states, durations
  fig_mixture.png   one- vs two-state beta-mixture fit of delay confidence
  fig_tuning.png    cross-validated held-out tuning curves in On vs Off
usage: python scripts/final_figures.py --data data/X_session --K 50 --out figures/final
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
from scipy import optimize, stats  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_wm import panichello as pa  # noqa: E402
from gamma_wm.analysis import beta_mixture_cv, loo_confidence_dual  # noqa: E402
from gamma_wm.network import A, B, INH, NS  # noqa: E402

C_ON, C_OFF, C_A, C_B, C_NS, C_I, INK, INK2, GRID = ('#2a78d6', '#eb6834', '#2a78d6', '#eb6834', '#52514e',
                                                      '#1baf7a', '#0b0b0b', '#52514e', '#e4e3df')
plt.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'legend.frameon': False})
P_OFF_MEAN, P_ON_MEAN = 146.1, 192.4


def runs(mask):
    a = np.r_[False, mask, False].astype(int)
    return list(zip(np.where(np.diff(a) == 1)[0], np.where(np.diff(a) == -1)[0]))


def load(prefix):
    sdirs = sorted(glob.glob(prefix + '*'))
    meta = json.load(open(os.path.join(sdirs[0], 'meta.json')))
    group = np.load(os.path.join(sdirs[0], 'net.npz'))['group']
    trials = []
    for sd in sdirs:
        for f in sorted(glob.glob(os.path.join(sd, 'trial*.npz'))):
            d = dict(np.load(f))
            d['cue'] = str(d['cue'])
            trials.append(d)
    return meta, group, trials


def fit_betas(c):
    c = np.clip(c, 1e-3, 1 - 1e-3)
    r1 = optimize.minimize(lambda th: -stats.beta.logpdf(c, *np.exp(th)).sum(), np.zeros(2), method='Nelder-Mead')

    def nll2(th):
        a1, b1, a2, b2 = np.exp(th[:4])
        w = 1 / (1 + np.exp(-th[4]))
        return -np.log(w * stats.beta.pdf(c, a1, b1) + (1 - w) * stats.beta.pdf(c, a2, b2) + 1e-300).sum()
    best = None
    for init in ([1, 0, 0, 1, 0], [2, 0, 0, 0, 0], [0.5, 0.5, 1.5, -0.5, 0.0]):
        r = optimize.minimize(nll2, np.array(init, float), method='Nelder-Mead', options=dict(maxiter=6000))
        if best is None or r.fun < best.fun:
            best = r
    return np.exp(r1.x), best.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--K', type=int, default=50)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-shuffle', type=int, default=10)
    ap.add_argument('--lam', type=float, default=10.0)
    ap.add_argument('--label', default='')
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--frac-selective', type=float, default=None,
                    help='draw this fraction of the sample from the memory pools (Panichello: 0.44)')
    ap.add_argument('--silence-pop', default='E', choices=['E', 'NS'],
                    help='population whose silences are measured: all E cells, or only the non-pool E cells')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meta, group, trials = load(args.data)
    T = meta['task']
    t_cue = T['t_pre']
    t_don = t_cue + T['cue_dur']
    n_t = int(t_don + T['t_delay'])
    y = np.array([d['cue'] == 'A' for d in trials], int)
    sizes = np.bincount(group, minlength=6)
    S = dict(model=meta['params'], K=args.K, n_trials=len(trials))

    # ------------------------------------------------------------ silences
    ns_only = args.silence_pop == 'NS'

    def silence_rate(r):
        if ns_only:
            return uniform_filter1d(r[:, NS].astype(float), 5)
        return uniform_filter1d(r[:, :3] @ sizes[:3] / sizes[:3].sum(), 5)

    sil, after = [], []
    for d in trials:
        r = d['rates'][:n_t]
        e5 = silence_rate(r)
        C, U = (A, B) if d['cue'] == 'A' else (B, A)
        for a_, b_ in runs(e5[int(t_don):] < 1.0):
            a_ += int(t_don)
            b_ += int(t_don)
            if b_ - a_ >= 20:
                sil.append(b_ - a_)
                if ns_only:  # memory pools keep firing: is the cued pool ahead during the silence?
                    after.append((b_ - a_, r[a_:b_, C].sum() > r[a_:b_, U].sum()))
                elif b_ + 50 < n_t:
                    after.append((b_ - a_, r[b_:b_ + 50, C].sum() > r[b_:b_ + 50, U].sum()))
    sil = np.array(sil)
    after = np.array(after, float) if after else np.zeros((0, 2))
    dl_s = len(trials) * (n_t - t_don) / 1000
    S['silences'] = dict(per_s=len(sil) / dl_s, mean=float(sil.mean()), median=float(np.median(sil)),
                         p95=float(np.percentile(sil, 95)), max=float(sil.max()),
                         frac_time=float(sil.sum() / (len(trials) * (n_t - t_don))))
    end_ok = np.mean([(d['rates'][n_t - 300:n_t, A].mean() > d['rates'][n_t - 300:n_t, B].mean()) == (d['cue'] == 'A')
                      for d in trials])
    S['memory_correct_end'] = float(end_ok)

    fig = plt.figure(figsize=(11, 6.6))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.35, height_ratios=[1.3, 1])
    # example: a trial with memory correct at the end and a silence fraction at the 80th percentile
    sfrac = np.array([(silence_rate(t_['rates'][:n_t])[int(t_don):] < 1).mean() for t_ in trials])
    ok_ = np.array([(t_['rates'][n_t - 300:n_t, A].mean() > t_['rates'][n_t - 300:n_t, B].mean()) == (t_['cue'] == 'A')
                    for t_ in trials])
    cand = np.where(ok_)[0]
    d = trials[int(cand[np.argmin(np.abs(sfrac[cand] - np.percentile(sfrac, 80)))])]
    a = fig.add_subplot(gs[0, :])
    C, U = (A, B) if d['cue'] == 'A' else (B, A)
    y0 = 0
    for gi, col, lab in ((C, C_A, 'cued pool'), (U, C_B, 'uncued pool'), (NS, C_NS, 'other E'), (INH, C_I, 'inhibitory')):
        idx = np.where(group == gi)[0][:60]
        m = np.isin(d['spk_i'], idx)
        lut = np.full(len(group), -1)
        lut[idx] = np.arange(len(idx))
        a.scatter(d['spk_t'][m], lut[d['spk_i'][m]] + y0, s=1.5, c=col, lw=0, rasterized=True)
        a.text(n_t + 20, y0 + 30, lab, fontsize=8, color=INK2, va='center')
        y0 += 64
    e5 = silence_rate(d['rates'][:n_t])
    for a_, b_ in runs(e5 < 1.0):
        if b_ - a_ >= 20 and a_ > t_don:
            a.axvspan(a_, b_, color=GRID, lw=0, zorder=0)
    a.axvline(t_cue, color=INK2, lw=0.8)
    a.set_yticks([])
    a.set_xlim(0, n_t)
    if ns_only:
        a.set_xlabel('time (ms); grey = silence of the non-pool E cells (rate < 1 Hz for ≥ 20 ms)')
        a.set_title(f'Example trial (cue {d["cue"]}): the background goes silent; the memory pool keeps firing or pauses and restarts')
    else:
        a.set_xlabel('time (ms); grey = network-wide silence (E rate < 1 Hz for ≥ 20 ms)')
        a.set_title(f'Example trial (cue {d["cue"]}): spontaneous network-wide silences during the delay')
    a = fig.add_subplot(gs[1, 0])
    a.hist(sil, np.arange(0, max(400, np.percentile(sil, 99)) + 20, 20), color=INK2)
    a.set_yscale('log')
    a.set_xlabel('silence duration (ms)')
    a.set_ylabel('count')
    a.set_title(f'Silences: mean {sil.mean():.0f} ms, max {sil.max():.0f} ms')
    a = fig.add_subplot(gs[1, 1])
    bins = [(20, 50), (50, 100), (100, 200), (200, 400), (400, 10 ** 6)]
    ps, ns = [], []
    for lo, hi in bins:
        m = (after[:, 0] >= lo) & (after[:, 0] < hi) if len(after) else np.zeros(0, bool)
        ps.append(after[m, 1].mean() if m.sum() else np.nan)
        ns.append(int(m.sum()))
    a.bar(range(len(bins)), ps, color=C_A)
    for i, (p_, n_) in enumerate(zip(ps, ns)):
        if n_:
            a.text(i, p_ + 0.02, f'n={n_}', ha='center', fontsize=7)
    a.axhline(0.5, color=INK2, ls=':', lw=0.8)
    a.set_xticks(range(len(bins)), ['20-50', '50-100', '100-200', '200-400', '>400'], fontsize=8)
    a.set_ylim(0, 1.1)
    a.set_xlabel('silence duration (ms)')
    a.set_ylabel('P(cued pool > uncued during silence)' if ns_only else 'P(cued pool restarts first)')
    a.set_title('Memory held through silences' if ns_only else 'Memory survives silences (NMDA)')
    a = fig.add_subplot(gs[1, 2])
    a.bar([0], [end_ok], color=C_A, width=0.5)
    a.text(0, end_ok + 0.02, f'{end_ok:.2f}', ha='center')
    a.set_xticks([0], ['memory correct\nat end of delay'])
    a.set_ylim(0, 1.1)
    a.axhline(0.5, color=INK2, ls=':', lw=0.8)
    a.set_title(f'{len(trials)} trials')
    fig.savefig(os.path.join(args.out, 'fig_silences.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------ decoding
    rng = np.random.default_rng(args.seed)
    if args.frac_selective is None:
        sample = np.sort(rng.choice(len(group), args.K, replace=False))
    else:
        sel_c = np.where(np.isin(group, [A, B]))[0]
        oth = np.where(~np.isin(group, [A, B]))[0]
        ns_ = int(round(args.frac_selective * args.K))
        sample = np.sort(np.r_[rng.choice(sel_c, ns_, replace=False), rng.choice(oth, args.K - ns_, replace=False)])
    res = pa.run(trials, sample, y, n_t, t_cue, delay_start=t_don + 500, count_window=(t_cue, t_cue + 1500),
                 n_shuffle=args.n_shuffle, seed=args.seed, lam=args.lam, split_half=False)
    conf, state, t = res['conf'], res['state'], res['t']
    st = res['stats']
    dmask = t >= t_don + 500
    acc_t = (conf > 0.5).mean(0)
    S['decoding'] = dict(n_selective=int(np.isin(group[sample], [A, B]).sum()), acc_delay=res['acc_delay'],
                         acc_on=res['acc_on'], acc_off=res['acc_off'],
                         on_per_trial=float(st['n_on'].mean()), off_per_trial=float(st['n_off'].mean()),
                         on_mean=float(st['on_dur'].mean()) if len(st['on_dur']) else None,
                         on_median=float(np.median(st['on_dur'])) if len(st['on_dur']) else None,
                         off_mean=float(st['off_dur'].mean()) if len(st['off_dur']) else None,
                         off_median=float(np.median(st['off_dur'])) if len(st['off_dur']) else None,
                         frac_on=res['frac_on_delay'], frac_off=res['frac_off_delay'])
    # ground truth during Off
    truth = []
    for k, d in enumerate(trials):
        r = d['rates'][:n_t]
        diff = (r[:, A] - r[:, B]) * (1 if d['cue'] == 'A' else -1)
        truth.append(np.convolve(diff, np.ones(100) / 100, mode='same'))
    truth = np.array(truth)[:, np.clip(t.astype(int), 0, n_t - 1)]
    offd = (state == -1) & dmask[None, :]
    S['decoding']['memory_correct_in_off'] = float((truth[offd] > 0).mean()) if offd.any() else None

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(3, 3, hspace=0.6, wspace=0.35)
    a = fig.add_subplot(gs[0, :2])
    a.plot(t, acc_t, color=INK)
    a.axhline(0.5, color=INK2, ls=':', lw=0.8)
    a.axvline(t_cue, color=C_OFF, lw=0.8)
    a.set_ylim(0.3, 1.02)
    a.set_ylabel('decoding accuracy\n(all trials)')
    a.set_title(f'Trial-averaged decoding: {res["acc_delay"]:.0%} in the delay '
                f'({args.K} recorded cells, {S["decoding"]["n_selective"]} pool cells)')
    a = fig.add_subplot(gs[0, 2])
    bins_d = np.arange(0, 801, 40)
    if len(st['on_dur']):
        a.hist(st['on_dur'], bins_d, color=C_ON, alpha=0.8, label=f'On: {np.mean(st["on_dur"]):.0f} ms')
    if len(st['off_dur']):
        a.hist(st['off_dur'], bins_d, histtype='step', lw=1.6, color=C_OFF, label=f'Off: {np.mean(st["off_dur"]):.0f} ms')
    a.axvline(P_ON_MEAN, color=C_ON, ls='--', lw=0.9)
    a.axvline(P_OFF_MEAN, color=C_OFF, ls='--', lw=0.9)
    a.set_xlabel('state duration (ms)')
    a.set_ylabel('count')
    a.set_title('Durations (dashed: Panichello means)')
    a.legend(fontsize=7.5)
    a = fig.add_subplot(gs[1, :])
    im = a.imshow(conf, aspect='auto', cmap='Blues', vmin=0, vmax=1, origin='lower', interpolation='nearest',
                  extent=[t[0] - 5, t[-1] + 5, -0.5, len(conf) - 0.5])
    a.set_ylabel('trial')
    a.set_title('Single-trial decoder confidence (probability of the correct cue)')
    fig.colorbar(im, ax=a, fraction=0.02)
    a = fig.add_subplot(gs[2, :])
    # examples: trials whose memory is correct at the end, with the most Off states
    held = np.array([(d_['rates'][n_t - 300:n_t, A].mean() > d_['rates'][n_t - 300:n_t, B].mean()) == (d_['cue'] == 'A')
                     for d_ in trials])
    n_off = np.array([len(runs((state[k] == -1) & dmask)) for k in range(len(trials))])
    ex = np.argsort(-(n_off + 0.01 * ((state == 1) & dmask).sum(1)) - 100 * held)[:4]
    for j, k in enumerate(ex):
        a.plot(t, conf[k] + 1.15 * j, color=INK, lw=0.9)
        a.fill_between(t, 1.15 * j, 1.15 * j + 1, where=state[k] == 1, color=C_ON, alpha=0.25, lw=0, step='mid')
        a.fill_between(t, 1.15 * j, 1.15 * j + 1, where=state[k] == -1, color=C_OFF, alpha=0.3, lw=0, step='mid')
        a.axhline(1.15 * j + 0.5, color=INK2, ls=':', lw=0.6)
    a.set_yticks([])
    a.set_xlabel('time (ms)')
    a.set_title('Example trials: On (blue) and Off (orange; decoding at chance) states')
    fig.savefig(os.path.join(args.out, 'fig_decoding.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------ mixture
    c = np.clip(conf[:, dmask].ravel(), 1e-3, 1 - 1e-3)
    ll1, ll2 = beta_mixture_cv(c)
    p1, th2 = fit_betas(c)
    S['mixture_mbit'] = float((ll2 - ll1) * 1000)
    fig, a = plt.subplots(figsize=(5, 3.6))
    a.hist(c, np.linspace(0, 1, 31), density=True, color=GRID, edgecolor=INK2, lw=0.4)
    xx = np.linspace(0.005, 0.995, 300)
    a.plot(xx, stats.beta.pdf(xx, *p1), color=INK, lw=1.4, label='1 state')
    a1, b1, a2, b2 = np.exp(th2[:4])
    w = 1 / (1 + np.exp(-th2[4]))
    a.plot(xx, w * stats.beta.pdf(xx, a1, b1) + (1 - w) * stats.beta.pdf(xx, a2, b2), color=C_ON, lw=1.6, label='2 states')
    a.plot(xx, w * stats.beta.pdf(xx, a1, b1), color=C_ON, lw=0.8, ls='--')
    a.plot(xx, (1 - w) * stats.beta.pdf(xx, a2, b2), color=C_ON, lw=0.8, ls='--')
    a.set_xlabel('decoder confidence (delay)')
    a.set_ylabel('density')
    a.set_title(f'Two-state model better by {S["mixture_mbit"]:.0f} mbit/sample (CV)')
    a.legend(fontsize=8)
    fig.savefig(os.path.join(args.out, 'fig_mixture.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------ held-out tuning
    Call, tc = pa.counts_matrix(trials, sample, n_t, 100, 10)
    keep = tc >= t_cue - 100
    Cm, tk = Call[:, :, keep], tc[keep]
    valid = tk >= t_cue
    dmk = tk >= t_don + 500
    perm = rng.permutation(args.K)
    halves = (perm[: args.K // 2], perm[args.K // 2:])
    cur = {k: [] for k in ('on_p', 'on_n', 'off_p', 'off_n')}
    for h in (0, 1):
        Cl = np.ascontiguousarray(Cm[:, halves[h]])
        c1 = loo_confidence_dual(Cl, y, args.lam)
        n1 = np.array([loo_confidence_dual(Cl, y, args.lam, shuffle_rng=np.random.default_rng(900 + 10 * h + s))
                       for s in range(args.n_shuffle)])
        s1, _, _ = pa.label_states(c1, n1, valid)
        Xt = Cm[:, halves[1 - h]].astype(float)
        Z = (Xt - Xt.mean(0, keepdims=True)) / (Xt.std(0, keepdims=True) + 1e-9)

        def means(yy, lab):
            out = np.full((Z.shape[1], 2), np.nan)
            per = np.array([Z[k][:, (s1[k] == lab) & dmk].mean(1) if ((s1[k] == lab) & dmk).sum() else
                            np.full(Z.shape[1], np.nan) for k in range(len(yy))])
            for cc in (0, 1):
                m = (yy == cc) & ~np.isnan(per[:, 0])
                if m.sum():
                    out[:, cc] = per[m].mean(0)
            return out

        def align(on, off):
            pref = np.nanargmax(np.nan_to_num(on, nan=-1e9) + np.nan_to_num(off, nan=-1e9), axis=1)
            i = np.arange(len(pref))
            return np.c_[on[i, pref], on[i, 1 - pref]], np.c_[off[i, pref], off[i, 1 - pref]]
        on_a, off_a = align(means(y, 1), means(y, -1))
        nul = [align(means(ys, 1), means(ys, -1)) for ys in
               (np.random.default_rng(5000 + s).permutation(y) for s in range(20))]
        on_c = on_a - np.nanmean([u[0] for u in nul], 0)
        off_c = off_a - np.nanmean([u[1] for u in nul], 0)
        cur['on_p'] += list(on_c[:, 0])
        cur['on_n'] += list(on_c[:, 1])
        cur['off_p'] += list(off_c[:, 0])
        cur['off_n'] += list(off_c[:, 1])
    mv = {k: (np.nanmean(v), np.nanstd(v) / np.sqrt(np.sum(~np.isnan(v)))) for k, v in cur.items()}
    S['heldout_tuning'] = {k: float(v[0]) for k, v in mv.items()}
    S['heldout_tuning']['on_diff'] = float(mv['on_p'][0] - mv['on_n'][0])
    S['heldout_tuning']['off_diff'] = float(mv['off_p'][0] - mv['off_n'][0])
    fig, a = plt.subplots(figsize=(4.2, 3.8))
    for key, c_, lab in (('on', C_ON, 'On'), ('off', C_OFF, 'Off')):
        a.errorbar([0, 1], [mv[key + '_n'][0], mv[key + '_p'][0]], yerr=[mv[key + '_n'][1], mv[key + '_p'][1]],
                   color=c_, marker='o', ms=6, lw=1.6, capsize=3, label=lab)
    a.axhline(0, color=INK2, ls=':', lw=0.8)
    a.set_xticks([0, 1], ['non-preferred', 'preferred'])
    a.set_xlim(-0.4, 1.4)
    a.set_ylabel('held-out cells, delay activity\n(z-score, shuffle-corrected)')
    a.set_title(f'Cross-validated tuning (pref − non: On {S["heldout_tuning"]["on_diff"]:.2f}, '
                f'Off {S["heldout_tuning"]["off_diff"]:.2f})', fontsize=9)
    a.legend(fontsize=8)
    fig.savefig(os.path.join(args.out, 'fig_tuning.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    json.dump(S, open(os.path.join(args.out, 'summary.json'), 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in S.items() if k != 'model'}, indent=1, default=float))


if __name__ == '__main__':
    main()
