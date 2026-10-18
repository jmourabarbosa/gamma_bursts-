"""Analyse the working-memory sessions and make the figures.

usage: python scripts/analyze.py [--data data] [--fig figures]
Writes figures/*.png and figures/summary.json.
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

# palette (fixed categorical order): cued pool, uncued pool, inhibitory; NS in ink gray
C_CUED, C_UNCUED, C_INH, C_NS = '#2a78d6', '#eb6834', '#1baf7a', '#52514e'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'lines.linewidth': 1.2, 'legend.frameon': False})
FREQS = np.arange(10, 121, 2.0)
GAMMA = (40, 100)


def load(data):
    trials = []
    for sdir in sorted(glob.glob(os.path.join(data, 'session*'))):
        meta = json.load(open(os.path.join(sdir, 'meta.json')))
        net = np.load(os.path.join(sdir, 'net.npz'))
        for f in sorted(glob.glob(os.path.join(sdir, 'trial*.npz'))):
            d = dict(np.load(f))
            d['cue'] = str(d['cue'])
            d['session'] = sdir
            trials.append(d)
    return meta, dict(net), trials


class Obj(dict):
    __getattr__ = dict.__getitem__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--fig', default='figures')
    ap.add_argument('--n-decode', type=int, default=200, help='neurons in the decoder sample')
    ap.add_argument('--n-shuffle', type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.fig, exist_ok=True)
    meta, net, trials = load(args.data)
    P, T = meta['params'], meta['task']
    group, vrec_idx = net['group'], net['vrec_idx']
    t_cue = T['t_pre']
    t_don = t_cue + T['cue_dur']
    T_tot = t_don + T['t_delay']
    n_t = int(T_tot)
    tt = np.arange(n_t)
    DL = slice(int(t_don) + 500, n_t)          # analysed delay window (as Panichello)
    PRE = slice(200, int(t_cue))
    summary = dict(n_trials=len(trials), params=P, task=T)
    sizes = np.bincount(group, minlength=4)
    print(f'{len(trials)} trials')

    # ------------------------------------------------------------------ per-trial
    for d in trials:
        cue_col, unc_col = (A, B) if d['cue'] == 'A' else (B, A)
        d['r_cued'] = d['rates'][:n_t, cue_col]
        d['r_unc'] = d['rates'][:n_t, unc_col]
        d['r_E'] = d['rates'][:n_t, :3] @ sizes[:3] / sizes[:3].sum()
        d['P'] = an.morlet_power(d['lfp'][:n_t].astype(float), FREQS)
        # power statistics per frequency from the trial excluding the cue response
        ref = np.ones(n_t, bool)
        ref[int(t_cue):int(t_don) + 250] = False
        mu = d['P'][:, ref].mean(1, keepdims=True)
        sd = d['P'][:, ref].std(1, keepdims=True)
        bursts, keep = an.detect_bursts(d['P'], FREQS, tt.astype(float), mean=mu, sd=sd)
        d['gbursts'] = [b for b in bursts if GAMMA[0] <= b['f_peak'] <= GAMMA[1]]
        d['bbursts'] = [b for b in bursts if 15 <= b['f_peak'] < 35]
        d['gmask'] = an.burst_mask(d['gbursts'], tt)
        d['correct'] = d['r_cued'][DL].mean() > d['r_unc'][DL].mean()
    acc = np.mean([d['correct'] for d in trials])
    # memory read-out sign over time (full-network readout)
    readout = np.array([uniform_filter1d(d['r_cued'] - d['r_unc'], 50) for d in trials])
    frac_correct_t = (readout > 0).mean(0)
    summary['memory_accuracy_delay'] = float(acc)
    summary['min_frac_correct_over_delay'] = float(frac_correct_t[DL].min())
    summary['rates_delay'] = dict(cued=float(np.mean([d['r_cued'][DL].mean() for d in trials])),
                                  uncued=float(np.mean([d['r_unc'][DL].mean() for d in trials])),
                                  NS=float(np.mean([d['rates'][DL, NS].mean() for d in trials])),
                                  I=float(np.mean([d['rates'][DL, INH].mean() for d in trials])))

    # ------------------------------------------------------------ single neurons
    ex = trials[0]
    is_cued_pool = lambda d: group == (A if d['cue'] == 'A' else B)  # noqa: E731
    cvs_cued, cvs_ns, rates_cued, rates_ns = [], [], [], []
    for d in trials:
        res = dict(t=d['spk_t'], i=d['spk_i'])
        pool = np.where(is_cued_pool(d))[0][:200]
        nsi = np.where(group == NS)[0][:400]
        cvs_cued += list(an.isi_cv(res, pool, DL.start, DL.stop))
        cvs_ns += list(an.isi_cv(res, nsi, DL.start, DL.stop))
        dur = (DL.stop - DL.start) * 1e-3
        m = (d['spk_t'] >= DL.start) & (d['spk_t'] < DL.stop)
        cnt = np.bincount(d['spk_i'][m], minlength=len(group))
        rates_cued += list(cnt[pool] / dur)
        rates_ns += list(cnt[nsi] / dur)
    summary['cv_isi'] = dict(cued_mean=float(np.mean(cvs_cued)), ns_mean=float(np.mean(cvs_ns)),
                             cued_median=float(np.median(cvs_cued)), ns_median=float(np.median(cvs_ns)))
    # spectra
    spec_pre = np.mean([d['P'][:, PRE].mean(1) for d in trials], 0)
    spec_dl = np.mean([d['P'][:, DL].mean(1) for d in trials], 0)
    f_peak = float(FREQS[np.argmax(spec_dl * FREQS)])
    summary['lfp_peak_freq_delay'] = f_peak
    # cycles: fraction of gamma cycles in which a neuron of the cued pool fires
    summary['mean_rate_over_gamma_freq'] = summary['rates_delay']['cued'] / f_peak

    # ---------------------------------------------------------------- bursts
    gb = [b for d in trials for b in d['gbursts'] if DL.start <= b['t_peak'] < DL.stop]
    durs = np.array([b['dur'] for b in gb])
    fpk = np.array([b['f_peak'] for b in gb])
    width = np.array([b['f_hi'] - b['f_lo'] for b in gb])
    dl_len = (DL.stop - DL.start) / 1000
    cv2s = [an.cv2_intervals([b['t_peak'] for b in d['gbursts'] if DL.start <= b['t_peak'] < DL.stop])
            for d in trials]
    burst_rate_t = np.mean([d['gmask'] for d in trials], 0)
    gpow_t = np.mean([d['P'][(FREQS >= GAMMA[0]) & (FREQS <= GAMMA[1])].mean(0) for d in trials], 0)
    # spiking inside vs outside bursts during delay
    ins = np.concatenate([d['gmask'][DL] for d in trials])
    rc = np.concatenate([d['r_cued'][DL] for d in trials])
    ru = np.concatenate([d['r_unc'][DL] for d in trials])
    rE = np.concatenate([d['r_E'][DL] for d in trials])
    summary['gamma_bursts'] = dict(
        n=len(gb), per_s=float(len(gb) / (len(trials) * dl_len)), dur_mean=float(durs.mean()),
        dur_sd=float(durs.std()), f_mean=float(fpk.mean()), width_mean=float(width.mean()),
        frac_time=float(ins.mean()), cv2_ibi=float(np.nanmean(cv2s)),
        cued_rate_in=float(rc[ins].mean()), cued_rate_out=float(rc[~ins].mean()),
        uncued_rate_in=float(ru[ins].mean()), uncued_rate_out=float(ru[~ins].mean()),
        E_rate_in=float(rE[ins].mean()), E_rate_out=float(rE[~ins].mean()),
        corr_burstrate_power=float(np.corrcoef(burst_rate_t[200:], gpow_t[200:])[0, 1]))
    # burst-centre autocorrelogram
    lags = np.arange(-500, 501, 5)
    ac = np.zeros(len(lags))
    for d in trials:
        c = np.array([b['t_peak'] for b in d['gbursts'] if DL.start <= b['t_peak'] < DL.stop])
        for x in c:
            dd = c - x
            ac += np.histogram(dd, bins=np.r_[lags - 2.5, lags[-1] + 2.5])[0]
    ac[lags == 0] = 0

    # ------------------------------------------------- silence and hidden state
    # silent episodes: E population rate (5 ms boxcar) below 25% of its delay mean
    sil_durs, trig_cued, trig_unc, trig_rc, trig_ru = [], [], [], [], []
    win = np.arange(-40, 81)
    for d in trials:
        rs = uniform_filter1d(d['r_E'], 5)
        thr = 0.25 * rs[DL].mean()
        s = np.r_[False, rs[DL] < thr, False].astype(int)
        on = np.where(np.diff(s) == 1)[0] + DL.start
        off = np.where(np.diff(s) == -1)[0] + DL.start
        sil_durs += list(off - on)
        V = d['V'].astype(float)
        vc = V[:n_t, :len(vrec_idx) // 2] if d['cue'] == 'A' else V[:n_t, len(vrec_idx) // 2:]
        vu = V[:n_t, len(vrec_idx) // 2:] if d['cue'] == 'A' else V[:n_t, :len(vrec_idx) // 2]
        for a_, b_ in zip(on, off):
            if b_ - a_ >= 8 and a_ + win[0] >= DL.start and a_ + win[-1] < n_t:
                trig_cued.append(vc[a_ + win].mean(1))
                trig_unc.append(vu[a_ + win].mean(1))
                trig_rc.append(d['r_cued'][a_ + win])
                trig_ru.append(d['r_unc'][a_ + win])
    sil_durs = np.array(sil_durs)
    summary['silences'] = dict(n=int(len(sil_durs)), per_s=float(len(sil_durs) / (len(trials) * dl_len)),
                               dur_mean=float(sil_durs.mean()), dur_max=float(sil_durs.max()),
                               n_long_ge8ms=int(len(trig_cued)),
                               frac_time=float(sil_durs.sum() / (len(trials) * (DL.stop - DL.start))))
    trig_cued, trig_unc = np.array(trig_cued), np.array(trig_unc)
    trig_rc, trig_ru = np.array(trig_rc), np.array(trig_ru)
    summary['silences']['V_gap_mV_during'] = float((trig_cued - trig_unc)[:, 40:48].mean())
    summary['silences']['rate_after_cued'] = float(trig_rc[:, 50:80].mean())
    summary['silences']['rate_after_uncued'] = float(trig_ru[:, 50:80].mean())

    # ----------------------------------------------------------- decoding
    rng = np.random.default_rng(0)
    dec_idx = np.sort(rng.choice(len(group), args.n_decode, replace=False))
    win_ms, step_ms = 50, 20
    starts = np.arange(0, n_t - win_ms + 1, step_ms)
    t_dec = starts + win_ms / 2
    counts = []
    for d in trials:
        res = dict(t=d['spk_t'], i=d['spk_i'])
        H = an.spike_counts(res, dec_idx, np.arange(0, n_t + 1, step_ms))   # 20 ms bins
        # 50 ms windows ~ sum of 2.5 bins: use 60 ms (3 bins) for simplicity
        C3 = H[:, :-2] + H[:, 1:-1] + H[:, 2:]
        counts.append(C3)
    counts = np.array(counts)
    t_dec = (np.arange(counts.shape[2]) + 1.5) * step_ms
    y = np.array([d['cue'] == 'A' for d in trials], int)
    print('decoding...', flush=True)
    conf = an.loo_confidence(counts, y)
    nulls = np.array([an.loo_confidence(counts, y, shuffle_rng=np.random.default_rng(100 + k))
                      for k in range(args.n_shuffle)])
    z = (conf - nulls.mean(0)) / (nulls.std(0) + 1e-9)
    dmask = (t_dec >= DL.start) & (t_dec < DL.stop)
    c_delay = conf[:, dmask].ravel()
    ll1, ll2 = an.beta_mixture_cv(c_delay)
    # On / Off labelling: On = z > 1.96 for >= 3 consecutive bins (60 ms);
    # Off = z < 0.84 (one-sided P > 0.2) for >= 3 consecutive bins
    def runs(mask, min_len):
        out = []
        a = np.r_[False, mask, False].astype(int)
        for s, e in zip(np.where(np.diff(a) == 1)[0], np.where(np.diff(a) == -1)[0]):
            if e - s >= min_len:
                out.append((s, e))
        return out
    on_d, off_d, state = [], [], np.zeros_like(conf, dtype=int)
    for k in range(len(trials)):
        zk = np.where(dmask, z[k], np.nan)
        for s, e in runs(zk > 1.96, 3):
            on_d.append((e - s) * step_ms)
            state[k, s:e] = 1
        for s, e in runs(zk < 0.84, 3):
            off_d.append((e - s) * step_ms)
            state[k, s:e] = -1
    # gamma-burst occupancy and cued-pool rate during On vs Off decoder states
    g_on, g_off, rc_on, rc_off, ro_on, ro_off = [], [], [], [], [], []
    for k, d in enumerate(trials):
        for b, tb in enumerate(t_dec):
            if state[k, b] == 0:
                continue
            sl = slice(int(tb - 30), int(tb + 30))
            (g_on if state[k, b] == 1 else g_off).append(d['gmask'][sl].mean())
            (rc_on if state[k, b] == 1 else rc_off).append(d['r_cued'][sl].mean())
            (ro_on if state[k, b] == 1 else ro_off).append(d['r_unc'][sl].mean())
    summary['decoding'] = dict(
        n_neurons=int(args.n_decode), mean_conf_delay=float(c_delay.mean()),
        acc_delay=float((conf[:, dmask] > 0.5).mean()), ll_1beta=ll1, ll_2beta=ll2,
        n_on=len(on_d), n_off=len(off_d), on_dur_mean=float(np.mean(on_d)) if on_d else None,
        off_dur_mean=float(np.mean(off_d)) if off_d else None,
        on_per_trial=len(on_d) / len(trials), off_per_trial=len(off_d) / len(trials),
        frac_on=float((state[:, dmask] == 1).mean()), frac_off=float((state[:, dmask] == -1).mean()),
        gamma_occ_on=float(np.mean(g_on)) if g_on else None, gamma_occ_off=float(np.mean(g_off)) if g_off else None,
        cued_rate_on=float(np.mean(rc_on)) if rc_on else None, cued_rate_off=float(np.mean(rc_off)) if rc_off else None,
        uncued_rate_on=float(np.mean(ro_on)) if ro_on else None, uncued_rate_off=float(np.mean(ro_off)) if ro_off else None,
        frac_correct_readout_during_off=None)
    # does the full-network memory survive decoder Off states?
    corr_off = []
    for k in range(len(trials)):
        for b in np.where(state[k] == -1)[0]:
            corr_off.append(readout[k, int(t_dec[b])] > 0)
    summary['decoding']['frac_correct_readout_during_off'] = float(np.mean(corr_off)) if corr_off else None
    json.dump(summary, open(os.path.join(args.fig, 'summary.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))

    # =============================================================== FIGURES
    S = Obj(trials=trials, group=group, tt=tt, t_cue=t_cue, t_don=t_don, n_t=n_t, DL=DL)
    fig_example(S, ex, args.fig)
    fig_irregular(S, ex, cvs_cued, cvs_ns, rates_cued, rates_ns, spec_pre, spec_dl, f_peak, args.fig)
    fig_bursts(S, durs, fpk, width, burst_rate_t, gpow_t, lags, ac, summary, args.fig)
    fig_silence(S, sil_durs, win, trig_cued, trig_unc, trig_rc, trig_ru, frac_correct_t, args.fig)
    fig_decoding(S, conf, z, state, t_dec, dmask, c_delay, ll1, ll2, on_d, off_d, summary, args.fig)


def shade_task(ax, S):
    ax.axvspan(S.t_cue, S.t_don, color=GRID, lw=0, zorder=0)


def fig_example(S, d, out):
    g = S.group
    fig, ax = plt.subplots(5, 1, figsize=(11, 9.5), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2, 1, 1, 1.6, 0.9], hspace=0.25))
    cue_pool = A if d['cue'] == 'A' else B
    unc_pool = B if d['cue'] == 'A' else A
    rows = [(np.where(g == cue_pool)[0][:80], C_CUED, 'cued pool'),
            (np.where(g == unc_pool)[0][:80], C_UNCUED, 'uncued pool'),
            (np.where(g == NS)[0][:80], C_NS, 'non-selective E'),
            (np.where(g == INH)[0][:40], C_INH, 'inhibitory')]
    y0 = 0
    for idx, col, lab in rows:
        m = np.isin(d['spk_i'], idx)
        lut = np.full(len(g), -1)
        lut[idx] = np.arange(len(idx))
        ax[0].scatter(d['spk_t'][m], lut[d['spk_i'][m]] + y0, s=0.6, c=col, lw=0, rasterized=True)
        ax[0].text(S.n_t + 15, y0 + len(idx) / 2, lab, color=INK2, va='center', fontsize=8)
        y0 += len(idx) + 6
    ax[0].set_yticks([])
    ax[0].set_ylim(-3, y0)
    ax[0].set_title(f'Single trial (cue {d["cue"]}): raster, 280 of 12,500 neurons')
    sm = lambda x: uniform_filter1d(x, 10)  # noqa: E731
    ax[1].plot(S.tt, sm(d['r_cued']), color=C_CUED, label='cued pool')
    ax[1].plot(S.tt, sm(d['r_unc']), color=C_UNCUED, label='uncued pool')
    ax[1].plot(S.tt, sm(d['rates'][:S.n_t, INH]), color=C_INH, label='inhibitory', lw=0.9)
    ax[1].set_ylabel('rate (Hz)\n10 ms boxcar')
    ax[1].set_ylim(0, np.percentile(sm(d['r_cued']), 99.7) * 1.1)
    ax[1].legend(ncol=3, loc='upper right', fontsize=8)
    ax[2].plot(S.tt, d['lfp'][:S.n_t] / 1e3, color=INK, lw=0.4)
    ax[2].set_ylabel('LFP proxy\n(summed |PSP|)')
    ax[2].set_ylim(0, np.percentile(d['lfp'], 99.8) / 1e3 * 1.1)
    Z = np.log10(d['P'])
    ax[3].imshow(Z, aspect='auto', origin='lower', extent=[0, S.n_t, FREQS[0], FREQS[-1]], cmap='Blues',
                 vmin=np.percentile(Z, 5), vmax=np.percentile(Z, 99.5), rasterized=True)
    for b in d['gbursts']:
        ax[3].add_patch(plt.Rectangle((b['on'], b['f_lo']), b['dur'], b['f_hi'] - b['f_lo'] + 2,
                                      fill=False, ec=C_UNCUED, lw=1.0))
    ax[3].set_ylabel('frequency (Hz)')
    ax[3].set_title('LFP spectrogram (log power); boxes = detected gamma bursts (>2 SD, >=3 cycles)')
    ro = uniform_filter1d(d['r_cued'] - d['r_unc'], 50)
    ax[4].plot(S.tt, ro, color=C_CUED)
    ax[4].axhline(0, color=INK2, lw=0.8)
    ax[4].set_ylabel('memory readout\ncued - uncued (Hz)')
    ax[4].set_xlabel('time (ms); grey band = cue')
    for a in ax[[0, 1, 2, 4]]:
        shade_task(a, S)
    ax[4].set_xlim(0, S.n_t)
    fig.savefig(os.path.join(out, 'fig1_example_trial.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_irregular(S, d, cvs_cued, cvs_ns, rates_cued, rates_ns, spec_pre, spec_dl, f_peak, out):
    fig = plt.figure(figsize=(11, 6.8))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
    a = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 2, 41)
    a.hist(cvs_cued, bins, color=C_CUED, alpha=0.85, label=f'cued pool (mean {np.mean(cvs_cued):.2f})', density=True)
    a.hist(cvs_ns, bins, histtype='step', color=C_NS, lw=1.4, label=f'non-selective (mean {np.mean(cvs_ns):.2f})', density=True)
    a.axvline(1, color=INK2, ls=':', lw=0.8)
    a.set_xlabel('CV of ISI (delay)')
    a.set_ylabel('density')
    a.set_title('ISI variability: CV ≈ 0.6')
    a.legend(fontsize=7.5, loc='upper right', bbox_to_anchor=(1.0, 1.18))
    a = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, 50, 41)
    a.hist(rates_cued, bins, color=C_CUED, alpha=0.85, density=True, label='cued pool')
    a.hist(rates_ns, bins, histtype='step', color=C_NS, lw=1.4, density=True, label='non-selective')
    a.set_xlabel('delay firing rate (Hz)')
    a.set_title(f'... at rates far below {f_peak:.0f} Hz')
    a.legend(fontsize=7.5)
    a = fig.add_subplot(gs[0, 2])
    a.semilogy(FREQS, spec_pre, color=C_NS, label='pre-cue')
    a.semilogy(FREQS, spec_dl, color=C_CUED, label='delay')
    a.set_xlabel('frequency (Hz)')
    a.set_ylabel('LFP power (a.u.)')
    a.set_title('LFP spectrum: same gamma peak pre-cue and delay')
    a.legend(fontsize=7.5)
    # zoom: population oscillates, single neurons skip cycles
    a = fig.add_subplot(gs[1, :])
    t0 = S.DL.start + 300
    t1 = t0 + 300
    g = S.group
    cue_pool = A if d['cue'] == 'A' else B
    idx = np.where(g == cue_pool)[0][:60]
    m = np.isin(d['spk_i'], idx) & (d['spk_t'] >= t0) & (d['spk_t'] < t1)
    lut = np.full(len(g), -1)
    lut[idx] = np.arange(len(idx))
    a.scatter(d['spk_t'][m], lut[d['spk_i'][m]], s=3, c=C_CUED, lw=0)
    r = d['rates'][t0:t1, :3] @ np.bincount(g, minlength=4)[:3] / np.bincount(g, minlength=4)[:3].sum()
    a.plot(np.arange(t0, t1), 62 + r / r.max() * 30, color=INK, lw=0.9)
    a.text(t1 + 3, 77, 'E population\nrate', color=INK2, fontsize=8, va='center')
    a.text(t1 + 3, 30, '60 cued-pool\nneurons', color=INK2, fontsize=8, va='center')
    a.set_yticks([])
    a.set_xlim(t0, t1 + 45)
    a.set_xlabel('time (ms)')
    a.set_title('Delay, 300 ms zoom: the population oscillates, while each neuron fires on only a few cycles at irregular times')
    fig.savefig(os.path.join(out, 'fig2_irregular_oscillation.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_bursts(S, durs, fpk, width, burst_rate_t, gpow_t, lags, ac, summary, out):
    gbs = summary['gamma_bursts']
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.6, wspace=0.38)
    a = fig.add_subplot(gs[0, 0])
    a.hist(durs, np.arange(0, 301, 10), color=C_CUED)
    a.set_xlabel('burst duration (ms)')
    a.set_ylabel('count')
    a.set_title(f'Duration {gbs["dur_mean"]:.0f} ± {gbs["dur_sd"]:.0f} ms (n = {gbs["n"]})')
    a = fig.add_subplot(gs[0, 1])
    a.hist(fpk, np.arange(40, 102, 4), color=C_CUED)
    a.set_xlabel('burst peak frequency (Hz)')
    a.set_title(f'Peak frequency, mean {gbs["f_mean"]:.0f} Hz')
    a = fig.add_subplot(gs[0, 2])
    a.hist(width, np.arange(0, 62, 4), color=C_CUED)
    a.set_xlabel('half-power bandwidth (Hz)')
    a.set_title(f'Half-power width, mean {gbs["width_mean"]:.0f} Hz')
    # burst raster over trials
    a = fig.add_subplot(gs[1, :])
    for k, d in enumerate(S.trials):
        for b in d['gbursts']:
            a.plot([b['on'], b['off']], [k, k], color=C_CUED, lw=2.2, solid_capstyle='butt')
    shade_task(a, S)
    a.set_xlim(0, S.n_t)
    a.set_ylabel('trial')
    a.set_title(f'Detected 40-100 Hz bursts per trial (Lundqvist criterion): {gbs["per_s"]:.1f} per s of delay, {100 * gbs["frac_time"]:.1f}% of time')
    a = fig.add_subplot(gs[2, 0])
    a.plot(S.tt, uniform_filter1d(burst_rate_t, 25), color=C_CUED)
    shade_task(a, S)
    a.set_xlabel('time (ms)')
    a.set_ylabel('P(in burst)')
    a.set_title('Burst rate across trials')
    a = fig.add_subplot(gs[2, 1])
    a.plot(S.tt, uniform_filter1d(gpow_t, 25), color=INK)
    shade_task(a, S)
    a.set_xlabel('time (ms)')
    a.set_ylabel('gamma power (a.u.)')
    a.set_title(f'Trial-averaged power (r = {gbs["corr_burstrate_power"]:.2f})')
    a = fig.add_subplot(gs[2, 2])
    a.bar(lags, ac / ac.max(), width=5, color=C_CUED, lw=0)
    a.set_xlabel('lag between burst centres (ms)')
    a.set_ylabel('norm. count')
    a.set_title(f'Burst-centre autocorrelogram (CV2 = {gbs["cv2_ibi"]:.2f})')
    fig.savefig(os.path.join(out, 'fig3_gamma_bursts.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # inside vs outside bursts (separate small figure)
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.2))
    lab = ['outside\nbursts', 'inside\nbursts']
    x = np.arange(2)
    for a, key, title in [(ax[0], 'cued', 'Cued pool rate'), (ax[1], 'uncued', 'Uncued pool rate')]:
        v = [gbs[f'{key}_rate_out'], gbs[f'{key}_rate_in']]
        a.bar(x, v, width=0.55, color=C_CUED if key == 'cued' else C_UNCUED)
        for xi, vi in zip(x, v):
            a.text(xi, vi, f'{vi:.1f}', ha='center', va='bottom', color=INK, fontsize=9)
        a.set_xticks(x, lab)
        a.set_ylabel('Hz')
        a.set_title(title)
    fig.suptitle('Pool rates inside vs outside detected bursts (delay)', x=0.02, ha='left',
                 fontweight='bold', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig3b_spiking_in_bursts.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_silence(S, sil_durs, win, trig_cued, trig_unc, trig_rc, trig_ru, frac_correct_t, out):
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.38)
    a = fig.add_subplot(gs[0, 0])
    a.hist(sil_durs, np.arange(0, max(41, sil_durs.max() + 2), 1), color=C_NS)
    a.set_yscale('log')
    a.set_xlabel('silent episode duration (ms)')
    a.set_ylabel('count')
    a.set_title('Network-wide silences in the delay')
    a = fig.add_subplot(gs[0, 1])
    a.plot(win, trig_rc.mean(0), color=C_CUED, label='cued pool')
    a.plot(win, trig_ru.mean(0), color=C_UNCUED, label='uncued pool')
    a.axvline(0, color=INK2, lw=0.8, ls=':')
    a.set_xlabel('time from silence onset (ms)')
    a.set_ylabel('rate (Hz)')
    a.set_title(f'Silences ≥ 8 ms (n = {len(trig_cued)})')
    a.legend(fontsize=7.5)
    a = fig.add_subplot(gs[0, 2])
    a.plot(win, trig_cued.mean(0), color=C_CUED, label='cued pool')
    a.plot(win, trig_unc.mean(0), color=C_UNCUED, label='uncued pool')
    a.axvline(0, color=INK2, lw=0.8, ls=':')
    a.axhline(20, color=INK2, lw=0.6, ls='--')
    a.text(win[-1], 20, 'threshold', color=INK2, fontsize=7, ha='right', va='bottom')
    a.set_xlabel('time from silence onset (ms)')
    a.set_ylabel('mean membrane potential (mV)')
    a.set_title('Memory is held in sub-threshold V')
    a.legend(fontsize=7.5)
    a = fig.add_subplot(gs[1, :])
    a.plot(S.tt, frac_correct_t, color=C_CUED)
    shade_task(a, S)
    a.axhline(0.5, color=INK2, lw=0.8, ls=':')
    a.set_ylim(0, 1.05)
    a.set_xlim(0, S.n_t)
    a.set_xlabel('time (ms)')
    a.set_ylabel('fraction of trials')
    a.set_title('Full-network readout (cued − uncued pool rate, 50 ms) has the correct sign throughout the delay')
    fig.savefig(os.path.join(out, 'fig4_silence_hidden_state.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_decoding(S, conf, z, state, t_dec, dmask, c_delay, ll1, ll2, on_d, off_d, summary, out):
    dec = summary['decoding']
    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.38)
    a = fig.add_subplot(gs[0, :2])
    im = a.imshow(conf, aspect='auto', cmap='Blues', vmin=0, vmax=1, origin='lower',
                  extent=[t_dec[0] - 10, t_dec[-1] + 10, -0.5, len(conf) - 0.5], interpolation='nearest')
    a.axvline(S.t_cue, color=C_UNCUED, lw=1)
    a.axvline(S.DL.start, color=INK2, lw=0.8, ls=':')
    a.set_ylabel('trial')
    a.set_xlabel('time (ms)')
    a.set_title(f'Single-trial decoder confidence ({dec["n_neurons"]} random neurons, 60 ms windows)')
    cb = fig.colorbar(im, ax=a, fraction=0.025)
    cb.set_label('P(correct cue)')
    a = fig.add_subplot(gs[0, 2])
    a.hist(c_delay, np.linspace(0, 1, 26), color=C_CUED, density=True)
    a.set_xlabel('confidence (delay)')
    a.set_ylabel('density')
    a.set_title(f'2-state vs 1-state beta model:\n{(ll2 - ll1) * 1000:.0f} mbit/sample (CV)')
    a = fig.add_subplot(gs[1, :2])
    for j, k in enumerate(range(min(4, len(conf)))):
        y = conf[k]
        a.plot(t_dec, y + 1.2 * j, color=INK, lw=0.9)
        on = state[k] == 1
        off = state[k] == -1
        a.fill_between(t_dec, 1.2 * j, 1.2 * j + 1, where=on, color=C_CUED, alpha=0.25, lw=0, step='mid')
        a.fill_between(t_dec, 1.2 * j, 1.2 * j + 1, where=off, color=C_UNCUED, alpha=0.25, lw=0, step='mid')
        a.axhline(1.2 * j + 0.5, color=INK2, lw=0.5, ls=':')
    a.axvline(S.t_cue, color=C_UNCUED, lw=1)
    a.set_yticks([])
    a.set_xlabel('time (ms)')
    a.set_title('Example trials: decoder On (blue) and Off (orange) states')
    a = fig.add_subplot(gs[1, 2])
    bins = np.arange(0, 801, 40)
    if on_d:
        a.hist(on_d, bins, color=C_CUED, alpha=0.85, label=f'On ({np.mean(on_d):.0f} ms)')
    if off_d:
        a.hist(off_d, bins, histtype='step', color=C_UNCUED, lw=1.6, label=f'Off ({np.mean(off_d):.0f} ms)')
    a.set_xlabel('state duration (ms)')
    a.set_ylabel('count')
    a.set_title('On/Off state durations')
    a.legend(fontsize=7.5)
    fig.savefig(os.path.join(out, 'fig5_decoding_on_off.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
