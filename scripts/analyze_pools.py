"""Analysis for the pool-specific-inhibition model: each memory pool (its E cells and
their inhibitory partners) is treated as a recording site. Gamma bursts are detected
in each pool's population activity (Lundqvist et al. 2016 criterion) and compared
between the cued and the uncued pool.

usage: python scripts/analyze_pools.py --data data/poolI_a0.2_session --fig figures/pool_inhibition
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
from gamma_wm.network import A, B, IA, IB  # noqa: E402

C_CUED, C_UNCUED, C_INH, INK, INK2, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'lines.linewidth': 1.2, 'legend.frameon': False})
FREQS = np.arange(20, 121, 2.0)
GAMMA = (35, 100)


def site_signal(rates, pool):
    """Population activity of one site: E pool rate plus its inhibitory partners."""
    e, i = (A, IA) if pool == 'A' else (B, IB)
    return rates[:, e] + rates[:, i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='session directory prefix')
    ap.add_argument('--fig', required=True)
    args = ap.parse_args()
    os.makedirs(args.fig, exist_ok=True)
    sdirs = sorted(glob.glob(args.data + '*'))
    meta = json.load(open(os.path.join(sdirs[0], 'meta.json')))
    net = np.load(os.path.join(sdirs[0], 'net.npz'))
    group = net['group']
    T = meta['task']
    t_cue, t_don = T['t_pre'], T['t_pre'] + T['cue_dur']
    trials = []
    for sd in sdirs:
        for f in sorted(glob.glob(os.path.join(sd, 'trial*.npz'))):
            d = dict(np.load(f))
            d['cue'] = str(d['cue'])
            trials.append(d)
    n_t = int(t_don + T['t_delay'])
    tt = np.arange(n_t, dtype=float)
    DL = slice(int(t_don) + 500, n_t)
    dl_s = (DL.stop - DL.start) / 1000
    ref = np.ones(n_t, bool)
    ref[int(t_cue):int(t_don) + 250] = False

    for d in trials:
        cue, unc = d['cue'], ('B' if d['cue'] == 'A' else 'A')
        d['r_c'] = d['rates'][:n_t, A if cue == 'A' else B]
        d['r_u'] = d['rates'][:n_t, B if cue == 'A' else A]
        for key, pool in (('c', cue), ('u', unc)):
            x = site_signal(d['rates'][:n_t], pool).astype(float)
            P = an.morlet_power(x, FREQS)
            mu, sd = P[:, ref].mean(1, keepdims=True), P[:, ref].std(1, keepdims=True)
            b, _ = an.detect_bursts(P, FREQS, tt, mean=mu, sd=sd)
            b = [x_ for x_ in b if GAMMA[0] <= x_['f_peak'] <= GAMMA[1]]
            d['P_' + key], d['b_' + key] = P, b
            d['m_' + key] = an.burst_mask(b, tt)
        ro = uniform_filter1d(d['r_c'] - d['r_u'], 200)
        d['readout'] = ro
        wrong = np.where(ro[int(t_don) + 200:] < 0)[0]
        d['t_flip'] = (wrong[0] + int(t_don) + 200) if len(wrong) else np.nan

    S = {}
    S['n_trials'] = len(trials)
    S['params'] = meta['params']
    S['memory_correct_end'] = float(np.mean([d['readout'][-1] > 0 for d in trials]))
    S['memory_correct_whole_delay'] = float(np.mean([np.isnan(d['t_flip']) for d in trials]))
    S['rates_delay'] = dict(
        cued=float(np.mean([d['r_c'][DL].mean() for d in trials])),
        uncued=float(np.mean([d['r_u'][DL].mean() for d in trials])),
        I_cued=float(np.mean([d['rates'][DL, IA if d['cue'] == 'A' else IB].mean() for d in trials])),
        I_uncued=float(np.mean([d['rates'][DL, IB if d['cue'] == 'A' else IA].mean() for d in trials])),
        NS=float(np.mean([d['rates'][DL, 0].mean() for d in trials])))
    # only trials where the memory held throughout for burst statistics
    ok = [d for d in trials if np.isnan(d['t_flip'])]
    S['n_trials_held'] = len(ok)

    def bstats(key):
        bs = [b for d in ok for b in d['b_' + key] if DL.start <= b['t_peak'] < DL.stop]
        if not bs:
            return dict(n=0)
        du = np.array([b['dur'] for b in bs])
        return dict(n=len(bs), per_s=len(bs) / (len(ok) * dl_s), dur_mean=float(du.mean()), dur_sd=float(du.std()),
                    f_mean=float(np.mean([b['f_peak'] for b in bs])),
                    width_mean=float(np.mean([b['f_hi'] - b['f_lo'] for b in bs])),
                    frac_time=float(np.mean([d['m_' + key][DL].mean() for d in ok])),
                    cv2_ibi=float(np.nanmean([an.cv2_intervals([b['t_peak'] for b in d['b_' + key]
                                                                 if DL.start <= b['t_peak'] < DL.stop]) for d in ok])))
    S['bursts_cued_site'] = bstats('c')
    S['bursts_uncued_site'] = bstats('u')
    ins = np.concatenate([d['m_c'][DL] for d in ok])
    rc = np.concatenate([d['r_c'][DL] for d in ok])
    ru = np.concatenate([d['r_u'][DL] for d in ok])
    S['cued_rate_in_bursts'] = float(rc[ins].mean()) if ins.any() else None
    S['cued_rate_out_bursts'] = float(rc[~ins].mean())
    S['selectivity_in'] = float((rc - ru)[ins].mean()) if ins.any() else None
    S['selectivity_out'] = float((rc - ru)[~ins].mean())
    # cued-pool Off epochs: 20 ms smoothed rate below half its delay mean
    offs = []
    for d in ok:
        r20 = uniform_filter1d(d['r_c'], 20)[DL]
        m = np.r_[False, r20 < 0.5 * r20.mean(), False].astype(int)
        offs += list(np.where(np.diff(m) == -1)[0] - np.where(np.diff(m) == 1)[0])
    offs = np.array(offs)
    S['cued_off_epochs'] = dict(per_s=len(offs) / (len(ok) * dl_s), mean_ms=float(offs.mean()) if len(offs) else 0,
                                p95_ms=float(np.percentile(offs, 95)) if len(offs) else 0,
                                max_ms=float(offs.max()) if len(offs) else 0)
    # CV of ISIs in the cued pool
    cvs = []
    for d in ok:
        pool = np.where(group == (A if d['cue'] == 'A' else B))[0][:200]
        cvs += list(an.isi_cv(dict(t=d['spk_t'], i=d['spk_i']), pool, DL.start, DL.stop))
    S['cv_isi_cued'] = float(np.mean(cvs))
    json.dump(S, open(os.path.join(args.fig, 'summary.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in S.items() if k != 'params'}, indent=1))

    # ------------------------------------------------------------------ figures
    d = ok[0] if ok else trials[0]
    fig, ax = plt.subplots(5, 1, figsize=(11, 10), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2, 1, 1.4, 1.4, 0.9], hspace=0.3))
    cue = d['cue']
    rows = [(A if cue == 'A' else B, C_CUED, 'cued E'), (IA if cue == 'A' else IB, C_INH, 'cued I partners'),
            (B if cue == 'A' else A, C_UNCUED, 'uncued E'), (IB if cue == 'A' else IA, INK2, 'uncued I partners')]
    y0 = 0
    for gi, col, lab in rows:
        idx = np.where(group == gi)[0][:60]
        m = np.isin(d['spk_i'], idx)
        lut = np.full(len(group), -1)
        lut[idx] = np.arange(len(idx))
        ax[0].scatter(d['spk_t'][m], lut[d['spk_i'][m]] + y0, s=0.6, c=col, lw=0, rasterized=True)
        ax[0].text(n_t + 15, y0 + 30, lab, color=INK2, va='center', fontsize=8)
        y0 += 66
    ax[0].set_yticks([])
    ax[0].set_title(f'Single trial (cue {cue})')
    ax[1].plot(tt, uniform_filter1d(d['r_c'], 10), color=C_CUED, label='cued pool')
    ax[1].plot(tt, uniform_filter1d(d['r_u'], 10), color=C_UNCUED, label='uncued pool')
    ax[1].set_ylabel('rate (Hz)')
    ax[1].legend(ncol=2, loc='upper right', fontsize=8)
    for k, (key, lab) in enumerate([('c', 'cued site'), ('u', 'uncued site')]):
        a = ax[2 + k]
        Z = np.log10(d['P_' + key] + 1e-3)
        a.imshow(Z, aspect='auto', origin='lower', extent=[0, n_t, FREQS[0], FREQS[-1]], cmap='Blues',
                 vmin=np.percentile(np.log10(d['P_c'] + 1e-3), 20), vmax=np.percentile(np.log10(d['P_c'] + 1e-3), 99.7),
                 rasterized=True)
        for b in d['b_' + key]:
            a.add_patch(plt.Rectangle((b['on'], b['f_lo']), b['dur'], b['f_hi'] - b['f_lo'] + 2, fill=False,
                                      ec=C_UNCUED, lw=1.0))
        a.set_ylabel('Hz')
        a.set_title(f'Spectrogram, {lab} (boxes: gamma bursts)')
    ax[4].plot(tt, d['readout'], color=C_CUED)
    ax[4].axhline(0, color=INK2, lw=0.8)
    ax[4].set_ylabel('cued − uncued\n(200 ms, Hz)')
    ax[4].set_xlabel('time (ms); grey band = cue')
    for a in ax[[0, 1, 4]]:
        a.axvspan(t_cue, t_don, color=GRID, lw=0, zorder=0)
    ax[4].set_xlim(0, n_t)
    fig.savefig(os.path.join(args.fig, 'example_trial.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.38)
    a = fig.add_subplot(gs[0, :])
    br_c = uniform_filter1d(np.mean([x['m_c'] for x in ok], 0), 25)
    br_u = uniform_filter1d(np.mean([x['m_u'] for x in ok], 0), 25)
    a.plot(tt, br_c, color=C_CUED, label='cued site')
    a.plot(tt, br_u, color=C_UNCUED, label='uncued site')
    a.axvspan(t_cue, t_don, color=GRID, lw=0, zorder=0)
    a.set_xlim(0, n_t)
    a.set_ylabel('P(in gamma burst)')
    a.set_xlabel('time (ms)')
    a.legend(fontsize=8)
    a.set_title(f'Gamma burst rate by site ({len(ok)} trials where the memory held)')
    bc = [b for x in ok for b in x['b_c'] if DL.start <= b['t_peak'] < DL.stop]
    a = fig.add_subplot(gs[1, 0])
    a.hist([b['dur'] for b in bc], np.arange(0, 301, 10), color=C_CUED)
    a.set_xlabel('burst duration (ms)')
    a.set_ylabel('count')
    a.set_title('Cued-site bursts: duration')
    a = fig.add_subplot(gs[1, 1])
    a.hist([b['f_peak'] for b in bc], np.arange(GAMMA[0], GAMMA[1] + 2, 4), color=C_CUED)
    a.set_xlabel('peak frequency (Hz)')
    a.set_title('... peak frequency')
    a = fig.add_subplot(gs[1, 2])
    v = [S['cued_rate_out_bursts'], S['cued_rate_in_bursts'] or 0]
    a.bar([0, 1], v, width=0.55, color=C_CUED)
    for xi, vi in enumerate(v):
        a.text(xi, vi, f'{vi:.1f}', ha='center', va='bottom', fontsize=9)
    a.set_xticks([0, 1], ['outside\nbursts', 'inside\nbursts'])
    a.set_ylabel('cued-pool rate (Hz)')
    a.set_title('Spiking vs bursts')
    fig.savefig(os.path.join(args.fig, 'burst_stats.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
