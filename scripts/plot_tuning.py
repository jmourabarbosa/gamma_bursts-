"""Single-neuron tuning in the binary task: delay rate for cue A vs cue B, PSTHs of example
neurons, and the distribution of cue-explained variance (Panichello: 8.2% in On states).

usage: python scripts/plot_tuning.py --data data/session --label "Model 1" --out figures/tuning_model1.png
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_wm.network import A, B, IA, IB, INH, NS  # noqa: E402

C_A, C_B, C_NS, C_I, INK, INK2 = '#2a78d6', '#eb6834', '#52514e', '#1baf7a', '#0b0b0b', '#52514e'
plt.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'legend.frameon': False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    sdirs = sorted(glob.glob(args.data + '*'))
    meta = json.load(open(os.path.join(sdirs[0], 'meta.json')))
    group = np.load(os.path.join(sdirs[0], 'net.npz'))['group']
    T = meta['task']
    t_cue = T['t_pre']
    t_don = t_cue + T['cue_dur']
    n_t = int(t_don + T['t_delay'])
    d0, d1 = t_don + 500, n_t
    N = len(group)
    trials = []
    for sd in sdirs:
        for f in sorted(glob.glob(os.path.join(sd, 'trial*.npz'))):
            d = dict(np.load(f))
            trials.append(d)
    cue = np.array([str(d['cue']) == 'A' for d in trials], int)
    R = np.zeros((len(trials), N))
    for k, d in enumerate(trials):
        m = (d['spk_t'] >= d0) & (d['spk_t'] < d1)
        R[k] = np.bincount(d['spk_i'][m], minlength=N) / ((d1 - d0) * 1e-3)
    rA, rB = R[cue == 1].mean(0), R[cue == 0].mean(0)
    tot = R.var(0) + 1e-12
    ev = (0.5 * (rA - R.mean(0)) ** 2 + 0.5 * (rB - R.mean(0)) ** 2) / tot
    p = meta['params']

    fig = plt.figure(figsize=(12, 8.0))
    gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.32)
    a = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(0)
    for g_, c, lab in ((NS, C_NS, 'non-selective E'), (INH, C_I, 'inhibitory'), (A, C_A, 'pool A'), (B, C_B, 'pool B')):
        idx = np.where(group == g_)[0]
        idx = rng.choice(idx, min(400, len(idx)), replace=False)
        a.scatter(rB[idx], rA[idx], s=5, c=c, alpha=0.6, lw=0, label=lab)
    mx = np.percentile(np.r_[rA, rB], 99.8)
    a.plot([0, mx], [0, mx], color=INK2, lw=0.8, ls=':')
    a.set_xlim(0, mx)
    a.set_ylim(0, mx)
    a.set_xlabel('delay rate, cue B (Hz)')
    a.set_ylabel('delay rate, cue A (Hz)')
    a.set_title('Each neuron: cue A vs cue B')
    a.legend(fontsize=7.5, markerscale=2)

    a = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, 1, 41)
    for g_, c, lab in ((NS, C_NS, 'non-selective E'), (INH, C_I, 'inhibitory'), (A, C_A, 'pools A+B')):
        sel = np.isin(group, [A, B]) if g_ == A else group == g_
        v = ev[sel & (R.mean(0) > 0.5)]
        a.hist(v, bins, histtype='step', lw=1.5, color=c, density=True, label=f'{lab} (median {np.median(v):.3f})')
    a.axvline(0.082, color=INK, ls='--', lw=0.9, label='Panichello On-state mean (0.082)')
    a.set_yscale('log')
    a.set_xlabel('cue-explained variance of delay rate')
    a.set_ylabel('density')
    a.set_title('Strength of tuning')
    a.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)

    a = fig.add_subplot(gs[0, 2])
    allsel = R.mean(0) > 0.5
    frac = np.mean(ev[allsel] > 0.1)
    a.bar([0, 1, 2], [np.mean(np.isin(group, [A, B])), frac, 0.44], color=[C_A, INK2, '#eda100'], width=0.6)
    for i, v in enumerate([np.mean(np.isin(group, [A, B])), frac, 0.44]):
        a.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    a.set_xticks([0, 1, 2], ['pool cells', 'cells with\nEV > 0.1', 'Panichello\nselective'])
    a.set_ylim(0, 1)
    a.set_ylabel('fraction of all cells')
    a.set_title('How many cells are tuned')

    # PSTHs of example neurons
    ex = []
    for g_ in (A, B):
        idx = np.where(group == g_)[0]
        ex.append(idx[np.argmax(np.abs(rA - rB)[idx])])
    ns_idx = np.where((group == NS) & (R.mean(0) > 1))[0]
    o = ns_idx[np.argsort(-ev[ns_idx])]
    ex += [o[0], o[len(o) // 2]]
    tt = np.arange(0, n_t, 20)
    for j, n in enumerate(ex):
        a = fig.add_subplot(gs[1, :]) if False else fig.add_subplot(2, 4, 5 + j)
        for cv, c, lab in ((1, C_A, 'cue A'), (0, C_B, 'cue B')):
            H = np.zeros(len(tt) - 1)
            for k in np.where(cue == cv)[0]:
                d = trials[k]
                H += np.histogram(d['spk_t'][d['spk_i'] == n], tt)[0]
            H = H / (cue == cv).sum() / 0.02
            H = np.convolve(H, np.ones(5) / 5, mode='same')
            a.plot(tt[:-1] + 10, H, color=c, label=lab)
        a.axvspan(t_cue, t_don, color='#e4e3df', lw=0)
        kind = {A: 'pool A', B: 'pool B', NS: 'non-selective', INH: 'inhibitory', IA: 'I_A', IB: 'I_B'}[int(group[n])]
        a.set_title(f'{kind} #{n}: EV {ev[n]:.2f}', fontsize=9)
        a.set_xlabel('time (ms)')
        if j == 0:
            a.set_ylabel('rate (Hz, 100 ms)')
            a.legend(fontsize=7.5)
    fig.suptitle(f'{args.label}  (f = {p["f"]}, beta = {p["beta"]}, alpha = {p.get("alpha", 0)}, '
                 f'q_NMDA = {p.get("q_nmda", 0)})', x=0.01, ha='left', fontweight='bold')
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print('wrote', args.out, '| median EV pool %.3f NS %.3f | frac EV>0.1 %.3f' %
          (np.median(ev[np.isin(group, [A, B])]), np.median(ev[(group == NS) & allsel]), frac))


if __name__ == '__main__':
    main()
