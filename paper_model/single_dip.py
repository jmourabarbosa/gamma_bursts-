"""Memory vs the length of a single input dip, in the model fitted to all units.

No random dips: every trial gets one dip of fixed length L (0-1,000 ms) starting 500 ms after cue onset, with the
fitted depth (and, for comparison, a full dip, depth 1), 10 ms ramps as in model.common_input, and the fitted noise.
Memory held = NMDA criterion at the end of the delay (model.memory_correct).

usage: python -m paper_model.single_dip
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import model as M

OUT = os.path.join('figures', 'off_vs_memory')
ONSET, N_TR = 500.0, 300
LENGTHS = np.arange(0, 1001, 25)


def simulate_dip(p, length, depth, n_tr, seed, ramp=10.0):
    rng = np.random.default_rng(seed)
    dt = p['dt']
    n = int(round((p['t_pre'] + p['cue_dur'] + p['t_delay']) / dt))
    tt = np.arange(n) * dt - p['t_pre']
    box = np.where((tt >= ONSET) & (tt < ONSET + length), 1 - depth, 1.0)
    w = int(ramp / dt)
    m = np.convolve(np.r_[np.ones(w), box, np.ones(w)], np.ones(w) / w, mode='same')[w:-w]
    m = np.broadcast_to(m, (n_tr, n)).copy()
    cue = ((tt >= 0) & (tt < p['cue_dur'])).astype(float)
    xi = M._ou(rng.standard_normal((n_tr, n, 2)), np.exp(-dt / p['tau_xi']))
    rec = int(round(1 / dt))
    o = M._sim(m, cue, xi, np.array([float(p[k]) for k in M.KEYS]), rec)
    return dict(rc=o[..., 0], ru=o[..., 1], sc=o[..., 2], su=o[..., 3], m=m[:, rec - 1::rec],
                t=np.arange(n // rec) - p['t_pre'])


def main():
    p = M.params()
    os.makedirs(OUT, exist_ok=True)
    res = {}
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw=dict(width_ratios=(1.3, 1)))
    for depth, col in ((p['dip_depth'], 'k'), (1.0, 'C3')):
        mem, swap, none = [], [], []
        for i, L in enumerate(LENGTHS):
            s = simulate_dip(p, L, depth, N_TR, seed=1000 * int(depth * 100) + i)
            d = (s['sc'][:, -100:] - s['su'][:, -100:]).mean(1)
            mem.append((d > 0.05).mean()); swap.append((d < -0.05).mean()); none.append((np.abs(d) <= 0.05).mean())
        res[f'depth {depth:.2f}'] = dict(length_ms=LENGTHS.tolist(), memory=mem, other_item=swap, neither=none)
        ax[0].plot(LENGTHS / 1000, mem, '.-', color=col, ms=3, lw=1, label=f'memory held, depth {depth:.2f}')
        ax[0].plot(LENGTHS / 1000, swap, ':', color=col, lw=1, label=f'uncued item held, depth {depth:.2f}')
    ax[0].set_xlabel('dip length (s)'); ax[0].set_ylabel('fraction of trials'); ax[0].set_ylim(-0.02, 1.02)
    ax[0].legend(fontsize=7); ax[0].set_title(f'One dip, {ONSET:.0f} ms after cue onset; {N_TR} trials per length', fontsize=9)
    for L, a in zip((100, 300, 600), (0.9, 0.6, 0.3)):
        s = simulate_dip(p, L, p['dip_depth'], 1, seed=7)
        ax[1].plot(s['t'], s['sc'][0] - s['su'][0], color=str(1 - a), lw=1, label=f'{L} ms dip')
    ax[1].axhline(0.05, color='C3', lw=0.6, ls='--')
    ax[1].set_xlabel('time from cue onset (ms)'); ax[1].set_ylabel('NMDA cued - uncued'); ax[1].legend(fontsize=7)
    ax[1].set_title(f'example trials, depth {p["dip_depth"]:.2f}', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'memory_vs_single_dip_length.png'), dpi=200)
    json.dump(res, open(os.path.join(OUT, 'single_dip.json'), 'w'), indent=1)
    for k, v in res.items():
        print(k, ' '.join(f"{L}:{m:.2f}/{w:.2f}" for L, m, w in zip(v['length_ms'], v['memory'], v['other_item'])))


if __name__ == '__main__':
    main()
