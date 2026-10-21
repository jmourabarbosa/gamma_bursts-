"""Diagram of the mean-field model and fit: components, parameters, what is fitted and what is not.

usage: python scripts/mf_diagram.py [--res data/mf_refit_best_fresh.json]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanfield import model as M  # noqa: E402

INK, INK2, BG = '#0b0b0b', '#52514e', '#fcfcfb'
KIND = {  # fill, edge
    'fit': ('#fbe1d6', '#c9521f'),
    'fixed': ('#eeedea', '#8a897f'),
    'data': ('#d7f0e6', '#13865c'),
    'anchor': ('#e6e2fa', '#5242c0'),
}
plt.rcParams.update({'figure.facecolor': BG, 'savefig.facecolor': BG, 'font.size': 8})


def box(ax, x, y, w, h, kind, title, lines, fs=7.2):
    f, e = KIND[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.012', fc=f, ec=e, lw=1.2))
    ax.text(x + 0.008, y + h - 0.012, title, fontsize=fs + 0.6, weight='bold', va='top', color=INK)
    ax.text(x + 0.008, y + h - 0.034, '\n'.join(lines), fontsize=fs, va='top', color=INK, linespacing=1.35)


def arrow(ax, p, q, color=INK2, style='-|>', rad=0.0, lw=1.2, ls='-'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=2, shrinkB=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default='data/mf_refit_best_fresh.json')
    a = ap.parse_args()
    r = json.load(open(a.res))
    th = r['theta']
    S = M.STRUCT
    gains = {k: v['gain'] for k, v in r['sets'].items()}

    fig = plt.figure(figsize=(11, 8.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(0.02, 0.975, 'Mean-field model of decoder On/Off states: components, parameters, and what is fitted',
            fontsize=11, weight='bold', va='top')
    ax.text(0.02, 0.948, 'Best refit to the recorded read-out table (λ = 10,000; 7 unit sets). Box colour = how each '
            'parameter is set.', fontsize=8, color=INK2, va='top')

    # ---------------- column headers
    for x, t in ((0.02, '1  Shared input'), (0.27, '2  Memory circuit (rate model)'), (0.67, '3  Read-out (as the data analysis)')):
        ax.text(x, 0.905, t, fontsize=9.5, weight='bold', color=INK)

    # ---------------- 1. input
    box(ax, 0.02, 0.62, 0.21, 0.27, 'fit', 'Input dips  m(t)', [
        'Drive × m(t); m = 1 − depth in a dip,',
        '10 ms ramps. Same generator as the',
        'spiking network (task.shared_dips).',
        '',
        f'dip rate      {th["dip_rate"]:.1f} /s   (Poisson)',
        f'dip length  {th["dip_mean"]:.0f} ms mean',
        f'dip shape   k = {th["dip_shape"]:.1f}  (gamma)',
        f'dip depth   {100 * th["dip_depth"]:.0f}%'])
    # small dip trace
    ix = fig.add_axes([0.035, 0.635, 0.18, 0.05]); ix.axis('off')
    rng = np.random.default_rng(3)
    m = M.shared_dips(1, 1500, 1.0, th['dip_rate'], th['dip_mean'], th['dip_depth'], th['dip_shape'], rng)[0]
    ix.plot(m, color=KIND['fit'][1], lw=1); ix.set_ylim(-0.05, 1.1)

    box(ax, 0.02, 0.43, 0.21, 0.165, 'fixed', 'Drive and feed-forward inhibition', [
        f'external drive to pools  I₀ = {S["I_0"]:.2f}',
        f'feed-forward inhibition  w_ff = {S["w_ff"]:.1f},',
        f'   lags the drive by τ_ff = {S["tau_ff"]:.0f} ms',
        '→ after a dip, excitation returns first:',
        '   a brief kick to both pools (restart)'])

    # ---------------- 2. circuit
    cx_a, cx_b, cy = 0.355, 0.535, 0.735
    for cx, lab in ((cx_a, 'A'), (cx_b, 'B')):
        ax.add_patch(Circle((cx, cy), 0.052, fc=KIND['fixed'][0], ec=INK, lw=1.4))
        ax.text(cx, cy + 0.012, f'pool {lab}', ha='center', fontsize=9, weight='bold')
        ax.text(cx, cy - 0.016, 'rate r, NMDA s', ha='center', fontsize=7, color=INK2)
        arrow(ax, (cx - 0.03, cy + 0.05), (cx + 0.03, cy + 0.05), color=KIND['fit'][1], rad=-1.6, lw=1.6)
    ax.text((cx_a + cx_b) / 2, cy + 0.118, 'self-excitation: NMDA (slow) + AMPA (fast)', ha='center', fontsize=7.2, color=INK2)
    # inhibition
    ix_, iy = (cx_a + cx_b) / 2, 0.635
    ax.add_patch(Circle((ix_, iy), 0.035, fc='#dfe9f7', ec=INK, lw=1.2))
    ax.text(ix_, iy, 'I', ha='center', va='center', fontsize=10, weight='bold')
    for cx in (cx_a, cx_b):
        arrow(ax, (cx + (0.02 if cx < ix_ else -0.02), cy - 0.05), (ix_ + (-0.022 if cx < ix_ else 0.022), iy + 0.028), color=INK2)
        arrow(ax, (ix_ + (-0.03 if cx < ix_ else 0.03), iy + 0.012), (cx + (0.04 if cx < ix_ else -0.04), cy - 0.035),
              color='#2a78d6', style='-[', rad=0.25 if cx < ix_ else -0.25)
    ax.text(ix_ + 0.045, iy - 0.005, 'shared inhibition\n→ winner-take-all', ha='left', va='center', fontsize=7, color=INK2)
    # input arrows into pools
    arrow(ax, (0.23, 0.79), (cx_a - 0.055, cy + 0.01), color=KIND['fit'][1], lw=1.4)
    arrow(ax, (0.23, 0.79), (cx_b - 0.03, cy + 0.045), color=KIND['fit'][1], lw=1.4, rad=-0.25)
    arrow(ax, (0.23, 0.53), (ix_ - 0.036, iy), color=KIND['fixed'][1], lw=1.2, ls='--', rad=-0.15)
    # cue
    ax.text(cx_a - 0.095, cy - 0.065, 'cue (50 ms)', fontsize=7, color=INK2)
    arrow(ax, (cx_a - 0.06, cy - 0.052), (cx_a - 0.045, cy - 0.03), color=INK2)
    # noise
    for cx in (cx_a, cx_b):
        ax.text(cx, cy - 0.085, 'noise σ', ha='center', fontsize=7, color=KIND['fit'][1])
        arrow(ax, (cx, cy - 0.075), (cx, cy - 0.055), color=KIND['fit'][1])
    # parameter boxes
    box(ax, 0.255, 0.405, 0.16, 0.16, 'fit', 'Fitted (7 parameters)', [
        f'τ_N   NMDA decay  {th["tau_N"]:.0f} ms',
        '        (bound 50–150 ms)',
        f'w_s   NMDA strength  {th["w_s"]:.2f}',
        f'σ      fast noise  {th["sigma"]:.3f}',
        '+ the 4 dip parameters (left)'])
    box(ax, 0.425, 0.405, 0.225, 0.16, 'fixed', 'Fixed by design (not fitted)', [
        f'sigmoid: max {S["r_max"]:.0f} Hz, slope {S["kappa"]}',
        f'AMPA self-excitation w_a = {S["w_a"]}',
        f'NMDA saturation β = {S["beta"]} /Hz',
        f'shared inhibition {S["w_I"] * S["w_EI"]:.2f} /Hz, τ_I {S["tau_I"]:.0f} ms',
        f'τ_r = {S["tau_r"]:.0f} ms; noise corr. time {S["tau_xi"]:.0f} ms',
        '→ low 1 Hz / cued 110 Hz; saddle 7 Hz'])
    ax.text(0.255, 0.388, 'Background E (~5 Hz, follows the drive) is simulated but not used by the read-out.',
            fontsize=7, color=INK2)

    # ---------------- 3. read-out
    box(ax, 0.67, 0.735, 0.31, 0.155, 'fixed', 'Signal', [
        'Δr(t) = cued − uncued pool rate,',
        '100 ms windows every 10 ms',
        '',
        'z(t) = c · Δr(t) + noise',
        'z ≈ the decoder\'s z against the shuffled null'])
    box(ax, 0.67, 0.585, 0.148, 0.135, 'anchor', 'Gain c, per unit set', [
        'set so the model\'s μ_on', '= the data\'s μ_on', '(2-Gaussian mixture of',
        'delay z, as in the data)', f'top 20: {gains["top/20"]:.3f} z/Hz'])
    box(ax, 0.832, 0.585, 0.148, 0.135, 'data', 'Noise (measured)', [
        'unit variance;', '100 ms window overlap;', 'slow per-trial offset:',
        'ρ = 0.09 top, 0.11 random,', '0.35 all units'])
    box(ax, 0.67, 0.415, 0.31, 0.155, 'data', 'State labelling (paper rules)', [
        'On:  z > 1.96, cluster mass above the 95th',
        '       percentile of pure-noise clusters',
        'Off: ≥ 5 points (50 ms) with z < 0.84',
        'counted 0–1,400 ms after cue onset;',
        'accuracy = P(z > 0), 500–1,400 ms'])
    arrow(ax, (cx_b + 0.055, cy), (0.67, 0.81), color=INK2, lw=1.4)
    arrow(ax, (0.745, 0.735), (0.745, 0.72), color=INK2)
    arrow(ax, (0.905, 0.735), (0.905, 0.72), color=INK2)
    arrow(ax, (0.825, 0.585), (0.825, 0.57), color=INK2)

    # ---------------- 4. fit
    ax.plot([0.02, 0.98], [0.372, 0.372], color='#d8d7d2', lw=1)
    ax.text(0.02, 0.36, '4  Fit', fontsize=9.5, weight='bold')
    box(ax, 0.02, 0.06, 0.31, 0.285, 'data', 'Targets (fitted to)', [
        'Recorded read-out table, λ = 10,000, 25 sessions,',
        'for 7 unit sets: top 5/20/50, random 5/20/50, all:',
        '  • Off per trial, Off mean length, fraction Off',
        '  • On per trial, On mean length, fraction On',
        '  (each scaled: 0.3 per trial, 20 ms, 0.05)',
        '• top-20 Off and On length histograms',
        '  (20 ms bins, 40–600 ms; Hellinger² × 100)',
        '',
        'Constraint: memory correct at the end of the',
        'delay ≥ 98% (≥ 95% checked on fresh seeds)'])
    box(ax, 0.345, 0.06, 0.3, 0.285, 'fixed', 'Not fitted (checks)', [
        '• decoding accuracy per unit set',
        '   → model ~0.1 too high in every set',
        '• accuracy within On / Off / unlabelled,',
        '   confidently-incorrect periods',
        '   → data have 0.77 wrong-cue periods per trial',
        '      (top 20); model has none',
        '• histograms of the other 6 unit sets',
        '• population rates during Off (data: untuned',
        '   units unchanged; model: everything silent)'])
    box(ax, 0.66, 0.06, 0.32, 0.285, 'fixed', 'Method and result', [
        'CMA-ES on the 7 fitted parameters, 4 runs,',
        '1,000 evaluations each; 792 trials per',
        'evaluation, same seeds within a run',
        '',
        f'best loss {r["loss"]:.1f} (fresh seeds),  memory {100 * r["mem"]:.1f}%',
        'runs agree: τ_N 115–129 ms, w_s 1.63–1.66,',
        '   dip length 122–126 ms; dip rate 8.8–11/s,',
        '   shape 4.6–8.6 and depth 81–90% looser',
        '',
        'Code: meanfield/model.py, refit.py,',
        '      scripts/mf_refit.py'])

    # legend
    for lx, (kind, lab) in zip((0.02, 0.12, 0.33, 0.47), (('fit', 'fitted'), ('fixed', 'fixed by design / not fitted'),
                                                          ('data', 'from the data'), ('anchor', 'anchored per unit set'))):
        f, e = KIND[kind]
        ax.add_patch(FancyBboxPatch((lx, 0.015), 0.018, 0.018, boxstyle='round,pad=0.002', fc=f, ec=e, lw=1.2))
        ax.text(lx + 0.025, 0.024, lab, fontsize=7.5, va='center')
    out = os.path.join(os.path.dirname(__file__), '..', 'figures', 'meanfield', 'fig0_model_diagram.png')
    fig.savefig(out, dpi=170)
    print('wrote', out)


if __name__ == '__main__':
    main()
