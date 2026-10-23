"""Figure 3: time Off and memory, in the monkeys (top) and in the two models (bottom).

Off time = fraction of the 91 delay time points (500-1,400 ms after cue onset, 10 ms apart) labelled Off x 900 ms,
for the data and for the models alike.
a, fraction correct against Off time (session mean +- SEM in 10 bins of 0.1 of the delay), with the linear
   regression of each session's fraction correct on bin centre (red).
b, odds ratio of an error per within-session SD of each On/Off statistic (logistic GLMM fixed effect, 95% CI),
   with each session's weight as grey dots; stars from the likelihood-ratio p.
c, Off time of correct (grey) and error (red) trials, as a fraction of all trials (bins of 4 time points).
d, memory held against Off time in single-dip trials, NMDA and STP model (same bins as a, 95% Wilson intervals).
e, f, Off time of single-dip trials with the memory held (grey) or lost (red), STP model (e) and NMDA model (f),
   as a fraction of all trials of that model.

usage: python fig3_off_time.py <behaviour_handoff/data> <off_vs_memory dir with {nmda,fac}/single_dip_readout_trials.npz> <out.png>
data: jmourabarbosa/panichello, branch analysis-replication, analysis/behaviour_handoff/data
models: figures/off_vs_memory on branch paper-model (paper_model/single_dip_readout.py)
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

DATA, MODELS, OUT = sys.argv[1:4]
NPT, W = 91, 900          # delay time points; delay window length (ms)
C_ON, C_OFF, INK, GREY, LIGHT, C_STP = '#e8474a', '#3cb4c9', '#000000', '#7f7f7f', '#d9d9d9', '#2b83ba'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'],
                     'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK,
                     'lines.linewidth': 0.9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
EDGES = np.r_[np.arange(0, 1.0, 0.1), 1.0001]           # 10 bins of the fraction Off (a, d)
CENT = (EDGES[:-1] + EDGES[1:]) / 2
HEDGES = (np.arange(0, NPT + 2, 4) - 0.5) / NPT           # histogram bins of 4 time points (c, e, f)
HEDGES[-1] = 1.0001


def letter(ax, txt, x=-0.3, y=1.04):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=10, weight='bold', va='bottom')


def p_text(p):
    e = int(np.floor(np.log10(p)))
    return f'p = {p:.2f}' if p >= 0.01 else f'p = {p / 10 ** e:.0f}×10$^{{{e}}}$'


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def off_axis(ax, ylabel=None):
    ax.set_xlim(0, W); ax.set_xticks([0, 300, 600, 900])
    ax.set_xlabel('Off time in delay (ms)')
    if ylabel:
        ax.set_ylabel(ylabel)


def histogram(ax, frac, good, n_all, labels):
    hg, _ = np.histogram(frac[good], HEDGES)
    hb, _ = np.histogram(frac[~good], HEDGES)
    x, w = W * np.clip(HEDGES[:-1], 0, None), W * np.diff(np.clip(HEDGES, 0, 1))
    ax.bar(x, hg / n_all, width=w, align='edge', color=GREY, alpha=0.75, lw=0, label=labels[0])
    ax.bar(x, hb / n_all, width=w, align='edge', color=C_ON, alpha=0.75, lw=0, label=labels[1])
    off_axis(ax, 'Fraction of trials')
    ax.legend(loc='upper center', handlelength=1, fontsize=7, frameon=False)
    return hg, hb


glmm = pd.read_csv(f'{DATA}/joint_glmm.csv')
sw = pd.read_csv(f'{DATA}/joint_glmm_session_weights.csv')
trials = pd.read_csv(f'{DATA}/per_trial_correct_and_error.csv')
model = {}
for v in ('nmda', 'fac'):
    d = np.load(f'{MODELS}/{v}/single_dip_readout_trials.npz')
    model[v] = (np.round(d['off_time_s'] / 0.01) / NPT, d['memory_held'].astype(bool))

fig = plt.figure(figsize=(7.2, 4.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.15, 1], wspace=0.62, hspace=0.75,
                      left=0.1, right=0.98, top=0.9, bottom=0.1)
a, b, c = (fig.add_subplot(gs[0, j]) for j in range(3))
d, e, f = (fig.add_subplot(gs[1, j]) for j in range(3))
for ax, txt in ((a, 'Monkeys'), (d, 'Models')):
    ax.text(-0.42, 1.25, txt, transform=ax.transAxes, fontsize=9, weight='bold', va='bottom')

# a: fraction correct against Off time, with the regression over the non-overlapping bins
per = []
for _, t in trials.groupby('session'):
    i = np.digitize(t.frac_off, EDGES) - 1
    per.append([t.correct[i == j].mean() if (i == j).sum() >= 5 else np.nan for j in range(len(CENT))])
per = np.array(per)
mu, n = np.nanmean(per, 0), np.sum(~np.isnan(per), 0)
se = np.nanstd(per, 0, ddof=1) / np.sqrt(n)
a.fill_between(W * CENT, mu - se, mu + se, color=LIGHT, lw=0)
a.plot(W * CENT, mu, 'o-', color=INK, ms=3)
ok = ~np.isnan(per)   # one point per session and bin
reg = stats.linregress(np.broadcast_to(W * CENT, per.shape)[ok], per[ok])
a.plot([0, W], [reg.intercept, reg.intercept + reg.slope * W], color=C_ON, lw=1.2)
a.text(W * 0.97, 0.97, p_text(reg.pvalue), ha='right', va='top', fontsize=7.5, color=C_ON)
off_axis(a, 'Fraction correct')
a.set_ylim(0.5, 1); a.set_yticks([0.5, 0.75, 1], ['0.5', '', '1'])
a.yaxis.labelpad = -6
a.text(30, 0.53, 'Mean ± SEM over sessions', fontsize=7)
letter(a, 'a')

# b: odds ratio of an error per SD of each statistic
rows = [('frac_off', 'Total Off time', C_OFF), ('off_len_ms', 'Off length', C_OFF),
        ('n_off', 'Off states', C_OFF), ('frac_on', 'Fraction On', C_ON)]
rng = np.random.default_rng(0)
for k, (stat, name, col) in enumerate(rows):
    g = glmm[(glmm.group == 'all') & (glmm.model == f'M_single_{stat}')].iloc[0]
    s = sw[(sw.model == 'single') & (sw.predictor == stat)]
    b.plot(s.session_or, k + rng.uniform(-0.18, 0.18, len(s)), '.', ms=2.5, color=GREY, alpha=0.5, mew=0)
    b.plot([g.or_ci_lo, g.or_ci_hi], [k, k], color=col, lw=1.4, solid_capstyle='butt')
    b.plot(g.odds_ratio_per_sd, k, 'o', ms=4, color=col)
    b.text(g.odds_ratio_per_sd, k - (0.22 if g.p_lrt < 0.05 else 0.36), stars(g.p_lrt), ha='center', va='center',
           fontsize=7.5)
b.axvline(1, color=GREY, ls=(0, (3, 2)), lw=0.6)
b.set_xscale('log'); b.set_xlim(0.5, 2.2)
b.set_xticks([0.5, 1, 2], ['0.5', '1', '2']); b.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
b.set_yticks(range(len(rows)), [r[1] for r in rows]); b.set_ylim(len(rows) - 0.5, -0.5)
b.tick_params(axis='y', length=0)
b.set_xlabel('Odds ratio of an error per SD')
letter(b, 'b', x=-0.62)

# c: Off time of correct and error trials
correct = trials.correct.values.astype(bool)
hc, he = histogram(c, trials.frac_off.values, correct, len(trials), ('Correct', 'Error'))
c.set_title(f'{correct.sum():,} correct, {(~correct).sum():,} errors', fontsize=7.5, pad=3)
letter(c, 'c')

# d: memory held against Off time in the two models
for v, name, col in (('nmda', 'NMDA model', INK), ('fac', 'STP model', C_STP)):
    frac, held = model[v]
    i = np.digitize(frac, EDGES) - 1
    k = np.array([held[i == j].sum() for j in range(len(CENT))])
    m = np.array([(i == j).sum() for j in range(len(CENT))])
    lo, hi = wilson(k, m)
    d.fill_between(W * CENT, lo, hi, color=col, alpha=0.18, lw=0)
    d.plot(W * CENT, k / m, 'o-', color=col, ms=3, label=name)
off_axis(d, 'Fraction memory held')
d.set_ylim(0, 1.02); d.set_yticks([0, 0.5, 1], ['0', '', '1'])
d.yaxis.labelpad = -6
d.legend(loc='center right', handlelength=1.2, fontsize=7, frameon=False)
letter(d, 'd')

# e, f: Off time of model trials with the memory held or lost
for ax, v, name, L in ((e, 'fac', 'STP model', 'e'), (f, 'nmda', 'NMDA model', 'f')):
    frac, held = model[v]
    histogram(ax, frac, held, len(held), ('Memory held', 'Memory lost'))
    ax.set_title(f'{name.split()[0]}: {held.sum():,} held, {(~held).sum():,} lost', fontsize=7.5, pad=3)
    letter(ax, L)

fig.savefig(OUT, dpi=300)
fig.savefig(OUT.replace('.png', '.svg'))
print('fraction correct by bin:', np.round(mu, 3).tolist(), 'sessions per bin:', n.tolist())
print('regression: slope per 100 ms', 100 * reg.slope, 'p', reg.pvalue, 'n', ok.sum())
for v in ('nmda', 'fac'):
    frac, held = model[v]
    i = np.digitize(frac, EDGES) - 1
    print(v, 'memory held by bin:', [round(held[i == j].mean(), 3) for j in range(len(CENT))],
          'mean Off held/lost (ms):', W * frac[held].mean(), W * frac[~held].mean() if (~held).any() else None)
print('data mean Off correct/error (ms):', W * trials.frac_off[correct].mean(), W * trials.frac_off[~correct].mean())
