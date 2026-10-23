"""Figure 3: time Off and memory, in the models (a, b) and in the monkeys (c, d).

Off time = fraction of the 91 delay time points (500-1,400 ms after cue onset, 10 ms apart) labelled Off x 900 ms,
for the data and for the models alike.
a, memory held against Off time in the fitted models with their own random dips (100,000 trials per model),
   NMDA and STP model (10 bins of 0.1 of the delay, 95% Wilson intervals).
b, the same for the single-dip test (one dip of 0-1 s per trial, 12,300 trials per model).
c, fraction correct against Off time (session mean +- SEM in the same bins), with the linear regression of each
   session's fraction correct on bin centre (red).
d, odds ratio of an error per within-session SD of each On/Off statistic (logistic GLMM fixed effect, 95% CI),
   with each session's weight as grey dots; stars from the likelihood-ratio p.

usage: python fig3_off_time.py <behaviour_handoff/data> <off_vs_memory dir with {nmda,fac}/{natural,single_dip}_readout_trials.npz> <out.png>
data: jmourabarbosa/panichello, branch analysis-replication, analysis/behaviour_handoff/data
models: figures/off_vs_memory on branch paper-model (paper_model/natural_readout.py, paper_model/single_dip_readout.py)
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
EDGES = np.r_[np.arange(0, 1.0, 0.1), 1.0001]           # 10 bins of the fraction Off
CENT = (EDGES[:-1] + EDGES[1:]) / 2


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


glmm = pd.read_csv(f'{DATA}/joint_glmm.csv')
sw = pd.read_csv(f'{DATA}/joint_glmm_session_weights.csv')
trials = pd.read_csv(f'{DATA}/per_trial_correct_and_error.csv')
model = {}
for sim in ('natural', 'single_dip'):
    for v in ('nmda', 'fac'):
        d = np.load(f'{MODELS}/{v}/{sim}_readout_trials.npz')
        model[sim, v] = (np.round(d['off_time_s'] / 0.01) / NPT, d['memory_held'].astype(bool))

fig = plt.figure(figsize=(6.4, 4.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], wspace=0.6, hspace=0.75, left=0.1, right=0.97, top=0.88,
                      bottom=0.1)
ma, mb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])   # models
a, b = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])     # monkeys
for row, txt in ((ma, 'Models'), (a, 'Monkeys')):
    row.text(-0.27, 1.28, txt, transform=row.transAxes, fontsize=9, weight='bold', va='bottom')

# c: fraction correct against Off time, with the regression over the non-overlapping bins
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
letter(a, 'c')

# d: odds ratio of an error per SD of each statistic
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
letter(b, 'd', x=-0.62)

# a, b: memory held against Off time in the two models, fitted dips (a) and single-dip test (b)
for ax, sim, title, L in ((ma, 'natural', 'Fitted dips', 'a'), (mb, 'single_dip', 'Single dip (0-1 s)', 'b')):
    for v, name, col in (('nmda', 'NMDA model', INK), ('fac', 'STP model', C_STP)):
        frac, held = model[sim, v]
        i = np.digitize(frac, EDGES) - 1
        k = np.array([held[i == j].sum() for j in range(len(CENT))])
        m = np.array([(i == j).sum() for j in range(len(CENT))])
        lo, hi = wilson(k, m)
        ax.fill_between(W * CENT, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(W * CENT, k / m, 'o-', color=col, ms=3, label=name)
    off_axis(ax, 'Fraction memory held')
    ax.set_title(title, fontsize=8, pad=4)
    letter(ax, L)
ma.set_ylim(0.5, 1.01); ma.set_yticks([0.5, 0.75, 1], ['0.5', '', '1'])
mb.set_ylim(0, 1.02); mb.set_yticks([0, 0.5, 1], ['0', '', '1'])
for ax in (ma, mb):
    ax.yaxis.labelpad = -6
ma.legend(loc='lower left', handlelength=1.2, fontsize=7, frameon=False)

fig.savefig(OUT, dpi=300)
fig.savefig(OUT.replace('.png', '.svg'))
print('fraction correct by bin:', np.round(mu, 3).tolist(), 'sessions per bin:', n.tolist())
print('regression: slope per 100 ms', 100 * reg.slope, 'p', reg.pvalue, 'n', ok.sum())
for sim in ('natural', 'single_dip'):
    for v in ('nmda', 'fac'):
        frac, held = model[sim, v]
        i = np.digitize(frac, EDGES) - 1
        print(sim, v, 'held', held.mean(), 'by bin:', [round(held[i == j].mean(), 3) for j in range(len(CENT))],
              'n by bin:', [int((i == j).sum()) for j in range(len(CENT))],
              'mean Off held/lost (ms):', W * frac[held].mean(), W * frac[~held].mean() if (~held).any() else None)
