"""Proposed Figure 3: Off states and behaviour, in the style of the paper figures (paper_model/make_figures.py).

a, odds ratio of an error per within-session SD of each On/Off statistic (logistic GLMM fixed effect, 95% CI),
   with each session's weight as grey dots.
b, fraction correct against Off time in the delay (session mean +- SEM in 0.1 bins), with the trial-level
   linear regression of correct (0/1) on Off time (red).

usage: python fig3_behaviour.py <behaviour_handoff/data> <out.png>
data: jmourabarbosa/panichello, branch analysis-replication, analysis/behaviour_handoff/data
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

DATA, OUT = sys.argv[1:3]
W = 910   # ms: delay window 500-1,400 ms after cue onset, 91 points 10 ms apart; Off time = fraction Off x W
C_ON, C_OFF, INK, GREY, LIGHT = '#e8474a', '#3cb4c9', '#000000', '#7f7f7f', '#d9d9d9'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'],
                     'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK,
                     'lines.linewidth': 0.9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def letter(ax, txt, x=-0.3, y=1.04):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=10, weight='bold', va='bottom')


def p_text(p):
    e = int(np.floor(np.log10(p)))
    return f'p = {p:.2f}' if p >= 0.01 else f'p = {p / 10 ** e:.0f}×10$^{{{e}}}$'


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


glmm = pd.read_csv(f'{DATA}/joint_glmm.csv')
sw = pd.read_csv(f'{DATA}/joint_glmm_session_weights.csv')
trials = pd.read_csv(f'{DATA}/per_trial_correct_and_error.csv')

fig, (b, c) = plt.subplots(1, 2, figsize=(4.8, 2.1), gridspec_kw=dict(width_ratios=[1.25, 1.1], wspace=0.5,
                                                                     left=0.2, right=0.98, top=0.88, bottom=0.2))

# a: odds ratio of an error per SD of each statistic
rows = [('frac_off', 'Total Off time', C_OFF), ('off_len_ms', 'Off length', C_OFF),
        ('n_off', 'Off states', C_OFF), ('frac_on', 'Fraction On', C_ON)]
rng = np.random.default_rng(0)
for k, (stat, name, col) in enumerate(rows):
    g = glmm[(glmm.group == 'all') & (glmm.model == f'M_single_{stat}')].iloc[0]
    s = sw[(sw.model == 'single') & (sw.predictor == stat)]
    b.plot(s.session_or, k + rng.uniform(-0.18, 0.18, len(s)), '.', ms=2.5, color=GREY, alpha=0.5, mew=0)
    b.plot([g.or_ci_lo, g.or_ci_hi], [k, k], color=col, lw=1.4, solid_capstyle='butt')
    b.plot(g.odds_ratio_per_sd, k, 'o', ms=4, color=col)
    b.text(2.3, k + (0.1 if g.p_lrt < 0.05 else 0), stars(g.p_lrt), va='center', fontsize=7.5)
b.axvline(1, color=GREY, ls=(0, (3, 2)), lw=0.6)
b.set_xscale('log'); b.set_xlim(0.5, 2.2)
b.set_xticks([0.5, 1, 2], ['0.5', '1', '2']); b.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
b.set_yticks(range(len(rows)), [r[1] for r in rows]); b.set_ylim(len(rows) - 0.5, -0.5)
b.tick_params(axis='y', length=0)
b.set_xlabel('Odds ratio of an error per SD')
letter(b, 'a', x=-0.6)

# b: fraction correct against fraction of the delay Off
edges = np.r_[np.arange(0, 1.0, 0.1), 1.0001]
cent = (edges[:-1] + edges[1:]) / 2
per = []
for _, t in trials.groupby('session'):
    i = np.digitize(t.frac_off, edges) - 1
    v = [t.correct[i == j].mean() if (i == j).sum() >= 5 else np.nan for j in range(len(cent))]
    per.append(v)
per = np.array(per)
mu, n = np.nanmean(per, 0), np.sum(~np.isnan(per), 0)
se = np.nanstd(per, 0, ddof=1) / np.sqrt(n)
c.fill_between(W * cent, mu - se, mu + se, color=LIGHT, lw=0)
c.plot(W * cent, mu, 'o-', color=INK, ms=3)
reg = stats.linregress(W * trials.frac_off, trials.correct)
c.plot([0, W], [reg.intercept, reg.intercept + reg.slope * W], color=C_ON, lw=1.2)
c.text(W * 0.97, 0.97, p_text(reg.pvalue), ha='right', va='top', fontsize=7.5, color=C_ON)
c.set_xlim(0, W); c.set_xticks([0, 300, 600, 900])
c.set_ylim(0.5, 1); c.set_yticks([0.5, 0.75, 1], ['0.5', '', '1'])
c.set_xlabel('Off time in delay (ms)'); c.set_ylabel('Fraction correct', labelpad=-6)
c.text(30, 0.53, 'Mean ± SEM over sessions', fontsize=7, color=INK)
letter(c, 'b')

fig.savefig(OUT, dpi=300)
fig.savefig(OUT.replace('.png', '.pdf'))
print('per-bin n sessions:', n.tolist())
print('fraction correct by bin:', np.round(mu, 3).tolist())
print('regression: slope per 100 ms', 100 * reg.slope, 'intercept', reg.intercept, 'p', reg.pvalue, 'n', len(trials))
