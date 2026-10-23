"""Figure 3: Off states and errors in the monkeys, and the NMDA time constant fitted to their performance.

a, odds ratio of an error per within-session SD of each On/Off statistic (logistic GLMM fixed effect, 95% CI),
   with each session's weight as grey dots; stars from the likelihood-ratio p.
b, fraction correct against Off time, all 15,468 trials pooled (95% Wilson intervals), and the NMDA model with the
   best-fitting tau_N (red): P(correct) = p0 x [P(held) + G x P(lost)], G = 1/8 (a guess among the 8 cues).
c, log-likelihood of the fit against tau_N, relative to the best.
d, best tau_N in 1,000 bootstraps over sessions, with the 95% interval (dotted) and the fit to all data (dashed).
Model curves P(held | Off) for each tau_N come from the fitted NMDA model with its own dips, only tau_N changed
(paper_model/tau_sweep.py); the fit follows paper_model/fit_tau_pooled.py.

usage (from the root of branch paper-model): python paper/scripts/fig3_behaviour_tau.py <behaviour_handoff/data> <out.png>
data: jmourabarbosa/panichello, branch analysis-replication, analysis/behaviour_handoff/data
"""
import csv
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from paper_model.fit_tau_behaviour import DATA, EDGES, NB, P0, bins, model_curves, wilson  # noqa: E402
from paper_model.fit_tau_pooled import best, G, N_BOOT  # noqa: E402

GLMM, OUT = sys.argv[1:3]
C_ON, C_OFF, INK, GREY, C_FIT = '#e8474a', '#3cb4c9', '#000000', '#7f7f7f', '#d7301f'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'],
                     'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.minor.size': 1.5,
                     'xtick.direction': 'out', 'ytick.direction': 'out',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK,
                     'lines.linewidth': 0.9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def letter(ax, txt, x=-0.3, y=1.04):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=10, weight='bold', va='bottom')


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


def log_axis(ax):
    ax.set_xscale('log'); ax.set_xlim(20, 1500)
    ax.set_xticks([30, 100, 300, 1000], ['30', '100', '300', '1000'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


# data and model curves, fitted as in fit_tau_pooled.py
rows = list(csv.DictReader(open(DATA)))
b = bins(np.array([float(r['frac_off']) for r in rows]))
c = np.array([int(r['correct']) for r in rows])
sess = np.array([r['session'] for r in rows])
names = np.unique(sess)
ks = np.array([np.bincount(b[sess == s], weights=c[sess == s], minlength=NB) for s in names])
ns = np.array([np.bincount(b[sess == s], minlength=NB) for s in names])
k, n = ks.sum(0), ns.sum(0)
fine, Hf = model_curves('nmda')[2:]
i, j, ll = best(k, n, Hf)
tau, p0, prof = fine[i], P0[j], ll.max(1) - ll.max()
rng = np.random.default_rng(0)
boot = np.array([fine[best(ks[idx].sum(0), ns[idx].sum(0), Hf)[0]]
                 for idx in (rng.integers(0, len(names), len(names)) for _ in range(N_BOOT))])
med, lo95, hi95 = np.percentile(boot, (50, 2.5, 97.5))

fig = plt.figure(figsize=(7.2, 2.25))
gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1, 1, 1], wspace=0.7, left=0.13, right=0.99, top=0.84, bottom=0.21)
a, bx, cx, dx = (fig.add_subplot(gs[0, q]) for q in range(4))

# a: odds ratio of an error per SD of each statistic
glmm = pd.read_csv(f'{GLMM}/joint_glmm.csv')
sw = pd.read_csv(f'{GLMM}/joint_glmm_session_weights.csv')
stats_ = [('frac_off', 'Total Off time', C_OFF), ('off_len_ms', 'Off length', C_OFF),
          ('n_off', 'Off states', C_OFF), ('frac_on', 'Total On time', C_ON)]
jit = np.random.default_rng(0)
for q, (stat, name, col) in enumerate(stats_):
    g = glmm[(glmm.group == 'all') & (glmm.model == f'M_single_{stat}')].iloc[0]
    s = sw[(sw.model == 'single') & (sw.predictor == stat)]
    a.plot(s.session_or, q + jit.uniform(-0.18, 0.18, len(s)), '.', ms=2.5, color=GREY, alpha=0.5, mew=0)
    a.plot([g.or_ci_lo, g.or_ci_hi], [q, q], color=col, lw=1.4, solid_capstyle='butt')
    a.plot(g.odds_ratio_per_sd, q, 'o', ms=4, color=col)
    a.text(g.odds_ratio_per_sd, q - (0.22 if g.p_lrt < 0.05 else 0.36), stars(g.p_lrt), ha='center', va='center',
           fontsize=7.5)
a.axvline(1, color=GREY, ls=(0, (3, 2)), lw=0.6)
a.set_xscale('log'); a.set_xlim(0.5, 2.2)
a.set_xticks([0.5, 1, 2], ['0.5', '1', '2']); a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
a.set_yticks(range(len(stats_)), [r[1] for r in stats_]); a.set_ylim(len(stats_) - 0.5, -0.5)
a.tick_params(axis='y', length=0)
a.set_xlabel('Odds ratio of an error per SD')
letter(a, 'a', x=-0.75)

# b: fraction correct against Off time, data and best-fitting NMDA model
cen = 900 * (EDGES[:-1] + np.minimum(EDGES[1:], 1)) / 2
pk = k / n; lo, hi = wilson(k, n)
bx.plot(cen, p0 * (Hf[i] + G * (1 - Hf[i])), color=C_FIT, lw=1.2)
bx.errorbar(cen, pk, yerr=[pk - lo, hi - pk], fmt='o', ms=3, color=INK, elinewidth=0.6, capsize=0)
bx.set_xlim(0, 900); bx.set_xticks([0, 300, 600, 900]); bx.set_ylim(0.5, 1)
bx.set_yticks([0.5, 0.75, 1], ['0.5', '', '1']); bx.yaxis.labelpad = -6
bx.set_xlabel('Off time in delay (ms)'); bx.set_ylabel('Fraction correct')
bx.text(30, 0.62, 'Monkeys', fontsize=7, color=INK)
bx.text(30, 0.54, rf'NMDA model, $\tau_N$ = {tau:.0f} ms', fontsize=7, color=C_FIT)
letter(bx, 'b')

# c: likelihood profile over tau_N
cx.plot(fine, prof, color=C_FIT, lw=1.2)
log_axis(cx); cx.set_ylim(-30, 1); cx.set_yticks([-30, -20, -10, 0])
cx.set_xlabel(r'$\tau_N$ (ms)'); cx.set_ylabel('Log-likelihood\n(relative to best)')
letter(cx, 'c')

# d: bootstrap over sessions
edges = np.exp(np.linspace(np.log(20), np.log(1500), 41))
dx.hist(boot, bins=edges, color=C_FIT, alpha=0.8, lw=0)
log_axis(dx)
for x in (lo95, hi95):
    dx.axvline(x, color=C_FIT, lw=0.6, ls=(0, (1, 1.5)))
dx.axvline(tau, color=INK, lw=0.6, ls=(0, (3, 2)))
dx.set_xlabel(r'Best $\tau_N$ (ms)'); dx.set_ylabel('Bootstrap samples')
dx.text(0.62, 0.62, f'{tau:.0f} ms\n95% CI\n{lo95:.0f}-{hi95:.0f}', transform=dx.transAxes, ha='left', va='center',
        fontsize=6.5)
letter(dx, 'd')

fig.savefig(OUT, dpi=300)
print(f'best tau_N {tau:.1f} ms, p0 {p0:.3f}; bootstrap median {med:.0f}, 95% CI {lo95:.0f}-{hi95:.0f};'
      f' fraction correct by bin {np.round(pk, 3).tolist()}')
