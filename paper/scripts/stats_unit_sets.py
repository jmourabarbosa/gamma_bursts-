"""Paired tests of On/Off statistics between the k most cue-selective and k random neurons (k = 5, 20, 50), Fig. 2.

Random draws are averaged within each session; two-sided Wilcoxon signed-rank tests (and paired t-tests) over sessions.

usage (from the root of branch paper-model): python paper/scripts/stats_unit_sets.py
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

d = pd.read_csv('paper_model/data/readout_per_session_data_lam10000.csv')
g = d[d.set.isin(['top', 'random'])].groupby(['session', 'set', 'n']).mean(numeric_only=True).reset_index()
for col in ['frac_on', 'frac_off', 'off_per_trial', 'on_per_trial', 'mu_on']:
    print(col)
    for n in (5, 20, 50):
        t = g[(g.set == 'top') & (g.n == n)].set_index('session')[col]
        r = g[(g.set == 'random') & (g.n == n)].set_index('session')[col]
        t, r = t.align(r, join='inner')
        print(f'  k={n:2d}: top {t.mean():.3f} +- {t.sem():.3f}, random {r.mean():.3f} +- {r.sem():.3f}, '
              f'top > random in {(t > r).sum()}/{len(t)}, Wilcoxon p = {wilcoxon(t, r).pvalue:.1e}, '
              f'paired t p = {ttest_rel(t, r).pvalue:.1e}')
