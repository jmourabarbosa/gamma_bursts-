"""Logistic regression of memory loss on each On/Off statistic in the fitted models, as the monkeys' GLMM (Fig. 3c).

Data analysis (behaviour_handoff): error ~ z(stat) + t + t^2 + monkey + (1 + z(stat) || session), one model per
statistic, trials with >= 1 Off period, z-scored within session. Model version: error (memory lost) ~ z(stat), one
model per statistic, trials with >= 1 Off period, z-scored over trials (no sessions, monkeys or time in session).
Trials: each fitted model with its own dips (natural_readout.py, 100,000 trials per model). Odds ratio per SD with
95% Wald interval; p from the likelihood-ratio test against the intercept-only model.
Output: figures/paper_model/fig_model_glm.{png,svg} and fig_model_glm.csv.
Also at the data's sample size: N_DRAWS random draws of N_DATA = 15,468 model trials (the number of trials in the data),
refitted; median odds ratio with the 2.5-97.5 percentiles over draws, and the fraction of draws with p_LRT < 0.05
(fig_model_glm_matched.{png,svg,csv}).

usage: python -m paper_model.model_glm
"""
import csv
import os

import numpy as np
from scipy import stats

from .make_figures import plt, save, INK, GREY, C_ON, C_OFF, OUT

MODELS = (('nmda', 'NMDA model', INK), ('fac', 'STP model', '#2b83ba'))
ROWS = (('frac_off', 'Total Off time', C_OFF), ('off_len_ms', 'Off length', C_OFF),
        ('n_off', 'Off states', C_OFF), ('frac_on', 'Total On time', C_ON))


def logit_fit(X, y, n_iter=50):
    """Maximum-likelihood logistic regression (IRLS). Returns coefficients, their SEs and the log-likelihood."""
    b = np.zeros(X.shape[1])
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-15, 1 - 1e-15)
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return b, np.sqrt(np.diag(np.linalg.inv(H))), ll


def fit(v, stat):
    d = np.load(os.path.join('figures', 'off_vs_memory', v, 'natural_readout_trials.npz'))
    keep = d['n_off'] >= 1
    x, err = d[stat][keep], (~d['memory_held'].astype(bool))[keep].astype(float)
    z = (x - x.mean()) / x.std()
    b, se, ll = logit_fit(np.c_[np.ones_like(z), z], err)
    p0 = err.mean()
    ll0 = len(err) * (p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))
    p_lrt = stats.chi2.sf(2 * (ll - ll0), 1)
    return dict(n_trials=int(keep.sum()), n_errors=int(err.sum()), odds_ratio_per_sd=float(np.exp(b[1])),
                or_ci_lo=float(np.exp(b[1] - 1.96 * se[1])), or_ci_hi=float(np.exp(b[1] + 1.96 * se[1])),
                p_lrt=float(p_lrt), sd=float(x.std()))


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


def main():
    res = [dict(model=name, statistic=stat, **fit(v, stat)) for v, name, _ in MODELS for stat, _, _ in ROWS]
    fig, ax = plt.subplots(figsize=(2.6, 2.2))
    fig.subplots_adjust(left=0.36, right=0.96, bottom=0.2, top=0.88)
    for j, (v, name, col) in enumerate(MODELS):
        for k, (stat, lab, _) in enumerate(ROWS):
            r = next(r for r in res if r['model'] == name and r['statistic'] == stat)
            y = k + (j - 0.5) * 0.36
            ax.plot([r['or_ci_lo'], r['or_ci_hi']], [y, y], color=col, lw=1.4, solid_capstyle='butt')
            ax.plot(r['odds_ratio_per_sd'], y, 'o', ms=4, color=col, label=name if k == 0 else None)
            ax.text(min(r['or_ci_hi'], 9) * 1.08, y, stars(r['p_lrt']), va='center', fontsize=6.5, color=col)
    ax.axvline(1, color=GREY, ls=(0, (3, 2)), lw=0.6)
    ax.set_xscale('log')
    ax.set_xlim(0.1, 10)
    ax.set_xticks([0.1, 0.3, 1, 3, 10], ['0.1', '0.3', '1', '3', '10'])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks(range(len(ROWS)), [r[1] for r in ROWS]); ax.set_ylim(len(ROWS) - 0.5, -0.5)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Odds ratio of memory loss per SD')
    ax.legend(loc='lower center', bbox_to_anchor=(0.4, 1.0), ncol=2, handlelength=0.8, fontsize=6.5, columnspacing=0.8)
    save(fig, os.path.join(OUT, 'fig_model_glm.png'))
    with open(os.path.join(OUT, 'fig_model_glm.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, list(res[0]))
        w.writeheader()
        w.writerows(res)
    for r in res:
        print(f"{r['model']:10s} {r['statistic']:10s} n {r['n_trials']} errors {r['n_errors']:5d} OR {r['odds_ratio_per_sd']:.2f}"
              f" ({r['or_ci_lo']:.2f}-{r['or_ci_hi']:.2f}) p_LRT {r['p_lrt']:.1e}")




# ---- the same regression at the data's sample size ----
N_DATA, N_DRAWS = 15468, 1000      # trials in the data (13,962 of them with >= 1 Off period); random draws per model


def fit_arrays(x, err):
    """As fit(), on given arrays; returns (odds ratio per SD, p_LRT), or (nan, 1) without errors."""
    if err.sum() == 0:
        return np.nan, 1.0
    z = (x - x.mean()) / x.std()
    try:
        b, _, ll = logit_fit(np.c_[np.ones_like(z), z], err, n_iter=30)
    except np.linalg.LinAlgError:
        return np.nan, 1.0
    p0 = err.mean()
    ll0 = len(err) * (p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))
    return float(np.exp(np.clip(b[1], -20, 20))), float(stats.chi2.sf(max(2 * (ll - ll0), 0), 1))


def matched():
    """Draw N_DATA trials (without replacement) N_DRAWS times per model and refit each statistic."""
    rng = np.random.default_rng(3)
    out = []
    for v, name, _ in MODELS:
        d = np.load(os.path.join('figures', 'off_vs_memory', v, 'natural_readout_trials.npz'))
        lost = ~d['memory_held'].astype(bool)
        draws = [rng.choice(len(lost), N_DATA, replace=False) for _ in range(N_DRAWS)]
        for stat, _, _ in ROWS:
            r = []
            for idx in draws:
                idx = idx[d['n_off'][idx] >= 1]
                r.append((*fit_arrays(d[stat][idx], lost[idx].astype(float)), lost[idx].sum(), len(idx)))
            r = np.array(r)
            ok = ~np.isnan(r[:, 0])
            q = np.percentile(r[ok, 0], (2.5, 50, 97.5)) if ok.any() else (np.nan,) * 3
            out.append(dict(model=name, statistic=stat, n_trials=int(np.median(r[:, 3])), errors_median=float(np.median(r[:, 2])),
                            errors_lo=float(np.percentile(r[:, 2], 2.5)), errors_hi=float(np.percentile(r[:, 2], 97.5)),
                            draws_without_errors=float((r[:, 2] == 0).mean()), or_median=float(q[1]),
                            or_lo=float(q[0]), or_hi=float(q[2]), frac_significant=float((r[:, 1] < 0.05).mean())))
    return out


def matched_figure():
    res = matched()
    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    fig.subplots_adjust(left=0.31, right=0.8, bottom=0.2, top=0.88)
    for j, (v, name, col) in enumerate(MODELS):
        for k, (stat, lab, _) in enumerate(ROWS):
            r = next(r for r in res if r['model'] == name and r['statistic'] == stat)
            y = k + (j - 0.5) * 0.36
            if not np.isnan(r['or_median']):
                ax.plot([max(r['or_lo'], 0.1), min(r['or_hi'], 10)], [y, y], color=col, lw=1.4, solid_capstyle='butt')
                ax.plot(min(max(r['or_median'], 0.1), 10), y, 'o', ms=4, color=col, label=name if k == 0 else None)
            ax.text(1.02, y, f"{100 * r['frac_significant']:.0f}%", transform=ax.get_yaxis_transform(), va='center',
                    fontsize=6.5, color=col)
    ax.text(1.02, -0.75, 'p < 0.05', transform=ax.get_yaxis_transform(), va='center', fontsize=6.5)
    ax.axvline(1, color=GREY, ls=(0, (3, 2)), lw=0.6)
    ax.set_xscale('log'); ax.set_xlim(0.1, 10)
    ax.set_xticks([0.1, 0.3, 1, 3, 10], ['0.1', '0.3', '1', '3', '10'])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks(range(len(ROWS)), [r[1] for r in ROWS]); ax.set_ylim(len(ROWS) - 0.5, -0.5)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Odds ratio of memory loss per SD')
    ax.legend(loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=2, handlelength=0.8, fontsize=6.5, columnspacing=0.8)
    save(fig, os.path.join(OUT, 'fig_model_glm_matched.png'))
    with open(os.path.join(OUT, 'fig_model_glm_matched.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, list(res[0]))
        w.writeheader()
        w.writerows(res)
    for r in res:
        print(f"{r['model']:10s} {r['statistic']:10s} errors {r['errors_median']:.0f} ({r['errors_lo']:.0f}-{r['errors_hi']:.0f}),"
              f" no errors in {100 * r['draws_without_errors']:.0f}% | OR {r['or_median']:.2f} ({r['or_lo']:.2f}-{r['or_hi']:.2f})"
              f" | p<0.05 in {100 * r['frac_significant']:.0f}%")


if __name__ == '__main__':
    main()
    matched_figure()
