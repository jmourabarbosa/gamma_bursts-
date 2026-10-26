"""Normalise Procrustes shape similarity by its label-shuffle chance level:

    normalised = 1 - disparity / shuffle_disparity = (similarity - shuffle_mean) / (1 - shuffle_mean)

i.e. the distance (disparity) of the centroids from the ring expressed as a fraction of the distance obtained with
shuffled cue labels. 0 = chance, 1 = identical shape, < 0 = worse than chance. This puts spaces with different chance
levels (7 PCs, all units) and recordings/simulations on one scale. The shuffle 95th percentile is normalised the same way.

Usage: python procrustes_normalize.py procrustes_sessions.csv OUT_DIR [PLOT_PNG]
Writes procrustes_sessions_normalised.csv (adds `normalised`, `normalised_p95`) and procrustes_stats_normalised.json;
with PLOT_PNG also a quick check plot (pcs and full space; 0 = chance, so no shuffle lines are needed).
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp

PER = ["on", "off", "pre"]


def normalise(D):
    D = D.copy()
    D["normalised"] = (D.similarity - D.shuffle_mean) / (1 - D.shuffle_mean)
    D["normalised_p95"] = (D.shuffle_p95 - D.shuffle_mean) / (1 - D.shuffle_mean)
    return D


def stats(D):
    out = {}
    for space, g in D.groupby("space"):
        N = g.pivot(index="session", columns="period", values="normalised")[PER]
        N95 = g.pivot(index="session", columns="period", values="normalised_p95")[PER]
        p2 = lambda a, b: (lambda r: dict(t=float(r.statistic), p=float(r.pvalue), n=int((a.notna() & b.notna()).sum())))(ttest_rel(a, b, nan_policy="omit"))
        p1 = lambda a: (lambda r: dict(t=float(r.statistic), p=float(r.pvalue), n=int(a.notna().sum())))(ttest_1samp(a, 0, nan_policy="omit"))
        out[space] = dict(
            n_sessions=int(N.shape[0]),
            normalised_mean={p: float(N[p].mean()) for p in PER},
            normalised_sem={p: float(N[p].sem()) for p in PER},
            normalised_p95_mean={p: float(N95[p].mean()) for p in PER},
            ttest_on_vs_0=p1(N.on), ttest_off_vs_0=p1(N.off), ttest_pre_vs_0=p1(N.pre),
            ttest_on_vs_off=p2(N.on, N.off), ttest_off_vs_pre=p2(N.off, N.pre),
            n_sessions_above_p95={p: int((N[p] > N95[p]).sum()) for p in PER},
            n_sessions_above_0={p: int((N[p] > 0).sum()) for p in PER})
    return out


def check_plot(D, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MCOL = {"21": "#2a78d6", "22": "#eb6834", "24": "#1baf7a"}
    INK, MUTED = "#1f1f1e", "#6b6a64"
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(1, 2, figsize=(5.9, 2.6), constrained_layout=True, sharey=True)
    for a, sp, title in ((ax[0], "pcs", "7 cue-mean PCs"), (ax[1], "full", "all held-out units")):
        N = D[D.space == sp].pivot(index="session", columns="period", values="normalised")[PER]
        for s, r in N.iterrows():
            a.plot(range(3), r.values, color=MCOL[str(s)[:2]], lw=0.6, alpha=0.5, marker="o", ms=2, mew=0)
        a.axhline(0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
        a.errorbar(range(3), N.mean(), N.sem(), color=INK, lw=1.8, marker="o", ms=4, capsize=0, zorder=5)
        a.set_xticks(range(3), ["On", "Off", "pre-cue"])
        a.set_xlim(-0.5, 2.5)
        a.set_title(title, fontsize=7.5, loc="left")
    ax[0].set_ylabel("shape similarity (normalised)\n1 − disparity / shuffle disparity")
    for mk, c in (("A", MCOL["21"]), ("H", MCOL["22"]), ("J", MCOL["24"])):
        ax[1].plot([], [], color=c, lw=0.8, label=f"monkey {mk}")
    ax[1].plot([], [], color=MUTED, ls=(0, (3, 2)), label="chance")
    ax[1].legend(frameon=False, fontsize=5.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(png, dpi=200)


if __name__ == "__main__":
    D = normalise(pd.read_csv(sys.argv[1]))
    os.makedirs(sys.argv[2], exist_ok=True)
    D.to_csv(os.path.join(sys.argv[2], "procrustes_sessions_normalised.csv"), index=False)
    json.dump(stats(D), open(os.path.join(sys.argv[2], "procrustes_stats_normalised.json"), "w"), indent=1)
    if len(sys.argv) > 3:
        check_plot(D, sys.argv[3])
