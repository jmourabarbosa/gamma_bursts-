# Brief for the code-refactoring agent

Goal: turn the code behind the paper "Intermittent working-memory coding through NMDA-sustained persistent
activity" into a clean, readable public repository, **without changing any result**. Core modules will be
proofread by the author afterwards, so favour clarity over cleverness.

## Where everything is

| What | Repository | Branch | Path |
|---|---|---|---|
| Paper (LaTeX, figures, bib) | `jmourabarbosa/gamma_bursts-` | `claude/intermittent-rate-coding-paper-ced984` (same files on `paper-model`) | `paper/` (`main.tex`, `refs.bib`, `figures/`) |
| Rate model (NMDA and STP), read-out, fitting, Figs. 1–3, Supp. Figs. 1–3 | `jmourabarbosa/gamma_bursts-` | `paper-model` | `paper_model/` |
| Figure scripts for the paper (Fig. 3, Fig. 4, single-dip supp.) and their plotted data | `jmourabarbosa/gamma_bursts-` | `paper-model` | `paper/scripts/`, `paper/scripts/data/` |
| Spiking ring network (Compte et al. 2000 + changes), read-out, PCA / Procrustes, Fig. 4 data | `jmourabarbosa/gamma_bursts-` | `claude/gallant-volta-bkjojq` | `spiking/` (`PARAMETERS.md`, `NOTE.md`, `results/pca_onoff/counts_s*.npz` = stored spike counts), `figures/spiking/` |
| Data analyses (decoders, On/Off labelling, unit sets, behaviour GLMM, PCA / Procrustes) | `jmourabarbosa/panichello` | `analysis-replication` | `analysis/` (hand-offs in `analysis/*handoff*/`) |
| Recordings (25 sessions, Panichello et al. 2024) | `jmourabarbosa/panichello` | any | `*.mat` at the repo root (about 520 MB, tracked in git) |
| Hand-offs between agents (method specs) | `jmourabarbosa/gamma_bursts-` | `claude/intermittent-rate-coding-paper-ced984` | `paper/handoffs/` |

Other branches of `gamma_bursts-` (`paper-model-mongillo`, `paper-model-ww`, `paper-model-v1-custom-rate`,
`claude/gallant-goodall-45zpz2`, `ccr-*`) are earlier explorations; ignore them unless the author says otherwise.

The paper's Methods (`paper/main.tex`) are the specification: every number in the paper must be reproducible from
the refactored code.

## Rules
1. **Work on your own branch** (e.g. `publication`) in a new repository or in `gamma_bursts-`. Never push to the
   branches above; other agents and the author still use them.
2. **No change in results.** Before refactoring a module, run it and save its outputs. After refactoring, rerun and
   compare (numbers equal up to floating-point tolerance; figures visually identical). Keep a short log of these checks.
3. **Keep expensive steps cached.** The spiking simulations take about 9 min per session; use the stored counts in
   `spiking/results/pca_onoff/counts_s*.npz`. Rate-model fits use CMA-ES; keep the fitted parameters (Methods,
   "Fitting") and do not refit unless asked.
4. **Data paths are configurable** (one config file or command-line argument); do not hard-code `/home/user/...`.
5. **Structure** (suggestion):
   - `ratemodel/` (model, read-out, fitting, STP);
   - `spiking/` (network, read-out);
   - `analysis/` (data loading, decoders, On/Off labelling, behaviour, PCA / Procrustes);
   - `figures/` (one script per paper figure, reading saved results);
   - `README.md` mapping each figure and each number of the paper to the script that produces it.
6. **Remove dead code** and exploratory variants that the paper does not use, but list what was removed.
7. **Dependencies:** Python 3, numpy, scipy, numba, joblib, pandas, matplotlib, scikit-learn, cma. Pin versions in
   `requirements.txt`.
8. Do not edit `paper/` or the Google Doc; the paper agent maintains them. Report anything in the paper that does not
   match the code.
