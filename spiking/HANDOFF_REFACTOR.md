# Hand-off: refactoring the spiking-network code for publication

Everything below is on branch `claude/gallant-volta-bkjojq` of `jmourabarbosa/gamma_bursts-`, folder `spiking/`
(code, results) and `figures/spiking/` (outputs given to the paper agent). Background: `spiking/AGENT_INSTRUCTIONS.md`
(the original task), `spiking/NOTE.md` (what was built, tuning, results), `spiking/PARAMETERS.md` (every parameter
and its source).

## Goal and ground rules
- Make the code clean and reproducible for publication **without changing any result**. Every number in
  `figures/spiking/*.json|csv` must be reproduced exactly (same seeds, same trials) after the refactor.
- Keep the outputs the paper agent reads (file names and columns in `figures/spiking/`), or list every rename.
- Don't re-tune the model; parameters are final (`network.P` overridden by `run.TUNED`).

## What the code does (pipeline)
| step | module / entry point | inputs → outputs | time (4 cores) |
|---|---|---|---|
| network (LIF ring, NMDA 150 ms, dips, kick) | `network.py` (`Network`, `simulate`, `memory`; numba kernel `_run`) | — | 0.4 s per trial-second |
| read-out like the data (decoder, On/Off labels, stats) | `readout.py` (`read_session`, `decode`, `windows`); labelling and stats from `paper_model/readout.py` (`label`, `stats`, `runs`, `load_data`) | — | — |
| main simulation, 12 sessions × 480 trials | `python -m spiking.run sessions` | → `results/sessions_dips.json`, `results/sessions_dips_trials.npz` | ~80 min |
| no-dip control, single dip, stability, example | `python -m spiking.run nodips --n_sessions 6`, `single_dip`, `stability`, `example`, `examples_grid` | → `results/*.json`, `*.npz` | ~1 h |
| Brian2 cross-check | `python -m spiking.validate_brian2` | → `results/validate_brian2.json` | ~10 min |
| diagnostic figures and table | `python -m spiking.figures` | → `spiking/figures/fig_*.png`, `results/table.md` | 1 min |
| PDF report | `python -m spiking.make_report` | → `spiking/report.pdf` | 1 min |
| **paper figure** (trial, histograms, On/Off tuning) | `python -m spiking.fig_paper` | → `figures/spiking/fig_spiking*` | ~15 min (long-trial session cached) |
| state histograms (counts, full-range lengths) | `PYTHONPATH=. python spiking/fig_state_histograms.py` | → `spiking/figures/fig_state_histograms.*` | 1 min |
| **PCA On/Off** (data agent's analysis) | `python -m spiking.pca_onoff --sessions 0 … 9`; `paper_panels`, `figure`, `procrustes_panel`, `procrustes_k` | counts → `results/pca_onoff/session_s.npz` → `figures/spiking/fig_spiking_pca_onoff*`, `fig_spiking_pca_procrustes*` | ~3 min per session from cached counts |
| **Procrustes, paper spec** (Fig. 4e) | `python -m spiking.procrustes_run` (full + pcs), then `main_k(7)` (pcs7); normalised with `procrustes_normalize.py` | → `figures/spiking/procrustes_sessions*.csv`, `procrustes_stats*.json`, `net_panel_c.csv` | ~25 min |
| CCG / Fig. 5 (**abandoned**) | `ccg.py`, `run_ccg.sh` | sessions 0–1 only | — |

## Data and caches (all committed except the CCG spikes)
- `results/sessions_dips_trials.npz`: per session `s{s}_held`, `s{s}_cue`, `s{s}_state_all` (all-units states,
  correct trials), `s{s}_z_all`, `s{s}_off_time_all_trials`, and session 0's full `s0_counts`. **Quirk:**
  `s0_state_all` / `s0_z_*` hold only a 40-trial preview (the full session 0 states are recomputed in `ccg.py` and
  `pca_onoff.py`, same seed, identical result). Fix this in the refactor by storing full arrays for every session.
- `results/pca_onoff/counts_s{0..9}.npz`: 10 ms counts of the 330 recorded cells for sessions 0–9 (keys `counts`,
  `cue`, `held`), so no re-simulation is needed for the PCA / Procrustes analyses. Sessions 10–11 have no stored
  counts.
- `results/pca_onoff/states_allH1_s{s}.npz`: On/Off states from all cells of H1 (Procrustes spec).
- `results/fig_paper/long_session.npz`: the 2 s-delay session of the paper figure (panel a).
- Determinism: every simulation is seeded per batch of 64 trials (`run.run_trials`, seed `seed0 * 1000 + batch`) and
  per trial inside numba. Re-simulating a session with the same batch size gives identical spikes; scripts assert
  this (`pca_onoff.session_counts`, `fig_paper.example_raster`). **Changing BATCH changes the trials.**

## Things to preserve or know
- External inputs read by path:
  - `paper_model/` (same repo): `readout.py`, `data/*.csv` (data statistics and histograms).
  - `/home/user/panichello/analysis/pca_onoff_handoff/reference/pca_onoff_stats.csv` and
    `.../paper/panel_d_stats.csv` (data means drawn as red crosses), from repo `jmourabarbosa/panichello`, branch
    `analysis-replication`. Make this path configurable or copy the few numbers into the repo.
- Cell halves H1/H2 (`default_rng(1)` permutation), trial halves (`default_rng(0)`, stratified by cue), labelling
  rng `default_rng(s + 1)`: these define the published numbers.
- Pre-cue window centres −400 to −50 ms in the network (not −400 to 0 as the data), because the model's cue drives
  spikes within milliseconds.
- Two labellings coexist: top 20 of H1 (cosine PCA panels, `pca_onoff.run_session`) and all of H1 (Procrustes spec,
  `procrustes_run`). The data used the top 20. Don't merge them silently; ask the user which one the paper keeps.
- `procrustes_shape.py` and `procrustes_normalize.py` are copies of the paper / data agents' reference code (one
  addition each, documented in the docstrings); keep them identical to the references apart from those.

## Clean-up candidates (confirm with the user before deleting)
- `ccg.py`, `run_ccg.sh`, `results/ccg/`, `spiking/figures/fig_ccg.*` (CCG analysis stopped by the user).
- `results/pca_onoff/old/` (superseded PCA runs), `run_rest.sh` (one-off).
- Ad-hoc scripts that live as functions with hard-coded paths: `fig_state_histograms.py`, `pca_onoff.figure`
  (diagnostic), `procrustes_panel` / `procrustes_k` (superseded by `procrustes_run`).
- `spiking/figures/` (diagnostics) vs `figures/spiking/` (paper outputs): keep the distinction, or merge with a clear
  naming rule.

## Suggested checks after refactoring
1. Re-run `python -m spiking.procrustes_run` and `main_k(7)` from the cached counts: `figures/spiking/procrustes_*`
   must be byte-identical (or equal to 1e-12).
2. Re-run `python -m spiking.fig_paper` and `python -c "from spiking import pca_onoff as P; P.paper_panels(list(range(10)), ex=0)"`:
   the JSON/CSV outputs must match.
3. Re-simulate one batch of session 0 and compare with `s0_counts` (as `fig_paper.example_raster` does).
4. Optionally the full pipeline (`run_all.sh`, ~2.5 h) on a fresh checkout.
