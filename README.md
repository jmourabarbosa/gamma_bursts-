# Rate model of On/Off states (paper code)

Two bistable memory pools with NMDA self-excitation and shared inhibition. A common input drops briefly ("dips")
and silences the pools; lagging feed-forward inhibition then gives both pools a short kick when the input returns,
and the pool with more residual NMDA wins. The model is read out like the recordings (Panichello et al. 2024
On/Off labelling) and fitted to the all-units read-out of 25 sessions (decoder λ = 10,000).

| file (in `paper_model/`) | content |
|---|---|
| `model.py` | equations, fixed parameters (`FIXED`), fitted parameters (`FIT`), simulation, memory check |
| `readout.py` | z = gain × (cued − uncued rate, 100 ms windows) + measured noise; gain set from the data's μ_on for each unit set; On/Off labelling; statistics; data loading |
| `fit.py` | CMA-ES fit of `FIT` to the all-units On/Off statistics and length distributions, with memory ≥ 98% |
| `make_figures.py` | Figure 1 (`fig1_trial_histograms`) and Figure 2 (`fig2_unit_sets`) as PNG and editable SVG, plus `model_stats.json` |
| `model_ours_stp.py` | the same model with short-term facilitation (Tsodyks–Markram u, τ_f 1.5 s) in place of NMDA gating; `FIT_FAC` fitted on branch `paper-model-mongillo` |
| `variant.py` | chooses the model for the Off-time analyses (`PM_VARIANT=nmda` default, or `fac`) |
| `off_vs_memory.py`, `single_dip.py`, `single_dip_readout.py` | memory vs time Off in the delay: dip statistics varied, one dip of fixed length, and one dip read out like the data (`figures/off_vs_memory/<variant>/`) |
| `natural_readout.py` | per-trial time Off and memory held of each fitted model with its own dips (Figure 4) |
| `fig_off_time_models.py` | Figure 4 (`fig4_off_time_models`, fitted models) and supplementary `figS_off_time_models_single_dip`: memory vs time Off, NMDA vs STP model; hand-off note `figures/paper_model/fig4_off_time_models_note.md` |
| `fig_unit_sets_variant.py` | Figure 2 layout for either model (`PM_VARIANT`) |
| `data/` | the data's read-out per session and pooled state-length histograms (λ = 10,000), with the data agent's note |

Fitted values: τ_N 150 ms (upper bound), w_s 1.30 (lower bound), dips 10.2/s, mean length 90 ms (gamma shape 8.7),
depth 60%, noise σ 0.006. The other six unit sets are predictions: the same dynamics, read out with each set's gain
(from the data's μ_on) and noise share. Memory correct at the end of the delay: 98% of trials.

```
pip install numpy numba scikit-learn matplotlib cma
python -m paper_model.make_figures            # ~15 s; writes figures/paper_model/
python -m paper_model.fit --out fit_run0      # optional refit, ~1 h per run
```
