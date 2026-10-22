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
| `data/` | the data's read-out per session and pooled state-length histograms (λ = 10,000), with the data agent's note |

Fitted values: τ_N 150 ms (upper bound), w_s 1.30 (lower bound), dips 10.2/s, mean length 90 ms (gamma shape 8.7),
depth 60%, noise σ 0.006. The other six unit sets are predictions: the same dynamics, read out with each set's gain
(from the data's μ_on) and noise share. Memory correct: 97% of trials. A 300 ms response period follows the delay, with the attention signal at 1 and a
re-engagement transient at its onset (go_amp, ~3x the transient at the end of a dip); the memory is correct when
the cued pool fires above 20 Hz and above the uncued pool in its last 100 ms. (The fit, `FIT`, used an earlier
criterion: NMDA gating of the cued pool above the uncued pool's at the end of the delay; 98% by that criterion.)

```
pip install numpy numba scikit-learn matplotlib cma
python -m paper_model.make_figures            # ~15 s; writes figures/paper_model/
python -m paper_model.fit --out fit_run0      # optional refit, ~1 h per run
```
