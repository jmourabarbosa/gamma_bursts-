# Rate model of On/Off states (paper code)

The best-fit model: two bistable memory pools with NMDA self-excitation and shared inhibition. A common input
drops briefly ("dips") and silences the pools; lagging feed-forward inhibition then gives both pools a short kick
when the input returns, and the pool with more residual NMDA wins. The model is read out like the recordings
(Panichello et al. 2024 On/Off labelling) and fitted to the 25-session read-out table at decoder λ = 10,000.

| file | content |
|---|---|
| `model.py` | equations, fixed parameters (`FIXED`), best fit (`FIT`), fit bounds, simulation, memory check |
| `readout.py` | z = gain × (cued − uncued rate, 100 ms windows) + measured noise; gain anchored to the data's μ_on per unit set; On/Off labelling; statistics per unit set |
| `fit.py` | CMA-ES fit of `FIT` (loss: six On/Off statistics for each of 7 unit sets, top-20 length histograms, memory ≥ 98% on the fit's trials) |
| `make_figures.py` | Figure 1 (example trial + model top-20 length histograms) and Figure 2 (On/Off statistics of every unit set, model vs data), `model_stats.json` |
| `data/` | data read-out table and histograms (λ = 10,000), with the data agent's note |

Fitted: τ_N 125 ms, w_s 1.63, dips 9.9/s, mean length 122 ms (gamma shape 5.9), depth 82%, noise σ 0.012.
Memory correct at the end of the delay: 97.7% (2,112 new trials).

```
pip install numpy numba scikit-learn matplotlib cma
python -m paper_model.make_figures          # ~1 min; writes figures/paper_model/
python -m paper_model.fit --out fit_run0     # optional refit (hours)
```
