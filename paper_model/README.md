# Rate model of On/Off states (paper code)

The best-fit model: two bistable memory pools with NMDA self-excitation and shared inhibition. A common input
drops briefly ("dips") and silences the pools; lagging feed-forward inhibition then gives both pools a short kick
when the input returns, and the pool with more residual NMDA wins. The model is read out like the recordings
(Panichello et al. 2024 On/Off labelling) and fitted to the 25-session read-out table at decoder λ = 10,000.

| file | content |
|---|---|
| `model.py` | equations, fixed parameters (`FIXED`), best fit (`FIT`), fit bounds, simulation, memory check |
| `readout.py` | z = gain × (cued − uncued rate, 100 ms windows) + measured noise; gain anchored to the data's μ_on per unit set; On/Off labelling; statistics per unit set |
| `fit.py` | CMA-ES fit of `FIT` (loss: six On/Off statistics and the On/Off length histograms of each of 7 unit sets, memory ≥ 98% on the fit's trials) |
| `make_figures.py` | Figure 1 (example trial: 2 s delay without dips, then dips; all-unit decoding and length histograms, model vs data) and Figure 2 (On/Off statistics of every unit set, model vs data), `model_stats.json` |
| `data/` | data read-out table and histograms (λ = 10,000), with the data agent's note |

Fitted (to the On/Off statistics and length histograms of all 7 unit sets): τ_N 132 ms, w_s 1.61, dips 10.2/s,
mean length 119 ms (gamma shape 5.2), depth 84%, noise σ 0.012. Memory correct at the end of the delay: 98.3%
(2,112 new trials).

```
pip install numpy numba scikit-learn matplotlib cma
python -m paper_model.make_figures          # ~1 min; writes figures/paper_model/
python -m paper_model.fit --out fit_run0     # optional refit (hours)
```
