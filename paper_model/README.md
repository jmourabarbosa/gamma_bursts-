# Rate model of On/Off states (paper code)

The best-fit model: two bistable memory pools with NMDA self-excitation and shared inhibition. A common input
drops briefly ("dips") and silences the pools; lagging feed-forward inhibition then gives both pools a short kick
when the input returns, and the pool with more residual NMDA wins. The model is read out like the recordings
(Panichello et al. 2024 On/Off labelling) and fitted to the 25-session read-out table at decoder λ = 10,000.

| file | content |
|---|---|
| `model.py` | equations, fixed parameters (`FIXED`), best fit (`FIT`), fit bounds, simulation, memory check |
| `readout.py` | z = gain × (cued − uncued rate, 100 ms windows) + measured noise; gain anchored to the data's μ_on per unit set; On/Off labelling; statistics per unit set |
| `fit.py` | CMA-ES fit of `FIT` (loss: six On/Off statistics and On/Off length histograms of the chosen unit set(s), memory ≥ 98% on the fit's trials; `--fit-set 'top 20'` gives `FIT`) |
| `make_figures.py` | Figure 1 (example trial: 1 s delay without dips, then dips; all-unit decoding and length histograms, model vs data) and Figure 2 (On/Off statistics of every unit set, model vs data), `model_stats.json` |
| `data/` | data read-out table and histograms (λ = 10,000), with the data agent's note |

Fitted to the top-20 unit set only (its On/Off statistics and length histograms): τ_N 150 ms, w_s 1.59,
dips 11.4/s, mean length 115 ms (gamma shape 3.9), depth 75%, noise σ 0.007. The other six unit sets are predictions:
the same dynamics, read out with each set's gain (from the data's μ_on) and noise share. Memory correct at the end
of the delay: 98.2% (2,112 new trials).

```
pip install numpy numba scikit-learn matplotlib cma
python -m paper_model.make_figures          # ~1 min; writes figures/paper_model/
python -m paper_model.fit --out fit_run0     # optional refit (hours)
```
