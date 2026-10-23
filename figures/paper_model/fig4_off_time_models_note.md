# Hand-off: memory vs time Off in the delay (NMDA model vs STP model)

## For the paper agent: what changed
The model panel of paper Figure 3 should now use the **fitted models with their own random dips**, not the single-dip test.

**Why.** In the single-dip test, dip lengths are spread evenly up to 1 s, far beyond the fitted dips (mean 90 ms in the NMDA model). So 63% of trials lose the memory by design, and memory falls to about 5%. That exaggerates how fast performance decays. With the fitted dips, the NMDA model stays near the data's level and declines with time Off, while the STP model stays at 100%.

**What to change in `paper/scripts/fig3_off_time.py`:**
- Load `figures/off_vs_memory/{nmda,fac}/natural_readout_trials.npz` instead of `single_dip_readout_trials.npz`.
- The keys and units are the same: `off_time_s` in seconds and `memory_held`, now with 20,000 trials per model.
- Keep the single-dip test as a supplementary figure or a sentence in the Methods.

**Reference figures:**
- Main figure, fitted models with their own dips: `figures/paper_model/fig4_off_time_models.png` (panels as below).
- Supplementary, single-dip test: `figures/paper_model/figS_off_time_models_single_dip.png`.

## Files
All files are in `figures/paper_model/`.

| file | content |
|---|---|
| `fig4_off_time_models.png` (600 dpi), `.svg` (editable text) | main figure |
| `fig4_off_time_models_data.csv` | the plotted values: panel a, one row per window (`memory_held`, 95% CI, `n_trials`); panels b and c, one row per bin (`n_held`, `n_lost`, and the plotted `frac_held`, `frac_lost`) |
| `figS_off_time_models_single_dip.png`, `.svg`, `_data.csv` | the single-dip test |

## Reproduce (about 10 min)
```
python -m paper_model.natural_readout
PM_VARIANT=fac python -m paper_model.natural_readout
python -m paper_model.single_dip_readout
PM_VARIANT=fac python -m paper_model.single_dip_readout
python -m paper_model.fig_off_time_models
```

## Main figure: what it shows
**The question.** In the fitted models, is the memory lost more often in trials with more time Off, as in the data?

**Simulation**, the same for both models:
- 20,000 trials of each fitted model, with its own random dips of the common input (the fitted dip statistics). Nothing is manipulated.
- Each trial is read out like the data with the all-units read-out:
  - the fitted model's all-units gain;
  - the all-units noise, with a 0.35 slow share;
  - On/Off labelling as in the main analysis.

**Definitions:**
- **Time Off in delay:** the number of Off time points in 500–1,400 ms × 10 ms. That is 91 points, at most 0.91 s, the same window as the behavioural analysis. The paper's Off-time axis uses 900 ms, a difference of one 10 ms point.
- **Memory held:** the network itself, not the decoder. At the end of the delay (last 100 ms), the cued pool's slow variable exceeds the uncued pool's by more than 0.05. The slow variable is NMDA gating in the NMDA model and facilitation u in the STP model.
  - In the NMDA model, a lost memory means neither pool re-ignites after a dip; the uncued item is never held.

**Panels:**
- **a:** fraction of trials with the memory held against time Off. Black: NMDA model. Blue: STP model.
  - Sliding windows as the behavioural analysis, converted to seconds: 23 ms wide, steps of 2.3 ms, from 45 ms. Shading: 95% Wilson interval.
  - The y axis runs from 0.75 to 1.
- **b, c:** distribution of time Off for trials with the memory held (grey) or lost (red), for the NMDA model (b) and the STP model (c).
  - Each group is normalised to its own number of trials, because lost trials are few.
  - 35 ms bins.
  - The STP model has a single lost trial, so its histogram is not drawn.

## The two models

### NMDA model
The paper model (`model.py`, `model.FIT`).

### STP model
`model_ours_stp.py`, `FIT_FAC`. It is identical to the NMDA model except for the slow recurrent term.
- **Recurrent term:**
  - The NMDA gating variable s (τ_N 150 ms) is replaced by the same saturating synaptic activation without a time constant of its own: S = g/(1 + g), with g = β r u x / U.
  - It is weighted by Tsodyks–Markram short-term facilitation u (Mi, Katkov & Tsodyks 2017): du/dt = (U − u)/τ_f + U(1 − u) r, with U = 0.3 and τ_f = 1.5 s. This is the working-memory time scale of Mongillo et al. 2008 and Mi et al. 2017.
- **Facilitation only:** depression is switched off (x = 1). With depression (τ_d 0.3 s) the model fits the data badly (all-units loss 37 vs 10, three fits), because depression terminates each re-ignition and On states are too short (112 vs 172 ms).
- **Everything else is as in the NMDA model:** the fixed parameters, AMPA self-excitation, inhibition, common input and lagging feed-forward inhibition.
- **Fitting:** as for the NMDA model (CMA-ES, all-units read-out only, same loss), plus a penalty for population spikes in the 1 s before the cue. Six parameters were fitted, with τ_f fixed:
  - w_s = 0.69
  - dip rate 5.5/s, mean length 212 ms, gamma shape 0.70 (lower bound)
  - depth 0.39
  - σ = 0.005 (lower bound)
- Resting facilitation (u ≈ 0.5 at 1 Hz) already strengthens the recurrent synapses, hence the lower w_s.

### Both fit the On/Off statistics about equally well
Scored on 2,112 new trials:

| | NMDA model | STP model |
|---|---|---|
| all-units loss | 9.7 | 10.4 |
| loss summed over the 7 decoded populations (6 predicted) | 72.6 | 76.4 |
| Off: per trial / mean length / fraction of delay | 3.07 / 154 ms / 0.39 | 2.87 / 184 ms / 0.40 |
| On: per trial / mean length / fraction of delay | 1.86 / 184 ms / 0.18 | 2.14 / 168 ms / 0.21 |
| memory held | 98.5% | 100% |

- **Data:** Off 3.25 / 186 ms / 0.45; On 2.10 / 172 ms / 0.21.
- The STP model's version of Fig. 2 is in `figures/off_vs_memory/fac/fig2_unit_sets.png`. It is similar to the NMDA model's, with On states in the top-unit sets about 50 ms too long.

## Key numbers (main figure)

### NMDA model
- **Overall:** 19,703 of 20,000 trials held (98.5%).
- **Memory held by time Off:**

  | time Off | memory held |
  |---|---|
  | 0–0.1 s | 99.8% |
  | 0.4–0.5 s | 98.9% |
  | 0.6–0.7 s | 97.4% |
  | 0.8–0.91 s | 93.4% |

- **Lost trials have more time Off:** mean 0.62 s, against 0.36 s for held trials.

### STP model
- **Overall:** 19,999 of 20,000 trials held. Memory is 100% at every time Off, including 0.8–0.91 s.
- **Mean time Off:** 0.37 s.

### Data, for comparison
- About 91% correct with almost no time Off, falling to about 76% when nearly the whole delay is Off.
- The model's baseline (about 100%) is higher because it has no errors other than memory loss. The comparison is about the decline with time Off.

## Supplementary: the single-dip test (`figS_off_time_models_single_dip`)
**What it is.** Each trial has exactly one dip of fixed length, and no other dips:
- length 0–1,000 ms in 25 ms steps, 300 trials per length (12,300 per model);
- starting 500 ms after cue onset;
- at each model's fitted depth (NMDA 0.60, STP 0.39).

The read-out and definitions are as above. This controls how long the network is silent.

**NMDA model: the memory has a cut-off.**
- Held in 100% of trials for dips up to 350 ms, 29% at 375 ms, and 0% from 400 ms.
- With a full dip (depth 1) the cut-off is about 450 ms, because the feed-forward kick when the input returns is stronger.
- **Mechanism:** during a dip both pools are silent and the NMDA gating decays. When the input returns, the lagging feed-forward inhibition kicks both pools, and the cued pool re-ignites only if enough gating is left.
- **Through the read-out**, memory against time Off declines from about 0.88 to 0.05, because 63% of trials have dips beyond the cut-off by design.

**STP model: the memory is never lost**, at any dip length up to 1 s, and also with full dips (12,300 of 12,300 trials). Facilitation decays with τ_f 1.5 s, so the cued pool always re-ignites.

**Further panels** for both models are in `figures/off_vs_memory/{nmda,fac}/`:
- memory against dip length without the read-out;
- dip rate varied from 0 to 30/s and dip depth from 0.3 to 1.0. The STP model holds the memory in all 24,000 trials, even when Off covers half the delay.

## Interpretation, for the text
- **Both slow mechanisms reproduce the On/Off statistics**, but only the NMDA model reproduces the behavioural result.
- **NMDA model:** the memory is bridged by a slow-but-short trace (about 150 ms). It is lost when an Off period outlasts the trace, by about 375 ms of silence. So errors come with more time Off, and memory declines with time Off.
- **STP model:** facilitation on the working-memory time scale (1.5 s) bridges any Off period, so it predicts no relation between time Off and errors.

**Caveats:**
- **Trace time scale, not NMDA as such.** The argument concerns how long the trace lasts. An earlier unfitted test with facilitation at τ_f 150 ms gave nearly the same read-out as the NMDA model.
- **Smaller decline than the data.** The model goes from 100% to 93%, the data from 91% to 76%, and the model has no errors other than memory loss.
- **Why long Off times rarely cost the memory in the NMDA model:** they mostly consist of several short Off periods, each bridged by the trace.
- **Prediction for the data:** errors should depend on the longest Off period of a trial more than on total time Off, with a drop between about 350 and 450 ms of continuous silence.

## Code
In `paper_model/` on branch `paper-model`:

| file | role |
|---|---|
| `fig_off_time_models.py` | both figures |
| `natural_readout.py` | main-figure trials |
| `single_dip_readout.py` | single-dip trials |
| `variant.py` | chooses the model: `PM_VARIANT=fac` for the STP model, the default for the NMDA model |
| `model_ours_stp.py` | the STP model |

The STP fit script, `fit_ours_stp.py`, is on branch `paper-model-mongillo`.
