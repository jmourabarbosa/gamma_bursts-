# Hand-off: memory vs time Off in the delay (NMDA model vs STP model)

## For the paper agent: what changed
The model figure now shows **both simulations**, one per row:
- **Top row (a–c): fitted dips.** Each fitted model with its own random dips, 100,000 trials per model.
- **Bottom row (d–f): single dip.** One dip of 0–1 s per trial.

In both, the STP model holds the memory, while the NMDA model loses it with long silences.

**Why both.**
- **The single-dip test alone exaggerates how fast performance decays.** Dip lengths are spread evenly up to 1 s, far beyond the fitted dips (mean 90 ms in the NMDA model), so 63% of trials lose the memory by design.
- **The fitted-dips simulation is the comparison with the data.** The NMDA model stays near the data's level and declines with time Off; the STP model stays at 100%.
- **The single-dip test shows the mechanism.** The NMDA model has a cut-off at about 375 ms of continuous silence; the STP model has none.

**What to change in `paper/scripts/fig3_off_time.py`:**
- Its model panel reads `figures/off_vs_memory/{nmda,fac}/single_dip_readout_trials.npz`, which is unchanged.
- Add `natural_readout_trials.npz` from the same folders. It has the same keys and units: `off_time_s` in seconds and `memory_held`, with 100,000 trials per model.

**Reference figure:** `figures/paper_model/fig4_off_time_models.png`, panels as below.

## Files
All files are in `figures/paper_model/`.

| file | content |
|---|---|
| `fig4_off_time_models.png` (600 dpi), `.svg` (editable text) | the figure |
| `fig4_off_time_models_data.csv` | the plotted values: panels a and d, one row per window (`memory_held`, 95% CI, `n_trials`); panels b, c, e and f, one row per bin (`n_held`, `n_lost`, and the plotted `frac_held`, `frac_lost`) |

## Reproduce (about 10 min)
```
python -m paper_model.natural_readout
PM_VARIANT=fac python -m paper_model.natural_readout
python -m paper_model.single_dip_readout
PM_VARIANT=fac python -m paper_model.single_dip_readout
python -m paper_model.fig_off_time_models
```

## Top row (a–c), fitted dips: what it shows
**The question.** In the fitted models, is the memory lost more often in trials with more time Off, as in the data?

**Simulation**, the same for both models:
- 100,000 trials of each fitted model, with its own random dips of the common input (the fitted dip statistics). Nothing is manipulated.
- Each trial is read out like the data with the all-units read-out:
  - the fitted model's all-units gain;
  - the all-units noise, with a 0.35 slow share;
  - On/Off labelling as in the main analysis.

**Definitions:**
- **Time Off in delay:** the number of Off time points in 500–1,400 ms × 10 ms. That is 91 points, at most 0.91 s, the same window as the behavioural analysis. The paper's Off-time axis uses 900 ms, a difference of one 10 ms point.
- **Memory held:** the network itself, not the decoder. At the end of the delay (last 100 ms), the cued pool's slow variable exceeds the uncued pool's by more than 0.05. The slow variable is NMDA gating in the NMDA model and facilitation u in the STP model.
  - In the NMDA model, a lost memory means neither pool re-ignites after a dip; the uncued item is never held.

**Panels** (the bottom row d–f has the same layout for the single-dip test):
- **a:** fraction of trials with the memory held against time Off. Black: NMDA model. Blue: STP model.
  - Sliding windows as the behavioural analysis, converted to seconds: 23 ms wide, steps of 2.3 ms, from 45 ms. Shading: 95% Wilson interval.
  - The y axis runs from 0.75 to 1 (0 to 1 in panel d).
- **b, c:** distribution of time Off for trials with the memory held (grey) or lost (red), for the NMDA model (b) and the STP model (c).
  - Each group is normalised to its own number of trials, because lost trials are few.
  - 35 ms bins.
  - Lost trials are drawn only if there are at least 20 of them. The STP model has 9 lost trials of 100,000 in the top row and none in the bottom row.

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
- **Two more fits with other seeds agree** (scored on new trials):

  | seed | all-units loss | all 7 sets | memory held | dips |
  |---|---|---|---|---|
  | 1 | 10.9 | 76.0 | 100% | 4.9/s, mean 300 ms (bound), depth 0.44 |
  | 2 | 10.7 | 72.4 | 99.9% | 8.0/s, mean 164 ms, shape 0.7, depth 0.95, nearly full |

  - The dip statistics differ between seeds, but every fit holds the memory in practically all trials.
  - Seed 2 has frequent, long, nearly full dips and still keeps 99.9%, so the STP result doesn't depend on the seed.
  - The figure uses seed 0.

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
- The STP model's version of Fig. 2 is in `figures/off_vs_memory/fac/fig2_unit_sets.png`.
- **Supplementary figure for "it reproduced the On/Off statistics about as well as the NMDA model":** `figures/paper_model/figS_unit_sets_models.png` (with `.svg` and `.csv`).
  - Panels a–f: the six On/Off statistics of the 7 decoded populations for the data, the NMDA model, the STP model with facilitation only, and the STP model with facilitation and depression.
  - Panel g: fit loss per population.
  - Panel h: loss summed over the 7 populations (log axes): NMDA 74, STP with facilitation 79, STP with facilitation and depression 261.
  - Each model was fitted to the all-unit decoder only, evaluated on 2,112 trials, and the other six populations predicted by the read-out, as in Fig. 2.
  - The depression version shows why depression was switched off: On states too short, too much Off time, and identical top-unit sets because the read-out gain reaches its upper limit. It is similar to the NMDA model's, with On states in the top-unit sets about 50 ms too long.

## Key numbers, fitted dips (top row)

### NMDA model
- **Overall:** 98,517 of 100,000 trials held (98.5%).
- **Memory held by time Off:**

  | time Off | memory held |
  |---|---|
  | 0–0.1 s | 99.8% |
  | 0.2–0.3 s | 99.4% |
  | 0.4–0.5 s | 98.8% |
  | 0.6–0.7 s | 97.3% |
  | 0.8–0.91 s | 94.3% |

- **Lost trials have more time Off:** mean 0.60 s, against 0.36 s for held trials.

### STP model
- **Overall:** 99,991 of 100,000 trials held. Memory is about 100% at every time Off.
- **Mean time Off:** 0.37 s.

### Data, for comparison
- About 91% correct with almost no time Off, falling to about 76% when nearly the whole delay is Off.
- The model's baseline (about 100%) is higher because it has no errors other than memory loss. The comparison is about the decline with time Off.

## Bottom row (d–f): the single-dip test
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

## Supplementary: memory vs total dip time (`figS_dip_time_models`)
**What it is.** The same fitted-dips trials as the top row, with a different x axis.
- **x axis:** the total time the common input is in a dip during 500–1,400 ms, below the middle of the dip (1 − depth/2). This is the network's real silence, not the decoder's estimate.
- **Files:** `figS_dip_time_models.png`, `.svg`, `_data.csv`. The data CSV keeps the column name `x_off_time_s`, which here holds the dip time.

**NMDA model: memory depends steeply on dip time.**

| dip time | memory held |
|---|---|
| up to 0.4 s | 99.7–99.8% |
| 0.4–0.6 s | 98.6% |
| 0.6–0.8 s | 89% |
| about 0.75 s | about 50% (the last windows) |

- Lost trials average 0.57 s of dip time, held trials 0.43 s.
- **Time Off is a blurred proxy for dip time.** Decoded time Off correlates only moderately with dip time (r = 0.44), because 100 ms windows, read-out noise and the lagging kick blur the dips. That blurring flattens the relation in the main figure.
- **Implication for the data:** the measured relation between time Off and errors may underestimate how strongly errors depend on the network's actual silences.

**STP model:** memory is 100% at every dip time, up to 0.9 s of dips in the 0.9 s window.

## NMDA time constant fitted to the monkeys' performance (`fig_tau_fit_pooled`)
**Question.** If τ_N is left free, which value best explains how the monkeys' fraction correct falls with time Off?

**Files:** `figures/paper_model/fig_tau_fit_pooled.png`, `.svg`, and `.csv` (best fit on the data, then one row per bootstrap sample). Script: `paper_model/fit_tau_pooled.py`.

**Model curves:**
- The global best-fit NMDA model, with its own fitted dips and all-units read-out, with only τ_N changed.
- 17 values from 20 ms to 1.5 s, 30,000 trials each (`paper_model/tau_sweep.py`).
- For each τ_N: P(memory held) in the 10 Off-fraction bins of Fig. 3b, interpolated in log τ_N.
- Nothing else is refitted, so the On/Off statistics drift away from the fitted values as τ_N changes.

**Data:** all 15,468 trials pooled, correct or error, Off time in 500–1,400 ms. The data file is copied to `paper_model/data/behaviour_per_trial_correct_and_error.csv`.

**Link between model and behaviour:** P(correct) = p₀ × [P(held) + G × P(lost)].
- **G = 1/8:** after the memory is lost, the monkey guesses among the 8 cue locations.
- **p₀:** performance when the memory is held, one value for all trials (fitted 0.887). It absorbs every error that doesn't depend on Off time, such as saccades to a neighbouring location.
- **The model makes no response itself.** It has two pools, cued and uncued, matching the decoder that separates a cue from the opposite cue. Its only output is whether the memory is held.

**Fit and bootstrap:**
- Maximum likelihood over τ_N and p₀.
- Bootstrap over sessions: draw 25 sessions with replacement, pool their trials and refit, 1,000 times.

**Results:**
- **Best fit:** τ_N = 123 ms, a gain of 56 log-likelihood units over constant performance.
  - The model curve follows the pooled fraction correct, from about 0.88 with little time Off to about 0.78 at 900 ms Off (panel a).
- **Session bootstrap:** median 123 ms, quartiles 115–135 ms, 95% interval 101–190 ms.
  - About 2% of samples fall on a weak second likelihood peak near 1 s (panel b, 24 log-likelihood units below the best). At very long τ_N, Off states become long and frequent enough that memory fails again.
- **Conclusion:** the decay of performance with time Off asks for an NMDA-like trace of about 120 ms. That is close to measured NMDA kinetics and to the 150 ms of the model.

**Sensitivity to the guessing assumption.** The data contain only correct or error, not the chosen location. If a monkey kept partial information after a "loss", its guesses would be correct more often than 1/8. That gives a higher G, so more losses are needed to explain the same drop, and the fitted τ_N comes out shorter. The effect is small:

| G (P(correct) after a loss) | best τ_N | memory lost overall |
|---|---|---|
| 0 (always an error) | 129 ms | 3.9% |
| 1/8 (random among 8; used) | 123 ms | 4.5% |
| 1/4 | 118 ms | 5.2% |
| 1/2 | 102 ms | 8.4% |

For any plausible G, τ_N stays between about 100 and 130 ms, inside the bootstrap interval. With the chosen location on each trial, errors to neighbouring locations could be separated from true losses; these data don't allow it.

**Caveat:** the bootstrap covers variability across sessions, not uncertainty in the model's other fitted parameters.

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
| `fig_off_time_models.py` | the figure |
| `natural_readout.py` | top-row trials |
| `single_dip_readout.py` | bottom-row trials |
| `variant.py` | chooses the model: `PM_VARIANT=fac` for the STP model, the default for the NMDA model |
| `model_ours_stp.py` | the STP model |
| `tau_sweep.py`, `fit_tau_pooled.py` | τ_N sweep and pooled fit to the monkeys' performance |

The STP fit script, `fit_ours_stp.py`, is on branch `paper-model-mongillo`.
