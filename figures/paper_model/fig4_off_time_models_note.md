# Hand-off: Figure 4, memory vs time Off in the delay (NMDA model vs STP model)

Files (all in `figures/paper_model/`):
- `fig4_off_time_models.png` (600 dpi) and `.svg` (editable text)
- `fig4_off_time_models_data.csv`: the plotted values
  - panel a: one row per window, with `memory_held`, the 95% CI and `n_trials`
  - panels b and c: one row per bin, with `n_held` and `n_lost`

Reproduce (about 4 min):
```
python -m paper_model.single_dip_readout
PM_VARIANT=fac python -m paper_model.single_dip_readout
python -m paper_model.fig_off_time_models
```

## What the figure shows

**The question.** Does the memory survive in a model where Off states are long?

**Single-dip test**, run the same way for both models:
- Each trial has exactly one dip of the common input (attention signal).
  - Length: 0 to 1,000 ms, in 25 ms steps, with 300 trials per length (12,300 trials per model).
  - Start: 500 ms after cue onset.
  - Depth: each model's fitted depth (NMDA model 0.60, STP model 0.39). There are no other dips.
- Each trial is read out like the data, with the all-units read-out:
  - the fitted model's all-units gain, fixed and not recalibrated;
  - the all-units noise, with a 0.35 slow share;
  - On/Off labelling as in the main analysis.
- **Time Off in delay** = number of Off time points in 500–1,400 ms × 10 ms (91 points, at most 0.91 s). This is the same definition as the behavioural analysis in Fig. 3.
- **Memory held** = cued pool's slow variable exceeds the uncued pool's by more than 0.05 in the last 100 ms of the delay. The slow variable is NMDA gating in our model and facilitation u in the STP model.

**Panels:**
- **a:** fraction of trials with the memory held against time Off.
  - Sliding windows as the behavioural analysis, converted to seconds: 23 ms wide, steps of 2.3 ms, from 45 ms.
  - Shading: 95% Wilson interval.
  - Black: NMDA model. Blue: STP model.
- **b, c:** distribution of time Off for trials with the memory held (grey) or lost (red), for the NMDA model (b) and the STP model (c).

## The two models

- **NMDA model:** the paper model (`model.py`, `model.FIT`), fitted to the all-units read-out.
- **STP model** (`model_ours_stp.py`, `FIT_FAC`): identical to the NMDA model except for the slow recurrent term.
  - The NMDA gating variable s (τ_N 150 ms) is replaced by the same saturating synaptic activation without a time constant of its own, S = g/(1 + g), with g = β r u x / U.
  - It is weighted by Tsodyks–Markram short-term facilitation u (Mi, Katkov & Tsodyks 2017):
    - du/dt = (U − u)/τ_f + U(1 − u) r
    - U = 0.3, τ_f = 1.5 s (the working-memory time scale of Mongillo et al. 2008 and Mi et al. 2017)
  - **Facilitation only:** depression is switched off (x = 1). With depression (τ_d 0.3 s) the model fits the data badly (all-units loss 37 vs 10, three fits). Depression terminates each re-ignition, so On states are too short (112 vs 172 ms in the data).
  - All other fixed parameters are as in the NMDA model, including the AMPA self-excitation, inhibition, common input and lagging feed-forward inhibition.
  - Fitted like the NMDA model: CMA-ES, all-units read-out only, same loss, plus a penalty for population spikes in the 1 s before the cue. Six parameters, with τ_f fixed:
    - w_s = 0.69
    - dip rate 5.5/s, mean length 212 ms, gamma shape 0.70 (lower bound)
    - depth 0.39
    - σ = 0.005 (lower bound)
  - Resting facilitation (u ≈ 0.5 at 1 Hz) already strengthens the recurrent synapses, hence the lower w_s.

**Both models fit the On/Off statistics about equally well.** Scored on 2,112 new trials:

| | NMDA model | STP model |
|---|---|---|
| all-units loss | 9.7 | 10.4 |
| loss summed over the 7 decoded populations (6 predicted) | 72.6 | 76.4 |
| Off: per trial / mean length / fraction of delay | 3.07 / 154 ms / 0.39 | 2.87 / 184 ms / 0.40 |
| On: per trial / mean length / fraction of delay | 1.86 / 184 ms / 0.18 | 2.14 / 168 ms / 0.21 |
| memory held | 98.5% | 100% |

Data: Off 3.25 / 186 ms / 0.45; On 2.10 / 172 ms / 0.21.

## Key numbers

### NMDA model
- **Memory against dip length is a step.**
  - Held in 100% of trials for dips up to 350 ms, 29% at 375 ms, and 0% from 400 ms.
  - Without the read-out the step is the same: 100% to 350 ms, 34% at 375 ms, 0% at 400 ms.
  - With a full dip (depth 1) the step moves to about 450 ms, because the feed-forward kick when the input returns is stronger.
- **Why:** during a dip both pools are silent and the NMDA gating decays (τ_N 150 ms). When the input returns, the lagging feed-forward inhibition kicks both pools. The cued pool re-ignites only if enough gating is left.
- **A lost memory is not swapped:** neither pool re-ignites, and the uncued item is never held.
- **Through the read-out the step becomes graded** (panel a): memory held falls from about 0.88 at 0.06 s Off to about 0.05 at 0.9 s. The read-out spreads each dip length over a range of Off times:
  - dips under about 100 ms give no read-out Off;
  - dips of 100–350 ms (held) give a median Off of about 0.1–0.66 s;
  - dips of 400 ms and longer (lost) give a median of about 0.7 s, because the rest of the delay reads Off once the memory is lost.
- **Mean Off time:** 0.18 s in held trials, 0.65 s in lost trials.

### STP model
- **The memory is never lost:** 12,300 of 12,300 trials held, for every dip length up to 1 s, and also with full dips.
  - Facilitation decays with τ_f = 1.5 s, so after a 1 s dip the cued pool still has a much higher u than the uncued pool and re-ignites.
- **Same in the other tests:** with the dip rate varied from 0 to 30/s or the depth from 0.3 to 1.0, memory is 100% in all 24,000 trials. At 30 dips/s, Off covers 50% of the delay.
- **The read-out sees the dips:** median Off time rises from 0.08 s (100 ms dip) to 0.76 s (1 s dip). Mean Off time is 0.38 s.

## Interpretation, for the text

**Both slow mechanisms account for the On/Off statistics,** but they make opposite behavioural predictions:
- **NMDA model:** a slow-but-short trace (about 150 ms). The memory is lost when an Off period outlasts the trace, so errors come with more time Off. This matches Fig. 3: errors follow more total Off time and longer Off states, but not more Off states.
- **STP model:** facilitation on the working-memory time scale (1.5 s) bridges any Off period up to 1 s, so it predicts no relation between time Off and errors.

**Caveats:**
- The argument concerns the time scale of the trace, not NMDA as such. An earlier unfitted test with facilitation at τ_f 150 ms gave nearly the same read-out as the NMDA model.
- The single-dip test makes 63% of trials errors by design, because lengths are uniform up to 1 s. Panel a should be compared with the data in shape (a graded decline with time Off), not in level.
  - In the fitted NMDA model with its own random dips, memory is 98.9%. It still falls with time Off, but only at the high end (about 0.93 at 0.9 s), because long Off times there are mostly several short Off periods.
- **Prediction for the data:** performance should depend on the longest Off period of a trial more than on total Off time, with a drop between about 350 and 450 ms.

## Code
All code is in `paper_model/` on branch `paper-model`:
- `fig_off_time_models.py`: this figure.
- `single_dip_readout.py`: the per-trial data.
- `variant.py`: chooses the model; set `PM_VARIANT=fac` for the STP model.
- `model_ours_stp.py`: the STP model.

Also on `paper-model`:
- The fit script, `fit_ours_stp.py`, is on branch `paper-model-mongillo`.
- Supplementary analyses for both models are in `figures/off_vs_memory/{nmda,fac}/`:
  - memory against dip length without the read-out;
  - the dip-rate and dip-depth sweeps.
