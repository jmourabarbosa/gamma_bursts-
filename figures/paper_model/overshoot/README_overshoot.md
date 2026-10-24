# Hand-off: the model without feed-forward inhibition (attention overshoot)

## What changed
**Old version.** The paper model has a lagging feed-forward inhibition (w_ff = 0.8, τ_ff = 15 ms) driven by the attention signal.

**What the inhibition actually does.** Tests showed that only one of its effects matters: the brief kick it gives both pools when attention returns after a dip. That kick restarts the cued pool.
- Without the kick, memory survives in about 5–7% of trials.
- The extra silencing at dip onset is not needed.
- Removing the inhibition and refitting with stronger NMDA instead fits badly: all-units loss 15.7 vs 9.7, Off fraction 0.27 vs 0.45 in the data. Fits are in `figures/paper_model/no_ff/`.

**New version (`model.OVERSHOOT`, `ff_mode = 1`).** There is no inhibitory pathway. The attention signal itself briefly overshoots when it recovers from a dip:

  input to each pool = I_s m(t) + A [m(t) − m̄(t)]

- **m̄ (the slow copy of the signal)** follows m instantly into a dip and recovers with τ = 15 ms.
- **Overshoot term A [m − m̄]:** a positive pulse after each dip, and nothing at dip onset.
- **Constants:** I_s = 0.696 (the old I_0 − w_ff) and A = 0.8 (the old w_ff), so the steady input in and out of dips is unchanged. A and τ are fixed constants, not fitted.
- **Everything else is as in the paper model.**
- **Possible interpretation:** a transient re-engagement of attention.

## Fit
- **Procedure:** as the paper model. CMA-ES on the all-units read-out only, the same 7 fitted parameters and loss, 3 seeds; the best on the fitted set (seed 2) is used.
- **Files:** `fit.py --overshoot`; fits in `figures/paper_model/overshoot/fit_run*.json`.
- **Fitted parameters (`model.FIT_OVERSHOOT`):**
  - τ_N = 137 ms
  - w_s = 1.31
  - dip rate 11.4/s, mean dip length 96 ms, gamma shape 8.3, depth 0.68
  - σ = 0.011
- **Paper model, for comparison:** τ_N 150 ms (at the upper bound), w_s 1.30, 10.2/s, 90 ms, shape 8.7, depth 0.60, σ 0.006.

**Scored on 2,112 new trials:**

| | paper model (feed-forward inhibition) | overshoot model |
|---|---|---|
| all-units loss (fitted set) | 9.68 | 9.49 |
| loss summed over the 7 decoded populations | 73.2 | 69.5 |
| memory held | 98.3% | 97.8% |
| all units: Off per trial / mean / fraction | 3.13 / 152 ms / 0.39 | 3.16 / 150 ms / 0.40 |
| all units: On per trial / mean / fraction | 1.87 / 186 ms / 0.18 | 1.99 / 179 ms / 0.17 |

The other two seeds: all-units loss 10.1 and 11.2, 7-set loss 66.9 and 71.1.

## Figures for this version
Each file is the overshoot counterpart of an existing figure, made with `PM_VARIANT=os` or `PM_FIG4_NMDA=os`; numbers follow the list.

| figure | file |
|---|---|
| Fig. 1 | `figures/paper_model/fig1_trial_histograms_os.png` |
| Fig. 1 with the input distributions in panel b | `figures/paper_model/fig1_trial_histograms_os_inputs.png` |
| Fig. 2 (7 decoded populations) | `figures/off_vs_memory/os/fig2_unit_sets.png` |
| Fig. 4, both rows, vs the STP model | `figures/paper_model/fig4_off_time_models_os.png` |
| memory vs total dip time | `figures/paper_model/figS_dip_time_models_os.png` |
| τ_N fitted to the monkeys' performance | `figures/paper_model/fig_tau_fit_pooled_os.png` |
| logistic regression of memory loss | `figures/paper_model/fig_model_glm_os.png`, `fig_model_glm_matched_os.png` |
| dips vs Off states | `figures/paper_model/figS_dips_vs_states_os.png` |
| up periods vs On states | `figures/paper_model/figS_ups_vs_on_states_os.png` |

In Fig. 1, the attention trace is drawn as the pools receive it, with its overshoot after each dip, scaled to 1 outside dips.

### Key numbers, overshoot model (paper model in brackets)
**Fitted dips, 100,000 trials:**
- Memory held overall: 98.1% (98.5%).
- Memory by time Off:

  | time Off | memory held |
  |---|---|
  | 0–0.1 s | 99.5% (99.8%) |
  | 0.4–0.5 s | 98.1% (98.8%) |
  | 0.6–0.7 s | 96.8% (97.3%) |
  | 0.8–0.91 s | 94.2% (94.3%) |

- Mean time Off: held 0.37 s, lost 0.55 s (0.36 / 0.60 s).

**Single dip (cut-off):**
- At the fitted depth 0.68, memory held is 97% at 350 ms, 50% at 375 ms, 9% at 400 ms and 1% at 425 ms (paper model: 100%, 29%, 0% from 400 ms).
- With full dips, the cut-off is at about 425 ms.

**τ_N fitted to the monkeys' pooled performance** (guess 1/8 after a loss, session bootstrap):
- Best 110 ms (123 ms).
- Bootstrap median 110 ms, 95% interval 90–172 ms (101–190 ms).
- Gain of 57 log-likelihood units over constant performance (56).

**Logistic regression of memory loss per SD**, at the data's sample size (1,000 draws of 15,468 trials; about 284 errors per draw):

| statistic | odds ratio | draws with p < 0.05 |
|---|---|---|
| total Off time | 1.98 | 100% |
| Off length | 1.38 | 100% |
| number of Off states | 1.26 | 98% |
| total On time | 0.36 | 100% |

**Dips vs Off states:**
- 56% of dips are seen as an Off state (57%), and 97% of Off states overlap a dip (95%).
- Off length vs dip time inside it: r = 0.82 (0.82).
- Per trial, total dip time vs total Off time: r = 0.49 (0.46).

**Up periods vs On states:**
- 27% of up periods are seen as an On state (24%), and 85% of On states overlap one (83%).
- On length vs up time inside it: r = 0.68 (0.63).
- Per trial, total up time vs total On time: r = 0.62 (0.59).

## Conclusion
The overshoot model reproduces every result of the paper model, with equal or slightly better fits, and without introducing feed-forward inhibition:
- the On/Off statistics of all 7 decoded populations;
- memory held in about 98% of trials;
- the decline of memory with time Off;
- the cut-off at about 375–400 ms of silence;
- the fitted τ_N of about 110–120 ms;
- the contrast with the STP model.

The Methods can describe a single attention signal that dips and briefly overshoots on recovery.
