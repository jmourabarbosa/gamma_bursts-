# Request: what causes Off states? Mechanism checks in the recordings

**Goal.** The model explains Off states as short dips in an input shared by the whole network. Alternatives:
- the memory pool switching off on its own;
- adaptation or synaptic depression ending each burst;
- internally generated global silences;
- read-out failure only.

The analyses below separate these. Report the mean over sessions with session SEM, plus per-session values.

**Settings (same as `table_lambda10000/`).**
- 25 sessions, correct trials, decoder λ = 10,000 with the corrected intercept, approximate leave-one-out, 30 shuffles.
- The same On/Off labelling, with states counted 0–1,400 ms after cue onset.
- Unless stated otherwise, label states with the **top-20 units** (split-half ANOVA ranking, test trial never used).

**Unit split for held-out measures (analyses 1, 3, 4).** Per session, split units at random into halves H1 and H2, with 5 random splits.
- Choose top 20 within H1, label states from them, and measure activity in H2. Then swap the halves.
- Never measure activity in the units used for labelling.

**Tuned vs untuned units.**
- Tuned: one-way ANOVA across the 8 cues on the mean rate 500–1,400 ms after cue onset, p < 0.05, computed on half the trials (the same trial halves as the top-k ranking).
- Untuned: p > 0.5.
- Units in between are left out of tuned/untuned comparisons.

---

## 1. Population rate around Off states (highest priority)
For each Off state that starts 300–1,300 ms after cue onset and lasts ≥ 100 ms, align held-out activity (H2) to its onset, from −200 to +400 ms in 10 ms bins (20 ms boxcar). Do the same aligned to Off offset.

For each held-out unit, express the rate as a ratio to the unit's own mean rate from −400 to 0 ms (pre-cue). Report three groups:
- **tuned units:** on their preferred-cue trials, and separately on their opposite-cue trials (the preferred cue is decided on the other trial half);
- **untuned units,** on all trials;
- **all held-out units.**

Also report, for each group:
- the mean rate ratio over the middle of Off states (from onset + 50 ms to offset − 50 ms), and the same over On states;
- the onset latency of the drop, the first bin at least 2 SD below the −200 to −50 ms baseline, separately for tuned and untuned units.

*Interpretation:* if untuned units drop at Off onset, together with tuned units, the cause is global. If only tuned units drop, the memory pool switches on its own. If rates drop below the pre-cue level, that is a silence, not a return to spontaneous firing.

## 2. Population silences without a decoder: before the cue vs in the delay
Define silences from spiking only, with no decoder:
1. The population rate of **untuned units** (all of them), in 10 ms bins with a 20 ms boxcar, per trial.
2. A **silence** is ≥ 50 ms where this rate is below a threshold. The threshold is the session's 10th percentile of that rate over all pre-cue bins, from −600 to 0 ms or as far back as the data go.
   - Also report the 5th and 20th percentile thresholds.

Count silences, with the same definition, in two windows:
- pre-cue: −600 (or the earliest available) to 0 ms;
- delay: 500–1,400 ms.

Report per window: silences per second, the length distribution (20 ms bins), and the fraction of time silent.

Then, in the delay:
- the fraction of top-20 decoder Off time that overlaps a population silence;
- the fraction of population-silence time that is decoder Off;
- the same two numbers after circularly shifting the silence sequence within each trial, 20 random shifts, as the chance level.

*Interpretation:* shared input predicts silences before the cue at a similar rate to the delay, and decoder Off states that coincide with them. Memory-based mechanisms predict no silences before the cue.

## 3. Disjoint unit sets: do Off states coincide?
Label states separately with H1-top-20 and H2-top-20, and repeat with random 20 drawn from H1 and from H2. Report:
- Off coincidence: the fraction of H1 Off time also Off in H2, and the reverse;
- chance level: the same after pairing H1 with H2 from a different trial of the same cue, and after a circular shift within the trial (20 shifts);
- the same for On;
- the probability that an H1 Off state overlaps any H2 Off state, by H1 Off length: 50–100, 100–200, 200–400 and > 400 ms.

## 4. Regularity of state sequences
Using the top-20 labelling (all units, not split):
- the coefficient of variation (CV) of Off lengths and of On lengths;
- the CV of intervals between successive Off onsets within a trial;
- the correlation between each On length and the next Off length, and between each Off length and the next On length, within trials, pooled per session;
- the same CVs after shuffling state order within trials. This is a sanity check: CV should not change, while the correlations should go to ~0.

*Interpretation:* Poisson-like shared input gives CV ≈ 1 and no correlation between successive states. Adaptation or depression gives CV < 1 and a positive On-to-Off correlation (a long burst is followed by a long recovery).

## 5. Decoding accuracy within each state
For all units, top 20 and random 20:
- accuracy (fraction of confidence > 0.5) within On time points, within Off time points and within unlabelled time points, all within 500–1,400 ms;
- the fraction of delay time with **confidently incorrect** decoding: clusters of z < −1.96 that pass the same cluster-mass test applied to −z;
- the number of such periods per trial, and their length distribution.

This explains why the model's accuracy is ~0.1 higher than the data's at the same μ_on.

---

## Output
- One CSV per analysis: a summary (mean, SD and SEM over sessions) and per-session values.
- Analysis 1: the aligned rate curves.
- Analysis 2: silence length histograms for each window.
- A short note covering:
  - any deviation from this spec;
  - how many Off states and silences entered each average;
  - the earliest pre-cue time available in the data.
