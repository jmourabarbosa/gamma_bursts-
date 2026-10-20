# On/Off state statistics vs read-out (unit subsets)

Data: Panichello et al. 2024 (Dryad doi:10.5061/dryad.kkwh70sct; mirror github.com/jmourabarbosa/panichello, commit 1daab3d). Code: `readout_curve.py`.

## Files
- `readout_summary.csv`: one row per point (set, k).
  - Values are the mean over sessions, with the across-session SD (`_sd`) and SEM (`_sem`).
  - For random sets, the 10 draws are averaged within each session first.
  - `mean_units` is the number of units; for "all" it is the session's unit count, which averages 329.
- `readout_per_session.csv`: one row per session × set × k × draw. Each random draw is its own row.
- `readout_histograms.csv`: On and Off length histograms per point, pooled over sessions and draws.
  - 20 ms bins from 0 to 600 ms, plus a row for > 600 ms.
  - `count_per_trial` = count / (trials × draws).
- `readout_curve.png`: each statistic plotted against delay accuracy.
- `sessions/*.npz`: per-session cache. The CSVs can be rewritten from it without recomputing.

## Included data
- All 25 sessions: 10 from monkey A, 8 from H, 7 from J; 8,225 units.
- Correct trials only, all 8 cue locations: 13,190 trials. No other trial selection.
- k values larger than a session's unit count are skipped:
  - k = 100 uses the 20 sessions with ≥ 100 units (10,403 trials).
  - k = 200 uses the 14 sessions with ≥ 200 units (7,246 trials).
  - Averages at large k therefore come from a different, larger-session subset than at small k.

## Decoder (identical for every point)
- **Firing rates:** 100 ms boxcar, 10 ms steps, z-scored across trials per unit and time point.
- **Decoding:** binary, the test trial's cue vs the opposite cue (4 cue pairs), one decoder per time point.
- **Model:** L2 logistic regression with **λ = 30**, objective `sum_i logloss_i + (λ/2)·||w||²`.
  - The intercept is a constant feature of value 10, so its effective penalty is λ/100 = 0.3.
  - Fit to convergence by Newton's method, 8 iterations from zero. The unit-space or trial-space form is used, whichever is cheaper; both have the same objective.
  - No class rebalancing. Cue pairs are close to balanced.
  - λ was chosen once, on delay accuracy for all units and for random 20 units, in sessions 211015 and 221020. Accuracy was flat for λ from 10 to 100, and λ = 30 was best or tied. λ was not tuned per point.
- **Leave-one-trial-out:** approximate leave-one-out for ridge GLMs (Rad & Maleki 2018), `η_{-i} = η_i + (p_i − y_i)·h_i / (1 − v_i·h_i)`.
  - Checked against exact refits (session 221020, all units and 20 units, 3 time points).
  - At λ = 10 the mean absolute difference in confidence was ≤ 0.0004 and the maximum ≤ 0.009, with identical accuracy. Agreement improves as λ increases.
- **Confidence:** the probability the decoder gives to the trial's true label.
- **Null:** **30 shuffles** per cue pair; training labels are permuted within the pair.
  - Each test trial gets 30 null decoders, applied with the same leave-one-out. The paper re-shuffled for each test trial; here the 30 shuffled label sets are shared by all test trials of a pair.
  - z = (confidence − null mean) / null SD, per trial and time point.
  - `null_sd` = the SD of null confidence across the 30 shuffles, averaged over trials and time points from 500 to 1,400 ms.

## State labelling
- **On:** contiguous runs with z > 1.96 whose summed z exceeds the 95th percentile of the null distribution.
  - The null distribution is, for each of the 30 shuffled series, its largest positive cluster mass (0 if none).
  - Each shuffled series is z-scored against all other series, including the real one, as in the authors' `clusterMassOneSampZ.m`.
- **Off:** runs of at least 5 consecutive time points (≥ 50 ms) with z < 0.8416.
- **Window:**
  - States are labelled and counted from 0 to 1,400 ms after cue onset.
  - The data stop at 1,450 ms (the last full 100 ms window is centred at 1,400 ms), and don't record each trial's delay length (1,400–1,600 ms). States therefore end at 1,400 ms, not at each trial's own delay end.
  - A run's length is the number of points × 10 ms.
- **Counts and fractions:**
  - `on_per_trial` and `off_per_trial` count states over 0–1,400 ms, averaged over all trials.
  - `frac_on` and `frac_off` are fractions of the 500–1,400 ms time points.
- **Accuracy:** the fraction of 500–1,400 ms time points with confidence > 0.5.
- **Extra column `precue_frac_on`:** the same labelling applied separately to −400 to 0 ms, where any On is a false positive. It is 0.7–1.0% at every point.

## Top-k ranking (test trial never used)
- Trials are split in half at random, stratified by cue, with a fixed seed per session.
- Units are ranked by one-way ANOVA F across the 8 cues of the mean z-scored rate over 500–1,400 ms, computed on one half.
- Test trials in half A use units ranked on half B, and the reverse.
- The decoder itself is trained on all trials except the test trial, as for every other point.
- This replaces the earlier top-k analysis (`topk_onoff.py`). That one also ranked on the other half of the trials, but by SGD decoder weight, with a different decoder and null.

## Random k
- 10 draws per k, sampled uniformly without replacement, with a fixed seed per session.
- The same draw is used for all trials of a session.

## μ_on (read-out strength)
- Per session and point: a 2-component Gaussian mixture (sklearn `GaussianMixture`, `random_state = 0`) fit to all z values from 500 to 1,400 ms, over all trials. If there are more than 40,000 values, a random 40,000 are used.
- Reported per session:
  - `mu_on` and `w_on`: the higher component's mean and weight.
  - `mu_low` and `sd_low`: the lower component's mean and SD.
- These are averaged across sessions in the summary.
- Note: the lower component is **not** close to N(0, 1). `mu_low` is −0.2 to −0.9 and `sd_low` is 0.8–2.3. It is broadest for small top-k sets: top 5 has `sd_low` 2.3 and top 10 has 2.0. So the two-component split does not cleanly separate "null-like" from "On" z values, especially at small k. `mu_on` should be read with this in mind.

## Results (25 sessions; mean over sessions)
| set | k | accuracy | frac_off | frac_on | off/trial | on/trial | off mean ms | on mean ms | μ_on | null SD |
|---|---|---|---|---|---|---|---|---|---|---|
| all | (329) | 0.729 | 0.423 | 0.078 | 4.09 | 1.66 | 132 | 91 | 1.38 | 0.187 |
| top | 5 | 0.685 | 0.375 | 0.397 | 3.61 | 3.03 | 152 | 173 | 3.36 | 0.040 |
| top | 10 | 0.709 | 0.359 | 0.369 | 3.59 | 3.09 | 143 | 162 | 2.93 | 0.058 |
| top | 20 | 0.731 | 0.348 | 0.329 | 3.55 | 3.00 | 138 | 153 | 2.48 | 0.083 |
| top | 50 | 0.746 | 0.358 | 0.237 | 3.62 | 2.74 | 132 | 130 | 1.91 | 0.126 |
| top | 100 | 0.762 | 0.359 | 0.156 | 3.70 | 2.43 | 128 | 104 | 1.63 | 0.165 |
| top | 200 | 0.767 | 0.373 | 0.082 | 3.85 | 1.76 | 128 | 81 | 1.42 | 0.197 |
| random | 5 | 0.549 | 0.668 | 0.065 | 4.76 | 0.99 | 194 | 111 | 1.02 | 0.038 |
| random | 10 | 0.570 | 0.619 | 0.078 | 4.88 | 1.22 | 173 | 108 | 1.16 | 0.057 |
| random | 20 | 0.594 | 0.582 | 0.080 | 4.90 | 1.36 | 159 | 107 | 1.23 | 0.082 |
| random | 50 | 0.635 | 0.531 | 0.082 | 4.78 | 1.50 | 146 | 103 | 1.30 | 0.125 |
| random | 100 | 0.671 | 0.494 | 0.067 | 4.68 | 1.39 | 138 | 91 | 1.29 | 0.163 |
| random | 200 | 0.695 | 0.478 | 0.037 | 4.66 | 1.01 | 135 | 75 | 1.21 | 0.196 |

### Observations (descriptive, not interpreted against any model)
- **The null widens with k:** the null SD of confidence rises steeply with k, from 0.04 at k = 5 to 0.20 at k = 200. It is almost identical for top and random sets of the same size. With λ fixed, decoders with more units fit shuffled labels more, so the null widens regardless of which units are used.
- **More units, more accuracy, less On:** accuracy rises with k for both top and random sets. But On time *falls* with k for top units (0.40 → 0.08), and μ_on falls (3.4 → 1.4). For random units, On time stays flat at about 0.07. On labelling is therefore not monotone in read-out accuracy; null width pulls it the other way.
- **At equal k (equal null width), top units have:**
  - much more On time (for example 0.33 vs 0.08 at k = 20)
  - less Off time (0.35 vs 0.58)
  - fewer Off states per trial (3.6 vs 4.9)
  - higher μ_on (2.5 vs 1.2)
  - higher accuracy (0.73 vs 0.59)

  The random-k control isolates this signal effect.
- **Off time with top units:** it is about 0.35–0.37 for every k from 5 to 200, while accuracy goes from 0.69 to 0.77.
- **Pre-cue false-On rate:** ≤ 1.0% at every point.
