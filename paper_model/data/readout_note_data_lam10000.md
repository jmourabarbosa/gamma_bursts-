> Copied from the data analysis. In the CSV files here, the column `k` (number of decoded units) is renamed `n`,
> since k denotes the gamma shape of the dip lengths in the model; `k` in this note means the same as `n`.

# On/Off state statistics vs read-out, at λ = 10,000

The same request as `results_readout/` (λ = 30), with λ chosen by saturation. Code: `readout_curve.py`. The λ choice is documented in `../results_saturation/saturation_note.md`, and the noise check in `../results_noise/noise_note.md`.

## Files
- `readout_summary.csv`: one row per point (set, k).
  - Values are the mean over sessions, with the across-session SD (`_sd`) and SEM (`_sem`).
  - For random sets, the 10 draws are averaged within each session first.
- `readout_per_session.csv`: one row per session × set × k × draw, with `lam` and `max_grad` (the final Newton gradient).
- `readout_histograms.csv`: On and Off length histograms per point, pooled over sessions and draws.
  - 20 ms bins from 0 to 600 ms, plus a row for > 600 ms.
  - `count_per_trial` = count / (trials × draws).
- `readout_curve.png`: each statistic plotted against accuracy, with λ = 10,000 solid and the earlier λ = 30 run faded.
- `sessions/*.npz`: per-session cache.

All columns requested are included:
- accuracy
- `on_per_trial`, `off_per_trial`
- `on_mean_ms`, `on_median_ms`, `off_mean_ms`, `off_median_ms`
- `frac_on`, `frac_off`
- `mu_on`, `w_on`, `mu_low`, `sd_low`
- `null_sd`, `precue_frac_on`

Median lengths in the summary are the mean of the per-session medians.

## Unchanged from the λ = 30 run
- All 25 sessions; correct trials only, all cue locations (13,190 trials).
- Approximate leave-one-out (Rad & Maleki); 30 shuffled label sets per cue pair.
- Unit sets: all units; top k = 5, 10, 20, 50, 100, 200, ranked by split-half ANOVA so the test trial is never used; random k with 10 draws.
  - Same seeds, so the unit sets, half splits and shuffled labels are exactly the same as in the λ = 30 run.
  - k = 100 uses 20 sessions (10,403 trials) and k = 200 uses 14 sessions (7,246 trials).
- States are counted from 0 to 1,400 ms after cue onset; delay statistics use 500–1,400 ms.
- On and Off labelling, and the μ_on Gaussian mixture, are unchanged.

## Changed (not just λ)
1. **λ = 10,000**, the smallest λ from which every criterion changes by less than 5%.
2. **The intercept is now a constant feature of value 1, penalised like a unit weight.**
   - The λ = 30 run used a constant of 10 (penalty λ/100).
   - With a weakly penalised intercept, the decoder collapses to the class prior at large λ: accuracy falls to about 0.56 for every unit set.
   - At λ = 30 the old intercept made little difference (the same accuracy on a test session). But the λ = 30 table is not computed with exactly this decoder.
3. **Newton's method runs to a maximum gradient below 1e-9** (largest final value over all fits: 1.0e-9), with an exact logistic and no clipping. Confidences are never rounded or clipped, and the labelling SD floor is 1e-300.
   - The old fixed 8 iterations had also converged, to about 1e-13, so this change does not alter results.

## Results (25 sessions; mean over sessions)
| set | k | accuracy | frac_off | frac_on | off/trial | on/trial | off mean (ms) | on mean (ms) | μ_on | w_on | pre-cue false On |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | (329) | 0.707 | 0.446 | 0.211 | 3.25 | 2.10 | 186 | 172 | 2.18 | 0.44 | 2.1% |
| top | 5 | 0.693 | 0.358 | 0.439 | 3.50 | 3.14 | 151 | 184 | 3.94 | 0.55 | 1.2% |
| top | 10 | 0.713 | 0.339 | 0.432 | 3.38 | 3.20 | 145 | 181 | 3.93 | 0.52 | 1.4% |
| top | 20 | 0.730 | 0.328 | 0.425 | 3.25 | 3.11 | 144 | 188 | 3.85 | 0.51 | 1.2% |
| top | 50 | 0.740 | 0.336 | 0.387 | 3.15 | 2.92 | 145 | 192 | 3.48 | 0.49 | 1.4% |
| top | 100 | 0.752 | 0.343 | 0.353 | 3.03 | 2.74 | 152 | 189 | 3.22 | 0.46 | 1.9% |
| top | 200 | 0.748 | 0.377 | 0.289 | 2.97 | 2.36 | 173 | 182 | 2.73 | 0.44 | 2.2% |
| random | 5 | 0.552 | 0.646 | 0.074 | 4.80 | 1.07 | 186 | 114 | 1.14 | 0.49 | 1.1% |
| random | 10 | 0.573 | 0.606 | 0.091 | 4.77 | 1.32 | 174 | 114 | 1.28 | 0.48 | 1.2% |
| random | 20 | 0.596 | 0.572 | 0.105 | 4.65 | 1.51 | 166 | 120 | 1.42 | 0.47 | 1.3% |
| random | 50 | 0.633 | 0.526 | 0.137 | 4.27 | 1.75 | 165 | 135 | 1.69 | 0.46 | 1.6% |
| random | 100 | 0.663 | 0.502 | 0.147 | 3.93 | 1.79 | 173 | 141 | 1.81 | 0.45 | 1.8% |
| random | 200 | 0.677 | 0.507 | 0.121 | 3.71 | 1.62 | 191 | 141 | 1.62 | 0.44 | 1.9% |

Raw `null_sd` (in the CSVs) is between 2.5e-4 (random 5) and 3.7e-3 (all units). At this λ it reflects the 1/λ weight scale rather than the read-out.

## Comparison with λ = 30 (descriptive)
- **Accuracy:** nearly unchanged; within 0.022 at every point. Large sets are slightly lower: all units −0.022, top 200 −0.019, random 200 −0.018.
- **At equal k, top units still beat random units by a wide margin.** At k = 20, top vs random:

  | | top 20 | random 20 |
  |---|---|---|
  | frac_on | 0.43 | 0.11 |
  | frac_off | 0.33 | 0.57 |
  | Off states per trial | 3.3 | 4.7 |
  | μ_on | 3.9 | 1.4 |
- **The drop in On with k is much weaker.**
  - Top k: frac_on falls 0.44 → 0.29 from k = 5 to 200, vs 0.40 → 0.08 at λ = 30. μ_on falls 3.9 → 2.7, vs 3.4 → 1.4.
  - Random k: frac_on now rises with k (0.07 → 0.15 up to k = 100), where it was flat at λ = 30.
  - All units: frac_on is 0.21, vs 0.08 at λ = 30.
- **Off states per trial fall with accuracy for both top and random sets.** The fraction of delay time Off for top sets stays at 0.33–0.38.

## Deviations and caveats
- **The pre-cue false-On rate grows with the number of units:** from 1.1% for random 5 to 1.9–2.2% for 100–200 units and for all units. This matches the noise check, where z autocorrelation at long lags grows with λ and with set size (all units: 0.35 at 200 ms at λ = 10,000). Slow, trial-level offsets in z, probably shared baseline fluctuations, inflate On clusters for large sets. On fractions for large sets therefore include about 1–2 percentage points of false On.
- **The μ_on mixture's lower component is not N(0, 1).**
  - `mu_low` ranges from −1.07 (top 5) to +0.37 (top 200), and `sd_low` from 1.1 to 2.4.
  - `sd_low` is broadest for small top-k sets: 2.4 at top 5, 2.2 at top 20.
  - `w_on` is 0.44–0.55, so the two components split the delay z values roughly in half rather than isolating an "On" mode. μ_on should be read with this in mind.
- **States end at 1,400 ms:** the data do not record each trial's delay end, which varies from 1,400 to 1,600 ms.
- **Shared shuffles:** the 30 shuffled label sets are shared by all test trials of a cue pair; the paper re-shuffled for each test trial.
- **Approximate leave-one-out was checked against exact refits at λ = 10,000** with this decoder: session 221020, one cue pair, 4 time points including one before the cue, all units and 20 units. The maximum difference in confidence was 1.2e-8 (all units) and 1.1e-10 (20 units). The typical deviation of confidence from 0.5 is 3.5e-3 and 4.6e-4, so the approximation error is negligible, and the sign agreed for every trial. At λ = 30 the maximum difference was 3.5e-3 (all units) and 5.6e-3 (20 units).
