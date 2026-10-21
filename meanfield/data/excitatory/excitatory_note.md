# Off states in confidently excitatory units

Code: `excitatory.py` (analysis), `excitatory_figures.py` (figures). Data: 25 sessions, correct trials, λ = 10,000.

## Settings
The settings are the same as `mechanism.py` analysis 1, with identical unit splits (seed 1).
- **States:** labelled from the top-20 units of one unit half (H1), drawn from all units.
- **Measurement:** rates measured in the other half (H2), restricted to a unit class.
- **Splits:** 5 per session, both directions.
- **Tuning:** tuned (p < 0.05) or untuned (p > 0.5), with the preferred cue taken from the other trial half.
- **Rates:** 20 ms boxcar, as a ratio to each unit's own pre-cue rate (−400 to 0 ms).

**Check:** for session 240222, the "all" class reproduces the `mechanism.py` per-session values exactly.

## Classification (from `ei_classify.py`, `results_ei/`)
Units are classified from monosynaptic connections in jitter-corrected cross-correlograms. There are no waveforms. See `results_ei/ei_note.md`.

| class | criterion | units (all sessions) |
|---|---|---|
| **E_strict** (main) | E connection label (≥ 1 excitatory output, no inhibitory output) **and** ≥ 1 excitatory output that is not a sharp > 3× peak **and** not a likely multi-unit cluster (≤ 1% of ISIs < 1.5 ms) | 509 |
| E_any (sensitivity) | any E connection label | 1,250 |
| not_E | all other units | 6,975 |
| all | all units | 8,225 |

- **No putative inhibitory units could be identified.** So not_E means "not shown to be excitatory". It is a mix of interneurons and excitatory cells whose outputs went undetected.
- **The E label favours units with high firing rates and many partners.** Those give the correlograms enough power to detect a connection.

**Held-out units per labelling (median over sessions)**, from `unit_counts_per_session.csv`, which has per-session values for every class:

| class | all held-out | tuned | untuned |
|---|---|---|---|
| E_strict | 6.5 | 1.75 | 1.75 |
| E_any | — | 5.25 | 4.5 |

Sessions left with no E_strict units in a group:

| E_strict group | sessions with no units |
|---|---|
| any held-out unit | 3 |
| tuned | 4 |
| untuned | 7 |

Over sessions, mean E_strict counts per labelling are 2.8 tuned and 3.2 untuned. Sessions without units in a group are left out of that group's statistics; the `n_sessions` column gives the count.

## 1. Rate around Off states (`a1_*.csv`, `a1_off_vs_on_tests.csv`, `a1_excitatory.png`)
Mid-state ratio to pre-cue: mean over sessions. The p-value is a paired Wilcoxon test of mid-On vs mid-Off over sessions.

| group | E_strict Off / On (p) | E_any Off / On (p) | not_E Off / On (p) | all Off / On (p) |
|---|---|---|---|---|
| tuned, preferred cue | 1.21 / 1.74 (4e-4) | 1.19 / 2.14 (<1e-4) | 1.69 / 2.80 (<1e-4) | 1.60 / 2.63 (<1e-4) |
| tuned, opposite cue | 1.22 / 0.77 (1e-4) | 1.27 / 0.75 (<1e-4) | 1.19 / 0.81 (<1e-4) | 1.18 / 0.79 (<1e-4) |
| untuned | 1.00 / 0.96 (0.39) | 1.04 / 0.98 (0.014) | 1.05 / 1.04 (0.58) | 1.05 / 1.03 (0.31) |

- **The pattern in excitatory units matches all units.**
  - Tuned excitatory units show the push-pull: they are lower in Off on preferred-cue trials and higher in Off on opposite-cue trials.
  - Untuned excitatory units show **no dip** in Off. They are at the pre-cue level (1.00), and, if anything, slightly above their On level.
  - So the absence of a global dip in untuned units is **not** caused by interneurons diluting the effect.
- **Tuned E units are less strongly modulated than not_E units.** Both their On level and their Off–On difference are smaller.
- **Aligned curves** are in `a1_aligned_curves.csv`, with the number of sessions, total events and mean units per session in each row.
  - The range is −200 to +400 ms.
  - The onset-aligned curves use only Off onsets at 300–1,000 ms (curve names ending `_300to1000`). The offset-aligned curves use all Off states.
- **Untuned E units are flat** around both Off onset and Off offset.

## 2–3. Recovery after Off (`recovery_*.csv`, `recovery_E_strict.png`, `recovery_E_any.png`)
Off states that start 300–1,100 ms after cue onset and are followed by an On state within 200 ms. t50 and t90 are taken on the session-mean curves (10 ms bins), then averaged as mean ± SEM over sessions.

**E_strict**

| Off length | tuned, pref. cue: mid-Off → mid-On | t50 / t90 (ms) | untuned: mid-Off → mid-On | t50 / t90 (ms) |
|---|---|---|---|---|
| 50–100 | 1.22 → 1.56 | 22±5 / 36±9 | 0.97 → 0.97 | (13 / 23) |
| 100–200 | 1.27 → 1.60 | 17±4 / 50±13 | 1.00 → 0.95 | (10 / 21) |
| 200–400 | 1.14 → 1.55 | 20±5 / 37±9 | 1.00 → 1.00 | (6 / 9) |
| > 400 | 1.05 → 1.52 | 17±4 / 22±5 | 0.92 → 0.99 | (16 / 43) |

**Fraction of Off states followed within 200 ms** (all Off states starting 300–1,100 ms, tuned-pref analysis)

| Off length | by On (correct cue) | by confidently incorrect |
|---|---|---|
| 50–100 | 0.54 | 0.09 |
| 100–200 | 0.51 | 0.10 |
| 200–400 | 0.46 | 0.12 |
| > 400 | 0.35 | 0.12 |

- **Tuned E units recover almost immediately.** The rate is back near the On level within 20–40 ms of the Off offset, in every length bin. That is close to the time resolution (20 ms boxcar, 10 ms bins, state labels on 10 ms steps), so recovery is at least as fast as the analysis can resolve.
- **Recovery time does not grow with Off length.** The curves for > 400 ms states rest on few events (775) and are noisy.
- **Untuned E units have no Off–On difference to recover from.** Their t50/t90 values (in brackets) therefore just mark threshold crossings of a noisy flat curve and are not interpretable as recovery.
  - Recovery is therefore specific to tuned (memory) cells, not a global rebound.
- **Longer Off states are followed by On less often** (0.54 → 0.35). A confidently incorrect period follows in about 10% of cases, rising slightly with length.
- E_any gives the same picture (slightly slower t50, 20–30 ms).

## Deviations and caveats
- **Classes are small.** E_strict leaves ~2–3 tuned and ~3 untuned held-out units per labelling, and some sessions have none (see the counts above).
  - Per-session curves are therefore noisy. Summary statistics weight sessions equally.
  - E_any is reported as a larger, less strict sensitivity set.
- **Excitatory classification:**
  - It is not a cell-type label, because it depends on detecting a connection.
  - Waveforms would be needed to identify interneurons.
- **Mid levels:**
  - Mid-Off is the whole state for 50–100 ms Off states, which have no middle (onset + 50 to offset − 50 ms).
  - Mid-On is the middle of the following On state if it lasts ≥ 100 ms, and the whole On state otherwise.
- **t50/t90 definition:** the first 10 ms bin at or after the Off offset where the session-mean curve reaches 50% / 90% of the way from the mid-Off to the mid-On level. It is undefined (NaN) if the curve never reaches that level.
  - The bin at 0 ms already includes ±10 ms of smoothing.
- **Curve truncation:** recovery curves stop at the end of the data (1,440 ms bin centre). This affects some long Off states that start late.
- **The "followed by" fractions** come from the same states as the recovery curves. A state can count both as "followed by On" and as "followed by confidently incorrect" if both happen within 200 ms.
- **Units with a pre-cue rate of 0 Hz** are excluded, as in `mechanism.py`.
- **Added beyond the request:** paired Off-vs-On tests in `a1_off_vs_on_tests.csv`.

## Files
- `unit_counts_per_session.csv`
- `a1_per_session.csv`, `a1_summary.csv` (mean, SD, SEM, n_sessions), `a1_off_vs_on_tests.csv`, `a1_aligned_curves.csv`
- `recovery_per_session.csv`, `recovery_summary.csv`, `recovery_curves.csv` (with events and units per bin)
- `a1_excitatory.png`, `recovery_E_strict.png`, `recovery_E_any.png`
- `sessions/*.npz`: raw per-session results
