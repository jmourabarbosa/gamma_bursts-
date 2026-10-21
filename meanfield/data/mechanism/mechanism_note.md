# What causes Off states? Mechanism checks

Code: `mechanism.py`. Settings as `table_lambda10000/`:
- 25 sessions, 13,190 correct trials.
- L2 logistic regression, λ = 10,000, intercept penalised like a unit weight, approximate leave-one-out, 30 shuffles.
- The same On/Off labelling, with states over 0–1,400 ms.
- Unsplit labellings (analyses 2, 4 and 5) rebuild exactly the table's trial halves, unit sets and shuffled labels. Checked: analysis 5's accuracy, frac_on and frac_off reproduce the table to all digits.
- Unit splits (analyses 1 and 3) use 5 random halves H1/H2 per session (seed 1).
- Top 20 within a unit half are ranked by cue ANOVA over 500–1,400 ms on the other trial half, so the test trial is never used.
- Tuned (p < 0.05) / untuned (p > 0.5) and the preferred cue are computed on the other trial half.
- Rates for analyses 1 and 2: spikes in 10 ms bins with a 20 ms boxcar, every 10 ms, centred −490 to 1,440 ms.
- **The earliest pre-cue time available in the data is −500 ms**; the data end at +1,450 ms.

Summary CSVs give the mean, SD and SEM over sessions; per-session CSVs give every session.

## Files
| analysis | files |
|---|---|
| 1 | `a1_summary.csv`, `a1_per_session.csv`, `a1_aligned_curves.csv` (mean ± SEM per bin, with sessions and events per bin), `a1_aligned_curves_per_session.csv`, `a1_aligned_curves.png` |
| 2 | `a2_summary.csv`, `a2_per_session.csv`, `a2_silence_length_histograms.csv` |
| 3 | `a3_summary.csv`, `a3_per_session.csv` |
| 4 | `a4_summary.csv`, `a4_per_session.csv` |
| 5 | `a5_summary.csv`, `a5_per_session.csv`, `a5_confidently_incorrect_length_histogram.csv` |
| counts | `counts.json` |

## Counts entering the averages
- **Analysis 1:**
  - 199,029 Off states (onset 300–1,300 ms, length ≥ 100 ms) over 5 splits × 2 directions × 25 sessions. 152,134 of them have onsets at 300–1,000 ms.
  - 159,511 On states, used for the mid-state On ratios.
  - 116 units with zero pre-cue firing are excluded from the ratios (4.6 per session).
- **Analysis 2:**

  | threshold | silences before the cue | silences in the delay (500–1,400 ms) | silences in the delay (900–1,400 ms) |
  |---|---|---|---|
  | 5th percentile | 8,015 | 14,898 | 8,406 |
  | 10th percentile | 8,738 | 16,274 | 9,105 |
  | 20th percentile | 11,400 | 21,794 | 11,885 |

- **Analysis 3:** H1 Off states starting in the delay, all sessions and splits:

  | H1 Off length | top 20 | random 20 |
  |---|---|---|
  | 50–100 ms | 49,905 | 61,551 |
  | 100–200 ms | 50,562 | 69,003 |
  | 200–400 ms | 21,200 | 34,156 |
  | > 400 ms | 5,319 | 9,373 |

## Deviations from the spec
1. **Pre-cue window:** −500 to 0 ms, not −600 ms, because the data start at −500 ms.
2. **Analysis 1, late bins:**
   - Alignment windows run past the end of the data (1,450 ms) for late onsets and offsets. Those bins are left empty, and `a1_aligned_curves.csv` gives the event count per bin.
   - As requested, there are also curves from Off onsets at 300–1,000 ms only (`*_300to1000`). For these, every onset-aligned bin up to +400 ms uses the same states.
3. **Analysis 1, drop latency:**
   - Computed on each session's mean curve: the first bin from −40 ms onward that is ≥ 2 SD below the −200 to −50 ms baseline.
   - The criterion is met in only 14 of 25 sessions for tuned units on their preferred-cue trials, and 9 of 25 for untuned units. Most curves have no clear drop, so the latency values are weak; see the results.
4. **Analysis 2, threshold ties:** the untuned population fires sparsely (about 102 untuned units per session on average, about 4 Hz each before the cue).
   - The 5th, 10th and 20th percentile thresholds often tie at low rates; in some small sessions they are 0 Hz. A strict "below threshold" rule then finds nothing.
   - **A silent bin is therefore rate ≤ threshold.**
   - The effective percentiles, the fraction of pre-cue bins at or below threshold, are 23%, 27% and 34% on average for the nominal 5th, 10th and 20th. Per-session thresholds and effective percentiles are in `a2_per_session.csv`.
5. **Analysis 2, matched window:** as requested, delay statistics are also given for 900–1,400 ms (`delay500`), the same 500 ms length as the pre-cue window. The overlap with decoder Off is computed over the full delay window (500–1,400 ms).
6. **Analysis 3:**
   - Coincidence is measured over 500–1,400 ms, so the shared response to the cue does not inflate it.
   - Random 20 uses one draw from each unit half per split, so 5 pairs per session.
   - Overlap probability uses H1 Off states starting in 500–1,400 ms; their chance level uses H2 from a different trial of the same cue.
7. **Analysis 4:**
   - CVs of Off and On lengths are reported three ways: all states, including those cut at 0 or 1,400 ms; interior states only; and states starting 200–700 ms after the cue. Of the 200–700 ms states, 24 per session (about 2%) still reach the 1,400 ms edge.
   - The shuffle permutes the order of state segments (On, Off and unlabelled) within each trial; 20 shuffles.
8. **Analysis 5:**
   - Confidently incorrect decoding uses the same cluster-mass labelling applied to 1 − confidence, i.e. to −z.
   - Accuracy within On is 1.00 by construction: On requires z > 1.96 against a null centred at 0.5.

## Results (mean ± SEM over 25 sessions)
### 1. Population rate around Off states (held-out units; rate / own pre-cue rate)
| group | mid-Off | mid-On |
|---|---|---|
| tuned, preferred-cue trials | 1.60 ± 0.22 | 2.63 ± 0.19 |
| tuned, opposite-cue trials | 1.18 ± 0.04 | 0.79 ± 0.06 |
| untuned | 1.05 ± 0.02 | 1.03 ± 0.03 |
| all held-out units | 1.09 ± 0.02 | 1.17 ± 0.03 |

- **Untuned units do not drop at Off onset.** They stay within about 5% of their pre-cue rate throughout (see `a1_aligned_curves.png`).
- **Tuned units on preferred-cue trials:** the rate declines gradually into the Off state, about 1.8 → 1.5, and rises again afterwards.
  - It stays well above the pre-cue level: 1.60 mid-Off.
  - So the rate does not fall below the pre-cue level and does not return to spontaneous firing.
- **Tuned units on opposite-cue trials:** the rate goes *up* in Off (1.18) compared with On (0.79).
- **A brief transient:** about +50 ms after Off onset (and about −50 ms before offset), every group shows a small bump. It is largest for opposite-cue tuned units, about 1.2.
- **Drop latency:**
  - tuned, preferred-cue trials: −3 ± 10 ms (14 of 25 sessions had a detectable drop)
  - untuned: +48 ± 46 ms (9 of 25 sessions)

### 2. Population silences of untuned units, no decoder (10th-percentile threshold)
| window | silences per s | fraction of time silent | mean length |
|---|---|---|---|
| pre-cue (−500 to 0 ms) | 1.29 ± 0.32 | 0.150 ± 0.049 | 82 ms |
| delay (500–1,400 ms) | 1.34 ± 0.32 | 0.164 ± 0.051 | 87 ms |
| delay, matched 500 ms (900–1,400 ms) | 1.37 ± 0.34 | 0.159 ± 0.051 | 82 ms |

- Silences are about equally frequent before the cue and in the delay, at every threshold (see the CSV for 5th and 20th).
- Overlap with top-20 decoder Off, in the delay:

  | measure | observed | chance (circular shift) |
  |---|---|---|
  | fraction of Off time that is silent | 0.167 | 0.166 |
  | fraction of silent time that is Off | 0.336 | 0.343 |

  The results are the same at the 5th and 20th percentile thresholds. **Decoder Off states do not coincide with population silences beyond chance.**

### 3. Disjoint unit halves (delay, 500–1,400 ms)
| | top 20: observed | chance, other trial | chance, shift | random 20: observed | chance, other trial |
|---|---|---|---|---|---|
| Off in H1 also Off in H2 | 0.544 | 0.428 | 0.465 | 0.636 | 0.592 |
| On in H1 also On in H2 | 0.540 | 0.428 | 0.444 | 0.239 | 0.172 |

P(an H1 Off state overlaps an H2 Off state), top 20, observed vs chance:

| H1 Off length | observed | chance |
|---|---|---|
| 50–100 ms | 0.52 | 0.49 |
| 100–200 ms | 0.68 | 0.60 |
| 200–400 ms | 0.86 | 0.74 |
| > 400 ms | 0.95 | 0.85 |

States labelled from disjoint top-20 halves coincide above chance, by about 8–12 percentage points.

### 4. Regularity (top-20 labelling, all units)
| | all states | interior only | start 200–700 ms | shuffled order (interior) |
|---|---|---|---|---|
| CV of Off lengths | 0.83 | 0.79 | 0.83 | 0.80 |
| CV of On lengths | 0.86 | 0.78 | 0.93 | 0.82 |

- The three versions agree within about 0.15. All CVs are 0.78–0.93, below but close to 1, with no strong regularity.
- CV of intervals between successive Off onsets: 0.62 ± 0.01; after shuffling, 0.78.
- Correlations of successive state lengths:

  | | observed | shuffled |
  |---|---|---|
  | On → next Off | −0.04 ± 0.01 | −0.06 |
  | Off → next On | −0.07 ± 0.01 | −0.06 |

  There is **no positive On → Off correlation.** The small negative values match the shuffle, consistent with the fixed trial length.

### 5. Decoding accuracy within states (delay, 500–1,400 ms)
| set | accuracy in On | in Off | in unlabelled | overall | confidently incorrect: fraction of delay | per trial | mean length |
|---|---|---|---|---|---|---|---|
| all units | 1.00 | 0.36 | 0.98 | 0.71 | 2.3% | 0.26 | 110 ms |
| top 20 | 1.00 | 0.23 | 0.92 | 0.73 | 6.6% | 0.77 | 104 ms |
| random 20 | 1.00 | 0.32 | 0.95 | 0.60 | 2.9% | 0.42 | 92 ms |

**Accuracy within Off states is well below chance:** 0.23–0.36, not 0.5. During Off time points, the decoder more often favours the opposite cue. The confidently-incorrect periods sit inside Off states; there are 0.77 per trial with top 20.
