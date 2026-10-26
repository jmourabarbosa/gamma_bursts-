# Procrustes shape similarity (Fig. 4e), spiking network: note for the paper agent

Computed as in `procrustes_common_spec.md` and `handoff_spiking_procrustes.md` (branch
`claude/intermittent-rate-coding-paper-ced984`), with `spiking/procrustes_run.py` and a copy of
`procrustes_shape.py` (`spiking/procrustes_shape.py`).

## Method as run
- **Sessions:** 10 sessions (`run.sessions` seeds 0–9, the same trials as `results/pca_onoff/session_s.npz`), correct
  trials only (about 410 per session), 330 recorded cells.
- **Splits:** cells split with `default_rng(1)` (H1, H2 = 165 each); trial halves stratified by cue with
  `default_rng(0)`.
- **Labelling:** On/Off from **all cells of H1** (`pca_onoff.label_states(X, cue, H1, half, default_rng(s + 1), tz)`),
  as asked.
- **Activity:** H2 only, z-scored 100 ms window counts (step 10 ms).
  - Delay window centres 500–1,400 ms.
  - Pre-cue window centres −400 to −50 ms.
- **Procrustes:** `shape_similarity` exactly as in the reference, 50 shuffles.
  - `pcs` space: k = 3 or 4 PCs (≥ 90% of the cue-mean variance; k per half in the CSV).
- **Deviations from the spec:** one addition only. `shape_similarity` also returns the 95th percentile of the
  shuffles. Nothing else changed.
- **Counts:** the 10 ms counts of all 10 sessions are committed in `spiking/results/pca_onoff/counts_s{s}.npz`
  (keys `counts`, `cue`, `held`), so nothing needs re-simulation.

## Numbers (10 sessions, mean ± s.e.m.; shuffle mean in brackets)
| space | On | Off | pre-cue |
|---|---|---|---|
| `pcs` (k = 3–4) | 0.995 ± 0.000 (0.30) | 0.93 ± 0.01 (0.31) | 0.32 ± 0.04 (0.32) |
| `full` (165 units) | 0.994 ± 0.000 (0.46) | 0.64 ± 0.01 (0.59) | 0.62 ± 0.00 (0.61) |

Paired t-tests across sessions:

| | `pcs` | `full` |
|---|---|---|
| On vs Off | 3×10⁻⁴ | 3×10⁻¹⁰ |
| Off vs pre | 3×10⁻⁷ | 0.02 |
| Off vs its shuffle | 3×10⁻¹¹ | 0.001 |
| pre vs its shuffle | 0.97 | 0.24 |
| sessions with Off above its shuffle p95 | 10 of 10 | 6 of 10 |
| sessions with pre above its shuffle p95 | 2 of 10 | 0 of 10 |

## Read before plotting: Off is very small but ring-shaped
- **In `pcs` space the model's Off value (0.93) is far above the data's** (test session 210921: Off 0.14, below its
  shuffle).
- **Procrustes removes size.** The model's per-cue Off centroids are tiny: the root-mean-square spread of the 8
  Off centroids is 4% of that of the On centroids (pre-cue: 8%, unstructured). But they are consistently arranged
  like the ring, so after rescaling they fit it almost as well as On.
- **Why they are ring-shaped:** the 100 ms windows of Off points near the edges of an Off state still contain spikes
  of the bump just before it falls silent or just after it restarts, always at the cued location. Averaged over
  about 25 trials per cue, this weak residue keeps the ring's shape.
- **What this means for the comparison:** a high model Off value says "a small, cue-consistent residue", not "the
  memory is visible in Off activity". The data's low Off value says Off activity there has no consistent ring shape
  at this resolution. The model therefore does not reproduce the data on this measure in `pcs` space.
- **In `full` space** the model's Off is only slightly above its shuffle (0.64 vs 0.59), closer to the data's
  pattern. On the other hand, chance there is high (0.6), as the spec warns.
- **Recommendation:** if the panel shows the model, show it with its shuffle lines and with the size of the
  configurations (for example the Off/On spread ratio), or report the model in the text only.

## Panel d (2-D scatter, example session 0, all-H1 labelling)
`net_panel_c.csv`, one row per cue:
- `cue`, `angle_deg`;
- `ring_pc1`, `ring_pc2`: the mean of the two halves' aligned cue means;
- `on_pc1`, `on_pc2`, `n_on`, `off_pc1`, `off_pc2`, `n_off`: per-cue means of the trials' On and Off positions in
  the existing 2-PC plane (`split_proj`), with the trial counts.

With the all-H1 labelling, the Off positions of every cue collapse to nearly one point near the centre
(spread 0.13 against 9.7 for the On positions). This is the same small-but-structured residue as above.

## Files (in `figures/spiking/`)
| file | content |
|---|---|
| `procrustes_sessions.csv` | one row per session × period × space: `k_pcs` (half 0; half 1), `similarity`, `shuffle_mean`, `shuffle_p95`, `n_trials` |
| `procrustes_stats.json` | per space: means, s.e.m., shuffle means, paired t-tests, counts above the shuffle p95 |
| `net_panel_c.csv` | panel d values |

## Reproduce
```
python -m spiking.procrustes_run     # ~2 min per session, from the committed counts
```
