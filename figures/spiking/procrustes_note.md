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

## Numbers, spaces of the spec (10 sessions, mean ± s.e.m.; shuffle mean in brackets)
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

## Normalised to the shuffle (as the data hand-off; use this for the figure)
Same normalisation as the recordings (`analysis/procrustes_paper_handoff/` on `jmourabarbosa/panichello`,
branch `analysis-replication`), using their `procrustes_normalize.py` (copied to `spiking/procrustes_normalize.py`):

    normalised = 1 − disparity / shuffle disparity = (similarity − shuffle mean) / (1 − shuffle mean)

0 = chance, 1 = identical shape. Spaces renamed to match the data files: `pcs` = 7 PCs (our `pcs7`), `full` = all
165 held-out units; the earlier 90%-variance run is kept as `pcs_var90`.

| normalised, mean ± s.e.m. (data, 25 sessions) | On | Off | pre-cue |
|---|---|---|---|
| network, 7 PCs (10 sessions) | 0.99 ± 0.00 (0.50) | 0.84 ± 0.02 (0.36) | −0.01 ± 0.06 (0.01) |
| network, all held-out units | 0.99 ± 0.00 (0.49) | 0.13 ± 0.03 (0.32) | 0.01 ± 0.01 (0.00) |

- **7 PCs:** On vs Off p = 7×10⁻⁵; Off vs pre p = 9×10⁻⁷; Off > 0 p = 3×10⁻¹¹ (10/10 sessions above the shuffle
  p95); pre vs 0 p = 0.93.
- **All units:** On vs Off p = 2×10⁻¹⁰; Off vs pre p = 0.002; Off > 0 p = 0.001 (6/10 above the shuffle p95);
  pre vs 0 p = 0.24.
- **Order On > Off > pre-cue = 0 in both spaces, as in the data.**
- **Unlike the data, the two spaces disagree for Off** (0.84 vs 0.13; data 0.36 vs 0.32). The network's Off
  centroids are tiny (about 2% of the On spread in the PC plane). In the 7 cue-mean PCs their small,
  cue-consistent residue keeps the ring's shape; across all 165 units the noise in the other dimensions dominates.
  The data's Off activity keeps the ring's shape at a much larger size, so it survives in both spaces.
- **Labelling differs from the data:** the network uses all cells of H1, as the paper agent asked; the data README
  says the data used the top 20 units of H1. If the figure should compare like with like, one of the two needs
  re-running.

Files: `procrustes_sessions_normalised.csv` (adds `normalised`, `normalised_p95`), `procrustes_stats_normalised.json`
(same structure as the data's), `procrustes_normalised_check.png` (quick check, network vs data means; not a paper
figure).

## Third space, as requested: 7 PCs (`pcs7`)
The first 7 PCs of the cue means of each trial half, i.e. the whole cue-mean subspace (8 cue means span at most 7
dimensions). Same labels, splits and shuffles as above; only the number of PCs is fixed at 7 instead of the 90% rule
(`k_fixed=7` in the copy of `shape_similarity`). Rows with `space = pcs7` in `procrustes_sessions.csv`; key `pcs7` in
`procrustes_stats.json`.

| `pcs7` (10 sessions) | On | Off | pre-cue |
|---|---|---|---|
| similarity, mean ± s.e.m. | 0.994 ± 0.001 | 0.91 ± 0.01 | 0.48 ± 0.03 |
| shuffle mean | 0.38 | 0.47 | 0.49 |
| similarity minus shuffle | 0.61 ± 0.003 | 0.45 ± 0.01 | −0.00 ± 0.03 |

- On vs Off: p = 7×10⁻⁵ (minus shuffle: 1×10⁻⁶).
- Off vs pre: p = 8×10⁻⁷ (minus shuffle: 8×10⁻⁷).
- Off vs its shuffle: p = 8×10⁻¹¹; Off above its shuffle p95 in 10 of 10 sessions.
- Pre vs its shuffle: p = 0.95; pre above its shuffle p95 in 2 of 10 sessions.

**Plot it minus the shuffle.** In 7 dimensions chance differs between periods (On 0.38, Off 0.47, pre 0.49), so
raw values are not comparable across periods. Minus the shuffle the order is On > Off > pre = 0. The same caveat as in
`pcs` applies: the Off configuration is about 4% of the size of the On one; its high value reflects its shape only.

## Read before plotting: Off is very small but ring-shaped
- **In `pcs` space the model's Off value (0.93) is far above the data's** (test session 210921: Off 0.14, below its
  shuffle).
- **Procrustes removes size.** The model's per-cue Off centroids are tiny: the root-mean-square spread of the 8
  Off centroids is 4% of that of the On centroids (pre-cue: 8%, unstructured). But they are consistently arranged
  like the ring, so after rescaling they fit it almost as well as On.
- **Why they are ring-shaped (likely explanation, not yet verified):** the 100 ms windows of Off points near the
  edges of an Off state still contain spikes
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
| `procrustes_sessions.csv` | one row per session × period × space (`full`, `pcs`, `pcs7`): `k_pcs` (half 0; half 1), `similarity`, `shuffle_mean`, `shuffle_p95`, `n_trials` |
| `procrustes_stats.json` | per space: means, s.e.m., shuffle means, paired t-tests, counts above the shuffle p95 |
| `net_panel_c.csv` | panel d values |

## Reproduce
```
python -m spiking.procrustes_run                                       # full and pcs, ~2 min per session
python -c "from spiking import procrustes_run as R; R.main_k(7)"     # adds pcs7 (uses the stored labels)
```
