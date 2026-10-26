# Procrustes shape similarity of On, Off and pre-cue activity (Fig. 4e), common spec

Fig. 4e will show, per session, whether the eight per-cue centroids of activity during On moments, Off moments
and before the cue keep the shape of the ring of cue means. It replaces the per-trial cosine. The recordings and
the spiking network must be computed identically so the paper agent only plots.

**Do not run Procrustes in the 2-PC plane.** Two PCs are fitted to the ring, so a 2-D comparison mostly tests the
PCA itself. Use the high-dimensional activity of the held-out units, in two versions (both are reported, the paper
picks one).

## Method
1. **Splits** (same as the current PCA panels):
   - Cells: `pu = np.random.default_rng(1).permutation(n_cells)`, H1 = `pu[:n//2]`, H2 = `pu[n//2:]` (sorted).
   - Trials: halves stratified by cue, `rng = np.random.default_rng(0)`; for each cue, permute its trials and put
     the second half (`idx[len(idx)//2:]`) in half 1.
2. **Labelling.** On/Off from H1 with the paper's decoder and criteria (see each agent's note for which H1 cells).
   All other steps use H2 only.
3. **Activity.** z-scored rates in 100 ms windows (step 10 ms), as for the existing PCA.
   - Delay, window centres 500–1,400 ms.
   - Pre-cue window centres:
     - data: −400 ms to 0;
     - network: −400 to −50 ms.
4. **Per-trial vectors** (length n_H2). Mean activity over the trial's delay points labelled On, over those labelled
   Off, and over its pre-cue points. A trial with no On (or Off) point is left out of that period.
5. **Reference ring.** For trial half h, the 8 cue means C_h (8 × n_H2) are computed from the delay activity
   (all delay points) of the trials of the *other* half. They are centred by their mean over cues, μ_h.
6. **Centroids.** For the trials of half h, average the period's vectors per cue (8 × n_H2) and subtract μ_h. Drop
   cues with no trial; skip the half if fewer than 4 cues remain.
7. **Two spaces.**
   - `full`: all H2 units.
   - `pcs`: project both the centroids and C_h on the first k PCs of C_h. k is the smallest number of PCs holding
     ≥ 90% of the variance of C_h. Report k.
8. **Procrustes.** Use `scipy.spatial.procrustes(C_h, centroids)`, which allows translation, rotation (any
   orthogonal map) and uniform scaling. **Shape similarity = 1 − disparity.** Average the two trial halves.
9. **Chance.** Repeat with cue labels shuffled within each trial half, 50 shuffles. Report the shuffle mean and its
   95th percentile.

Reference implementation: `procrustes_shape.py` (this folder), `shape_similarity(Zd, Zpre, cue, state, half)`.
- Zd is trials × delay points × H2 units, Zpre is trials × pre points × H2 units, and state is +1 On / −1 Off / 0.
- It returns `{space: {period: (similarity, shuffle_mean, n_trials)}}` and k per half.
- Add the 95th percentile of the shuffles.

Expect chance to be high in `full` space: in one data session it was about 0.5, because 8 points in many
dimensions are easy to rotate onto any ring. In `pcs` space chance was about 0.2–0.26.

Test on data session 210921 (`pcs`, k = 3):

| period | similarity | shuffle |
|---|---|---|
| On | 0.82 | 0.20 |
| Off | 0.14 | 0.26 |
| pre | 0.23 | 0.26 |

## Outputs (commit to your own branch, then send the paths)
1. `procrustes_sessions.csv`, one row per session × period × space. Columns:
   - `session`, `period` (on/off/pre), `space` (full/pcs), `k_pcs`, `similarity`, `shuffle_mean`, `shuffle_p95`,
     `n_trials`.
2. `procrustes_stats.json`, per space:
   - n_sessions;
   - mean and s.e.m. of similarity per period;
   - shuffle mean per period;
   - paired t-tests across sessions: On vs Off, Off vs pre, Off vs its shuffle, pre vs its shuffle;
   - number of sessions with Off above its shuffle p95.
3. A short note (≤ 1 page): the method as run, any deviation from this spec, and the numbers.
4. Do not make a figure for the paper. A quick check plot is fine.
