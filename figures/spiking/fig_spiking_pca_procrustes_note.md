# Hand-off: Procrustes shape similarity of On, Off and pre-cue activity to the cue ring (`fig_spiking_pca_procrustes`)

## For the paper agent: what this figure is
An alternative to the cosine "alignment with the cue" panel (panel d of the PCA figure, and
`fig_spiking_pca_onoff_note.md`). Instead of a per-trial cosine, it asks, per session, whether the eight per-cue
average positions keep the shape of the ring of cue means:
- during **On** moments they do, almost perfectly;
- during **Off** moments they keep it partly: lower than On, but above pre-cue activity and above chance;
- **before the cue** they do not (at chance).

So the ordering On > Off > pre-cue = chance holds with a measure that does not depend on single-trial cosines.
This panel is computed on the spiking network only; the data counterpart has not been computed yet (see "For the
data").

**Reference figure:** `figures/spiking/fig_spiking_pca_procrustes.png` (300 dpi), `.svg`, `.pdf`. It is drawn in
the style of the data PCA panels, so it can replace panel d or sit next to it.

## Method
Same PCA as the cosine panel (independent units):
1. Cells split at random into halves. On/Off labelled from the top 20 cells of one half with the paper's decoder.
   PCA of the 8 cue-condition means of delay activity (500–1,400 ms) on the other half, cross-validated across
   trial halves.
2. Each trial's mean PC1–PC2 position is computed over its On moments, over its Off moments, and over its pre-cue
   activity (window centres −400 to −50 ms).
3. For each trial half, the 8 per-cue centroids of these positions are compared with the cue means of the other
   trial half (the half the trials were projected with). Comparison: Procrustes analysis allowing translation,
   rotation and uniform scaling (`scipy.spatial.procrustes`). The disparity is the residual sum of squares after the
   best fit, between 0 (same shape) and 1. **Shape similarity = 1 − disparity.**
4. The two halves are averaged, giving one value per session and period. Cues with no trial in a period are left
   out (at least 4 cues are required).
5. **Chance:** the same with cue labels shuffled within trial half (50 shuffles). Chance is not 0: 8 random points
   still fit an 8-point ring partly (about 0.2).

**Methods sentence (draft):**
> To compare the geometry of activity across states, we computed, for each session, the eight cue-specific mean
> positions in the cue-PCA plane during On moments, Off moments and before the cue, and compared each
> configuration with the cross-validated ring of cue means by Procrustes analysis (translation, rotation and
> uniform scaling). Shape similarity is 1 minus the Procrustes disparity; chance was estimated by shuffling cue
> labels within trial halves.

## Numbers (spiking network, 10 simulated sessions; `fig_spiking_pca_procrustes_stats.json`)
| shape similarity | On | Off | pre-cue |
|---|---|---|---|
| model (mean ± s.e.m.) | 0.98 ± 0.002 | 0.51 ± 0.02 | 0.22 ± 0.02 |
| shuffled cues | 0.33 | 0.21 | 0.22 |

Paired t-tests across sessions:
- On vs Off: p = 1×10⁻⁸.
- Off vs pre-cue: p = 4×10⁻⁶.
- Off vs its shuffle: p = 3×10⁻⁷ (Off keeps the ring's shape).
- Pre-cue vs its shuffle: p = 0.91 (at chance).

## How to draw it
- One grey line per session (On to Off), and a grey dot per session for pre-cue.
- Black: mean ± s.e.m.
- Red dashes: the shuffle mean for each period.
- Y axis 0 to 1, labelled "shape similarity to the cue ring (1 − Procrustes disparity)".
- Keep the shuffle lines: because chance is about 0.2–0.3, the values only mean something next to them.

## Caveats for the text
- **It measures shape, not size.** Scaling is removed, so it does not show that Off activity moves toward the centre
  of the ring. Keep panel c (On → Off arrows) or the projection on the cue direction for that.
- **On is near the ceiling in the model** (0.98) because model neurons are much more selective than recorded units;
  the data's On value is expected to be lower.
- **The model's Off value is probably inflated** by the labelling effect described in `fig_spiking_pca_onoff_note.md`.
  The 20 labelling cells sit near a few cue locations, so when the bump is elsewhere they call "Off" while the bump
  is still active.
- **One value per session.** The measure is computed on per-cue centroids, so there is one value per session and
  period, not per trial.

## For the data
To show the same panel for the recordings, the data agent needs to run the same computation on the data's per-trial
PCA positions (`pca_trials.csv` from `pca_onoff.py`, version "split"; periods on, off, pre) with the session's
cross-validated cue means (`split_cmeans`, `split_half` in `results_pca/sessions/*.npz`). The model code is
`procrustes_panel` in `spiking/pca_onoff.py` (about 40 lines). It uses only these inputs and can be copied.

## Files (all in `figures/spiking/`)
| file | content |
|---|---|
| `fig_spiking_pca_procrustes.png`, `.svg`, `.pdf` | the panel |
| `fig_spiking_pca_procrustes.csv` | per session and period: shape similarity and its shuffle mean |
| `fig_spiking_pca_procrustes_stats.json` | means, s.e.m., shuffle means and tests |

## Reproduce
```
python -c "from spiking import pca_onoff as P; P.procrustes_panel(list(range(10)))"
```
This needs the per-session PCA results in `spiking/results/pca_onoff/` (from `python -m spiking.pca_onoff
--sessions 0 1 2 3 4 5 6 7 8 9`).
