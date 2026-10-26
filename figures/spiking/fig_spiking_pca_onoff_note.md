# Hand-off: delay activity in the cue-PCA plane during On and Off states, spiking network (`fig_spiking_pca_onoff`)

## For the paper agent: what this figure is
The data analysis of the PCA hand-off (`jmourabarbosa/panichello`, branch `analysis-replication`,
`analysis/pca_onoff_handoff/`, panels c and d) applied to the spiking ring network
(`figures/spiking/fig_spiking_note.md`). The layout and the measures are the same as the data panels, so the two can
be placed side by side or the model panel can go to the supplement.

**Message:** as in the data, during Off states activity moves toward the centre of the ring of cue means but keeps a
reduced trace of the cue's direction; before the cue the same measure is 0. In the model the On alignment is higher
and the Off trace stronger than in the data (see caveats).

**Reference figure:** `figures/spiking/fig_spiking_pca_onoff.png` (300 dpi), `.svg`, `.pdf`.

## Panels
**Left (as data panel c). Example simulated session (session 0).**
- Grey: the ring of the 8 cue-condition means of delay activity in PC1–PC2.
- For each cue, the average position of trials during their On moments (filled) and during their Off moments
  (open), with an arrow from On to Off. Inset: colour key of the cue locations.
- Session 0 was chosen as the cleanest of the 10 (the On and Off positions of all 8 cues are separated); the other
  sessions show the same pattern.

**Right (as data panel d). Alignment with the cue, 10 simulated sessions.**
- Cosine between a trial's mean PC1–PC2 position and its cue's direction (the cue's mean from the other trial half),
  minus the same with cue labels shuffled within trial half (50 shuffles).
- On, Off, and the pre-cue null (activity before the cue projected on the same axes).
- Grey lines: sessions; black: mean ± s.e.m.; red crosses: data means (25 sessions).

## Method (as the data, independent units)
1. Activity: spike counts of the 330 recorded cells (264 E, 66 I) in 100 ms windows every 10 ms, z-scored across
   trials at each time point; correct trials (about 410 per session).
2. Cells split at random into halves. On/Off labelled from the top 20 cells of one half (ranked by 8-cue ANOVA of the
   delay rate on the other trial half) with the paper's decoder (cue vs opposite cue, λ = 10,000, leave-one-out,
   30 shuffles, On/Off labelling as the data). PCA and projections use the other half only.
3. PCA on the 8 cue-condition means of delay activity (500–1,400 ms), cross-validated across trial halves; the two
   planes aligned by an orthogonal Procrustes rotation fitted on the cue means.
4. Per trial: mean position over its On moments, over its Off moments, and over its pre-cue activity.
5. **Pre-cue window: centres −400 to −50 ms** (windows ending before cue onset). The data use −400 to 0 ms; in the
   model the cue drives spikes within a few ms (no visual latency), so windows reaching past 0 carry the cue (with
   them the "null" would be 0.38).
6. Differences from the data code: decoder features are spike counts (as in the model's main read-out) rather than
   z-scored rates; 10 sessions instead of 25.

## Numbers (10 sessions; data in brackets; `fig_spiking_pca_onoff_stats.json`)
| shuffle-corrected alignment | On | Off | pre-cue (null) |
|---|---|---|---|
| model | 0.78 ± 0.04 | 0.45 ± 0.03 | 0.02 ± 0.01 |
| data | 0.53 ± 0.04 | 0.19 ± 0.03 | 0.002 ± 0.006 |
| model, uncorrected / shuffle baseline | 0.99 / 0.21 | 0.48 / 0.04 | 0.03 / 0.01 |

- On vs Off: paired t-test p = 7×10⁻⁴.
- Off vs 0: p = 4×10⁻⁷; Off is above 0 in 10 of 10 sessions.
- Pre-cue vs 0: p = 0.11 (n.s.).
- Off vs pre-cue: p = 6×10⁻⁷.

## Caveats for the text
- **On alignment is higher than in the data** (uncorrected cosine 0.99): model neurons are far more selective than
  recorded units, so single-trial On activity sits exactly on its cue's direction. The shuffle baseline for On (0.21)
  is also larger than in the data (0.04) because the On positions lie far outside an irregular ring.
- **The Off trace is about twice the data's.** Part of it is a labelling effect specific to the model: the top 20
  labelling cells sit near a few cue locations, so when the bump is elsewhere they are silent and the decoder labels
  "Off" while the bump is still active, which the held-out half sees. When states are labelled from all cells
  instead (the "same units" version, partly circular), the Off alignment drops to near the data's (uncorrected cosine
  0.25 vs 0.09 in the data). In the model, true Off moments (the bump silenced by a dip) carry little cue information
  in firing; the memory is held in the NMDA gating.
- Suggested wording: "the spiking network reproduces the collapse of delay activity toward the centre of the cue
  ring during Off states", not "the residual cue trace during Off states".

## Files (all in `figures/spiking/`)
| file | content |
|---|---|
| `fig_spiking_pca_onoff.png`, `.svg`, `.pdf` | the figure |
| `fig_spiking_pca_onoff_panel_c.csv` | left panel: per cue, the ring position and the On and Off mean positions (session 0), with trial counts |
| `fig_spiking_pca_onoff_panel_d.csv` | right panel: per session and period (on, off, pre), the raw cosine, shuffle baseline and corrected value |
| `fig_spiking_pca_onoff_stats.json` | means, s.e.m. and tests |

## Reproduce
```
python -m spiking.pca_onoff --sessions 0 1 2 3 4 5 6 7 8 9     # re-simulates the sessions' counts (~7 min each)
python -c "from spiking import pca_onoff as P; P.paper_panels(list(range(10)), ex=0)"
```
Code: `spiking/pca_onoff.py` (functions `run_session`, `paper_panels`); the simulated sessions are those of
`spiking/run.py sessions`.
