# Hand-off: supplementary figure, read-out control without dips (reviewer point on Fig. 2)

## Reviewer point
The read-out is phenomenological: z = c·Δr + noise, with the noise statistics taken from the data. How much of the On/Off statistics comes from the noise model rather than the network? Control (a): apply the same read-out to the network with no dips (constant common input). How many Off states does noise alone produce?

## Files
All in `figures/paper_model/`:
- `figS_control_no_dips.png` (600 dpi) and `.svg` (editable text)
- `figS_control_no_dips.json`: every plotted value per unit set and condition, the read-out gains, and memory held

Script: `paper_model/control_no_dips.py`. Run with `python -m paper_model.control_no_dips` (about 2 min).

## What was done
**Model:** the fitted NMDA model of Figs. 1–2 (`model.FIT`), with the common input held constant (dip depth 0, no dips). Nothing else changed.

**Read-out:** exactly as in Fig. 2, for each of the 7 decoded populations: same noise model (unit variance with the autocorrelation of overlapping 100 ms windows, plus the slow per-trial share measured in the data), same cluster-mass On threshold, same Off criterion, same windows. 2,112 trials.

**Read-out gain c.** For the fitted model with dips, "same gain" and "recalibrated" are the same thing: its gain is the one calibrated to the data's μ_on. Only the network without dips has two versions:
1. **Same gain:** each population's gain from the fitted model. This is the identical read-out applied to a network that never falls silent.
2. **Recalibrated gain:** the gain set again from each population's μ_on in the data, as the pipeline would do for this model. This gives noise alone the best chance to reproduce the data.

## Figure layout
Same layout as Fig. 2, with the six On/Off statistics against decoded population:
- **Panels:** a, Off states per trial; b, Off median length; c, fraction of delay Off; d, On states per trial; e, On median length; f, fraction of delay On.
- **Loss panels:** g, fit loss per decoded population; h, loss summed over the 7 populations. Log axes, one bar per condition.
- **Colours:** black, data (mean ± 2 SEM over sessions); red, fitted model with dips; blue, no dips with the same gain; purple, no dips with the recalibrated gain.

## Key numbers
**Off states per trial / fraction of the delay Off:**

| population | data | fitted model | no dips, same gain | no dips, recalibrated |
|---|---|---|---|---|
| top 5 | 3.50 / 0.36 | 2.49 / 0.24 | 0.00 / 0.00 | 0.08 / 0.00 |
| top 20 | 3.25 / 0.33 | 2.51 / 0.25 | 0.00 / 0.00 | 0.10 / 0.00 |
| top 50 | 3.15 / 0.34 | 2.68 / 0.27 | 0.00 / 0.00 | 0.27 / 0.01 |
| rand 5 | 4.80 / 0.65 | 4.48 / 0.59 | 2.10 / 0.14 | 4.77 / 0.57 |
| rand 20 | 4.65 / 0.57 | 4.18 / 0.52 | 0.53 / 0.03 | 4.50 / 0.45 |
| rand 50 | 4.27 / 0.53 | 3.91 / 0.47 | 0.11 / 0.00 | 3.95 / 0.34 |
| all (fitted) | 3.25 / 0.45 | 3.10 / 0.39 | 0.00 / 0.00 | 2.37 / 0.20 |

**On, fraction of the delay:**
- No dips, same gain: 1.00 for the top sets and all units, 0.27–0.83 for the random sets.
- No dips, recalibrated: 0.72–0.86 for the top sets (data 0.39–0.44), 0.21 for all units.

**Fit loss** (the fit loss, as in Fig. 2; panels g, h):

| population | fitted model (dips) | no dips, same gain | no dips, recalibrated |
|---|---|---|---|
| top 5 | 13 | 827 | 97 |
| top 20 | 10 | 818 | 82 |
| top 50 | 9 | 814 | 59 |
| rand 5 | 12 | 61 | 17 |
| rand 20 | 11 | 117 | 17 |
| rand 50 | 9 | 148 | 19 |
| all | 10 | 463 | 22 |
| **sum over 7** | **74** | **3,248** | **313** |

**Memory held:** 98.4% in the fitted model, 100% without dips.

## Interpretation, for the text
- **With the identical read-out, noise alone produces almost no Off states.** The top-unit and all-unit decoders are On throughout the delay; only random 5 shows some Off states (2.1 per trial, 14% of the delay). In the fitted model, Off states come from the network falling silent during dips.
- **Even with the gain recalibrated, noise alone cannot reproduce the data.** It matches the random populations, but gives almost no Off states for the top-unit decoders (0.1–0.3 vs 3.2–3.5 per trial, On 72–86% of the delay vs 39–44%) and too little Off time for all units (0.20 vs 0.45). No single setting of the read-out lets noise explain strong and weak decoders together (summed loss 313, or 3,248 with the same gain, vs 74).
- **For weakly decoding populations, read-out noise contributes substantially to Off states.** This is consistent with, and directly illustrates, the paper's point that Off-state frequency depends on how well the decoded population reads out the memory.

## Draft caption (to adapt)
**Supplementary Figure X | Off states are not produced by the read-out noise alone.** On/Off statistics of the seven decoded populations, as in Fig. 2.
- **Black:** data (mean ± 2 SEM over 25 sessions).
- **Red:** fitted model.
- **Blue:** the same model with a constant common input (no dips), read out with the identical read-out (same gain, noise and labelling) for each population.
- **Purple:** the same no-dip model with the read-out gain recalibrated to each population's decoding strength in the data.

Without dips, the read-out produced no Off states for the top-unit and all-unit decoders and few for random neurons. With the gain recalibrated, noise alone reproduced the Off states of the random populations but not those of the top-unit decoders (summed fit loss over the seven populations 3,248 with the same gain and 313 recalibrated, vs 74 for the fitted model; g, h).

## Draft sentence for the response to the reviewer
> With a constant common input and the identical read-out (same gain, noise model and labelling), the model produced no Off states for the top-unit or all-unit decoders, and few for random neurons (Supplementary Fig. X). Even when the read-out gain was recalibrated to the data, noise alone reproduced the Off states of the weakly decoding random populations but not those of the top-unit or all-unit decoders (top sets: 0.1–0.3 vs 3.2–3.5 Off states per trial; summed loss 313 vs 74 with dips). The Off states of strong decoders therefore require the network to fall silent, whereas for weak decoders read-out noise contributes substantially, consistent with our conclusion that Off-state frequency depends on the decoded population.

## Note
This uses the paper's current model, with the lagging feed-forward inhibition. Without dips, the overshoot version (`figures/paper_model/overshoot/README_overshoot.md`) behaves identically, because both mechanisms act only around dips.
