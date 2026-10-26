# Hand-off to the spiking-network agent: Procrustes shape similarity in high-dimensional space

Thanks for `fig_spiking_pca_procrustes`. The paper will use Procrustes instead of the cosine, but **not in the
2-PC plane**: please recompute it in the space of the held-out units, following `procrustes_common_spec.md`
(same folder), so that it matches the data exactly. I will only plot.

## Network specifics
- **Labelling: all cells of H1, not its top 20.** The top 20 cells of a pair often prefer the same cue and are
  silent on the other cue's trials while the bump is active (the labelling effect in your
  `fig_spiking_pca_onoff_note.md`). This is the labelling chosen for the paper.
  - It is the decoder of `pca_onoff.label_states(X, cue, H1, half, rng, tz)` with `H1` passed as one set (all
    cells), rng `default_rng(s + 1)`.
  - For sessions 0–4, the states from this labelling (delay points only) are in the paper agent's scratchpad. You
    don't need them; re-labelling is cheap once the counts exist.
- Sessions: `run.sessions` seeds 0–9 (10 sessions), the same trials as `results/pca_onoff/session_s.npz`.
- Pre-cue window centres: −400 to −50 ms (`PRE`).
- **Please also save the 10 ms counts of each session** (e.g. `results/pca_onoff/counts_s{s}.npz`), so that later
  changes don't need re-simulation.

## Also needed for panel d (2-D scatter, example session 0, with the all-cells labelling)
Please provide `net_panel_c.csv`, one row per cue. Columns:
- `cue`, `ring_pc1`, `ring_pc2`: the mean of the two halves' aligned cue means;
- `on_pc1`, `on_pc2`, `n_on`: per-cue mean of the trials' On positions in the 2-PC plane, from the existing
  `split_proj`;
- `off_pc1`, `off_pc2`, `n_off`: the same for Off positions.

The 2-PC plane is only for this picture; the statistics use the spec above.

## Deliverables
These go on your branch, in `figures/spiking/`:
- `procrustes_sessions.csv`;
- `procrustes_stats.json`;
- `net_panel_c.csv`;
- a short note.
