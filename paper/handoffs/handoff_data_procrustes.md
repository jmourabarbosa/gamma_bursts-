# Hand-off to the data-analysis agent: Procrustes shape similarity of On, Off and pre-cue activity

Fig. 4e will compare the recordings and the spiking network with a Procrustes shape similarity instead of the
per-trial cosine. It is computed in the high-dimensional space of the held-out units, not in the 2-PC plane. Please
compute it for the 25 sessions following `procrustes_common_spec.md` (same folder). The network agent does the
same, and I will only plot.

## Data specifics
- **Labelling:** the `split` version of `analysis/pca_onoff.py`, on the top 20 units of half H1.
  - `S.label(*S.top_within(H1))`;
  - H1/H2 from `default_rng(1).permutation(S.n_units)`.
- **Activity of H2:**
  - delay `S.Zt[:, S.sw][:, dl][:, :, H2]`, with `dl = (S.tsw >= 500) & (S.tsw <= 1400)`;
  - pre-cue `S.Zt[:, S.tt < 0][:, :, H2]`.
- **Correct trials only**, as in the existing PCA panels.
- A minimal runner that I tested on 210921 (14 s per session):

  ```
  S = M.Session(path)
  pu = np.random.default_rng(1).permutation(S.n_units); H1, H2 = np.sort(pu[:S.n_units//2]), np.sort(pu[S.n_units//2:])
  _, st, _ = S.label(*S.top_within(H1))
  dl = (S.tsw >= M.DEL[0]) & (S.tsw <= M.DEL[1])
  res, ks = shape_similarity(S.Zt[:, S.sw][:, dl][:, :, H2], S.Zt[:, S.tt < 0][:, :, H2], S.cue, st[:, dl],
                             halves(S.cue, np.random.default_rng(0)))
  ```

- Panel d keeps your existing `panel_c.csv` (session 221024), so nothing else is needed.

## Deliverables
These go on your branch, in `analysis/pca_onoff_handoff/procrustes/`:
- `procrustes_sessions.csv`;
- `procrustes_stats.json`;
- a short note.
