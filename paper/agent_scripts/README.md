# Paper agent's helper scripts

These scripts were written in the paper agent's session scratchpad and are kept here for reproducibility. Paths inside
them point to that scratchpad and to `/home/user/...`, so they need editing before they run elsewhere.

| Folder | What it does | Feeds the paper? |
|---|---|---|
| `doc_sync/` | Converts the Google Doc (read with the Docs API) into `paper/main.tex`. `rd.py` flattens a doc read into text, flags and indices. `sync_tex.py` rewrites Abstract, Introduction, Results, Discussion and captions in `main.tex`, and `post_sync.py` restores formatting. The Methods in `main.tex` were maintained by hand. `make_figures.py` and `mf_orig.py` are earlier figure scripts. | Yes, `paper/main.tex` text |
| `doc_edits/` | One-off builders of Google Docs batchUpdate requests (suggested edits to the doc). | No, edit history only |
| `spiking_topbottom/` | Exploratory diagnostics of top-20 vs all-unit decoding in the spiking network. These are earlier versions of `paper/scripts/spiking_topbottom.py` (branch `paper-model`), which produces `paper/scripts/data/spiking/fig_spiking_b_top20.csv` for Fig. 4b. `example_top20.py` made the top-20 trace that was removed from Fig. 4a. | No (superseded) |
| `pca_procrustes/` | `pca_halfall.py` was an all-cells relabelling of the network PCA and was stopped before it finished. `procr.py` is the reference Procrustes shape-similarity code given to the data and spiking agents (also `paper/handoffs/procrustes_shape.py`). `run_data.py` was a one-session test on the recordings. | No. Fig. 4d,e use the agents' outputs (`spiking/procrustes_run.py` on `claude/gallant-volta-bkjojq`, `analysis/procrustes_paper_handoff/` on `jmourabarbosa/panichello`) |

Paper figures are produced by the scripts in `paper/scripts/` on branch `paper-model` (Figs. 3, 4 and the single-dip
supplementary figure) and by `paper_model/` (Figs. 1, 2 and the other supplementary figures).
