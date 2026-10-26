# Session archive (not used by the paper code)

Files that existed only in the working container of the modelling session, kept for reference.

- `fits/`: raw CMA-ES runs (best parameters `best.json`, progress logs), one folder per fit series. The values used by the
  paper code are already in `paper_model/model.py` (FIT, FIT_OVERSHOOT) and `paper_model/model_ours_stp.py`
  (FIT_FAC, FIT_DEP); fits are also copied in `figures/paper_model/{overshoot,no_ff}/` and, for the STP models, in
  `figures/mongillo/` on branch `paper-model-mongillo`.
  - `fits_all`, `fits_one`: fits of the paper NMDA model; `fits_ww`, `fits_ww2`: abandoned Wong-Wang variant;
    `fits_stp`: Mi et al. (2017) STP model; `fits_hyb`: our model with STP (facilitation + depression); `fits_fac`:
    facilitation only (tau_f 1.5 s); `fits_noff`: without feed-forward inhibition; `fits_os`: attention overshoot;
    `fits_session`: per-session fits (stopped early, incomplete).
- `scratch/`: one-off test scripts and intermediate files, not part of the analysis pipeline, e.g.
  `ff/ff_test.py` (which edge of the feed-forward lag matters; kick-size scan), `model_response_*.py` (memory read out in
  a response period), `beh/per_trial.csv` (monkey behaviour file).
