# Spiking ring attractor with intermittent (On/Off) decoding

## What was built
- **Network (`network.py`).** A ring of 2,048 pyramidal cells and 512 interneurons (leaky integrate-and-fire), as in
  Compte, Brunel, Goldman-Rakic & Wang (2000):
  - external Poisson AMPA drive;
  - recurrent excitation only through NMDA receptors (voltage-dependent Mg²⁺ block, saturating gating), with the
    **NMDA decay at 150 ms**;
  - GABA_A inhibition, broad (all-to-all, untuned);
  - E→E weights tuned in angle (Gaussian footprint).

  Every value and its source is in `PARAMETERS.md`.
- **Common input m(t) to all E cells.** It scales their external Poisson rate:
  - Poisson dips (gamma-distributed lengths, 10 ms ramps, none before the cue);
  - an overshoot when the input recovers (option A of the instructions): m_eff = m + A (m − m̄).
- **Heterogeneity**, fixed per network:
  - firing thresholds (SD 0.3 mV);
  - tuning of the E→E input: cell i receives a fraction h_i ~ U(0, 1) through a broad, weakly tuned kernel
    instead of the Gaussian footprint.

  Both are drawn stratified along the ring, so local averages are uniform and the bump is not pinned.
- **Simulator.** A batched LIF integrator in numba (`network._run`). Trials run in parallel, and the E→E sum is
  applied through the exact Fourier series of the ring kernel (24 harmonics, error 1e-14).
  - **Cross-check:** the same equations in Brian2 with explicit all-to-all synapses (`validate_brian2.py`).
  - **Speed:** about 0.4 s per trial-second on 4 cores.
- **Read-out (`readout.py`)**, as in the data:
  - 330 recorded cells per session (264 E, 66 I) across the ring;
  - 100 ms windows every 10 ms;
  - L2 logistic regression, λ = 10,000, cue vs opposite cue, correct trials, leave-one-out;
  - 30 label shuffles; z against the shuffle null;
  - On/Off labelling and statistics: the same code as the rate model (`paper_model.readout`);
  - decoded populations: all units, top 5/20/50 (split-half ANOVA), random 5/20/50 (10 draws).
- **Memory held**, from the whole E population in the last 100 ms of the delay:
  - the rate bump if it is active, otherwise the NMDA gating profile;
  - within 22.5° of the cue.

  Also reported:
  - a **response probe**: 300 ms after the delay with the input at 1, asking whether the bump restarts at the cue;
  - restarts at the wrong place.

(Results, tuning steps and model-vs-data table: below, filled from `results/`.)
