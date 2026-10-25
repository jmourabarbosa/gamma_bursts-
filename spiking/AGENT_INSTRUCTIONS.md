# Instructions: spiking ring-attractor network with intermittent (On/Off) decoding

## 1. Goal
Build a spiking network model of spatial working memory. It must:
1. be a **ring attractor** of excitatory and inhibitory leaky integrate-and-fire neurons that holds a cued location as a bump of persistent activity;
2. have recurrent excitation through **NMDA receptors with a 150 ms decay**;
3. when decoded like the monkey data, show **intermittent decoding**: On and Off states whose numbers, lengths and length histograms match the Panichello et al. (2024) data **qualitatively**;
4. keep the memory in most trials (above about 90%) despite the Off states.

An exact quantitative fit is not required. The spiking network should reproduce the same phenomenon, through the same mechanism, as our fitted rate model (section 4).

## 2. Context and repository
- **Repository:** `jmourabarbosa/gamma_bursts-`.
  - Branch `paper-model` holds the paper's rate model (`paper_model/`), the data summaries (`paper_model/data/`) and the paper (`paper/`, owned by another agent; **do not edit `paper/`**).
  - Work on a new branch, `spiking-ring`, created from `paper-model`. Put everything in a new folder `spiking/`. Commit and push regularly; don't open a pull request.
- **Background reading in the repository:**
  - `README.md`: overview of the rate model.
  - `paper_model/model.py`: the rate model equations.
  - `paper_model/readout.py`: how the model is read out like the data.
  - `paper_model/data/readout_note_data_lam10000.md`: how the data were decoded and labelled.
  - `figures/paper_model/fig4_off_time_models_note.md` and `figures/paper_model/overshoot/README_overshoot.md`: what the model predicts and why.
- **Simulator:** Brian2 is suggested (`pip install brian2`), but any well-tested LIF simulator is fine. Check what is installed first.
- **Published parameters:** take conductances, time constants and connectivity from the papers, not from memory, and record the source of every value. Hosts such as journal websites, PMC and arXiv may be blocked in this environment; if so, ask the user to upload the PDFs.

## 3. Task and data targets
**Task (Panichello et al. 2024):**
- 8 possible cue locations on a circle.
- Cue shown for about 50 ms.
- Delay ends 1.4–1.6 s after cue onset.
- All analyses use 0–1,400 ms after cue onset; "delay" statistics use 500–1,400 ms.

**How the data were decoded** (`paper_model/data/readout_note_data_lam10000.md`):
- L2 logistic regression with λ = 10,000, trained on correct trials to separate a cue from the opposite cue.
- Spike counts in 100 ms windows, stepped by 10 ms.
- Approximate leave-one-out fits, and a null from 30 label shuffles per cue pair.
- The decoder output for the cued location is z-scored against that null at each time point.
- **On state:** a run of z > 1.96 (p < 0.05) whose summed z exceeds the 95th percentile of the same quantity in the null.
- **Off state:** at least 5 consecutive points (50 ms) with z < 0.84 (p > 0.20).

**Decoded populations:** all units (about 329 per session); the top 5–200 most selective units; random 5–200 units (10 draws).

**Targets, mean over 25 sessions** (`paper_model/data/readout_per_session_data_lam10000.csv`):

| population | Off per trial | Off median (ms) | fraction Off (500–1,400 ms) | On per trial | On median (ms) | fraction On | μ_on |
|---|---|---|---|---|---|---|---|
| all units | 3.25 | 112 | 0.45 | 2.10 | 134 | 0.21 | 2.18 |
| top 5 | 3.50 | 109 | 0.36 | 3.14 | 126 | 0.44 | 3.94 |
| top 20 | 3.25 | 103 | 0.33 | 3.11 | 133 | 0.43 | 3.85 |
| top 50 | 3.15 | 102 | 0.34 | 2.92 | 139 | 0.39 | 3.48 |
| random 5 | 4.80 | 129 | 0.65 | 1.07 | 99 | 0.07 | 1.14 |
| random 20 | 4.65 | 117 | 0.57 | 1.51 | 102 | 0.11 | 1.42 |
| random 50 | 4.27 | 112 | 0.53 | 1.75 | 110 | 0.14 | 1.70 |

- **Length histograms** (pooled over sessions, 20 ms bins up to 600 ms): `paper_model/data/readout_histograms_data_lam10000.csv`.
  - Both On and Off histograms start at about 40–60 ms (the labelling minimum), peak at 60–120 ms, and have long tails to 600 ms and beyond.
  - All units, fraction per 20 ms bin from 0 ms:
    - Off: 0, 0, .11, .19, .14, .12, .08, .06, .04, .03, …
    - On: 0, .01, .05, .11, .11, .13, .09, .08, .07, .06, …
- **Behaviour** (`paper_model/data/behaviour_per_trial_correct_and_error.csv`): fraction correct falls from about 0.88 to about 0.78 as the Off time in the delay goes from 0 to 900 ms, and error trials have more and longer Off time.
- **Qualitative targets, in order of importance:**
  1. a few Off and On states per trial, each lasting tens to a few hundred milliseconds, with length histograms of the shapes above;
  2. top-unit decoders more On and less Off than random-unit decoders, with all units in between;
  3. memory kept in most trials;
  4. optionally, errors more likely with more Off time.

## 4. Mechanism to implement (what worked in the rate model)
Our fitted rate model (`paper_model/model.py`) produces these statistics through the following mechanism. Implement the same one in the spiking network.

1. **Bistable memory with NMDA.** The cued population holds persistent activity through recurrent NMDA excitation, while the rest of the ring stays at low spontaneous rates. Shared inhibition (the ring's broad inhibition) provides the competition.
2. **A common input that briefly drops ("attention dips").**
   - **What it is:** all excitatory neurons receive a common external drive m(t). Normally it is constant; occasionally it drops for a short time, enough to silence the bump.
   - **Fitted dip statistics in the rate model:**
     - onsets Poisson at about 10 per second;
     - lengths gamma-distributed with mean about 90 ms and shape about 8.7 (SD about 31 ms);
     - depth about 60% of the drive;
     - 10 ms ramps at both edges.
   - **During a dip:** the bump falls silent, so decoding drops to chance and the decoder labels an Off state. The cued neurons' NMDA gating decays with τ_N = 150 ms and stays higher than elsewhere on the ring.
3. **A brief kick when the input returns (essential).** When the drive recovers, the network needs a brief extra push to restart the bump. The neurons with the most residual NMDA, the cued location, win and the bump restarts in the same place.
   - **Rate model, without the kick:** the bump does not restart. Memory falls to about 5–7%, and Off covers two thirds of the delay.
   - **Rate model, with a constant stronger drive instead of a kick:** the resting state becomes unstable, and a bump forms spontaneously before the cue.
   - **Option A (simplest, recommended):** the common input overshoots briefly on recovery. Add a pulse A·(m(t) − m̄(t)), where m̄ follows m instantly into a dip and recovers with τ ≈ 15 ms. In the rate model, A equal to the normal drive above the dip level worked: a peak extra drive of about 0.7 times the normal drive above baseline, decaying with 15 ms.
   - **Option B:** lagging feed-forward inhibition driven by m(t) with τ ≈ 15 ms (the original rate model). Only its effect on recovery matters.
   - **Kick size controls the On/Off balance.** Too weak and the memory is lost. Too strong and Off states become too short and too rare.
4. **Long silences cost the memory.** With τ_N = 150 ms, silences longer than about 375 ms (2.5 τ_N, when the NMDA trace has fallen to about 8% of its active value) are not bridged. This links Off time to errors. Check that your network shows a similar cut-off.
5. **Different decoded populations come only from the read-out.** The network dynamics are the same for every population; top units decode the memory more strongly than random units, so they show more On and less Off. In the spiking network this should come out naturally from decoding different numbers and selectivities of neurons, with no change to the network.

## 5. Network design (starting point; verify every value in the papers)
- **Base model:** a ring of LIF neurons, about 2,000 excitatory and 500 inhibitory, as in Compte, Brunel, Goldman-Rakic & Wang (2000, Cerebral Cortex), with receptor kinetics from Wang (1999, J Neurosci; 2001, TINS).
  - E→E connections are tuned in angle (Gaussian footprint); I connections are broad.
  - Synapses are AMPA, NMDA (voltage-dependent Mg²⁺ block, saturating gating) and GABA_A.
  - Background drive is Poisson.
  - Use the published values as starting points and record sources in `spiking/PARAMETERS.md`.
- **NMDA decay:** set to 150 ms. That is a design requirement; it is also close to measured cortical NMDA kinetics, and close to the decay that best explains the monkeys' behaviour in our rate model (about 120 ms, 95% interval 101–190 ms). Retune the recurrent NMDA conductance so the bump is stable without dips.
- **Common input:** implement m(t) as a modulation of the external Poisson rate (or of an excitatory conductance) to all E cells, with the dip statistics and kick of section 4. Generate a new m(t) per trial; there are no dips before the cue.
- **Cue:** 50 ms of extra input to the E neurons around one of 8 locations, 45° apart.
- **Run order:**
  1. Bump stable for 1.6 s without dips, and no spontaneous bump before the cue (at least 1 s of pre-cue time).
  2. Add the dips and tune depth and length until the bump falls silent in a dip.
  3. Add the kick and tune its size until the bump restarts at the cued location after typical dips.
  4. Tune the dip rate and lengths for the On/Off statistics.
- **Heterogeneity:** include some heterogeneity (in thresholds or in connectivity) so neurons differ in selectivity. This makes "top" and "random" units meaningful.
- **Bump drift:** keep it small during the delay, for example through heterogeneity or a slightly stronger tuning of the connections. Drift isn't a target, but strong drift makes decoding unstable.

## 6. Read-out (mimic the data pipeline)
- **"Recorded" neurons:** sample about 330 neurons per simulated session from the E population, and optionally some I neurons, across the whole ring, like a recording array. Simulate several sessions, for example 10–25 sessions of 400–600 trials with cues balanced over the 8 locations.
- **Decoding:**
  - Spike counts in 100 ms windows every 10 ms.
  - For each cue and its opposite cue, an L2 logistic regression with λ chosen as in the data (λ = 10,000 on spike counts in the same units), with held-out trials (cross-validation or leave-one-out).
  - z-score the decoder output for the cued location against a label-shuffle null (at least 30 shuffles).
- **Labelling:** On = runs of z > 1.96 whose summed z exceeds the 95th percentile of the null's summed-z distribution; Off = at least 5 consecutive points with z < 0.84. Use 0–1,400 ms, with fractions over 500–1,400 ms.
- **Decoded populations:** all sampled neurons; top 5/20/50 ranked by selectivity on held-out data (split-half ANOVA, as in the data); random 5/20/50 (10 draws).
- **Memory held:** decode the bump location from the whole E population in the last 100 ms of the delay, or from the NMDA gating if the bump happens to be in a dip there. Score it correct if it is within 22.5° of the cue. Also report how often the bump restarts at the wrong location; this was never seen in the rate model, but a ring could drift or jump.
- **Optional behaviour:** fraction correct (memory held) against total Off time in 500–1,400 ms, in 10 bins of 90 ms, as the paper's Figure 3.

## 7. Deliverables (in `spiking/` on branch `spiking-ring`)
1. `network.py` (model), `readout.py` (decoding and labelling), `run.py` (simulations), with parameters in one place.
2. `PARAMETERS.md`: every parameter, its value and its source.
3. Figures, each as PNG plus SVG:
   - an example trial: raster of the E population sorted by preferred angle, the common input, and the decoded z with On/Off shading;
   - On/Off statistics of the 7 decoded populations against the data, as in `figures/paper_model/fig2_unit_sets.png`;
   - On and Off length histograms (all units) against the data;
   - memory held against single-dip length, to check the ~375 ms cut-off;
   - optionally, memory held against Off time.
4. `NOTE.md`: what was built, the tuning steps, a table of model against data, what matches and what doesn't, and known limitations.

## 8. Pitfalls we ran into in the rate model
- **The kick is essential.** A constant stronger drive can't replace it: it either fails to restart the bump or destabilises rest before the cue.
- **Facilitation-type STP with τ ≈ 1.5 s instead of NMDA** reproduces the On/Off statistics just as well but predicts no relation between Off time and errors. Synaptic depression breaks the On/Off fit (On states too short). Don't add STP to this network.
- **Read-out noise alone gives Off states only for weak decoders.** With constant drive (no dips), the identical read-out gives almost no Off states for top-unit or all-unit decoders. Show this control for the spiking network too: run without dips and decode the same way.
- **Shallow dips are dangerous.** Dips that don't fully silence the bump give an unreliable restart and many memory errors. Deep dips (at least 60% here) behave better.
- **"Memory held" must be read out at the end of the delay.** Near the cut-off, the cued neurons can sit in a slow, intermediate state (5–20 Hz) for hundreds of milliseconds before either recovering or fading. Report both a rate-based and an NMDA-based criterion.
- **Simulation cost:** a spiking ring of this size with NMDA takes a while per trial. Develop with short runs and few trials, then scale up. Keep random seeds so runs can be repeated.

## 9. When done
Report back with the figures, the model-vs-data table, and the main differences from the data, for example histograms that are too narrow, drift, or wrong-location restarts. Say which steps were not completed and why.
