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

Results, tuning steps and the model-vs-data table are below; the numbers come from `results/`.

## How to run
```
pip install brian2 numba scikit-learn matplotlib scipy
spiking/run_all.sh            # sessions, example, no-dip control, single dip, stability, figures (~2.5 h, 4 cores)
python -m spiking.validate_brian2
```

## Tuning steps
1. **Implementation check, published values** (τ_N = 100 ms, J⁺ = 1.62, g_EE = 0.381 nS, no heterogeneity):
   - spontaneous state about 3 Hz;
   - the cue starts a bump of about 50 Hz peak, held for 1.6 s, drifting a few degrees.
2. **τ_N = 150 ms.** g_EE was scanned for a stable spontaneous state plus a cue-triggered bump:
   - window only 0.36–0.375 nS;
   - threshold heterogeneity of 1 mV (unstratified) destroyed it: spontaneous bumps formed at clumps of
     low-threshold cells.
3. **Wider bistability.** J⁺ = 2 gave a window of 0.30–0.36 nS.
4. **Kick (step 3 of the instructions).** Single dips silence the bump. Findings:
   - **without the kick:** the bump restarts after 100 ms dips but not after 200–250 ms ones;
   - **sharp kick (τ 15 ms, A ≥ 1):** worse. It drives a synchronous burst in every E cell. With Wang's kinetics
     (α τ_x = 1) one spike nearly saturates a cell's NMDA gating, so the burst erases the cued cells' NMDA advantage,
     and global inhibition then shuts the network down;
   - **gentle, slow kick (A = 0.5–1, τ = 100–150 ms):** moves the cut-off to about 400 ms.
5. **First read-out.** The decoders were far too strong, and random 5 cells decoded as well as top 5 (every model
   cell is sharply tuned). Two kinds of heterogeneity were added to the tuning of the E→E input:
   - h_i ~ U(0, 1) through a broad kernel with κ = 0.25 (κ ≥ 0.5 makes the spontaneous state unstable);
   - J⁺ raised to 3 to keep the bump.

   Clumps in the heterogeneity pinned the bump about 15° away from the cue. Stratified draws (each block of 16
   neighbouring cells holds the same quantiles) removed the pinning.
6. **Dips (step 4).** With the rate model's dips (10/s, 90 ms):
   - the decoded z falls to chance in each dip but rarely stays below 0.84 for the 50 ms an Off state needs, since
     100 ms windows bridge 90 ms silences;
   - memory was 56–84%.

   Longer, sparser dips (6–7/s, 150 ms, depth 0.5) and g_EE = 0.33 nS gave the final values in `run.TUNED`. Raising
   the dip rate or length further gave more Off time but lower memory (5/s × 190 ms: memory 77%).
7. **Higher baselines were tried and failed.** Stronger external drive and larger threshold spread did not raise E
   baselines above about 1 Hz before spontaneous bumps appeared.

## Results (12 sessions × 480 trials, 330 recorded cells each)
Model, with the data in brackets (mean over sessions; lengths in ms):

| population | Off/trial | Off median | frac Off | On/trial | On median | frac On | μ_on |
|---|---|---|---|---|---|---|---|
| top 5 | 2.25 (3.50) | 112 (109) | 0.54 (0.36) | 2.04 (3.14) | 202 (126) | 0.35 (0.44) | 7.00 (3.94) |
| top 20 | 2.35 (3.25) | 116 (103) | 0.51 (0.33) | 1.98 (3.11) | 211 (133) | 0.36 (0.43) | 7.12 (3.85) |
| top 50 | 2.84 (3.15) | 111 (102) | 0.38 (0.34) | 2.24 (2.92) | 215 (139) | 0.41 (0.39) | 6.11 (3.48) |
| rand 5 | 3.15 (4.80) | 122 (129) | 0.43 (0.65) | 2.25 (1.07) | 188 (99) | 0.38 (0.07) | 5.80 (1.14) |
| rand 20 | 3.13 (4.65) | 103 (117) | 0.29 (0.57) | 2.97 (1.51) | 202 (102) | 0.52 (0.11) | 5.86 (1.42) |
| rand 50 | 3.06 (4.27) | 100 (112) | 0.26 (0.53) | 3.21 (1.75) | 203 (110) | 0.56 (0.14) | 5.78 (1.69) |
| all | 3.00 (3.25) | 98 (112) | 0.26 (0.45) | 3.37 (2.10) | 202 (134) | 0.59 (0.21) | 5.67 (2.18) |

**Memory, end of delay, rate bump or NMDA trace: 83.2%** (sessions 79.6–87.1%).
- Rate bump active and at the cue in the last 100 ms: 53% (the other trials end in or just after a dip).
- NMDA criterion alone: 83.0%.
- Response probe (300 ms at full input after the delay), bump restarts at the cue: 88.7%.
- Bump at the wrong place: 10.1% at the end of the delay, 11.3% in the probe.

**Stability** (no dips, 2 s before the cue, `results/stability.json`):
- no spontaneous bump (smoothed peak before the cue 1.1 Hz; E 0.6 Hz, I 4.5 Hz);
- bump peak 58 Hz;
- drift about 1° per second;
- 15/16 held; one bump formed 27° from the cue.

**No-dip control** (6 sessions × 480 trials, same read-out; `results/sessions_nodips.json`, purple crosses in
`fig_unit_sets`):
- memory 100%;
- all units, top 20 and top 50: no Off states, On for the whole 1.4 s (μ_on 11–13);
- read-out noise alone gives Off states only for the weakest decoders: random 5, 1.07 Off per trial (34% of the
  delay); random 20, 0.17 per trial.

  This is the instructions' control, and it shows the same as in the rate model: the dips, not read-out noise, make
  the Off states of the strong decoders.

**Brian2 cross-check** (`validate_brian2.py`, `results/validate_brian2.json`). The same equations in Brian2 (explicit
all-to-all synapses, cython) against the numba simulator, on a reduced ring (512 E + 128 I, conductances × 4, no
heterogeneity, no dips):

| | numba (8 trials) | Brian2 (3 trials) |
|---|---|---|
| E spontaneous | 16.4 Hz | 18.1 Hz |
| I spontaneous | 9.9 Hz | 9.8 Hz |
| E, delay | 24.4 Hz | 26.2 Hz |
| I, delay | 12.9 Hz | 12.9 Hz |
| bump peak | 108 Hz | 112 Hz |

The two agree to about 10%; the reduced ring without heterogeneity is not bistable at these settings in either
simulator (it forms a bump before the cue). A check of the full-size network in Brian2 was not run (about 4 million
synapses; too slow here).

**Single dip** (`results/single_dip.json`, `fig_single_dip`):
- memory held 100% up to 400 ms, 91% at 450 ms, 53% at 550 ms, about 20% at 650–700 ms;
- lost memories are mostly restarts at the wrong place.

**Behaviour analogue** (`fig_memory_off_time`; all-units decoder trained on all trials):
- memory held falls from 95% at little time Off to near 0 at 0.8 s;
- mean time Off 0.22 s in trials with the memory held, 0.39 s in lost trials.

## What matches and what doesn't
**Matches:**
- Dips of the common input silence the bump. It restarts at the cued location when the input returns, through the
  residual NMDA gating and the kick. The decoded z goes to chance in each dip (`fig_example_trial`,
  `fig_populations`).
- **Off states:** 2.3–3.2 per trial; medians 98–122 ms against 102–129 ms in the data, in every population. The
  all-units Off length histogram has the data's shape: rise at 40–60 ms, peak at 60–80 ms, long tail
  (`fig_histograms`).
- More time Off goes with memory loss, as in the data, and the restart-from-NMDA mechanism has a length cut-off.

**Does not match:**
1. **Decoders too strong:** μ_on 5.7–7 against 1.1–3.9. Consequences:
   - all units are On 59% and Off 26% of the delay, against 21% and 45%;
   - On states are too long (median about 200 ms against 100–140), with almost none shorter than 100 ms.

   Model cells are far more selective than PFC units (`fig_tuning`: E cells below 1 Hz at the non-preferred cue,
   10–50 Hz at the preferred one).
2. **Top vs random inverted:** top-unit decoders have more Off and fewer On states than random or all-unit
   decoders; in the data it is the reverse.
   - The top cells of a cue pair (split-half ANOVA) all prefer the same one of the two cues (`fig_tuning` c).
   - On trials of the other cue they are silent. Without dips that silence is informative: the top-unit decoders are
     On 94–100% of the delay in the no-dip control.
   - With dips, the preferred cue's trials are silent half of the time too, so silence no longer identifies the cue.
     Trials of the non-preferred cue then sit at z ≈ 0, Off throughout (`fig_populations`, red).
   - Random 5 cells, by contrast, usually include a cell near either cue.
   - In PFC, cells have baselines of several Hz, so a drop below baseline stays informative even when the bump is
     silent elsewhere.
3. **Memory 83%**, below the target of about 90%. More Off time costs more memory than in the rate model, and
   failures are often restarts at the wrong place (about 10% of trials), which the rate model never shows.
4. **Behaviour relation too steep:** memory falls from 95% to about 0 over 0–0.8 s of Off time, against 0.88 → 0.78
   in the data.
5. **Cut-off:** the single-dip cut-off (50% at about 575 ms) is later than 2.5 τ_N = 375 ms, because the slow kick
   also bridges long silences.
6. **Dip statistics differ from the rate model's fit:** 7/s, mean 150 ms, depth 0.5, kick τ 150 ms, against 10/s,
   90 ms, 0.6, 15 ms.

## Known limitations
- **Parameters unchecked against the papers.** Values from Compte et al. (2000) and Wang (1999) could not be checked
  against the papers (hosts blocked); see `PARAMETERS.md`.
- **Low spontaneous rates.** E spontaneous rates (about 0.6 Hz) are far below PFC baselines; this is the main reason
  for differences 1 and 2.
- **Ranking choice.** Top units are ranked per cue pair (split-half ANOVA between the two cues). Whether the data
  ranked per pair or over all 8 cues is not stated in the data note. An 8-cue ranking is available in `readout.py`
  (`rank_by='all8'`), and a balanced ranking as a diagnostic (`'pair_balanced'`).
- **Decoder approximation.** The decoder uses the second-order (one Newton step) solution of the L2 logistic
  regression; with λ = 10,000 its leave-one-out confidences match full Newton fits with approximate or exact
  leave-one-out to about 1e-4 in a data-like regime.
- **Delay length.** Trials have a 1.5 s delay (the task: 1.4–1.6 s).
