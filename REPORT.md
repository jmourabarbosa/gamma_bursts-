# Decoder "Off states" without synaptic plasticity: what works, what does not

**Question.** Panichello et al. (2024, *Nature*) find that single-trial decoding of a
spatial working memory from prefrontal populations alternates between On states and Off
states. During Off states decoding is at chance, tuning is nearly gone and rates are back
to spontaneous. They argue that the memory must be carried through Off states by
short-term synaptic plasticity. Can spiking networks with **no plasticity of any kind**
produce the same decoder Off states, while the memory is in fact never lost?

**Short answer.**
- **Yes, for the decoder statistics Panichello use as evidence:**
  - the number of Off states (≈ 3.7 per trial);
  - decoding falling to chance during them;
  - the two-state (beta-mixture) test;
  - Off states of roughly the right length (mean 96 ms, median 85 ms, vs 146 / 100 ms).

  A Brunel-style network with a rank-one memory and fixed weights produces all of these
  when the decoder sees a small recorded sample. The full network still holds the memory
  in 99% of those Off bins.
- **Not yet for one signature:** the cross-validated loss of tuning in *held-out* neurons
  during Off states (12-fold in their data, at most 2.2-fold here). That signature requires
  genuinely coordinated, slow (>100 ms) dips of the memory population. None of the
  plasticity-free mechanisms tried here produced them robustly (details below).
- **NMDA is enough to carry the memory through spontaneous network-wide silences of up to
  ~200 ms,** with no plasticity. That covers Panichello's Off durations, but in those
  regimes decoding stayed too good to show Off states.

---

## 1. The decoding analysis (`gamma_wm/panichello.py`)

The pipeline follows Panichello's Methods as closely as a binary task allows.
- **Counts:** 100 ms boxcar spike counts every 10 ms.
- **Decoder:** a leave-one-trial-out L2 logistic classifier per time point. Confidence is
  the probability assigned to the correct cue.
- **Null:** decoders trained on shuffled cue labels (10 shuffles).
- **On states:** runs of z > 1.96 with cluster mass above the 95th percentile of the null.
- **Off states:** at least 5 consecutive points with z < 0.84 (one-sided P > 0.20).
- **Counting window:** states counted from cue onset to 1.5 s after it, as in their
  "cue to end of delay".
- **Split-half test:** On/Off labelled from one half of the sample; tuning
  (cue-explained variance) of the held-out half measured in On vs Off.
- **Two-state test:** a 1- vs 2-component beta mixture, cross-validated.
- **Ground truth (not available in recordings):** the sign of the full-network readout
  (cued minus uncued pool rate, 100 ms) during decoder Off states.

The fitting is exact and fast. The decoder is solved in the span of the training trials,
and the numba version matches a direct fit to 1e-15.

**Regularization is a hidden analysis parameter.** With the same data and the same
decoding accuracy (0.99), the share of time labelled On/Off changes a lot with the L2
penalty λ:

| λ | Null SD | Time On | Time Off |
|---|---|---|---|
| 1 | 0.33 | 6% | 19% |
| 10 | 0.21 | 27% | 14% |
| 100 | 0.06 | 69% | 7% |

Weak regularization makes the shuffled-label decoders overconfident, which widens the null
and shrinks the On states. Panichello used MATLAB's `fitclinear` default (λ = 1/n, very
weak). Everything below uses λ = 10.

## 2. What works: Off states from a small recorded sample, no plasticity

**Model 1** (`gamma_wm/config.py`): Brunel (2000) model A, with 10,000 E and 2,500 I LIF
neurons, C_E = 1,000, delta synapses, D = 4 ms, g = 8 and ν_ext = 1.4·ν_thr (the SI
regime). Memory: J_ij·(1 + 0.9 ξ_i ξ_j) on E→E, with two pools of 10% of E cells each.
Weights are fixed, and there is no NMDA and no adaptation. 48 trials: cue A or B, then a
2 s delay. Memory correct in 48/48 trials; delay rates are 12.9 Hz cued, 3.3 Hz uncued,
7.2 Hz non-selective.

Decoder On/Off statistics vs number of recorded neurons (`figures/subsampling_v1/`,
mean of 2 random samples per size; Panichello in the last row):

| Recorded cells (selective) | Decoding accuracy | Off per trial | Off mean / median (ms) | Accuracy during Off | **Memory correct during Off (full network)** | Held-out tuning, On / Off | 2- vs 1-state (mbit) |
|---|---|---|---|---|---|---|---|
| 25 (≈2) | 0.72 | **4.33** | **96 / 85** | 0.37 | **0.99** | 0.083 / 0.113 | +7 |
| 50 (≈10) | 0.95 | 1.11 | 73 / 65 | 0.59 | 0.98 | 0.172 / 0.151 | +25 |
| 100 (≈18) | 0.96 | 0.67 | 78 / 70 | 0.62 | 0.98 | 0.144 / 0.104 | +39 |
| 200 (34) | 0.98 | 0.43 | 80 / 70 | 0.70 | 0.93 | 0.144 / 0.086 | +48 |
| 400 (≈72) | 0.99 | 0.23 | 86 / 75 | 0.68 | 0.85 | 0.177 / 0.082 | +96 |
| *Panichello* (≈330 units, 44% selective) | – | *3.74* | *146 / 100* | *~chance* | *?* | *0.082 / 0.007* | *positive in 19/25 sessions* |

Figures:
- `figures/subsampling_v1/subsampling.png`: all panels against sample size.
- `figures/subsampling_v1/examples_K25.png`: three trials with 25 recorded cells. The
  decoder falls into Off states again and again while the full-network pool rates show
  the memory held throughout. These are the three trials with the most Off states; in
  trial 34 the memory weakens toward the end of the delay.

What this shows:
1. **The Off-state count depends steeply on how many cells are recorded,** from 4.3 per
   trial at 25 cells to 0.2 at 400. With few informative cells, decoding is not stable:
   it repeatedly drops to chance for ~100 ms although the memory never goes away.
   Sampling noise in a small population is enough for Panichello's Off-state counts.
2. **The two-state test passes everywhere**, more strongly with more cells. A model with
   no discrete memory states passes it, so it is not evidence for two states.
3. **Off durations are set mostly by the 100 ms smoothing window,** at about 70–100 ms.
   That is 0.6× Panichello's mean and 0.85× their median.
4. **The held-out test separates the two explanations.** With few cells, Off states are
   sampling noise and held-out neurons are *not* less tuned (0.08 vs 0.11). With ≥ 200
   cells, the rare remaining Off states include real coordinated dips of the memory:
   held-out tuning drops ~2× and the memory is correct during only 85–93% of Off bins.
   Panichello report 12×.

**Model 1 + NMDA** (τ = 100 ms, q = 0.5, β = 0.8; 40/40 correct; 14.2 / 4.2 / 8.6 Hz)
gives the same picture (`figures/subsampling_nmda080/`). At 25 cells: 2.6–6.1 Off states
per trial, 77–96 ms, memory correct in 100% of Off bins, no held-out effect. NMDA does
not lengthen decoder Off states.


**Cross-validated held-out tuning curves** (Panichello Fig. 4d/e style;
`scripts/heldout_tuning.py`; figures `figures/heldout_model1.png` and
`figures/heldout_nmda.png`). States are labelled with one half of the recorded sample.
The plotted values are the held-out half's delay activity, z-scored across trials and
corrected with shuffled cue labels, for the preferred vs non-preferred cue.

| Recorded cells | Model 1: tuning (pref − non), On / Off | Model 1 + NMDA: On / Off |
|---|---|---|
| 25 | 0.23 / 0.26 (no loss) | 0.20 / 0.21 (no loss) |
| 100 | 0.27 / 0.16 (0.6×) | 0.17 / 0.14 (0.8×) |
| 400 | 0.22 / 0.09 (0.4×) | 0.17 / 0.09 (0.5×) |

With small samples the Off states are pure sampling noise, and the held-out cells keep
their full tuning in Off. With larger samples the rarer Off states increasingly coincide
with real dips in the memory, and held-out tuning falls to 0.4–0.5× of On. Panichello
report it nearly gone (about 12× in variance, roughly 3.5× in this difference measure).
Held-out rates in Off are close to the pre-cue level, as in their data. Here, though, the
pre-cue period already holds the previous trial's memory, so this baseline is not a clean
"spontaneous" level.

## 3. NMDA bridges spontaneous network-wide silences (no plasticity)

This network sits below threshold drive (ν_ext = 0.86·ν_thr, g = 7) and goes silent on its
own, network-wide and irregularly. It has small memory pools (2% of E cells each) with
strong within-pool synapses and NMDA (τ = 150–200 ms, q = 0.4); pool input counts are
exact, so neither pool is favoured. After each natural silence, does the *cued* pool
restart first? (`figures/screens/nmda_bridging.png`, from exploratory screens with 4 runs
per setting.)

- **Silences of 20–100 ms:** the cued pool restarts first 97–100% of the time, over
  hundreds of silences. Without a stored memory it is 43–50%.
- **Silences of 100–200 ms:** 67–100% (2–4 silences per setting).
- **Silences longer than 200 ms:** 50–75%, approaching chance.

This matches τ·ln(I₀/I_c) ≈ 1–2 τ_NMDA. The setting that holds the memory throughout
(α = 2.0, τ = 150 ms: 40/40 and 42/42 trials) has 11 silences per second, mean 32–34 ms.
But its pool cells are so selective (≈ 25 Hz vs 0.1 Hz on a ~1 Hz background) that
decoding is 98–100% correct from 10 cells up, with no Off states. Letting non-pool cells
draw random pool inputs (weak mixed selectivity) did not change this.

## 4. What was tried and did not work

Each line is one mechanism; numbers are from screens (2–4 runs) unless stated.

| Mechanism (all without plasticity) | Result | Why it fails |
|---|---|---|
| Brunel SI regime for gamma bursts (model 1) | Gamma at 58 Hz, sustained; Lundqvist 2 SD bursts rare (0.5/s), broadband | Finite-size noise gives noise-like amplitude modulation, not intermittency |
| Pool-specific inhibition (low-rank E→I/I→E) | Gamma becomes local to the cued pool but stays sustained; intermittency only when both pools are active (no memory) | Local inhibition removes the competition that stores the bit |
| Pool inhibition + NMDA | One trade-off curve: memory reliability up ⇔ intermittency down | NMDA smooths and stabilizes the active state |
| Near-critical rank-one memory + NMDA (f = 0.2) | β ≤ 0.38: flips; β ≥ 0.39: stable, no dips | Pools of 2,000 cells average out the noise; the window is very narrow |
| Spike-frequency adaptation, slow (τ_w 150–300 ms) | Memory lost or no dips | Adaptation leaves the cued pool less excitable than the uncued one |
| Fast adaptation (60 ms) + slow NMDA (150 ms), large pools | Memory with no dips, or dips with flips | Large pools settle to a fixed point instead of cycling |
| Fast adaptation + NMDA, small bistable pools (400 cells) | Same either/or | Same |
| Adaptation-driven global Up/Down (g = 8) | No Up/Down; memory erased | Inhibition-dominated network |
| Bistable small pools (100 cells) + lateral inhibition | Apparent bridging of second-long Off states turned out to be an artifact: random wiring gave pool A 7% more within-pool inputs. With exact in-degrees: chance | Quenched heterogeneity, now fixed with `fixed_group_indegree` |
| Denser / smaller networks (ε up to 0.8) | No dips at ε ≤ 0.4; memory flips at 0.8 | Coordinated fluctuations too weak until the attractor breaks |
| Near-quiescent drive + NMDA, realistic rates | Robust memory, no dips | NMDA stabilizes (Wang 1999) |
| Search for large spontaneous population events | None found (> 4 SD) | – |

The same pattern recurs throughout. With fixed connectivity, a memory that is strong enough
to persist is also smooth: its strength fluctuates by a few Hz on ~10–100 ms time scales.
Anything that makes it dip deeply and slowly either breaks it or is too rare. NMDA extends
the bridgeable gap to ~200 ms but also stabilizes the active state.

## 5. What this means for the Panichello argument

- **Their Off-state counts, the near-chance decoding during Off and the two-state mixture
  test are not evidence for synaptic storage.** A plasticity-free attractor network
  reproduces all three when the decoder has few informative neurons, while the full
  network keeps the memory throughout. Their counts also depend on decoder regularization
  and on sample size.
- **Their strongest evidence is the held-out tuning loss, together with the longer Off
  durations.** In these models, sampling-noise Off states fail the held-out test and are
  only ~100 ms long. A plasticity-free account then needs a source of *coordinated* slow
  fluctuations of the memory population. Section 3 shows NMDA can carry the memory across
  such gaps of up to ~200 ms. Candidate sources not modelled here are fluctuations in
  neuromodulatory or arousal state, or in input from other areas, which Panichello
  themselves discuss. Driving the network with such a slowly fluctuating input is the
  obvious next test. It was out of scope here because the silences were required to arise
  from the network itself.
- **Their Off states also retain information.** Held-out tuning during Off is 0.7%, not
  zero. A weak residual rate bias like the one in these models would leave about that much.

## 6. Caveats

- **Task and sampling differences:** binary task (theirs had 8 locations), 40–48 trials
  per model, 2 random samples per size, λ = 10.
- **Pending:** the 800- and 1,600-cell samples for model 1 were still running at the time
  of writing.
- **Exploratory screens:** sections 3 and 4 use 2–4 runs per setting; small bins are noisy.
- **Model simplifications:** NMDA is linear (no Mg²⁺ block), and synapses are current-based.

## Files

- `gamma_wm/network.py` (simulator), `gamma_wm/panichello.py` (decoding pipeline).
- `scripts/subsample_decoding.py`, `scripts/plot_subsampling.py`,
  `scripts/plot_offstate_examples.py`.
- Screens: `scripts/exploration/*.py`, with results in `figures/screens/*.jsonl`.
- Reproducing model 1: `scripts/run_session.py` ×4 sessions (README), then
  `python scripts/subsample_decoding.py --data data/session --out figures/subsampling_v1 --sizes 25 50 100 200 400 --lam 10`.

## Stationary memory with Off states (slow shared input noise)

The final model is summarised in `REPORT_SHORT.md`. The earlier fading-memory model is replaced
by a stable memory core: α = 2.8, NMDA 300 ms. A slow fluctuation of the external drive, shared
by all cells (OU, τ 500 ms, SD 45%), silences the weakly tuned background cells for 50 ms to over
1 s. Decoders trained on those background cells show:
- Off states of about 200 ms;
- a two-state mixture that beats the one-state model;
- held-out tuning that drops in Off (0.37 vs 0.10 with 100 cells; 0.47 vs 0.02 with 200).

Meanwhile the memory stays stationary (97.5% correct at the end of the delay).
