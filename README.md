# Gamma-burst working memory without short-term plasticity?

Question: can a Brunel (2000) sparse E/I spiking network in the irregular, oscillatory
(synchronous-irregular, SI) regime, plus a low-rank perturbation of its connectivity,
store a binary working memory **without any short-term plasticity**, and does its
delay activity look like the gamma bursts and On/Off states described by
Lundqvist et al. (2016, *Neuron*) and Panichello et al. (2024, *Nature*)?

Short answer from the simulations so far:

| Claim | Result |
|---|---|
| Binary memory held for 2 s with fixed weights (no STP, no adaptation) | **Yes.** 48/48 trials; cued pool 12.9 Hz vs uncued 3.3 Hz |
| Single neurons irregular, firing far below the population rhythm | **Partly.** CV(ISI) ≈ 0.63; 13 Hz vs a 58 Hz rhythm (a cell fires on ~1 in 5 cycles) |
| Network-wide silences in the delay, with the memory surviving them | **Yes, but short.** Silences of 5 ms on average (17 ms at most), 17% of the delay; the memory survives in sub-threshold V (cued pool ≈ 2.2 mV closer to threshold) |
| Lundqvist-style gamma bursts (brief, intermittent, memory-related) | **No, not in this model.** The gamma rhythm is sustained (same spectrum before the cue and in the delay); 2 SD events are rare (0.5/s), broadband and mostly at the 100 Hz edge; burst rate is not tied to memory |
| Panichello-style On/Off states | **Only in a subsampled decoder.** A decoder on 200 random cells shows "Off" states of ~80 ms, but the full-network readout is correct during 98% of them, so they are sampling noise, not a loss of the memory |

So the "silence" mechanism works without STP, but only on the time scale of a gamma
cycle (well under τ_m = 20 ms). To get the 50–150 ms bursts and Off states seen in the
data, something slow has to push the network in and out of the oscillatory regime.
In this homogeneous network, nothing does (see *Negative results* below).

## Model

`gamma_wm/network.py`: Brunel (2000) model A, simulated with numba.

- N_E = 10,000, N_I = 2,500 LIF neurons; τ = 20 ms, θ = 20 mV, V_r = 10 mV, τ_rp = 2 ms.
- Fixed in-degree: C_E = 1,000, C_I = 250, plus C_ext = C_E external Poisson synapses.
- Delta synapses: J = 0.1 mV, inhibitory −gJ, delay D.
- **Memory structure (rank one):** E cells carry ξ_i ∈ {+1 (pool A), −1 (pool B), 0},
  with 10% of E cells in each pool. Existing E→E synapses are rescaled to
  J_ij = J(1 + β ξ_i ξ_j). The mean connectivity is therefore the random matrix plus
  β·εJ ξξᵀ. Dale's law holds for β ≤ 1. Weights are fixed during simulation.
- Working point (`gamma_wm/config.py`): g = 8, ν_ext = 1.4 ν_thr, D = 4 ms, β = 0.9.
  This is the inhibition-dominated SI regime, with a ~58 Hz population rhythm set
  mainly by D.
- Task (`gamma_wm/task.py`): 1 s pre-cue, then a 150 ms cue (+10% external rate onto
  pool A or B), then a 2 s delay. Trials run back to back from the previous trial's
  membrane potentials.

The rank-one term makes a symmetric pitchfork, giving two attractors: "A high, B low"
and the reverse. Over the range scanned (f = 0.1–0.5, β = 0.2–1.0, g = 5–8), there was
**no** parameter where a symmetric spontaneous state coexists with the memory states.
Below β_c the network stays symmetric; above it, the network always holds one bit. So
the pre-cue state is the previous trial's memory, and the cue overwrites it (reliably:
48/48).

The simulator reproduces Brunel's Table 1 (`scripts/validate_brunel.py`):

| State | rate (sim / paper) | global frequency (sim / paper) |
|---|---|---|
| B, SI fast (g=6, ν=4) | 58.7 / 60.7 Hz | 178 / 180 Hz |
| C, AI (g=5, ν=2) | 37.2 / 37.7 Hz | – |
| D, SI slow (g=4.5, ν=0.9) | 5.6 / 5.5 Hz | 22 / 22 Hz |

## Results (48 trials, 4 sessions; `figures/summary.json`)

- `figures/fig1_example_trial.png`: one trial (raster, pool rates, LFP proxy, spectrogram with
  detected bursts, memory readout).
- `figures/fig2_irregular_oscillation.png`: CV(ISI), rate distributions, LFP spectrum
  before the cue vs in the delay, and a 300 ms zoom (the population oscillates while each
  cell fires on a few cycles at irregular times).
- `figures/fig3_gamma_bursts.png`, `figures/fig3b_spiking_in_bursts.png`: Lundqvist
  burst analysis (2 SD above the mean per frequency, ≥ 3 cycles; statistics exclude the
  cue response).
- `figures/fig4_silence_hidden_state.png`: network-wide silences, pool rates and membrane
  potentials around them, and the fraction of trials whose readout has the correct sign.
- `figures/fig5_decoding_on_off.png`: Panichello-style leave-one-trial-out logistic
  decoder on 200 random neurons, single-trial confidence, a 1- vs 2-component beta
  mixture, and On/Off labelling against a shuffled-label null.

Key numbers:
- Delay rates: cued 12.9 Hz, uncued 3.3 Hz, non-selective E 7.2 Hz, I 7.2 Hz.
- LFP peak 58 Hz.
- Silences: E rate (5 ms boxcar) below 25% of its delay mean, 37 per second, mean
  4.7 ms. After a silence the cued pool restarts first (14.3 vs 3.9 Hz) because its
  membrane potentials stayed higher. This is an "activity-silent" carrier with no
  synaptic mechanism, but it lasts only ~τ_m.
- Detected 40–100 Hz bursts: 0.5 per second, 54 ± 28 ms, mean peak 84 Hz,
  half-power width 38 Hz.
  - These are rare, broadband events, not narrow-band gamma episodes.
  - Cued-pool rate inside vs outside bursts: 14.2 vs 12.8 Hz.
- Decoder: 91% correct in the delay. The 2-beta mixture beats the 1-beta model by
  70 mbit/sample (cross-validated). 2.6 Off states per trial, ~81 ms each; the
  full-network readout stays correct in 98% of Off bins.

## Negative results

1. **Strength of the oscillation.** I scanned D = 2.5–4 ms, g = 5–8, a delay spread of
   0–3.5 ms and several ν_ext. The gamma envelope's CV stayed at or below 0.48, the
   value expected for Gaussian narrow-band noise (≈0.52). The LFP coherence time was
   9–34 ms (1–2 cycles). Near and inside SI, the finite-size network makes noise-like
   gamma, not intermittent bursts with silent gaps.
2. **Gating gamma with the memory mode.** Bringing the rank-one mode close to its
   critical point produces slow fluctuations in the pool rate. The gamma envelope was
   uncorrelated with them (r = −0.07 to 0.13), and the memory degraded (some trials lost
   or flipped it) before any gating appeared. Gamma in this network is a global property
   of (g, D, ν_ext), shared by all cells, so the memory pool can't switch it.
3. **Spread delays as an "AI control"** (uniform in 0.5–7.5 ms) did not remove the
   rhythm at g = 8.

## Follow-up: synaptic kinetics and NMDA, with silences that arise on their own

`Params` now accepts exponential AMPA, NMDA and GABA_A currents (`tau_ampa`,
`tau_nmda`, `tau_gaba`, and `q_nmda`, the fraction of E charge carried by NMDA; NMDA is
linear, with no Mg²⁺ block). Each spike delivers the same total charge J as in the delta
model, which is recovered when all time constants are 0. With τ = 0.1 ms the rate is
36.8 Hz vs 37.4 Hz for the delta model. No silences are imposed: they come only from
finite-size noise and recurrent inhibition. The scans below are exploratory, with 1–2
trials per setting, run with scratch scripts that are not in the repo.

Settings: τ_AMPA = 2 ms, τ_GABA = 5 ms, τ_NMDA = 100 ms, D = 1.5 ms.

- **Unstructured network.** Long, irregular silences occur only at the edge of the
  quiescent state.
  - Example: ν_ext = 0.88·ν_thr, g = 7, q_NMDA = 0.4. Silence lasts 51 ms on average
    (95th percentile 191 ms, maximum 358 ms), with a CV of durations of 1.17 and a CV of
    intervals of 1.1. But the network is silent 92% of the time and fires at 0.7 Hz.
  - Moving away from that edge makes the silences clock-like: CV < 0.5 and periodic at
    6–15 Hz.
- **With the memory pools,** two things trade off:
  - A strong memory (g = 6, β = 0.9: cued pool 12.4 Hz vs 0.2 Hz, readout correct
    98% of the time) keeps the network active. Silences drop to 21% of the time, with a
    maximum of 37 ms.
  - Weaker memories (β = 0.75–0.85) keep silences of up to ~100–135 ms. But the cued
    pool fires at only 1–4 Hz, the readout is correct only 54–69% of the time, and after
    silences of 30 ms or more the cued pool restarts first in only 36–72% of cases
    (chance is 50%). The NMDA trace scales with the cued pool's rate before the silence,
    which is too low here to bias the restart.

So far, a self-sustaining memory and long natural silences have been mutually exclusive
in this network. The memory state itself is what keeps the network out of the quiescent
regime.

## Follow-up: pool-specific inhibition

Inhibitory partner pools I_A and I_B (`f_I` = 10% of I cells each) carry ξ = ±1, so the
E→I and I→E blocks get their own low-rank terms. Every block is scaled as
w_ij·(1 + α|ξ_i||ξ_j| + β ξ_i ξ_j): within-pool connections by 1 + α + β, between-pool
connections by 1 + α − β (Dale's law requires 1 + α − β ≥ 0). The model uses delta
synapses with no NMDA, at the same working point as above (g = 8, ν_ext = 1.4·ν_thr,
D = 4 ms).

Exploratory scans (1–2 trials per setting) found three regimes:
- **Pure rank-one E↔I structure** (α = 0): local inhibition caps the pools and removes
  the competition between them, so the memory is lost. Cued and uncued pools fire at
  the same rate.
- **Strong local inhibition or large α:** both pools are active together. Each pool's
  gamma then switches on and off by itself: Off epochs of 20–40 ms (maximum ~55 ms),
  envelope CV 0.3–0.45, gamma tracking the pool's rate (r ≈ 0.6–0.8). But no memory
  is stored.
- **Intermediate** (local β_ie = β_ei = 0.3, α = 0.2–0.25): the memory holds, and gamma
  is local to the cued pool.

Sessions at the intermediate setting (`scripts/analyze_pools.py`, 20 trials each, each
pool treated as a recording site; figures in `figures/pool_inhibition_alpha*`):

| | α = 0.2 | α = 0.25 |
|---|---|---|
| memory correct over the whole delay | 20/20 | 20/20 |
| delay rate, cued / uncued pool (Hz) | 24.7 / 2.4 | 33.6 / 1.5 |
| gamma bursts (2 SD criterion), cued site / uncued site (per s) | 1.1 / 0.23 | 0.93 / 0.30 |
| fraction of delay time in a burst, cued / uncued | 6.1% / 1.8% | 6.5% / 2.3% |
| cued-pool rate inside / outside bursts (Hz) | 26.6 / 24.6 | 32.5 / 33.7 |
| cued-pool Off epochs (rate below half its mean, 20 ms smoothing) | max 11 ms | max 9 ms |
| CV(ISI), cued pool | 0.54 | 0.48 |

So pool-specific inhibition makes gamma **site-specific**: it is strong where the item
is held and weak elsewhere, which is Lundqvist's "gamma at informative sites". But at the
cued site it is **sustained**, not bursty. The cued pool never switches off, and detected
bursts are rare, broadband and barely change the spike rate. The intermittent regime
only appears once the competition between pools, and with it the memory, is gone.

## What would probably be needed (not implemented)

- **Pool-specific inhibition.** Add a rank-one E↔I component so each memory pool has its
  own feedback loop (as in Lundqvist's modular model). Gamma would then be generated
  where, and only when, the pool is active.
- **A slow variable other than STP that switches the regime**, for example slow NMDA
  currents or a slowly fluctuating external drive. This would give On/Off epochs of
  100+ ms; which mechanisms count as "no STP" is up to you.

## Reproduce

```bash
pip install numpy scipy matplotlib numba
python scripts/validate_brunel.py
for s in 0 1 2 3; do python scripts/run_session.py --session $s --n-trials 12 & done; wait
python scripts/analyze.py            # ~20 min, mostly the shuffled-label decoders
```
One trial (12,500 neurons, 3.15 s at dt = 0.1 ms) takes ~18 s on one core.
