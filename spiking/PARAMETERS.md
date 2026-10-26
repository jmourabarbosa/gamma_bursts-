# Parameters of the spiking ring network

All values are in `network.P` (defaults) and `run.TUNED` (values changed during tuning, which override `network.P`).

> **Sources not checked against the PDFs.** Journal sites, PMC, arXiv and the mirrors I tried are blocked in this
> environment. The "published" values below are the standard values of Compte, Brunel, Goldman-Rakic & Wang (2000,
> Cereb Cortex 10:910) and Wang (1999, J Neurosci 19:9587), written from memory. They are marked **(verify)** and
> should be checked against the papers (please upload the PDFs). Running the network with these values (τ_N = 100 ms,
> no heterogeneity, J+ = 1.62) reproduces the published behaviour: spontaneous state, a cue-triggered bump of about 50 Hz
> peak, slow drift (tuning log in `NOTE.md`), which is some evidence that they are right.

Abbreviations: C00 = Compte et al. 2000; W99 = Wang 1999; JS90 = Jahr & Stevens 1990 (Mg²⁺ block, as used in W99 and C00).

## Neurons
| parameter | symbol | value | source |
|---|---|---|---|
| number of pyramidal cells | N_E | 2,048 | C00 (verify) |
| number of interneurons | N_I | 512 | C00 (verify) |
| pyramidal capacitance | C_E | 0.5 nF | C00 (verify) |
| pyramidal leak conductance | gL_E | 25 nS (τ_m = 20 ms) | C00 (verify) |
| interneuron capacitance | C_I | 0.2 nF | C00 (verify) |
| interneuron leak conductance | gL_I | 20 nS (τ_m = 10 ms) | C00 (verify) |
| resting potential | E_L | −70 mV | C00 (verify) |
| threshold | V_th | −50 mV (mean) | C00 (verify) |
| reset | V_reset | −60 mV | C00 (verify) |
| refractory period E / I | | 2 / 1 ms | C00 (verify) |
| threshold heterogeneity | sd_vth | 0.3 mV (SD), stratified | **this work**: added heterogeneity; larger values (1 mV, unstratified) seed spontaneous bumps |

## Synapses
| parameter | symbol | value | source |
|---|---|---|---|
| excitatory reversal | E_E | 0 mV | C00 (verify) |
| inhibitory reversal | E_I | −70 mV | C00 (verify) |
| Mg²⁺ block | B(V) | 1 / (1 + [Mg²⁺] exp(−0.062 V/mV) / 3.57), [Mg²⁺] = 1 mM | JS90 via W99 / C00 (verify) |
| AMPA decay | τ_AMPA | 2 ms | W99 / C00 (verify) |
| NMDA rise | τ_x | 2 ms | W99 / C00 (verify) |
| NMDA saturation rate | α | 0.5 ms⁻¹ | W99 / C00 (verify) |
| **NMDA decay** | **τ_N** | **150 ms** | **design requirement** (AGENT_INSTRUCTIONS.md); C00 / W99 use 100 ms |
| GABA_A decay | τ_GABA | 10 ms | W99 / C00 (verify) |
| E→E NMDA conductance (per synapse) | g_EE | **0.32 nS** | **retuned** (C00: 0.381 nS at τ_N = 100 ms); bistable range 0.30–0.33 nS with the values below |
| E→I NMDA | g_EI | 0.292 nS | C00 (verify) |
| I→E GABA_A | g_IE | 1.336 nS | C00 (verify) |
| I→I GABA_A | g_II | 1.024 nS | C00 (verify) |
| recurrent AMPA | | none (all recurrent excitation via NMDA) | C00 control network (verify) |

## Connectivity
| parameter | symbol | value | source |
|---|---|---|---|
| E→E footprint | W(Δθ) | J⁻ + (J⁺ − J⁻) exp(−Δθ²/2σ²), mean 1 | C00 (verify) |
| footprint width | σ | 18° | C00 (verify) |
| footprint peak | J⁺ | **3.0** | **retuned** (C00: 1.62). Needed once the tuning heterogeneity below halves the mean tuned input |
| footprint floor | J⁻ | 0.71 (from normalisation) | derived |
| E→I, I→E, I→I | | all-to-all, uniform | C00 (verify) |
| tuning heterogeneity | h_i | h_max · Beta(1, 1) = U(0, 1), stratified | **this work**: cell i receives (1 − h_i) of its E→E input through W and h_i through the broad kernel 1 + κ cos Δθ |
| broad kernel modulation | κ (kappa_broad) | 0.25 | **this work** (κ ≥ 0.5 makes the spontaneous state unstable) |
| Fourier truncation of W | K | 24 harmonics (max error 1e-14) | numerical |

## External drive and cue
| parameter | symbol | value | source |
|---|---|---|---|
| external Poisson rate | ν_ext | 1.8 kHz per cell (1,000 inputs × 1.8 Hz, lumped) | C00 (verify) |
| external AMPA conductance onto E / I | g_ext | 3.1 / 2.38 nS | C00 (verify) |
| cue | | current to E cells, 500 pA × exp(−Δθ²/2·(18°)²), 50 ms | task (50 ms, Panichello et al. 2024); amplitude **this work** (C00 used a longer, weaker cue; 200 pA × 50 ms did not reliably start a bump) |
| cue locations | | 8, 45° apart | task |

## Common input m(t) ("attention dips") and kick
| parameter | symbol | value | source |
|---|---|---|---|
| applied to | | external Poisson rate of all E cells (× m_eff) | instructions §5 |
| dip onsets | | Poisson, **7 per s** (gap after a dip exponential) | **tuned** (rate model fit: 10.2/s) |
| dip lengths | | gamma, shape 8.7, mean **150 ms** | **tuned** (rate model fit: 90 ms, shape 8.7); 90 ms dips are bridged by the 100 ms windows |
| dip depth | | **0.5** of the drive | rate model fit: 0.60; tuned here (0.6 loses the memory more often) |
| ramps | | 10 ms at both edges | rate model |
| kick | A (m − m̄) | A = **1.0**, m̄ recovers with τ_kick = **150 ms** | **this work**; rate model: A ≈ 0.8, τ = 15 ms. A short kick of A ≥ 1 makes every E cell fire, saturates their NMDA gating and erases the memory |
| no dips before the cue | | | instructions §5 |

## Trial and simulation
| parameter | value | source |
|---|---|---|
| pre-cue time | 500 ms (stability test: 2 s) | |
| delay (cue onset to end) | 1,500 ms | task (1.4–1.6 s) |
| response probe after the delay | 300 ms, m = 1 | this work (memory read-out check) |
| time step | 0.1 ms, forward Euler (V, NMDA s), exact decay (AMPA, x, GABA) | C00 used 0.02 ms |
| external Poisson input | exact time-rescaling of an inhomogeneous Poisson train per cell | |

## Read-out (as the data, `readout.py`)
| parameter | value | source |
|---|---|---|
| recorded cells per session | 330 (264 E + 66 I), random over the ring | data: ~329 units per session |
| sessions × trials | 12 × 480 (60 per cue); no-dip control 6 × 480 | |
| windows | 100 ms, step 10 ms | data |
| decoder | L2 logistic, λ = 10,000, intercept as a penalised feature of value 1, cue vs opposite cue, correct trials | data note |
| leave-one-out | exact for the second-order expansion of the loss (checked against Newton + Rad–Maleki ALO and exact refits) | this work |
| null | 30 label shuffles per cue pair; z per time point against mean and SD of the null over trials and shuffles | data |
| On / Off | z > 1.96 with summed z above the null's 95th percentile / ≥ 5 points of z < 0.84; 0–1,400 ms; fractions 500–1,400 ms | data |
| top-k ranking | split-half ANOVA between the two cues of each decoded pair | this work (the data note does not say whether the ranking is per pair; see NOTE.md) |
