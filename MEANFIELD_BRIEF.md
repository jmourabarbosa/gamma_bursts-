# Brief: mean-field model of decoder On/Off states without synaptic plasticity

## Goal
Build a reduced (mean-field / rate) model of a spiking network we already simulated. Use it to fit
the On/Off state-length histograms of Panichello et al. 2024 (Nature, Fig. 2c) by changing a few
physiologically meaningful parameters. The model must:
1. hold a binary working memory (cue A or B) as attractor activity, with no synaptic plasticity;
2. go through spontaneous population "Off" states, where a decoder falls to chance, caused by
   dips in the shared input;
3. carry the memory across Off states through slow NMDA-like synaptic gating;
4. produce On and Off state-length histograms cheaply, so that the parameters can be fitted.

## What the spiking model showed
- **Network:** a Brunel-style sparse E/I LIF network, 10,000 E and 2,500 I cells, 10% connectivity,
  g = 7, external drive 0.9× threshold.
- **Memory pools:** two pools A and B, each 2% of the E cells. Within-pool weights are
  (2 + 2α)·J, between-pool weights are 0, and α ≈ 2.8–3.0.
  - The cued pool fires ~90–125 Hz; the uncued pool fires < 1 Hz.
  - Background E cells fire a few Hz; some are weakly tuned through random pool inputs.
- **Synapses:** AMPA 2 ms, GABA 5 ms, NMDA linear (no Mg block) carrying 40% of the excitatory
  charge, with decay time τ_N.
- **Off states** come from shared dips of the external drive. A dip silences the whole network,
  memory pool included, for ~50–500 ms, and decoders on any sample fail.
- **Recovery:** when the drive returns, the pool with residual NMDA gating (the cued one)
  restarts first, so the memory survives without plasticity.
- **Without input dips** there are no Off states.
- **τ_N sets whether the memory survives a pause.** With slow dips (Ornstein-Uhlenbeck input,
  τ = 500 ms, pauses with median ~400 ms), τ_N = 300 ms was needed: 93% of pauses were followed by
  a correct restart, against ~50% at 100 ms.
- **The current best spiking setting** uses brief Poisson dips (3/s, gamma-distributed lengths
  with mean 120 ms, drive −80%), α = 3.0, τ_N = 100 ms and a 1.5 s delay. With random 100-cell
  samples it gives:

  | | Spiking model | Panichello |
  |---|---|---|
  | Memory correct at end of delay | 95% | – |
  | Trial-averaged decoding | 85% | ~70% |
  | Off states per trial | 1.6 | 3.7 |
  | Off length, mean / median | 191 / 130 ms | 146 / 100 ms |
  | On states per trial | 2.7 | 2.4 |
  | On length, mean | 94 ms | 192 ms |

  So Off states are too long and too few, and On states are too short. Shorter, sparser dips
  are being tested.

## Targets (Panichello et al. 2024)
From the text, as summarised in our earlier work:
- Off states: 3.7 per trial, mean 146 ms, median 100 ms.
- On states: 2.4 per trial, mean 192 ms.
- Trial-averaged decoding about 70%. A two-state beta-mixture beats a one-state fit (19 of 25
  sessions).

**Check against the paper.** In Fig. 2c one histogram has n = 3,890 states and peaks ~100 ms with
a tail to 600 ms. The other has n = 6,174 states and peaks ~60–70 ms. Read the legend to see which
colour is On and which is Off. The counts suggest the more numerous, shorter histogram is Off,
which would fit "Off median 100 ms, mean 146 ms; On mean 192 ms". Then digitise both histograms,
20 ms bins from 0 to 600 ms, as the fitting targets. Also take from the paper the delay length,
the number of trials per session and the number of neurons per session.

## How Panichello define On/Off states (copy this exactly when validating)
- Spike counts in 100 ms windows stepped by 10 ms.
- Leave-one-trial-out L2 logistic decoder trained separately per time bin. We use λ = 10 on
  z-scored counts.
- Null distribution: the same decoder with shuffled labels (10–20 shuffles). z-score the
  confidence against it.
- **On:** contiguous runs with z > 1.96, cluster-mass corrected.
- **Off:** runs of ≥ 5 consecutive bins (≥ 50 ms) with z < 0.84, i.e. confidence at chance.
- State lengths are counted within the delay.

Reference code (copy these files into the new session):
- `gamma_wm/panichello.py`
- `gamma_wm/analysis.py`, which contains `loo_confidence_dual`
- `scripts/nmda_sweep.py`, which pools state lengths

## Suggested mean-field model (starting point; feel free to improve)
**Populations:** pools A and B (excitatory), background E, and one shared inhibitory population I.
A smaller version merges background E and I into a single inhibitory feedback term.

For each pool X ∈ {A, B}:

    τ_r dr_X/dt = −r_X + Φ( w_s·s_X + w_a·r_X − w_I·r_I + w_bg·r_bg + I_ext·m(t) + σ·ξ_X(t) )
    ds_X/dt     = −s_X/τ_N + γ (1 − s_X) r_X           # NMDA gating, saturating; τ_N is a parameter
    τ_I dr_I/dt = −r_I + Φ_I( w_EI (r_A + r_B + c·r_bg) + I_ext,I·m(t) )
    τ_r dr_bg/dt = −r_bg + Φ_bg( w_bb·r_bg − w_bI·r_I + I_ext·m(t) + σ_bg·ξ_bg(t) )

- **Φ** is a sigmoid or threshold-power-law transfer function, chosen so that each pool is
  bistable (low ~1 Hz and high ~50–100 Hz) at m = 1. The strength of the recurrent term plays
  the role of α (call it w_s or α).
- **Shared inhibition** makes A and B compete (winner-take-all): only one pool can be high.
- **m(t)** is the shared input multiplier. Keep both versions:
  - (a) Poisson dips: rate λ_d per s, gamma-distributed lengths with mean μ_d and shape k,
    depth δ (m = 1 − δ inside a dip), 10 ms ramps;
  - (b) Ornstein-Uhlenbeck input with correlation time τ_m and SD σ_m, clipped at 0.
- **ξ** is fast finite-size noise (white or OU with τ ≈ 5–20 ms), which makes restarts after dips
  stochastic.
- **Cue:** a brief input to the cued pool at delay onset.

**Key behaviour to check:** during a dip both pools drop to the low state. After the dip the pool
with larger residual s_X restarts. The probability that the cued pool wins should depend on
dip length relative to τ_N and on the noise σ.

## Turning rates into decoder On/Off states
1. **Full validation (slow):** simulate K recorded cells as Poisson units. A fraction f_sel
   (~3%) are memory cells with rate r_A or r_B. The rest are background cells with rate
   r_bg + ε_i (r_A − r_B), where ε_i is a weak random tuning (SD to calibrate; ours arises from
   random pool inputs). Run Panichello's pipeline above on 40 trials × several sessions.
2. **Fast surrogate (for fitting):** in each 100 ms window, the expected decoder signal-to-noise
   is roughly
   `SNR(t) ≈ sqrt( Σ_i (Δλ_i(t)·0.1 s)² / λ_i(t)·0.1 s )`, where Δλ_i is cue A minus cue B.
   Map it to confidence and to the z-score against the null, then apply the On/Off rules.
   Calibrate this mapping once against the full pipeline, then fit with the surrogate.

## Fitting
- **Parameters to fit:**
  - NMDA: τ_N (constrain to 50–150 ms; 100 ms is the standard model value, e.g. Wang 1999,
    Compte et al. 2000);
  - recurrent strength α (w_s);
  - dip process: λ_d, μ_d, k, δ (or τ_m, σ_m);
  - fast noise σ;
  - recording: K and the background tuning SD.
- **Loss:** distance between model and paper histograms of On and Off lengths (e.g. KL or
  Wasserstein on the 20 ms bins), plus penalties to match:
  - Off states per trial (3.7);
  - On states per trial (2.4);
  - trial-averaged decoding (~70%);
  - memory correct at the end of the delay (≥ 95%), so the memory may not fade.
- **Optimiser:** Bayesian optimisation, CMA-ES or simulation-based inference (e.g. `sbi`). A grid
  over (τ_N, μ_d, λ_d, δ) is a good first look.
- **Deliverables:**
  - best-fit parameters with uncertainty;
  - a model vs paper histogram figure (On and Off);
  - the minimal τ_N that still keeps memory ≥ 95% while matching the histograms;
  - which parameters set Off length (expected: dip length) and which set On length (expected:
    gaps between dips).

## Files to give the new session
1. The Panichello et al. 2024 paper (PDF).
2. This brief (`MEANFIELD_BRIEF.md`).
3. `REPORT_SHORT.md`: spiking-model results, including the NMDA sweep and the attractor test.
4. `gamma_wm/panichello.py` and `gamma_wm/analysis.py`: the decoder pipeline to reuse for
   validation.
5. Optional: `gamma_wm/task.py`, with the exact dip and OU input generators (`shared_dips`,
   `slow_input_noise`) so the mean-field input matches the spiking model.
