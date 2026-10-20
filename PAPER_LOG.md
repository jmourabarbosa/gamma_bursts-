# Paper log: intermittent rate coding without short-term plasticity

Running log of the paper's structure and decisions. Newest entries at the bottom of the *Log*
section; the outline above it is kept current.

## Core claim
Single-trial working-memory decoding that switches between On and Off states (Panichello et al.
2024) does not need short-term plasticity. Brief shared dips in the input silence the network
(Off), and when the input recovers, the cued attractor restarts first because of residual NMDA
gating. The memory is carried by ordinary rate coding with fixed synapses.

## Outline

### 1. Introduction
- Persistent activity vs activity-silent working memory (STP: Mongillo et al. 2008; Lundqvist
  et al. 2016; Panichello et al. 2024).
- Panichello's evidence: intermittent single-trial decoding, On/Off states, loss of tuning and
  rate in Off, cue-specific CCG connectivity in Off. Their conclusion: memory is held in synapses.
- Our question: can the same signatures arise from a rate code with fixed synapses?

### 2. Mean-field model: input switching + recovery (Figure 1–2)
- Mechanism: shared input that switches between an "on" and an "off" state (in the repo:
  *shared input dips*; Poisson onsets, gamma-distributed lengths, depth δ. Could also be
  described as a two-state / telegraph input). During an off state all pools fall silent;
  on recovery, the pool with larger residual NMDA gating s_X wins.
- Show that it works: memory survives dips, and survival depends on dip length relative to τ_N
  and on fast noise σ.
- Fit to Panichello's On/Off state-length histograms (Fig. 2c / Fig. 4b,c), plus states per
  trial, trial-averaged decoding (~70%), memory correct ≥ 95%.
- Output: best-fit parameters (τ_N, α, λ_d, μ_d, k, δ, σ, K), with uncertainty; which parameters
  set Off length (dip length) and which set On length (gaps between dips).
- Spec: [`MEANFIELD_BRIEF.md`](MEANFIELD_BRIEF.md). **Status: not built yet.**

### 3. Spiking model instantiated with the fitted parameters (Figure 3–4)
- Brunel-type E/I LIF network, two memory pools, NMDA τ_N, shared dips taken from the
  mean-field fit.
- Reproduce with Panichello's pipeline: single-trial decoding, On/Off histograms, two-state
  beta mixture, held-out tuning and rate in On vs Off.
- Mechanism figure: background silenced by dips, memory pool pauses and restarts.
- Existing material (hand-tuned, not yet from the fit): model 1
  (`m1_r4_s4_d85`), see [`PAPER_COMPARISON.md`](PAPER_COMPARISON.md),
  `figures/summaries/m1_r4_s4_d85/`. **Status: to be re-run with fitted parameters.**

### 4. Cross-correlograms with the spiking model (Figure 5)
- Panichello Fig. 5 analogue: cue-specific CCG connectivity over the whole delay and in Off
  states, from activity rather than potentiated synapses.
- Explanation: whole-delay edges come from background cells sharing input from the active pool;
  Off-state edges involve memory cells that keep firing while the background is silent.
- Existing material: `scripts/ccg_analysis.py`, `figures/ccg/`. **Status: needs several
  independent sessions and more trials for proper statistics.**

### 5. Discussion
- What would distinguish STP from this account experimentally.
- Possible sources of the input dips (e.g. beta bursts, network-wide inhibitory events).
- Limits: binary cue instead of 8 locations; On states shorter than in the data.

## Open questions
- Name for the on/off input process in the paper (dips, two-state input, telegraph input?).
- Match Panichello's delay length, trial count and neurons per session before final fits.
- Digitise Fig. 2c histograms (20 ms bins, 0–600 ms) as fitting targets.

## Log
- **2026-10-19.** Paper structure set: intro → mean-field model of input switching and recovery,
  fitted to On/Off histograms → spiking model with the fitted parameters → CCGs from the spiking
  model. Existing spiking and CCG results (`PAPER_COMPARISON.md`) serve as preliminary material.
- **2026-10-20.** Short draft written as a shared doc, titled "Intermittent Rate Coding Without
  Short-Term Plasticity Can Support Working Memory":
  https://claude.ai/code/artifact/4cc2f6a8-f451-40d9-bc32-58838fb49369. Spiking and CCG numbers
  come from model 1 (`PAPER_COMPARISON.md`). The mean-field fit results are still placeholders.
