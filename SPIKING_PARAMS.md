# Brief: spiking-network parameters from the mean-field fit

Goal: run the spiking network (`gamma_wm/`) with parameters carried over from the mean-field fit, and match the
On/Off statistics of the recordings while keeping the memory in ≥ 95% of trials. This brief gives a best-guess
starting point, a small grid around it, the analysis to run on the spiking output (matched to the data analysis),
and the targets.

Background in this repository:
- `MEANFIELD_BRIEF.md`: the spiking network and the mean-field model it motivated.
- `meanfield/`: the rate model and the fits.
- `figures/meanfield/fig6_dynamics_fits.png` and `fig7_readout_data.png`.
- `meanfield/data/readout_*_data.csv`: the data analysis by unit subset.

## 1. What the mean-field model says (what carries over, what doesn't)

**Carries over.**
- **Off states are pauses caused by shared input dips.** Their count follows the dip rate and their length follows
  the dip length.
- **The memory survives a pause through slow NMDA plus competition at restart.** When the drive returns, both
  pools get a brief kick and the one with more residual NMDA wins. In the rate model the cued pool restarts first
  after silent pauses up to about 3·τ_N, beyond which it becomes a coin flip.
- **Keeping memory ≥ 95% at the observed number of Off states needs the longest allowed NMDA time constant.**
  Every fit, in every variant, pushed τ_N to its 150 ms ceiling.

**Does not carry over directly.**
- The rate model's recurrent strength w_s and fast noise σ have no one-to-one spiking counterpart. Recurrent
  strength maps to α; σ is the network's own finite-size noise.
- Rate-model time constants are faster than the spiking network's recovery. In the spiking network an Off state
  lasts about dip length + ~60 ms: with dips of mean 120 ms, Off states had mean 191 ms and median 130 ms.
  **Spiking dips therefore need to be shorter than the rate-model dips.**

**Fitted dip parameters (rate model, same generator as `gamma_wm.task.shared_dips`):**

| | Fit 1: read-out gain fitted | Fit 2: read-out-free (network silences) |
|---|---|---|
| τ_N | 150 ms | 150 ms |
| Dip rate | 7.7 /s (7.1–8.0 across 4 runs) | 4.7 /s |
| Dip mean length | 122 ms (104–137) | 122 ms |
| Dip shape k (gamma) | 3.8 (2.9–6) | 4.1 |
| Dip depth | −92% | −93% |
| Off per trial / Off fraction (0–1,400 ms) | 3.45 / 0.34 | 2.13 / 0.23 |
| Memory correct (fresh seeds) | 97.4% | 96.6% |

Fit 1 gets about a third of its Off states from read-out failures during weak activity. Fit 2 is the network's
own silences only, and it produces too few (2.1 vs 3.6 per trial). **The spiking run should aim between them:
about 3.6 real pauses per 1.4 s, at the data's Off lengths.**

## 2. Starting point (best guess)

Start from the configuration of the current best spiking run (`figures/nmda100_dips_a3.0`: NMDA 100 ms,
α = 3.0, dips 3/s, mean 120 ms, shape 2, depth 0.8). Keep the network, connectivity, synaptic time constants,
NMDA charge fraction (40%), pool size (2% each) and parameterisation. Change only:

| Parameter | Current spiking run | **Start here** | Reason |
|---|---|---|---|
| `tau_nmda` | 100 ms | **150 ms** | Every rate-model fit went to the 150 ms ceiling to keep memory at this many pauses |
| `dip_rate` | 3 /s | **6 /s** | Need ~3.6 Off per 1.4 s; not every dip produces an Off state, and long dips merge |
| `dip_mean` | 120 ms | **80 ms** | Spiking Off ≈ dip + ~60 ms recovery; target Off mean ~135–150 ms, median ~100 ms |
| dip shape k | 2 (hard-coded) | **4** | Fits chose k ≈ 3–6: fewer very short dips (no Off) and fewer very long ones (memory loss) |
| `dip_depth` | 0.8 | **0.9** | Fits chose 0.92–0.93; deep enough to silence the pools |
| α | 3.0 | **3.0** | Keep; it held memory at 95% with τ_NMDA 100 ms. Raise to 3.1 only if restarts fail |
| cue duration | 150 ms | **50 ms** | As in the paper; raise the cue strength if 50 ms does not reliably start the memory |
| delay | 1.5 s | **1.6 s simulated** | Paper delay 1,400–1,600 ms; the analysis counts states 0–1,400 ms after cue onset |

**Code change needed:** `gamma_wm.task.run_trial` calls `shared_dips` without passing `shape`, so the dip shape
is fixed at 2. Add a `dip_shape` argument to `run_trial` and pass it through.

## 3. Small grid (if the starting point misses)

Change one or two at a time, in this order of expected effect:

| Parameter | Values | Main effect |
|---|---|---|
| `dip_rate` | 4, 6, 8 /s | Off states per trial, fraction of time Off; higher costs memory |
| `dip_mean` | 60, 80, 100 ms | Off length; longer costs memory |
| `tau_nmda` | 125, 150 ms | Memory survival across pauses (stay within 50–150 ms) |
| `dip_depth` | 0.85, 0.9, 0.95 | Whether a dip silences the pools at all |
| dip shape k | 3, 4, 6 | Width of the Off length distribution |
| α | 3.0, 3.1 | Restart reliability; too strong and pools stop going silent in dips |

If no setting gives ≥ 3.3 Off per trial with memory ≥ 95% and τ_NMDA ≤ 150 ms, report that. The rate model
predicts this tension, and it is a result in itself.

## 4. Analysis of the spiking output (match the data analysis exactly)

**Sessions and trials**
- Binary task (cue A or B), **66 trials per cue (132 per session)**.
- At least **5 sessions**: different network seeds, or one network with independent trial seeds and independent
  unit samples.

**Recorded units**
- Per session, **329 units** sampled uniformly at random from the whole network (E and I, pools included).
- From the same session also form **top-20** sets (units ranked by a two-sample test between cues on the mean
  rate 500–1,400 ms after cue onset, on one half of the trials and applied to the other half; never using the test
  trial) and **random-20** sets (10 draws).

**Decoder** (as the data analysis, `meanfield/data/readout_note_data.md`)
- 100 ms boxcar firing rates in 10 ms steps, z-scored across trials per unit and time point.
- Leave-one-trial-out L2 logistic regression (approximate leave-one-out is fine), one decoder per time point.
- Run **λ = 30** (as the data) and a saturated λ (≥ 300; check that the results stop changing).
- 30 shuffled-label nulls; z = (confidence − null mean) / null SD.
- **On:** runs of z > 1.96 above the cluster-mass threshold. **Off:** ≥ 5 points with z < 0.84.
- States counted **0–1,400 ms after cue onset**. Accuracy over 500–1,400 ms.
- Reusable code: `meanfield/recording.py` (`full_pipeline`, `state_stats`), `meanfield/decoder.py`
  (`alo_confidence`), `meanfield/surrogate.py` (`label_states`, compiled).

**Network checks** (from the simulated rates, not the decoder)
- Memory correct at the end of the delay: cued pool ahead (rate, or NMDA gating if both are paused) over the last
  100–300 ms.
- Pool pauses: both pools < 2 Hz for ≥ 50 ms (as `scripts/nmda_sweep.py`). Report count and length, and whether
  the cued pool restarts first, as a function of pause length.
- Background population rate during Off states vs before the cue (the paper reports Off ≈ spontaneous).

## 5. Targets (data analysis: 25 sessions, λ = 30, states 0–1,400 ms)

**Primary: Off states with top-20 units** (robust: nearly the same for top 5 to top 200)

| Statistic | Target | Range over top 5–200 |
|---|---|---|
| Off states per trial | **3.55** | 3.55–3.85 |
| Fraction of delay Off | **0.35** | 0.35–0.37 |
| Off length, mean / median | **138 / 100 ms** | 128–152 / 95–108 ms |
| Off length histogram | `readout_histograms_data.csv`, set = top, k = 20 | |
| Decoding accuracy | 0.73 | 0.69–0.77 |

**Hard constraint:** memory correct at the end of the delay **≥ 95%**.

**Secondary: read-out checks.** These should follow if the network matches the primary targets and the recorded
sample is realistic.

| Set | Off per trial | Off mean | Fraction Off | Accuracy |
|---|---|---|---|---|
| All units (329) | 4.09 | 132 ms | 0.42 | 0.73 |
| Random 20 | 4.90 | 159 ms | 0.58 | 0.59 |

**Report but do not fit: On states.** They depend strongly on λ and on the number of units (null width). Data,
top 20 at λ = 30: 3.0 per trial, mean 153 ms, fraction On 0.33. All units: 1.66 per trial, 91 ms, 0.08.
Paper Fig. 4c (different analysis, counted to the end of the delay): On 2.35 per trial, mean 192 ms.

## 6. What to send back

- Per parameter setting: the tables in §5 (top 20, random 20, all units; λ = 30 and saturated λ), memory correct,
  and the network pause statistics.
- Off and On length histograms (20 ms bins, 0–600 ms plus > 600) for top 20 and all units.
- One example trial per setting: pool rates, input dips, decoder z and state labels.
- The exact parameters and seeds of every run.
