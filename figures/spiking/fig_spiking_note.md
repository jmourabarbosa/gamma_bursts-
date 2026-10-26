# Hand-off: spiking network figure (`fig_spiking`)

## For the paper agent: what this figure is
A spiking-network version of the rate model's mechanism.
- **Network:** a ring attractor of 2,048 excitatory and 512 inhibitory leaky integrate-and-fire neurons, based on
  Compte et al. 2000. Recurrent excitation is carried only by NMDA receptors, with a 150 ms decay.
- **Common input:** an input to all excitatory cells briefly drops ("dips"). Each dip silences the bump.
- **Restart:** when the input returns, the bump restarts at the cued location. It is driven by the cued cells'
  residual NMDA gating and a brief overshoot of the input (the "kick").
- **Read-out:** the network is read out like the recordings, and shows the data's On/Off states.

It is meant to show that the rate model's mechanism works in a network of spiking neurons. It is **not** fitted to
the data.

**Reference figure:** `figures/spiking/fig_spiking.png` (600 dpi), `.svg` (editable text).

## Panels
**a. One trial.**
- **Trial structure:** 500 ms of fixation, then a 50 ms cue at 180° (the middle of the ring), then 1 s of delay with
  a constant common input, then 1 s with dips.
- **Top:** spikes of all 2,048 excitatory cells, sorted by preferred location, drawn as counts in 5 ms × 8-cell bins.
- **Middle:** the common input to the excitatory cells, with its overshoot after each dip (1 = normal level).
- **Bottom:** the all-units decoder z of this trial, with On (red) and Off (blue) states and the two thresholds.

How the decoder was trained:
- the data's procedure (L2 logistic regression, λ = 10,000, cue vs opposite cue, leave-one-out, 30 label
  shuffles);
- on a session of 480 trials of this structure (60 per cue; 450 correct trials used), with 330 recorded cells
  (264 E, 66 I) sampled across the ring.

What it shows:
- with constant input the bump is stable and z stays high (one long On state);
- with dips the bump is silenced, z drops to chance (Off), and the bump restarts at the cued place when the input
  returns.

**b. State lengths (all units).**
- **Model** (bars): 12 simulated sessions × 480 trials of the main simulation (1.5 s delay with dips from cue onset);
  14,385 Off and 16,182 On states.
- **Data** (black line): the 25 sessions, all units, λ = 10,000 (`paper_model/data/readout_histograms_data_lam10000.csv`).
- 20 ms bins, 0–600 ms. States longer than 600 ms are not drawn (Off: data 4%, model 0%; On: data 2%, model 3%).

**c. Cross-validated memory tuning functions in On and Off states** (as Panichello et al. 2024, Fig. 4d and Methods
"Tuning curves").
- Firing rates in 100 ms windows (10 ms steps), z-scored across trials at each time point.
- Recorded cells split at random into halves. On/Off states are labelled with the decoder of one half.
- For each held-out cell: mean rate during the delay (500–1,400 ms) per cue, separately for On and Off time points.
  This gives two 8-point tuning functions.
- Each cell is aligned to its preferred cue: the largest value of On + Off.
- The same computation with shuffled cue labels, aligned to its own maximum, is subtracted to remove the alignment
  bias.
- Halves swapped; 10 random splits, averaged per cell; mean ± s.e.m. over the 330 cells of session 0 of the main
  simulation.

What it shows:
- **On states:** held-out cells are strongly tuned. +1.26 z at the preferred location, −0.58 z opposite.
- **Off states:** tuning is gone. +0.02 z at the preferred location, −0.01 z opposite.

This is as in the data, where On is tuned and Off is flat.

## Suggested legend
**Spiking network model.**
- **a,** Example trial of a spiking ring attractor with NMDA-mediated recurrent excitation (τ_NMDA = 150 ms).
  - Top: spikes of the 2,048 excitatory neurons, sorted by preferred location. Grey band: cue (180°).
  - Middle: common input to the excitatory neurons. It is constant for the first second of the delay and contains
    brief dips for the second; each dip is followed by a transient overshoot.
  - Bottom: single-trial decoding of the cued location from 330 simulated neurons, using the same decoder, null and
    labelling as the recordings. Shading: On (red) and Off (blue) states; dashed lines: P < 0.05 and P > 0.20.
- **b,** Distribution of Off and On state lengths (all units; model, coloured bars, 12 simulated sessions; data,
  black lines, 25 sessions).
- **c,** Cross-validated memory tuning functions of held-out simulated neurons during On (red) and Off (blue) states,
  computed as in Fig. 4d (mean ± s.e.m., n = 330 neurons).

## Methods text (draft, edit freely)
**Spiking network.**
- **Network:** a ring of 2,048 pyramidal neurons and 512 interneurons (leaky integrate-and-fire), with parameters
  from Compte et al. (2000).
- **Excitation:** recurrent excitation is mediated only by NMDA receptors (voltage-dependent Mg²⁺ block, saturating
  gating, decay 150 ms). E→E weights decay with the difference in preferred location (Gaussian, σ = 18°); all other
  connections are uniform.
- **Heterogeneity:** neurons are heterogeneous in firing threshold and in how sharply their recurrent input is tuned.
- **Common input:** all excitatory neurons receive external Poisson input whose rate is scaled by a common signal
  m(t). Dips of m(t) occur at random:
  - Poisson onsets, 7 per s;
  - gamma-distributed lengths, mean 150 ms;
  - 50% depth;
  - 10 ms ramps;
  - none before the cue.

  Each dip is followed by a transient overshoot that decays with 150 ms.
- **Cue:** a 50 ms current to excitatory neurons around one of 8 locations.
- **Read-out:** sessions of 480 trials (60 per cue) were simulated. Spikes of 330 randomly sampled neurons were read
  out exactly as the recordings, with L2 logistic regression, leave-one-out, the label-shuffle null and the On/Off
  labelling.

## Key numbers (main simulation, 12 sessions × 480 trials; data in brackets)
**All units:**
- Off states per trial 3.0 (3.25); Off median 98 ms (112 ms); fraction of delay Off 0.26 (0.45).
- On states per trial 3.4 (2.10); On median 202 ms (134 ms); fraction On 0.59 (0.21).

**Memory:**
- held in 83% of trials;
- 10% of trials end with the bump at the wrong location.
- Without dips: 100% held, and no Off states for the all-units decoder.

## Caveats for the text (please keep the claims modest)
- **Dips differ from the rate model's fit.** The spiking model uses longer, sparser, shallower dips (7/s, 150 ms,
  depth 0.5) and a slower overshoot (150 ms) than the fitted rate model (10/s, 90 ms, 0.6, 15 ms).
  - With 90 ms dips the 100 ms decoding windows bridge most silences, so too few Off states.
  - A sharp overshoot makes all excitatory neurons fire synchronously, which saturates their NMDA gating and erases
    the memory.
- **Model decoders are much stronger than the data's.**
  - Single-trial z ≈ 10 in panel a; μ_on 5.7 against 2.2.
  - On states are about 70 ms too long and rarely shorter than 100 ms (panel b, right).
  - Tuning in On states is about 6× the data's in z units (1.26 vs about 0.2).

  Model neurons are more selective than recorded units and nearly silent away from their preferred location.
- **Top vs random units inverted.** Top-unit decoders are not better than random ones in the model (the reverse of
  the data). The figure therefore shows only the all-units read-out.
- **Fig. 5 not reproduced.** Cue-specific functional connectivity (Fig. 5 analysis) is present in On states but
  absent in Off states in the model, the opposite of the data (`spiking/NOTE.md`). Do not use this figure to claim
  the model explains Fig. 5.
- **Parameters unchecked against the papers.** Published parameter values were taken from memory and not yet checked
  against the papers (`spiking/PARAMETERS.md`).

## Files (all in `figures/spiking/`)
| file | content |
|---|---|
| `fig_spiking.png`, `.svg` | the figure |
| `fig_spiking_a_trial.npz` | panel a: spike times (ms) and cells, cell preferred angle, common input (1 ms), decoder z, states (+1 On, −1 Off) |
| `fig_spiking_b_histograms.csv` | panel b: model and data fractions and counts per 20 ms bin (last row: ≥ 600 ms) |
| `fig_spiking_c_tuning.csv` | panel c: mean and s.e.m. per cue offset, On and Off |
| `fig_spiking_meta.json` | example trial, numbers of states and cells, tuning values |

## Reproduce
```
python -m spiking.run sessions          # main simulation (~80 min, 4 cores), if spiking/results/ is empty
python -m spiking.fig_paper             # long-trial session (~15 min) and this figure
```
Code: `spiking/fig_paper.py`. Network: `spiking/network.py`; read-out: `spiking/readout.py`.
