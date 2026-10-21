# Request: Off states in confidently excitatory units

**Why.** In the model, the memory pools and the background population are all excitatory. Its predictions
therefore apply directly to excitatory cells. The earlier mechanism check (analysis 1) mixed all units, so a
lack of change in untuned units could have come from interneurons diluting the effect. Restricting to
confidently excitatory units removes that possibility.

**Settings.** Identical to `mechanism.py` analysis 1, unless stated:
- 25 sessions, correct trials, λ = 10,000;
- states from top-20 units in one unit half (H1), with activity measured in the other half (H2); 5 splits per
  session, both directions;
- tuned (p < 0.05) / untuned (p > 0.5) and the preferred cue computed on the other trial half;
- rates from raw spikes in 10 ms bins with a 20 ms boxcar, as ratios to each unit's own pre-cue rate (−400 to 0
  ms).

Restrict the **measured** (held-out) units to those classified as excitatory with confidence. The labelling
units (H1 top 20) stay as before, drawn from all units. Report how units were classified and how many
excitatory units fall in each group per session.

---

## 1. Population rate around Off states, excitatory units only
Same as analysis 1, for three groups of held-out excitatory units:
- tuned, on preferred-cue trials;
- tuned, on opposite-cue trials;
- untuned, on all trials.

Report:
- mid-Off and mid-On rate ratios (Off and On states ≥ 100 ms; mid = onset + 50 ms to offset − 50 ms);
- curves aligned to Off onset and to Off offset, −200 to +400 ms. Use Off onsets in 300–1,000 ms for the
  onset-aligned curves, so every bin uses the same states;
- for comparison, the same three groups using **all** held-out units (as before) and using the units **not**
  classified as excitatory, with identical splits.

## 2. Recovery after Off states (tuned excitatory units, preferred-cue trials)
For each Off state that starts 300–1,100 ms after cue onset and is followed by an On state within 200 ms:
- the held-out rate (as a ratio to pre-cue) from Off offset to +300 ms;
- the **recovery time:** the first bin after Off offset where the rate reaches half-way from its mid-Off level
  to its mid-On level, and the time to 90% of the way. Compute these on curves averaged per session, separately
  for each Off-length bin below.

Bin by Off length: 50–100, 100–200, 200–400 and > 400 ms. Report, per bin:
- the recovery times;
- the mean aligned curve;
- the fraction of Off states followed by an On state (correct cue) within 200 ms, against the fraction followed by
  a confidently incorrect period within 200 ms.

## 3. Same as 2, untuned excitatory units
To see whether recovery after Off is memory-specific (only tuned cells recover, or recover more) or global.

---

## Output
- CSVs per analysis, with summary (mean, SD and SEM over sessions) and per-session values;
- aligned curves with the number of events and units per bin;
- a short note: the classification method and its criteria, counts per group, any deviations.
