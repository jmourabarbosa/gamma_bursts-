"""Short PDF report of the spiking ring network: methods, results, all figures (spiking/report.pdf).

    python -m spiking.make_report
"""
import json
import os

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, 'figures')
RES = os.path.join(HERE, 'results')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_B = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
pdfmetrics.registerFont(TTFont('DV', FONT))
pdfmetrics.registerFont(TTFont('DVB', FONT_B if os.path.exists(FONT_B) else FONT))
pdfmetrics.registerFontFamily('DV', normal='DV', bold='DVB', italic='DV', boldItalic='DVB')

BODY = ParagraphStyle('body', fontName='DV', fontSize=8.6, leading=11.4, spaceAfter=4)
BUL = ParagraphStyle('bul', parent=BODY, leftIndent=10, bulletIndent=2, spaceAfter=1.5)
H1 = ParagraphStyle('h1', fontName='DVB', fontSize=15, leading=19, spaceAfter=4)
H2 = ParagraphStyle('h2', fontName='DVB', fontSize=11, leading=14, spaceBefore=8, spaceAfter=4)
CAP = ParagraphStyle('cap', parent=BODY, fontSize=7.8, leading=10, textColor=colors.HexColor('#333333'))
SMALL = ParagraphStyle('small', parent=BODY, fontSize=7.4, leading=9.4)
W = A4[0] - 3.2 * cm


def p(t, s=BODY):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(t, BUL, bulletText='•') for t in items]


def figure(name, caption, width=W, head=None):
    path = os.path.join(FIG, name + '.png')
    w, h = Image.open(path).size
    return KeepTogether(([head] if head else []) + [RLImage(path, width=width, height=width * h / w), Spacer(1, 3), p(caption, CAP),
                         Spacer(1, 8)])


def table(rows, widths, header=True):
    t = Table([[Paragraph(str(c), SMALL) for c in r] for r in rows], colWidths=widths)
    st = [('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#bbbbbb')), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
          ('TOPPADDING', (0, 0), (-1, -1), 1.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5)]
    if header:
        st.append(('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eeeeee')))
    t.setStyle(TableStyle(st))
    return t


def stats_table():
    """Model vs data per decoded population, from results/table.md."""
    lines = [l for l in open(os.path.join(RES, 'table.md')).read().split('\n') if l.startswith('| ')]
    n = len(lines[0].split('|')) - 2
    head = [c.strip() for c in lines[0].split('|')[1:-1]]
    head = [h.replace('mu_on', 'μ<sub>on</sub>') for h in head]
    body = [[c.strip() for c in l.split('|')[1:-1]] for l in lines[1:8]]
    return table([head] + body, [2.0 * cm] + [(W - 2.0 * cm) / (n - 1)] * (n - 1))


def control_table():
    lines = [l for l in open(os.path.join(RES, 'table.md')).read().split('\n') if l.startswith('| ')]
    head = [c.strip().replace('mu_on', 'μ<sub>on</sub>') for c in lines[0].split('|')[1:-1]]
    body = [[c.strip().replace('nan', '–') for c in l.split('|')[1:-1]] for l in lines[9:16]]
    n = len(head)
    return table([head] + body, [2.0 * cm] + [(W - 2.0 * cm) / (n - 1)] * (n - 1))


def main():
    rows = json.load(open(os.path.join(RES, 'sessions_dips.json')))['rows']
    mem = {k: sum(r[k] for r in rows) / len(rows) for k in ('mem_held', 'mem_held_probe', 'wrong_place')}
    stab = json.load(open(os.path.join(RES, 'stability.json')))
    br = json.load(open(os.path.join(RES, 'validate_brian2.json')))
    sd = json.load(open(os.path.join(RES, 'single_dip.json')))
    story = [p('A spiking ring attractor with intermittent (On/Off) decoding', H1),
             p('Spiking-network counterpart of the rate model of On/Off states (Panichello et al. 2024 data). '
               'Code, parameters and notes: <font name="DV">spiking/</font> on branch claude/gallant-volta-bkjojq '
               '(network.py, readout.py, run.py, PARAMETERS.md, NOTE.md).', CAP),
             Spacer(1, 4),
             p('Summary', H2)]
    story += bullets([
        'A ring of 2,048 excitatory and 512 inhibitory LIF neurons (Compte et al. 2000) with NMDA-only recurrent '
        'excitation (τ<sub>N</sub> = 150 ms) holds the cued location as a bump. A common input to all E cells briefly '
        'drops ("attention dips"); each dip silences the bump, and it restarts at the cued place when the input '
        'returns, from the residual NMDA gating plus a brief overshoot of the input ("kick").',
        'Read out like the recordings (330 cells per session, L2 logistic decoder, λ = 10,000, shuffle null, the '
        'same On/Off labelling), the network gives about 3 Off states per trial with median lengths of 98–122 ms, '
        'close to the data (102–129 ms) in all 7 decoded populations, and an Off-length histogram of the data\'s '
        'shape. Without dips there are no Off states for the strong decoders.',
        f'Not matched: decoders are much too strong (μ<sub>on</sub> 5.7–7 vs 1.1–3.9), so On time is too high '
        f'and On states too long; top-unit decoders are worse, not better, than random ones; memory is held in '
        f'{100 * mem["mem_held"]:.0f}% of trials (target ≥ 90%), with about 10% restarts at the wrong place; and '
        f'memory falls with time Off far more steeply than the monkeys\' performance. The main cause is that model '
        f'cells are almost silent outside their preferred location (below 1 Hz), unlike PFC units.'])

    story += [p('Methods', H2), p('<b>Network.</b>', BODY)]
    story += bullets([
        'LIF neurons: C = 0.5 / 0.2 nF, g<sub>L</sub> = 25 / 20 nS (E / I), E<sub>L</sub> = −70, V<sub>th</sub> = −50, '
        'V<sub>reset</sub> = −60 mV, refractory 2 / 1 ms. External Poisson AMPA drive, 1.8 kHz, 3.1 / 2.38 nS.',
        'Synapses: AMPA (2 ms, external only), NMDA (rise 2 ms, α = 0.5 ms<super>−1</super>, decay <b>150 ms</b>, '
        'Mg<super>2+</super> block), GABA<sub>A</sub> (10 ms). E→E tuned in angle (Gaussian, σ = 18°, '
        'J<super>+</super> = 3.0), all other connections uniform. g<sub>EE</sub> = 0.33 nS (retuned for '
        'τ<sub>N</sub> = 150 ms; published 0.381 nS at 100 ms), g<sub>EI</sub> 0.292, g<sub>IE</sub> 1.336, '
        'g<sub>II</sub> 1.024 nS.',
        'Heterogeneity (fixed, stratified along the ring so the bump is not pinned): thresholds (SD 0.3 mV), and a '
        'fraction h<sub>i</sub> ~ U(0, 1) of each E cell\'s recurrent input through a broad kernel '
        '1 + 0.25 cos Δθ instead of the Gaussian footprint (narrowly to broadly tuned cells).',
        'Common input m(t) scales the external rate of all E cells: dips with Poisson onsets (7 per s), gamma '
        'lengths (mean 150 ms, shape 8.7), depth 50%, 10 ms ramps, none before the cue. Kick: '
        'm<sub>eff</sub> = m + A(m − m̄), m̄ follows m down instantly and recovers with 150 ms, A = 1.',
        'Task: 500 ms before the cue, 50 ms cue (current to E cells around one of 8 locations), 1.5 s delay, then a '
        '300 ms "response probe" with the input at 1. Batched LIF integration in numba, dt = 0.1 ms; the E→E sum '
        'uses the exact Fourier series of the ring kernel.',
        'Published values (Compte et al. 2000; Wang 1999) were written from memory because the journal hosts are '
        'blocked here; they are marked for verification in PARAMETERS.md.'])
    story += [p('<b>Read-out (as the data).</b>', BODY)]
    story += bullets([
        '12 sessions × 480 trials (60 per cue), 330 recorded cells per session (264 E, 66 I) across the ring. '
        'Spike counts in 100 ms windows every 10 ms.',
        'For each cue vs the opposite cue: L2 logistic regression, λ = 10,000, intercept as a penalised feature, '
        'trained on correct trials, leave-one-out (exact for the second-order expansion of the loss, which matches '
        'full Newton fits to ~1e-4 at this λ); null from 30 label shuffles; decoder output for the cued location '
        'z-scored against the null at each time point.',
        'On = runs of z > 1.96 whose summed z exceeds the null\'s 95th percentile; Off = ≥ 5 points of z < 0.84; '
        'states in 0–1,400 ms, fractions in 500–1,400 ms.',
        'Populations: all cells; top 5/20/50 ranked by split-half ANOVA between the two cues of each pair; random '
        '5/20/50 (10 draws).',
        'Memory held: bump location from the whole E population in the last 100 ms of the delay (spike rate if the '
        'bump is active, else NMDA gating), within 22.5° of the cue.'])

    story += [p('Tuning: what was changed to reach this regime', H2),
              p('Starting point: the published ring (Compte et al. 2000; Wang 1999) and, for the common input, the '
                'rate model\'s fitted dips and kick. With the published values and τ<sub>N</sub> = 100 ms the '
                'network reproduces the published behaviour (spontaneous state ~3 Hz, bump ~50 Hz, slow drift), '
                'which served as an implementation check. Each change below was made for the reason given, in this '
                'order (tests on 8–32 trials per setting, then 240-trial sessions).', BODY),
              table([['Parameter', 'Start', 'Final', 'Why'],
                     ['NMDA decay τ<sub>N</sub>', '100 ms', '150 ms', 'Design requirement.'],
                     ['E→E NMDA g<sub>EE</sub>', '0.381 nS', '0.33 nS',
                      'With τ<sub>N</sub> = 150 ms the published value gives a bump before the cue. Scanned for a '
                      'stable spontaneous state plus a cue-triggered bump; final window 0.30–0.33 nS.'],
                     ['Footprint peak J<super>+</super>', '1.62', '3.0',
                      'At 1.62 the bistable window was only 0.36–0.375 nS. Raised to 2, then to 3 to compensate the '
                      'weaker tuned input once the heterogeneity below was added.'],
                     ['Tuning heterogeneity h<sub>i</sub>, κ', 'none', 'U(0, 1), κ = 0.25',
                      'Every cell was sharply tuned, so 5 random cells decoded as well as the top 5. A fraction '
                      'h<sub>i</sub> of each E cell\'s input goes through a broad kernel 1 + κ cos Δθ. κ ≥ 0.5 made '
                      'the spontaneous state unstable.'],
                     ['Threshold SD', '0', '0.3 mV',
                      'Heterogeneity required. 1 mV (random draws) seeded spontaneous bumps at clumps of '
                      'low-threshold cells.'],
                     ['Stratified draws', '–', 'blocks of 16 cells',
                      'Random draws pinned the bump ~15° from the cue; each block of 16 neighbouring cells now holds '
                      'the same quantiles in random order.'],
                     ['Cue', '250 ms, weaker', '50 ms, 500 pA',
                      'Task cue is 50 ms; 200 pA for 50 ms did not reliably start a bump.'],
                     ['Kick A, τ<sub>kick</sub>', '0.8, 15 ms', '1.0, 150 ms',
                      'A sharp kick makes every E cell fire at once; one spike nearly saturates its NMDA gating, the '
                      'cued cells lose their advantage and global inhibition silences the ring (memory lost, A ≥ 1). '
                      'A slow kick restarts the bump; without any kick it fails after 200–250 ms dips.'],
                     ['Dip depth', '0.60', '0.50',
                      'Both silence the bump; 0.6 lost the memory more often (59–81% vs 84–94%).'],
                     ['Dip rate, mean length', '10/s, 90 ms', '7/s, 150 ms',
                      'With 90 ms dips the decoded z rarely stays below 0.84 for 50 ms (100 ms windows bridge the '
                      'silence): too few Off states. Longer, sparser dips give the data\'s Off counts and lengths; '
                      'longer still (5/s × 190 ms) costs memory (77%).'],
                     ['Response probe', '–', '300 ms at m = 1',
                      'Added to check whether a silent bump at the end of the delay can still restart at the cue.']],
                    [3.1 * cm, 2.0 * cm, 2.3 * cm, W - 7.4 * cm]),
              Spacer(1, 4),
              p('Tried and not kept: stronger external drive and larger threshold spread, to raise E baselines to '
                'PFC-like rates; neither raised them above ~1 Hz before bumps formed spontaneously.', BODY),
              figure('fig_examples_grid', '<b>Figure 1. Example trials at four cue locations (0°, 90°, 180°, '
                     '270°), without (left) and with (right) dips.</b> Each pair has the same neuron noise; only '
                     'the common input differs. Raster: E cells by preferred angle; red line: bump position '
                     '(population vector, 20 ms bins, shown when the bump is active); red dotted line: cue; grey '
                     'bar: cue; dashed line: end of the delay, followed by the response probe. Below each raster: '
                     'common input m(t) (grey) and the drive to E cells with the kick (black). Without dips the '
                     'bump holds the cued place for the whole delay. With dips it is silenced in every dip and '
                     'restarts at the same place; dips may start at cue onset, in which case the bump first '
                     'appears at the next recovery. In the 90° trial it restarts about 18° from the cue and stays '
                     'there (within the 22.5° criterion).', width=0.95 * W, head=p('Results', H2)),
              figure('fig_example_trial', '<b>Figure 2. Example trial (one trial in detail, with its decoding).</b> a, raster of the E population sorted '
                     'by preferred angle (red dotted line: cue). b, common input m(t) and the drive to E cells with '
                     'the kick. c, all-units decoder z, On (red) and Off (blue) states. Each dip silences the whole '
                     'bump and z falls to chance; the bump restarts at the cued place when the input returns.',
                     width=0.82 * W),
              figure('fig_unit_sets', '<b>Figure 3. On/Off statistics of the 7 decoded populations.</b> Black: data '
                     '(mean ± 2 SEM, 25 sessions); colour: model (12 sessions); purple crosses: the same network '
                     'and read-out without dips. Off counts and lengths are close to the data; the model has too '
                     'much On time and too little Off time for the strong decoders, and top units are worse than '
                     'random ones (reverse of the data). Without dips, the strong decoders have no Off states.'),
              p('<b>Table 1.</b> Model (data in brackets), mean over sessions; lengths in ms.', CAP), stats_table(),
              Spacer(1, 4),
              p(f'Memory held at the end of the delay: <b>{100 * mem["mem_held"]:.1f}%</b> (sessions 79.6–87.1%); '
                f'bump restarts at the cue in the response probe: {100 * mem["mem_held_probe"]:.1f}%; bump at the '
                f'wrong place: {100 * mem["wrong_place"]:.1f}%.', BODY),
              p('<b>Table 2.</b> No-dip control (6 sessions × 480 trials), same read-out. Memory 100%.', CAP),
              control_table(), Spacer(1, 8),
              figure('fig_histograms', '<b>Figure 4. State lengths, all units</b> (20 ms bins, pooled). The Off '
                     'histogram matches the data (rise at 40–60 ms, peak at 60–80 ms, long tail). On states are '
                     'about 70 ms too long and almost never shorter than 100 ms.', width=0.85 * W),
              figure('fig_populations', '<b>Figure 5. Decoder z of three populations on the same trials</b> (grey: '
                     'dips). All units and random 5 cells follow the bump; the top 5 cells of this pair prefer the '
                     'other cue and stay silent, so z ≈ 0 and the trial is Off throughout.', width=0.72 * W),
              figure('fig_tuning', '<b>Figure 6. Why top units are not better.</b> a, delay rate of the recorded E '
                     'cells against cue: narrow tuning. b, highest vs lowest cue rate: E cells are almost silent '
                     'at their non-preferred cues (below 1 Hz); interneurons are weakly modulated. c, the top 5 '
                     'cells for the 0° vs 180° pair all prefer 180°. Without dips their silence identifies the '
                     'other cue; with dips the preferred cue\'s trials are often silent too, so silence carries no '
                     'information. In PFC, cells have baselines of several Hz, so suppression stays informative.'),
              figure('fig_single_dip', f'<b>Figure 7. Memory after a single dip</b> ({sd["n_rep"]} trials per '
                     'length, dip 600 ms after the cue, no other dips). The memory survives dips up to 400 ms and '
                     'is lost by about 575 ms (50%), later than 2.5 τ<sub>N</sub> = 375 ms because the slow kick '
                     'also bridges long silences. Lost memories mostly restart at the wrong place.', width=0.55 * W),
              figure('fig_memory_off_time', '<b>Figure 8. Memory held against time Off in the delay</b> (all-units '
                     'decoder trained on all trials, 5,760 trials). a, model (black) and the monkeys\' fraction '
                     'correct (grey). b, distribution of time Off for held and lost trials. More time Off goes '
                     'with memory loss, as in the data, but far more steeply.', width=0.85 * W),
              p('Checks', H2)]
    story += bullets([
        f'Stability (no dips, 2 s before the cue): no spontaneous bump (E {stab["E_spont_Hz"]:.1f} Hz, I '
        f'{stab["I_spont_Hz"]:.1f} Hz), bump peak {stab["bump_peak_Hz"]:.0f} Hz, drift about '
        f'{stab["drift_deg_per_s_median"]:.0f}° per s, {stab["held"] * 16:.0f}/16 held.',
        f'Brian2 cross-check (same equations, explicit all-to-all synapses, reduced 512 E + 128 I ring): E rate '
        f'{br["numba"]["E_spont"]:.1f} vs {br["brian2"]["E_spont"]:.1f} Hz, I {br["numba"]["I_spont"]:.1f} vs '
        f'{br["brian2"]["I_spont"]:.1f} Hz, bump peak {br["numba"]["bump_peak"]:.0f} vs '
        f'{br["brian2"]["bump_peak"]:.0f} Hz (numba vs Brian2): agreement within about 10%.'])
    story += [p('Main differences from the rate model and the data', H2)]
    story += bullets([
        'A sharp kick (15 ms, as in the rate model) makes every E cell fire at once; one spike nearly saturates the '
        'NMDA gating, which erases the memory. The kick had to be slow (150 ms) and the dips longer and sparser '
        '(7 per s, 150 ms, depth 0.5, vs 10 per s, 90 ms, 0.6) for Off states to last ≥ 50 ms in 100 ms windows.',
        'Model cells are far more selective than PFC units and have almost no baseline; raising baselines (stronger '
        'drive, more threshold spread) made the bump form before the cue. This explains the too-strong decoders and '
        'the inverted top vs random order. A more strongly driven ring, with baselines of several Hz, would be the '
        'next step.',
        'Memory 83% (target ≥ 90%), with wrong-place restarts (~10%), which the rate model never shows; more Off '
        'time costs more memory than in the rate model.',
        'Open questions: whether the data ranked top units per cue pair or over all 8 cues; parameter values to be '
        'checked against the PDFs of Compte et al. (2000) and Wang (1999).'])
    doc = SimpleDocTemplate(os.path.join(HERE, 'report.pdf'), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.4 * cm, title='Spiking ring attractor: On/Off decoding',
                            author='')
    doc.build(story)


if __name__ == '__main__':
    main()
