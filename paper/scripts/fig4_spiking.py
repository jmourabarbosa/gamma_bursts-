"""Figure 4: the mechanism in a spiking network.

A ring attractor of 2,048 excitatory and 512 inhibitory leaky integrate-and-fire neurons (Compte et al. 2000) with
NMDA-only recurrent excitation (tau_NMDA = 150 ms) and a common input to the excitatory cells that briefly drops.
Not fitted to the data. Simulations and read-out: branch claude/gallant-volta-bkjojq (spiking/, figures/spiking/,
fig_spiking_note.md); this script only redraws the saved panel data in the style of the other figures.

a, example trial: spikes of the excitatory cells sorted by preferred location (5 ms x 8-cell bins); the common input
   (attention signal) with its overshoot after each dip; the all-units decoder z with On/Off states.
b, Off and On length distributions, model (12 sessions x 480 trials) and data (25 sessions, all units).
c, cross-validated tuning of held-out simulated neurons in On and Off states (as Panichello et al. 2024, Fig. 4d).

usage (from the root of branch paper-model): python paper/scripts/fig4_spiking.py <out.png>
"""
import csv
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402

from paper_model.make_figures import plt, save, letter, C_ON, C_OFF, INK, GREY, LIGHT  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), 'data', 'spiking')
QUIET, XL = 1000, (-300, 2000)      # end of the dip-free delay; time axis (ms from cue onset)


def shade(ax, t, state, step):
    for lab, c in ((1, C_ON), (-1, C_OFF)):
        m = np.r_[0, (state == lab).astype(int), 0]
        for a, b in zip(np.where(np.diff(m) == 1)[0], np.where(np.diff(m) == -1)[0]):
            ax.axvspan(t[a] - step / 2, t[b - 1] + step / 2, color=c, alpha=0.15, lw=0)


def main(out):
    tr = np.load(os.path.join(DATA, 'fig_spiking_a_trial.npz'))
    hist = list(csv.DictReader(open(os.path.join(DATA, 'fig_spiking_b_histograms.csv'))))
    tun = list(csv.DictReader(open(os.path.join(DATA, 'fig_spiking_c_tuning.csv'))))

    fig = plt.figure(figsize=(7.2, 3.4))     # double column, 183 mm
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 0.35, 1.6], hspace=0.22, left=0.08, right=0.58, top=0.9, bottom=0.13)
    gb = fig.add_gridspec(1, 2, wspace=0.35, left=0.74, right=0.99, top=0.9, bottom=0.6)
    gc = fig.add_gridspec(1, 1, left=0.74, right=0.99, top=0.42, bottom=0.13)
    a, m, d = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    # a, top: spikes as counts in 5 ms x 8-cell bins, cells sorted by preferred location
    order = np.argsort(np.argsort(tr['cell_angle_deg']))
    te, ce = np.arange(-500, 2001, 5), np.arange(0, 2049, 8)
    H, _, _ = np.histogram2d(tr['spike_t_ms'], order[tr['spike_cell']], bins=(te, ce))
    a.imshow(H.T, origin='lower', aspect='auto', cmap='Greys', vmin=0, vmax=np.percentile(H[H > 0], 95),
             extent=(te[0], te[-1], 0, 360), interpolation='nearest')
    a.axvspan(0, 50, color=LIGHT, alpha=0.6, lw=0, zorder=0)
    a.set_ylim(0, 360); a.set_yticks([0, 180, 360]); a.set_ylabel('Preferred\nlocation (deg)')
    xa = a.get_xaxis_transform()
    for dx, label in ((1, 'With attention fluctuations'), (-1, 'Without attention fluctuations')):
        a.annotate('', xy=(QUIET + dx * 160, 1.06), xytext=(QUIET + dx * 15, 1.06), xycoords=xa, textcoords=xa,
                   arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.6, mutation_scale=5), annotation_clip=False)
        a.text(QUIET + dx * 180, 1.06, label, transform=xa, fontsize=7.5, va='center', ha='left' if dx > 0 else 'right')
    a.text(25, 340, 'Cue', fontsize=7, ha='center', va='top', color=GREY)
    letter(a, 'a')

    # a, middle: common input with its overshoot after each dip
    m.plot(tr['t_ms'], tr['common_input'], color=INK, lw=0.7)
    m.set_ylim(0.3, max(1.6, tr['common_input'].max() + 0.05)); m.set_yticks([])
    for s in ('left', 'bottom'):
        m.spines[s].set_visible(False)
    m.tick_params(bottom=False)
    m.text(-0.01, 0.5, 'Attention\nsignal', transform=m.transAxes, ha='right', va='center', fontsize=7.5)

    # a, bottom: all-units decoder z with On/Off states
    tz, z, st = tr['z_t_ms'], tr['z'], tr['state']
    shade(d, tz, st, 10)
    d.plot(tz, z, color=INK)
    for y, c, txt in ((0, GREY, ''), (1.96, C_ON, 'P < 0.05'), (0.84, C_OFF, 'P > 0.20')):
        d.axhline(y, color=c, ls=(0, (3, 2)), lw=0.6)
        d.text(1.005, y, txt, transform=d.get_yaxis_transform(), color=c, fontsize=7, va='bottom' if y > 1 else 'top')
    d.set_ylim(-3, 12); d.set_yticks([0, 5, 10])
    d.set_ylabel('Decoding z')
    d.set_xlabel('Time from cue onset (ms)')
    for ax in (a, m, d):
        ax.axvline(QUIET, color=GREY, ls=(0, (3, 2)), lw=0.6)
        ax.set_xlim(*XL)
    for ax in (a, m):
        ax.tick_params(labelbottom=False)

    # b, state lengths, model (bars) and data (line), 20 ms bins up to 600 ms
    for j, (name, c) in enumerate((('off', C_OFF), ('on', C_ON))):
        h = fig.add_subplot(gb[j])
        rows = [r for r in hist if r['state'] == name and float(r['bin_lo_ms']) < 600]
        lo = np.array([float(r['bin_lo_ms']) for r in rows])
        mf, df = np.array([float(r['model_fraction']) for r in rows]), np.array([float(r['data_fraction']) for r in rows])
        h.bar(lo, mf, width=20, align='edge', color=c, edgecolor='white', lw=0.3)
        h.step(np.r_[lo, 600], np.r_[df, df[-1]], where='post', color=INK, lw=0.8)
        h.set_xlim(0, 600); h.set_xticks([0, 300, 600])
        h.set_ylim(0, 0.25); h.set_yticks([0, 0.125, 0.25], ['0', '', '0.25'] if j == 0 else [])
        h.text(0.97, 0.95, f'{name.capitalize()} states', transform=h.transAxes, ha='right', va='top', color=c, fontsize=7.5)
        h.text(0.97, 0.78, 'Model', transform=h.transAxes, ha='right', va='top', color=c, fontsize=7)
        h.text(0.97, 0.64, 'Data', transform=h.transAxes, ha='right', va='top', color=INK, fontsize=7)
        if j == 0:
            h.set_ylabel('Fraction\nof states', labelpad=-14)
            letter(h, 'b', x=-0.62, y=1.0)
        h.set_xlabel('Length (ms)')

    # c, cross-validated tuning in On and Off states
    c = fig.add_subplot(gc[0])
    x = np.array([float(r['cue_rel_deg']) for r in tun])
    for name, col, lab in (('on', C_ON, 'On states'), ('off', C_OFF, 'Off states')):
        mu = np.array([float(r[name + '_mean']) for r in tun]); se = np.array([float(r[name + '_sem']) for r in tun])
        c.errorbar(x, mu, yerr=se, color=col, marker='o', ms=2.5, lw=0.9, capsize=0)
        c.text(-185, 1.3 if name == 'on' else 1.1, lab, color=col, fontsize=7.5, va='top')
    c.axhline(0, color=GREY, ls=(0, (3, 2)), lw=0.6)
    c.set_xticks([-180, -90, 0, 90, 180]); c.set_xlim(-195, 195)
    c.set_ylim(-0.8, 1.4); c.set_yticks([-0.5, 0, 0.5, 1.0])
    c.set_xlabel('Cue relative to preferred (deg)')
    c.set_ylabel('Firing rate (z)')
    letter(c, 'c', x=-0.33, y=1.0)
    save(fig, out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'figures/paper_model/fig4_spiking.png')
