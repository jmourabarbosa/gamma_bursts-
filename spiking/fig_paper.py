"""Paper figure of the spiking ring network (hand-off to the paper agent: figures/spiking/fig_spiking_note.md).

a  One trial: 500 ms of fixation, a 50 ms cue at 180 deg, then 1 s of delay with a constant common input (stable bump)
   and 1 s with dips. Raster of the E population sorted by preferred angle, the common input to the E cells, and the
   all-units decoder z of that trial with its On / Off states. The decoder is trained, as for the data, on a session of
   480 such trials (60 per cue, correct trials, L2 logistic, lambda = 10,000, leave-one-out, 30 label shuffles).
b  On and Off state lengths (all units, 12 sessions x 480 trials of the main simulation), with the data.
c  Cross-validated memory tuning functions in On and Off states (as Panichello et al. 2024, Fig. 4d): states labelled
   with the decoder of one half of the recorded cells; the held-out half's firing rates (100 ms windows, z-scored
   across trials at each time point), averaged per cue over delay time points (500-1,400 ms) in On or in Off states;
   aligned to each cell's preferred cue (largest On + Off value); minus the same computed with shuffled cue labels;
   halves swapped. Session 0 of the main simulation.

    python -m spiking.fig_paper            # ~15 min (simulation of the long-trial session) on 4 cores
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from paper_model import readout as PR  # noqa: E402
from . import network as N, readout as R, run as RU  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures', 'spiking')
RES = os.path.join(RU.OUT, 'fig_paper')
QUIET, DIPS = 1000.0, 1000.0           # delay without dips, then with dips (ms after cue onset)
CUE_EX = 4                             # example cue: 180 deg, the middle of the raster
C_ON, C_OFF, INK, GREY, LIGHT = '#e8474a', '#3cb4c9', '#000000', '#7f7f7f', '#d9d9d9'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'],
                     'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK,
                     'lines.linewidth': 0.9, 'legend.frameon': False, 'legend.fontsize': 7.5,
                     'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic',
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=600)
    fig.savefig(os.path.join(OUT, name + '.svg'))
    plt.close(fig)


def letter(ax, txt, x=-0.08, y=1.04):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=10, weight='bold', va='bottom')


# ---------------------------------------------------------------------------------------------------------- panel a
def long_session(n_trials=480, seed=50):
    """Session of long trials (delay QUIET + DIPS, dips only after QUIET); 10 ms counts of the recorded cells."""
    f = os.path.join(RES, 'long_session.npz')
    if os.path.exists(f):
        return dict(np.load(f))
    os.makedirs(RES, exist_ok=True)
    net = RU.make_net(t_delay=QUIET + DIPS, t_probe=0.0)
    rng = np.random.default_rng(seed)
    rec = RU.record(net, rng)
    cue = rng.permutation(np.repeat(np.arange(8), n_trials // 8))
    counts, m_eff, held = [], [], []
    for i in range(0, n_trials, RU.BATCH):
        sim = N.simulate(net, 45.0 * cue[i:i + RU.BATCH], seed=seed * 1000 + i // RU.BATCH, rec_idx=rec, dip_free=QUIET)
        counts.append(sim['counts']); m_eff.append(sim['m_eff'].astype(np.float32))
        held.append(N.memory(sim, net)['held'])
        print(f'long session: {i + RU.BATCH} trials', flush=True)
    out = dict(counts=np.concatenate(counts), m_eff=np.concatenate(m_eff), held=np.concatenate(held), cue=cue,
               rec=rec, t10=sim['t10'], t1=sim['t1'], seed=seed)
    np.savez_compressed(f, **out)
    return out


def decode_long(d, seed=0):
    """All-units decoder z and On/Off states (0 to the end of the delay) of the correct trials of the long session."""
    held, cue = d['held'], d['cue'][d['held']]
    k = R.WIN // R.STEP
    c = np.cumsum(np.concatenate([np.zeros_like(d['counts'][held][:, :1], dtype=np.int32),
                                  d['counts'][held].astype(np.int32)], 1), 1)
    X = (c[:, k:] - c[:, :-k]).astype(float)
    tz = d['t10'][:X.shape[1]] + R.WIN / 2
    keep = tz >= -300
    X, tz = X[:, keep], tz[keep]
    rng = np.random.default_rng(seed)
    active = np.where(X.sum((0, 1)) > 0)[0]
    z, zn = R.decode(X, cue, rng, active)
    valid = tz >= 0
    thr = R.null_threshold(zn, valid)
    state = PR.label(np.nan_to_num(z), valid, thr)
    return z, state, tz, np.where(held)[0]


def pick_example(d, z, state, tz, idx):
    """Trial with cue CUE_EX, memory held, On through the whole stable second, 2-5 Off states with dips."""
    quiet = (tz >= 100) & (tz <= QUIET)
    dips = tz > QUIET + 50
    best = None
    for r, i in enumerate(idx):
        if d['cue'][i] != CUE_EX:
            continue
        n_off = sum(1 for a, b in PR.runs(state[r] == -1) if tz[a] > QUIET)
        n_on = sum(1 for a, b in PR.runs(state[r] == 1) if tz[a] > QUIET)
        ok = (state[r, quiet] == 1).mean() > 0.95 and 2 <= n_off <= 5 and n_on >= 2
        if ok and (best is None or (state[best[0], dips] == -1).mean() < (state[r, dips] == -1).mean()):
            best = (r, i)
    return best


def example_raster(d, i):
    """Re-simulate the batch of trial i with rasters (same seeds, so the same trial); E spike times and cells."""
    f = os.path.join(RES, f'raster_trial{i}.npz')
    if os.path.exists(f):
        return dict(np.load(f))
    net = RU.make_net(t_delay=QUIET + DIPS, t_probe=0.0)
    b0 = (i // RU.BATCH) * RU.BATCH
    sim = N.simulate(net, 45.0 * d['cue'][b0:b0 + RU.BATCH], seed=int(d['seed']) * 1000 + i // RU.BATCH,
                     rec_idx=d['rec'], raster=True, dip_free=QUIET)
    assert np.array_equal(sim['counts'], d['counts'][b0:b0 + RU.BATCH])
    t, c = np.nonzero(sim['raster'][i - b0])
    out = dict(t=sim['t1'][t].astype(np.float32), c=c.astype(np.int16), m=sim['m'][i - b0], m_eff=sim['m_eff'][i - b0],
               t1=sim['t1'])
    np.savez_compressed(f, **out)
    return out


# ---------------------------------------------------------------------------------------------------------- panel c
def tuning_on_off(s=0, n_rep=10, seed=0):
    """Cross-validated On and Off tuning functions of held-out cells (session s of the main simulation)."""
    d = np.load(os.path.join(RU.OUT, 'sessions_dips_trials.npz'))
    held, cue = d[f's{s}_held'], d[f's{s}_cue']
    counts = d[f's{s}_counts']
    X, tz = R.windows(counts[held], np.arange(counts.shape[1]) * 10.0 - RU.TUNED['t_pre'])
    cue = cue[held]
    zr = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)          # z-scored across trials, per time and cell
    valid, delay = (tz >= 0) & (tz <= 1400), (tz >= 500) & (tz <= 1400)
    rng = np.random.default_rng(seed)
    curves = {'on': [], 'off': []}
    ids = []
    n_cells = X.shape[2]
    for rep in range(n_rep):
        perm = rng.permutation(n_cells)
        halves = (perm[:n_cells // 2], perm[n_cells // 2:])
        for h in (0, 1):
            lab, test = halves[h], halves[1 - h]
            z, zn = R.decode(X, cue, rng, lab)
            st = PR.label(np.nan_to_num(z), valid, R.null_threshold(zn, valid))[:, delay]
            zt = zr[:, delay][:, :, test]
            for shuffled in (False, True):
                cu = rng.permutation(cue) if shuffled else cue
                tun = {}
                for nm, v in (('on', 1), ('off', -1)):
                    w = (st == v).astype(float)                          # (trials, time)
                    tun[nm] = np.array([(zt[cu == c] * w[cu == c][:, :, None]).sum((0, 1)) /
                                        np.maximum(w[cu == c].sum(), 1) for c in range(8)])   # (8, cells)
                pref = np.argmax(tun['on'] + tun['off'], 0)
                al = {nm: np.array([np.roll(tun[nm][:, j], -pref[j]) for j in range(len(test))]) for nm in tun}
                if not shuffled:
                    true = al
                else:
                    for nm in curves:
                        curves[nm].append(true[nm] - al[nm])
                    ids.append(test)
    ids = np.concatenate(ids)
    # one curve per cell (mean over the random splits in which it was held out), preferred cue at index 0
    out = {nm: np.array([np.concatenate(v)[ids == j].mean(0) for j in range(n_cells)]) for nm, v in curves.items()}
    np.savez_compressed(os.path.join(RES, 'tuning_on_off.npz'), **out)
    return out


# ---------------------------------------------------------------------------------------------------------- figure
def figure():
    os.makedirs(RES, exist_ok=True)
    d = long_session()
    z, state, tz, idx = decode_long(d)
    r, i = pick_example(d, z, state, tz, idx)
    ras = example_raster(d, i)
    tun = tuning_on_off()
    main = json.load(open(os.path.join(RU.OUT, 'sessions_dips.json')))['rows']
    h_model = {st: np.sum([row['sets']['all'][f'h_{st}'] for row in main], 0).astype(float) for st in ('on', 'off')}
    _, h_data = PR.load_data()

    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(3, 3, width_ratios=[3.3, 1.0, 1.0], height_ratios=[2.4, 0.55, 1.6], wspace=0.5,
                          hspace=0.14, left=0.085, right=0.985, top=0.93, bottom=0.1)
    t_end = QUIET + DIPS
    xl = (-500, t_end)
    # a: raster as a spike-count image (5 ms x 8 cells)
    a = fig.add_subplot(gs[0, 0])
    tb = np.arange(xl[0], t_end + 5, 5)
    cb = np.arange(0, N.P['N_E'] + 8, 8)
    H, _, _ = np.histogram2d(ras['t'], ras['c'].astype(float), bins=(tb, cb))
    a.imshow(H.T, origin='lower', aspect='auto', cmap='Greys', vmin=0, vmax=np.percentile(H[H > 0], 98),
             extent=(tb[0], tb[-1], 0, 360), interpolation='nearest', rasterized=True)
    a.axvspan(0, 50, color=C_ON, alpha=0.12, lw=0)
    a.set_xlim(*xl); a.set_ylim(0, 360); a.set_yticks([0, 90, 180, 270, 360])
    a.set_ylabel('E cells, preferred\nlocation (deg)')
    a.tick_params(labelbottom=False)
    a.axvline(QUIET, color=GREY, lw=0.6, ls=(0, (3, 2)))
    for x_, txt, ha in ((-250, 'Fixation', 'center'), (60, 'Cue', 'left'), (QUIET - 30, 'Constant input', 'right'),
                        (QUIET + 30, 'With dips', 'left')):
        a.text(x_, 1.02, txt, transform=a.get_xaxis_transform(), ha=ha, va='bottom', fontsize=7.5)
    letter(a, 'a', x=-0.12)
    # a: common input
    a2 = fig.add_subplot(gs[1, 0], sharex=a)
    a2.plot(ras['t1'], ras['m_eff'], color=INK, lw=0.7)
    a2.set_ylim(0.35, 1.65); a2.set_yticks([0.5, 1, 1.5])
    a2.set_ylabel('Common\ninput', fontsize=7.5)
    a2.tick_params(labelbottom=False)
    a2.axvline(QUIET, color=GREY, lw=0.6, ls=(0, (3, 2)))
    # a: decoding z of this trial
    a3 = fig.add_subplot(gs[2, 0], sharex=a)
    zt, st = z[r], state[r]
    a3.fill_between(tz, -3, 20, where=st == 1, color=C_ON, alpha=0.22, lw=0, step='mid')
    a3.fill_between(tz, -3, 20, where=st == -1, color=C_OFF, alpha=0.3, lw=0, step='mid')
    a3.plot(tz, zt, color=INK, lw=0.9)
    a3.axhline(1.96, color=C_ON, lw=0.7, ls=(0, (3, 2)))
    a3.axhline(0.84, color=C_OFF, lw=0.7, ls=(0, (3, 2)))
    a3.axvline(QUIET, color=GREY, lw=0.6, ls=(0, (3, 2)))
    a3.text(-490, 1.96, 'P < 0.05', color=C_ON, fontsize=7, va='bottom', ha='left')
    a3.text(-490, 0.84, 'P > 0.20', color=C_OFF, fontsize=7, va='top', ha='left')
    a3.set_ylim(-3, max(8, np.ceil(np.nanmax(zt[tz <= t_end]) + 1)))
    a3.set_xlim(*xl)
    a3.set_xlabel('Time from cue onset (ms)')
    a3.set_ylabel('Decoding z\n(all units)')
    # b: state lengths, model (bars) and data (line)
    e = np.arange(31) * 20
    for k, (stn, col, nm) in enumerate((('off', C_OFF, 'Off'), ('on', C_ON, 'On'))):
        b = fig.add_subplot(gs[0, 1 + k])
        hm, hd = h_model[stn][:30], h_data['all', stn][:30]
        b.bar(e[:30] + 10, hm / h_model[stn].sum(), width=20, color=col, lw=0, label='Model')
        b.step(e[:30] + 10, hd / h_data['all', stn].sum(), where='mid', color=INK, lw=0.8, label='Data')
        b.set_xlim(0, 600); b.set_xticks([0, 200, 400, 600], ['0', '', '400', ''])
        b.set_ylim(0, 0.22); b.set_yticks([0, 0.1, 0.2])
        b.set_xlabel(f'{nm} length (ms)')
        b.legend(loc='upper right', fontsize=7, handlelength=1.0, borderaxespad=0.2)
        if k == 0:
            b.set_ylabel('Fraction of states')
            letter(b, 'b', x=-0.55)
        else:
            b.set_yticklabels([])
    # c: tuning functions
    c = fig.add_subplot(gs[2, 1:3])
    x = np.arange(-4, 5) * 45
    for nm, col, lab in (('on', C_ON, 'On states'), ('off', C_OFF, 'Off states')):
        v = tun[nm]
        v = np.c_[v[:, 4:], v[:, :5]]                                    # -180 .. 180, preferred cue at 0
        mu, se = v.mean(0), v.std(0) / np.sqrt(len(v))
        c.errorbar(x, mu, yerr=se, color=col, marker='o', ms=2.5, lw=0.9, elinewidth=0.6, capsize=0, label=lab)
    c.axhline(0, color=GREY, lw=0.5, ls=(0, (3, 2)))
    c.set_xticks([-180, -90, 0, 90, 180]); c.set_xlim(-195, 195)
    c.set_xlabel('Cue location relative to preferred (deg)')
    c.set_ylabel('Cross-validated\nfiring rate (z)')
    c.legend(loc='upper right', handlelength=1.2, fontsize=7)
    letter(c, 'c', x=-0.24)
    save(fig, 'fig_spiking')
    # plotted values, for re-plotting in the paper's style
    with open(os.path.join(OUT, 'fig_spiking_b_histograms.csv'), 'w') as fh:
        fh.write('state,bin_lo_ms,bin_hi_ms,model_fraction,data_fraction,model_count,data_count\n')
        for stn in ('off', 'on'):
            for k in range(31):
                fh.write(f"{stn},{20 * k},{20 * k + 20 if k < 30 else ''},{h_model[stn][k] / h_model[stn].sum():.6f},"
                         f"{h_data['all', stn][k] / h_data['all', stn].sum():.6f},{int(h_model[stn][k])},"
                         f"{int(h_data['all', stn][k])}\n")
    with open(os.path.join(OUT, 'fig_spiking_c_tuning.csv'), 'w') as fh:
        fh.write('cue_rel_deg,on_mean,on_sem,off_mean,off_sem,n_cells\n')
        for k, xd in enumerate(np.arange(-4, 5) * 45):
            col = (k + 4) % 8
            fh.write(f"{xd}," + ','.join(f'{tun[nm][:, col].mean():.5f},{tun[nm][:, col].std() / np.sqrt(len(tun[nm])):.5f}'
                                            for nm in ('on', 'off')) + f',{len(tun["on"])}\n')
    np.savez_compressed(os.path.join(OUT, 'fig_spiking_a_trial.npz'), spike_t_ms=ras['t'], spike_cell=ras['c'],
                        cell_angle_deg=np.arange(N.P['N_E']) * 360 / N.P['N_E'], t_ms=ras['t1'], common_input=ras['m_eff'],
                        dips_m=ras['m'], z_t_ms=tz, z=z[r], state=state[r])
    meta = dict(example_trial=int(i), example_cue_deg=float(d['cue'][i] * 45), n_trials_long_session=int(len(d['cue'])),
                n_correct_long_session=int(d['held'].sum()),
                n_states_model=dict(on=int(h_model['on'].sum()), off=int(h_model['off'].sum())),
                n_tuning_curves=int(len(tun['on'])),
                tuning_peak_on=float(tun['on'][:, 0].mean()), tuning_peak_off=float(tun['off'][:, 0].mean()),
                tuning_opposite_on=float(tun['on'][:, 4].mean()), tuning_opposite_off=float(tun['off'][:, 4].mean()))
    json.dump(meta, open(os.path.join(OUT, 'fig_spiking_meta.json'), 'w'), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    figure()
