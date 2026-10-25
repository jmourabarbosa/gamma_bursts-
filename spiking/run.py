"""Simulations of the spiking ring network (network.py), read out like the data (readout.py).

    python -m spiking.run stability             # step 1: 2 s without cue, then cue and 1.5 s delay, no dips
    python -m spiking.run sessions [--tag T]    # simulated sessions with dips, read out (spiking/results/sessions_T.npz)
    python -m spiking.run nodips                # control: the same sessions without dips
    python -m spiking.run single_dip            # memory held against the length of one dip
    python -m spiking.run example               # one example trial with its raster (for the figure)

Network parameters: network.P with the overrides in TUNED (all values and sources in PARAMETERS.md).
"""
import argparse
import json
import os
import time

import numpy as np

from . import network as N, readout as R

OUT = os.path.join(os.path.dirname(__file__), 'results')
# values changed from network.P during tuning (NOTE.md, "Tuning steps")
TUNED = dict(J_plus=2.0, g_EE=0.33, sd_vth=0.3, cue_amp=500.0, kick_A=1.0, tau_kick=150.0, dip_depth=0.5, t_pre=500.0,
             t_delay=1500.0, t_probe=300.0)
N_REC = 330           # recorded E cells per session
N_SESSIONS = 12
N_TRIALS = 480        # per session, 60 per cue
BATCH = 64


def make_net(**over):
    return N.Network(N.params(**{**TUNED, **over}))


def run_trials(net, cue_idx, seed, rec_idx, **kw):
    """Simulate trials in batches; returns concatenated arrays and the memory read-out."""
    parts = []
    for i in range(0, len(cue_idx), BATCH):
        parts.append(N.simulate(net, 45.0 * np.asarray(cue_idx[i:i + BATCH]), seed=seed * 1000 + i // BATCH,
                                rec_idx=rec_idx, **{k: (v[i:i + BATCH] if k == 'fixed_dips' else v) for k, v in kw.items()}))
    sim = {k: np.concatenate([p[k] for p in parts]) for k in ('counts', 'popv', 's_end', 'c_end', 'm', 'm_eff', 'cue_deg')}
    sim['t10'] = parts[0]['t10']
    sim['dips'] = sum([p['dips'] for p in parts], [])
    sim['mem'] = N.memory(sim, net)
    return sim


def off_time(state, tz):
    """Time Off (s) in 500-1,400 ms per trial."""
    d = (tz >= 500) & (tz <= 1400)
    return (state[:, d] == -1).sum(1) * R.STEP / 1000


def sessions(tag='dips', n_sessions=N_SESSIONS, n_trials=N_TRIALS, seed0=0, **over):
    os.makedirs(OUT, exist_ok=True)
    net = make_net(**over)
    rows, keep = [], {}
    for s in range(n_sessions):
        t0 = time.time()
        rng = np.random.default_rng(seed0 + 100 + s)
        rec = np.sort(rng.choice(net.p['N_E'], N_REC, replace=False))
        cue = rng.permutation(np.repeat(np.arange(8), n_trials // 8))
        sim = run_trials(net, cue, seed0 + s, rec)
        mem = sim['mem']
        res, zs, tz = R.read_session(sim['counts'], sim['t10'], cue, mem['held'], seed=seed0 + s, keep_z=True)
        row = dict(session=s, n_trials=n_trials, mem_held=float(mem['held'].mean()),
                   mem_held_rate=float(mem['held_rate'].mean()), mem_held_nmda=float(mem['held_nmda'].mean()),
                   wrong_place=float((mem['wrong_rate'] | (~mem['active_rate'] & mem['wrong_nmda'])).mean()),
                   sets={k: {a: (b.tolist() if isinstance(b, np.ndarray) else b) for a, b in v.items()} for k, v in res.items()})
        rows.append(row)
        # per-trial records for the behaviour analysis: all-units states on all trials need a decoder trained on
        # correct trials; error trials are decoded with the same procedure including them (see NOTE.md)
        keep[f's{s}_z_all'] = zs['all']['z'].astype(np.float32)
        keep[f's{s}_state_all'] = zs['all']['state'].astype(np.int8)
        keep[f's{s}_held'] = mem['held']
        keep[f's{s}_cue'] = cue
        if s == 0:
            keep['tz'] = tz
            keep['s0_m'] = sim['m'][:20].astype(np.float32)
            keep['s0_m_eff'] = sim['m_eff'][:20].astype(np.float32)
            keep['s0_counts20'] = sim['counts'][:20]
            keep['s0_rec'] = rec
            for lab in zs:
                keep[f's0_z_{lab}'] = zs[lab]['z'][:40].astype(np.float32)
                keep[f's0_state_{lab}'] = zs[lab]['state'][:40].astype(np.int8)
        a = res['all']
        print(f"session {s}: memory {row['mem_held']:.3f}  all: Off {a['n_off']:.2f}/tr {a['off_median']:.0f} ms "
              f"frac {a['frac_off']:.2f} | On {a['n_on']:.2f}/tr {a['on_median']:.0f} ms frac {a['frac_on']:.2f} "
              f"mu_on {a['mu_on']:.2f}  ({time.time() - t0:.0f} s)", flush=True)
        json.dump(dict(params=net.p, rows=rows), open(os.path.join(OUT, f'sessions_{tag}.json'), 'w'), indent=1,
                  default=float)
        np.savez_compressed(os.path.join(OUT, f'sessions_{tag}_trials.npz'), **keep)
    return rows


def all_trials_offtime(tag='dips', n_sessions=N_SESSIONS, n_trials=N_TRIALS, seed0=0, **over):
    """Behaviour analogue: decode all trials (held and lost) with the all-units decoder and relate time Off to
    memory held. The decoder is trained on all trials of the session (error trials included), since in the model
    lost trials are a small minority."""
    net = make_net(**over)
    out = dict(off=[], held=[])
    for s in range(n_sessions):
        rng = np.random.default_rng(seed0 + 100 + s)
        rec = np.sort(rng.choice(net.p['N_E'], N_REC, replace=False))
        cue = rng.permutation(np.repeat(np.arange(8), n_trials // 8))
        sim = run_trials(net, cue, seed0 + s, rec)
        res, zs, tz = R.read_session(sim['counts'], sim['t10'], cue, np.ones(len(cue), bool), seed=seed0 + s,
                                     sets=('all',), keep_z=True)
        out['off'].append(off_time(zs['all']['state'], tz))
        out['held'].append(sim['mem']['held'])
        print(f'session {s}: memory {sim["mem"]["held"].mean():.3f}', flush=True)
    np.savez_compressed(os.path.join(OUT, f'offtime_{tag}.npz'), off_time_s=np.concatenate(out['off']),
                        memory_held=np.concatenate(out['held']))


def single_dip(lengths=tuple(range(0, 701, 50)), n_rep=32, onset=600.0, seed=7, **over):
    """One dip of fixed length per trial, starting `onset` ms after cue onset (no other dips); memory at the end of
    the delay (rate and NMDA criteria). t_delay is extended so that the read-out window starts >= 200 ms after the
    longest dip."""
    os.makedirs(OUT, exist_ok=True)
    net = make_net(t_delay=max(1500.0, onset + max(lengths) + 300.0), **over)
    L = np.repeat(lengths, n_rep).astype(float)
    cue = np.tile(np.arange(8), len(L) // 8 + 1)[:len(L)]
    fixed = [(net.p['t_pre'] + onset, l) for l in L]
    sim = run_trials(net, cue, seed, np.arange(0), fixed_dips=fixed)
    mem = sim['mem']
    res = dict(lengths=list(map(float, lengths)), n_rep=n_rep, onset=onset, t_delay=net.p['t_delay'],
               held=[float(mem['held'][L == l].mean()) for l in lengths],
               held_rate=[float(mem['held_rate'][L == l].mean()) for l in lengths],
               held_nmda=[float(mem['held_nmda'][L == l].mean()) for l in lengths],
               wrong=[float(mem['wrong_rate'][L == l].mean()) for l in lengths],
               kick_A=net.p['kick_A'], tau_kick=net.p['tau_kick'])
    json.dump(res, open(os.path.join(OUT, f'single_dip{"_" + "_".join(f"{k}{v}" for k, v in over.items()) if over else ""}.json'), 'w'), indent=1)
    for l, h, hr, hn, w in zip(lengths, res['held'], res['held_rate'], res['held_nmda'], res['wrong']):
        print(f'dip {l:4.0f} ms: held {h:.2f} (rate {hr:.2f}, NMDA {hn:.2f}), wrong place {w:.2f}')
    return res


def stability(n=16, seed=3):
    """Step 1: no dips; 2 s before the cue (spontaneous state), cue, 1.5 s delay."""
    net = make_net(t_pre=2000.0, dip_rate=0.0)
    cue = np.arange(n) % 8
    sim = run_trials(net, cue, seed, np.arange(net.p['N_E'] + net.p['N_I']))
    c, t = sim['counts'].astype(float), sim['t10']
    pre, dl = (t >= -1800) & (t < 0), (t >= 500) & (t < 1500)
    rE_pre, rI_pre = c[:, pre, :2048].mean(1) * 100, c[:, pre, 2048:].mean(1) * 100
    rE_d, rI_d = c[:, dl, :2048].mean(1) * 100, c[:, dl, 2048:].mean(1) * 100
    sm = N._smooth(rE_pre, 28)
    pv = sim['popv'].reshape(n, -1, 10, 6).sum(2)
    ang = np.rad2deg(np.arctan2(pv[..., 1], pv[..., 0]))
    t100 = t[::10]
    err = N.angdiff(ang, sim['cue_deg'][:, None])
    drift = np.abs(err[:, (t100 >= 1400) & (t100 < 1500)][:, 0] - err[:, (t100 >= 400) & (t100 < 500)][:, 0])
    res = dict(E_spont_Hz=float(rE_pre.mean()), I_spont_Hz=float(rI_pre.mean()),
               max_smoothed_E_pre_Hz=float(sm.max(1).mean()), E_delay_mean_Hz=float(rE_d.mean()),
               I_delay_Hz=float(rI_d.mean()), bump_peak_Hz=float(N._smooth(rE_d, 28).max(1).mean()),
               held=float(sim['mem']['held'].mean()), drift_deg_per_s_median=float(np.median(drift)),
               drift_deg_per_s_max=float(drift.max()), end_error_deg=sim['mem']['err_rate'].round(1).tolist())
    json.dump(res, open(os.path.join(OUT, 'stability.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


def example(seed=11, n=24):
    """Candidate example trials with rasters (with dips), saved for the figure script."""
    net = make_net()
    rng = np.random.default_rng(seed)
    rec = np.sort(rng.choice(net.p['N_E'], N_REC, replace=False))
    cue = np.arange(n) % 8
    sim = N.simulate(net, 45.0 * cue, seed=seed, rec_idx=rec, raster=True)
    np.savez_compressed(os.path.join(OUT, 'example_trials.npz'), raster=sim['raster'], m=sim['m'], m_eff=sim['m_eff'],
                        counts=sim['counts'], rec=rec, cue=cue, t1=sim['t1'], t10=sim['t10'],
                        held=N.memory(sim, net)['held'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--n_sessions', type=int, default=N_SESSIONS)
    ap.add_argument('--n_trials', type=int, default=N_TRIALS)
    ap.add_argument('--set', nargs='*', default=[], help='parameter overrides k=v')
    a = ap.parse_args()
    over = {k: float(v) for k, v in (s.split('=') for s in a.set)}
    os.makedirs(OUT, exist_ok=True)
    if a.what == 'sessions':
        sessions(a.tag or 'dips', a.n_sessions, a.n_trials, **over)
    elif a.what == 'nodips':
        sessions(a.tag or 'nodips', a.n_sessions, a.n_trials, dip_rate=0.0, **over)
    elif a.what == 'offtime':
        all_trials_offtime(a.tag or 'dips', a.n_sessions, a.n_trials, seed0=1000, **over)
    elif a.what == 'single_dip':
        single_dip(**over)
    elif a.what == 'stability':
        stability()
    elif a.what == 'example':
        example()
