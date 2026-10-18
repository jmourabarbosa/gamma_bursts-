"""Screen: memory strength kappa(t) = cued - uncued pool rate (100 ms boxcar) during a 4 s
delay. Reports how often kappa dips to near zero (Off-like: below 25% of its delay
median for >= 50 ms), dip durations, sign flips, and end-of-delay correctness."""
import numpy as np, sys, json
sys.path.insert(0, '/home/user/gamma_bursts-')
from gamma_wm.network import Params, build_network, NS, A, B, INH
from gamma_wm.task import run_trial
from gamma_wm import analysis as an

for c in sys.argv[1:]:
    kw = json.loads(c); T_delay = kw.pop('T_delay', 4000.0); cs = kw.pop('cue_strength', 0.2)
    net = build_network(Params(**kw))
    out = dict(dips=[], flips=0, ok=0, kmed=[], rc=[], ru=[], rns=[], frac_dip=[])
    for cue, seed in (('A', 1), ('B', 2), ('A', 3), ('B', 4)):
        r = run_trial(net, cue, t_pre=500, cue_dur=150, t_delay=T_delay, cue_strength=cs, seed=seed)
        pc = an.pop_rates(r, net)
        C, U = (A, B) if cue == 'A' else (B, A)
        k = np.convolve(pc[:, C] - pc[:, U], np.ones(100) / 100, mode='same')
        d = slice(650 + 500, pc.shape[0] - 50)
        kd = k[d]
        med = np.median(kd)
        thr = 0.25 * max(med, 1e-6)
        m = np.r_[False, kd < thr, False].astype(int)
        on, off = np.where(np.diff(m) == 1)[0], np.where(np.diff(m) == -1)[0]
        du = [b - a for a, b in zip(on, off) if b - a >= 50]
        out['dips'] += du
        out['frac_dip'].append(sum(du) / len(kd))
        f = np.r_[False, kd < -thr, False].astype(int)
        out['flips'] += sum(1 for a, b in zip(np.where(np.diff(f) == 1)[0], np.where(np.diff(f) == -1)[0]) if b - a >= 200)
        out['ok'] += int(kd[-300:].mean() > 0)
        out['kmed'].append(med); out['rc'].append(pc[d, C].mean()); out['ru'].append(pc[d, U].mean()); out['rns'].append(pc[d, NS].mean())
    du = np.array(out['dips'])
    T_s = 4 * (T_delay - 550) / 1000
    print(json.dumps(dict(kw=kw, rc=round(float(np.mean(out['rc'])), 1), ru=round(float(np.mean(out['ru'])), 1), rns=round(float(np.mean(out['rns'])), 1),
                          kmed=round(float(np.mean(out['kmed'])), 1), dips_per_s=round(len(du) / T_s, 2),
                          dip_mean=round(float(du.mean()), 0) if len(du) else 0, dip_med=float(np.median(du)) if len(du) else 0,
                          frac_dip=round(float(np.mean(out['frac_dip'])), 2), flips=out['flips'], end_ok=out['ok'])), flush=True)
