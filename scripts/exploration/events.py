"""Screen for rare, large spontaneous population events followed by long suppression,
with memory loaded: per run, E population rate (5 ms bins); events = rate > mean + 4 SD;
after each event, how long until the cued pool rate (20 ms smoothing) recovers above
half its delay median; and cued-uncued difference in that window (memory loss?)."""
import numpy as np, sys, json
sys.path.insert(0, '/home/user/gamma_bursts-')
from gamma_wm.network import Params, build_network, NS, A, B, INH
from gamma_wm.task import run_trial
from gamma_wm import analysis as an
from scipy.ndimage import uniform_filter1d
for c in sys.argv[1:]:
    kw = json.loads(c); T_delay = kw.pop('T_delay', 4000.0)
    net = build_network(Params(**kw))
    sizes = np.bincount(net.group, minlength=6)
    ev_n, sup, okk, rc, ru, ns = 0, [], 0, [], [], []
    for cue, seed in (('A', 1), ('B', 2)):
        r = run_trial(net, cue, t_pre=500, cue_dur=150, t_delay=T_delay, cue_strength=0.2, seed=seed)
        pc = an.pop_rates(r, net)
        C, U = (A, B) if cue == 'A' else (B, A)
        d0 = 1150
        e5 = uniform_filter1d(pc[:, :3] @ sizes[:3] / sizes[:3].sum(), 5)[d0:]
        thr = e5.mean() + 4 * e5.std()
        rc20 = uniform_filter1d(pc[:, C], 20)[d0:]
        med = np.median(rc20)
        above = np.where((e5[1:] > thr) & (e5[:-1] <= thr))[0]
        last = -1000
        for t in above:
            if t - last < 50 or t + 400 > len(e5): continue
            last = t; ev_n += 1
            after = rc20[t + 5:t + 400]
            rec = np.where(after > 0.5 * med)[0]
            sup.append(int(rec[0]) + 5 if len(rec) else 400)
        okk += int(pc[-300:, C].mean() > pc[-300:, U].mean())
        rc.append(pc[d0:, C].mean()); ru.append(pc[d0:, U].mean()); ns.append(pc[d0:, NS].mean())
    T_s = 2 * (T_delay - 650) / 1000
    print(json.dumps(dict(kw=kw, rc=round(float(np.mean(rc)), 1), ru=round(float(np.mean(ru)), 1), ns=round(float(np.mean(ns)), 1),
                          events_per_s=round(ev_n / T_s, 2), supp_mean=round(float(np.mean(sup)), 0) if sup else 0,
                          supp_med=float(np.median(sup)) if sup else 0, frac_supp_ge50=round(float(np.mean(np.array(sup) >= 50)), 2) if sup else 0,
                          end_ok=okk)), flush=True)
