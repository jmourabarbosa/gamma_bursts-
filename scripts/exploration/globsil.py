"""Global spontaneous silences + small NMDA memory pools. Per config, 4 runs (cue A/B x2):
global silences (E rate 5 ms < 1 Hz, >= 20 ms), memory kappa (cued-uncued, 100 ms),
fraction of delay with kappa > 0, and after each silence >= 20 ms which pool fires more in
the following 50 ms (cued = memory retained)."""
import numpy as np, sys, json
sys.path.insert(0, '/home/user/gamma_bursts-')
from gamma_wm.network import Params, build_network, NS, A, B
from gamma_wm.task import run_trial
from gamma_wm import analysis as an
from scipy.ndimage import uniform_filter1d
for c in sys.argv[1:]:
    kw = json.loads(c); cs = kw.pop('cue_strength', 0.3)
    net = build_network(Params(**kw)); sizes = np.bincount(net.group, minlength=6)
    sil, after, kpos, rc, ru, rns, ok = [], [], [], [], [], [], 0
    for cue, seed in (('A', 1), ('B', 2), ('A', 3), ('B', 4)):
        r = run_trial(net, cue, t_pre=500, cue_dur=150, t_delay=4000, cue_strength=cs, seed=seed)
        pc = an.pop_rates(r, net); C, U = (A, B) if cue == 'A' else (B, A)
        d0 = 1150
        e5 = uniform_filter1d(pc[:, :3] @ sizes[:3] / sizes[:3].sum(), 5)
        m = np.r_[False, e5[d0:] < 1.0, False].astype(int)
        on, off = np.where(np.diff(m) == 1)[0] + d0, np.where(np.diff(m) == -1)[0] + d0
        for a_, b_ in zip(on, off):
            if b_ - a_ >= 20 and b_ + 50 < pc.shape[0]:
                sil.append(int(b_ - a_)); after.append((int(b_ - a_), bool(pc[b_:b_ + 50, C].sum() > pc[b_:b_ + 50, U].sum())))
        k = uniform_filter1d(pc[:, C] - pc[:, U], 100)[d0:]
        kpos.append((k > 0).mean()); ok += int(pc[-500:, C].mean() > pc[-500:, U].mean())
        rc.append(pc[d0:, C].mean()); ru.append(pc[d0:, U].mean()); rns.append(pc[d0:, NS].mean())
    a = np.array(after) if after else np.zeros((0, 2))
    def pb(lo, hi):
        mm = (a[:, 0] >= lo) & (a[:, 0] < hi) if len(a) else []
        return (round(float(a[mm, 1].mean()), 2), int(mm.sum())) if len(a) and mm.sum() else None
    print(json.dumps(dict(kw=kw, rc=round(float(np.mean(rc)), 1), ru=round(float(np.mean(ru)), 1), rns=round(float(np.mean(rns)), 1),
          sil_per_s=round(len(sil) / (4 * 3.35), 2), sil_mean=round(float(np.mean(sil)), 0) if sil else 0, sil_max=max(sil) if sil else 0,
          kpos=round(float(np.mean(kpos)), 2), end_ok=ok, after_20_50=pb(20, 50), after_50_100=pb(50, 100), after_100_200=pb(100, 200), after_200=pb(200, 10**6))), flush=True)
