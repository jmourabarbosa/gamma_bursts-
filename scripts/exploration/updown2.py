"""Up/Down switching of the memory pools with hysteresis labelling (50 ms smoothing,
Up > 10 Hz, Down < 4 Hz). After a cue to A, free run; record each pool's Up episodes,
global silences (no pool Up) and which pool comes back after each silence."""
import numpy as np, sys, json
sys.path.insert(0, '/home/user/gamma_bursts-')
from gamma_wm.network import Params, build_network, NS, A, B, INH
from gamma_wm.task import run_trial
from gamma_wm import analysis as an
from scipy.ndimage import uniform_filter1d

def hyst(x, hi=10.0, lo=4.0):
    s = np.zeros(len(x), bool); up = False
    for k, v in enumerate(x):
        if up and v < lo: up = False
        elif not up and v > hi: up = True
        s[k] = up
    return s

for c in sys.argv[1:]:
    kw = json.loads(c); T_free = kw.pop('T_free', 15000.0)
    net = build_network(Params(**kw))
    agg = dict(bridges=[], sil=[], upc=[], upu=[], n_ign_c=0, n_ign_u=0, frac_c=[], frac_u=[], uprate=[])
    for cue, seed in (('A', 1), ('B', 2)):
        r = run_trial(net, cue, t_pre=300, cue_dur=150, t_delay=T_free, cue_strength=0.3, seed=seed)
        pc = an.pop_rates(r, net); t0 = 450
        C, U = (A, B) if cue == 'A' else (B, A)
        uc = hyst(uniform_filter1d(pc[:, C], 50)[t0:]); uu = hyst(uniform_filter1d(pc[:, U], 50)[t0:])
        lab = np.where(uc & ~uu, 1, np.where(uu & ~uc, 2, np.where(uc & uu, 3, 0)))
        ch = np.r_[0, np.where(np.diff(lab) != 0)[0] + 1, len(lab)]
        ep = [(int(lab[s_]), int(e - s_)) for s_, e in zip(ch[:-1], ch[1:])]
        agg['sil'] += [n for l, n in ep if l == 0]
        agg['upc'] += [n for l, n in ep if l == 1]; agg['upu'] += [n for l, n in ep if l == 2]
        starts = np.r_[0, np.cumsum([n for _, n in ep])]
        for k in range(1, len(ep) - 1):
            if ep[k][0] == 0 and ep[k - 1][0] in (1, 2) and ep[k + 1][0] in (1, 2):
                sl = slice(t0 + starts[k], t0 + starts[k + 1])
                prevpool = C if ep[k - 1][0] == 1 else U
                other = U if prevpool == C else C
                agg['bridges'].append((ep[k][1], ep[k - 1][0] == ep[k + 1][0]))
                agg.setdefault('resid', []).append(float(pc[sl, prevpool].mean() - pc[sl, other].mean()))
                agg.setdefault('offrate', []).append(float(pc[sl, prevpool].mean()))
        agg['n_ign_c'] += sum(1 for k in range(1, len(ep)) if ep[k][0] == 1 and ep[k - 1][0] == 0)
        agg['n_ign_u'] += sum(1 for k in range(1, len(ep)) if ep[k][0] == 2 and ep[k - 1][0] == 0)
        agg['frac_c'].append(uc.mean()); agg['frac_u'].append(uu.mean())
        if uc.any(): agg['uprate'].append(pc[t0:, C][uc].mean())
    b = agg['bridges']
    print(json.dumps(dict(kw=kw, T=T_free, frac_c=round(float(np.mean(agg['frac_c'])), 2), frac_u=round(float(np.mean(agg['frac_u'])), 2),
          up_rate=round(float(np.mean(agg['uprate'])), 1) if agg['uprate'] else 0,
          n_up_c=len(agg['upc']), n_up_u=len(agg['upu']), up_mean=round(float(np.mean(agg['upc'] + agg['upu'])), 0) if agg['upc'] + agg['upu'] else 0,
          n_ign_c=agg['n_ign_c'], n_ign_u=agg['n_ign_u'],
          n_sil=len(agg['sil']), sil_mean=round(float(np.mean(agg['sil'])), 0) if agg['sil'] else 0, sil_med=float(np.median(agg['sil'])) if agg['sil'] else 0,
          sil_max=max(agg['sil']) if agg['sil'] else 0,
          resid=round(float(np.mean(agg.get('resid', [np.nan]))), 2), offrate=round(float(np.mean(agg.get('offrate', [np.nan]))), 2),
          n_bridge=len(b), p_same=round(float(np.mean([x for _, x in b])), 2) if b else None, bridges=b)), flush=True)
