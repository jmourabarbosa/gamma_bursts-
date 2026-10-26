"""Re-simulate the long session of fig_paper (same seeds) and decode the example trial with all units and with the
20 most cue-selective neurons (ranked per cue pair by split-half ANOVA, as in Fig. 4b)."""
import os, numpy as np
import spiking.fig_paper as FP
from spiking import readout as R
from paper_model import readout as PR
D = os.path.dirname(os.path.abspath(__file__))
FP.RES = D
d = FP.long_session()
z_all, st_all, tz, idx = FP.decode_long(d)
held = d['held']; cue = d['cue'][held]
k = R.WIN // R.STEP
c = np.cumsum(np.concatenate([np.zeros_like(d['counts'][held][:, :1], dtype=np.int32), d['counts'][held].astype(np.int32)], 1), 1)
X = (c[:, k:] - c[:, :-k]).astype(float); t = d['t10'][:X.shape[1]] + R.WIN / 2
X = X[:, t >= -300]
rng = np.random.default_rng(1)
valid, rankwin = tz >= 0, (tz >= 0) & (tz <= 1400)
half = np.zeros(len(cue), int)
for cc in range(8):
    i = rng.permutation(np.where(cue == cc)[0]); half[i[len(i) // 2:]] = 1
tot = X[:, rankwin].sum(1)
active = set(np.where(X.sum((0, 1)) > 0)[0])
rank = [[np.array([u for u in np.argsort(-np.nan_to_num(R.anova_f(tot[(half == h) & (cue % 4 == p)], cue[(half == h) & (cue % 4 == p)]), nan=0)) if u in active])[:20]
         for p in range(4)] for h in (0, 1)]
z = np.full(z_all.shape, np.nan); zn_all = []
for h in (0, 1):
    zh, zn = R.decode(X, cue, rng, rank[1 - h]); z[half == h] = zh[half == h]; zn_all.append(zn[:, half == h])
thr = R.null_threshold(np.concatenate(zn_all, 1), valid)
st = PR.label(np.nan_to_num(z), valid, thr)
r = int(np.where(idx == 27)[0][0])
saved = np.load('/tmp/claude-0/-home-user/51e93236-a4df-5b7b-99f9-872ebc6e17b8/scratchpad/wt_pm/paper/scripts/data/spiking/fig_spiking_a_trial.npz')
print('check all-units z vs saved: max abs diff', np.nanmax(np.abs(z_all[r][:len(saved['z'])] - saved['z'])))
np.savez_compressed(os.path.join(D, 'fig_spiking_a_top20.npz'), z_t_ms=tz, z_top20=np.nan_to_num(z[r]), state_top20=st[r],
                    z_all=np.nan_to_num(z_all[r]), state_all=st_all[r])
print('done', r, 'top20 states On/Off in trial:', (st[r] == 1).mean(), (st[r] == -1).mean())
