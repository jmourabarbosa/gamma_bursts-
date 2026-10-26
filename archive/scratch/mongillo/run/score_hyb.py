"""Score fits of model_ours_stp (or our model) on fresh trials, all 7 unit sets."""
import json, sys
import numpy as np
from paper_model import model as M, model_ours_stp as H, readout as R
from paper_model.fit_stp import loss
from paper_model.fit import STATS

ses, hist = R.load_data()
name = sys.argv[1]
if name == 'current':
    R.M = M; p = M.params(t_pre=1000.0)
else:
    d = json.load(open(f'/home/user/{name}/best.json'))['theta']
    R.M = H; p = H.params(t_pre=1000.0, depression=0.0 if 'fac' in name else 1.0, **d)
out = R.evaluate(p, ses, n_trials=2112, seed=11)
per = {}
for lab in R.SETS:
    r, dd = out['sets'][lab], ses[lab]
    st = np.mean([((r[mk] - dd[dk].mean()) / sc) ** 2 for dk, mk, sc in STATS])
    h = sum(50 * np.sum((np.sqrt(r['h_' + s][2:] / max(r['h_' + s][2:].sum(), 1)) -
                         np.sqrt(hist[lab, s][2:] / hist[lab, s][2:].sum())) ** 2) for s in ('on', 'off'))
    per[lab] = st + h
r = out['sets']['all']
print(f"{name:18s} mem {out['mem']:.3f} | loss(all) {loss(out, ses, hist):.2f} | 7-set {sum(per.values()):.1f} |",
      ' '.join(f'{k} {v:.1f}' for k, v in per.items()),
      f"| all: Off {r['n_off']:.2f}/{r['off_mean']:.0f}ms/{r['frac_off']:.2f} On {r['n_on']:.2f}/{r['on_mean']:.0f}ms/{r['frac_on']:.2f}", flush=True)
