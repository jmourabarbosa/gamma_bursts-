"""Score a parameter set on fresh trials: memory, fit loss for all units, and the same per-set loss for all 7 sets."""
import json, sys
import numpy as np
from paper_model import model as M, readout as R
from paper_model.fit import loss, STATS

ses, hist = R.load_data()
name = sys.argv[1]
if name == 'nodips':
    th = dict(dip_depth=0.0, w_ff=0.0)
elif name == 'current':
    th = {}
else:
    th = json.load(open(f'/home/user/fits_ww/{name}/best.json'))['theta']
out = R.evaluate(M.params(**th), ses, n_trials=2112, seed=11)
per = {}
for lab in R.SETS:
    r, d = out['sets'][lab], ses[lab]
    st = np.mean([((r[mk] - d[dk].mean()) / sc) ** 2 for dk, mk, sc in STATS])
    h = sum(50 * np.sum((np.sqrt(r['h_' + s][2:] / max(r['h_' + s][2:].sum(), 1)) -
                         np.sqrt(hist[lab, s][2:] / hist[lab, s][2:].sum())) ** 2) for s in ('on', 'off'))
    per[lab] = st + h
r = out['sets']['all']
print(name, f"mem {out['mem']:.3f} | loss(all, fresh) {loss(out, ses, hist):.2f} | 7-set total {sum(per.values()):.1f} |",
      ' '.join(f'{k} {v:.1f}' for k, v in per.items()),
      f"| all: Off {r['n_off']:.2f}/{r['off_median']:.0f}ms/{r['frac_off']:.2f} On {r['n_on']:.2f}/{r['on_median']:.0f}ms/{r['frac_on']:.2f}",
      flush=True)
