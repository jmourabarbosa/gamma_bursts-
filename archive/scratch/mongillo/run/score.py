"""Score fits on fresh trials: memory, all-units loss, per-set loss for the 7 sets, all-units statistics."""
import json, sys
import numpy as np
from paper_model import model as M, model_stp as S, readout as R
from paper_model import fit_stp as F
from paper_model.fit import STATS

ses, hist = R.load_data()
for name in sys.argv[1:]:
    if name == 'current':
        R.M = M; p = M.params(t_pre=1000.0)
    else:
        R.M = S; p = S.params(t_pre=1000.0, **json.load(open(f'/home/user/fits_stp/{name}/best.json'))['theta'])
    out = R.evaluate(p, ses, n_trials=2112, seed=11)
    per = {}
    for lab in R.SETS:
        r, d = out['sets'][lab], ses[lab]
        st = np.mean([((r[mk] - d[dk].mean()) / sc) ** 2 for dk, mk, sc in STATS])
        h = sum(50 * np.sum((np.sqrt(r['h_' + s][2:] / max(r['h_' + s][2:].sum(), 1)) -
                             np.sqrt(hist[lab, s][2:] / hist[lab, s][2:].sum())) ** 2) for s in ('on', 'off'))
        per[lab] = st + h
    L = F.loss(out, ses, hist)
    r = out['sets']['all']
    print(f"{name:8s} mem {out['mem']:.3f} spont {out['spont']:.3f} | loss(all) {L:.2f} | 7-set {sum(per.values()):.1f} |",
          ' '.join(f'{k} {v:.1f}' for k, v in per.items()),
          f"| all: Off {r['n_off']:.2f}/{r['off_mean']:.0f}ms/{r['frac_off']:.2f} On {r['n_on']:.2f}/{r['on_mean']:.0f}ms/{r['frac_on']:.2f}",
          flush=True)
d = ses['all']
print(f"data all: Off {d['off_per_trial'].mean():.2f}/{d['off_mean_ms'].mean():.0f}ms/{d['frac_off'].mean():.2f} On {d['on_per_trial'].mean():.2f}/{d['on_mean_ms'].mean():.0f}ms/{d['frac_on'].mean():.2f}")
