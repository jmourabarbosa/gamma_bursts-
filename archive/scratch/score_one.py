"""Score a parameter set on all 7 unit sets with fresh trials: per-set loss (stats + histograms) and memory."""
import json, sys
import numpy as np
from paper_model import readout as R, model as M
from paper_model.fit import loss
name = sys.argv[1]
th = M.FIT if name == 'current' else json.load(open(f'/home/user/fits_one/{name}/best.json'))['theta']
summ, hist = R.load_data()
rows = {}
for seed in (11, 12):
    out = R.evaluate(th, n_trials=2112, seed=seed, summ=summ)
    for st, lab in zip(R.SETS, R.LABELS):
        L, t = loss(out, summ, hist, (st,))
        rows.setdefault(lab, []).append(L - t['mem'])
    rows.setdefault('mem', []).append(out['mem'])
res = {k: float(np.mean(v)) for k, v in rows.items()}
json.dump(dict(theta=th, scores=res), open(f'/tmp/claude-0/-home-user/55b58474-28bf-5b68-b09e-8186611c5a8b/scratchpad/score_{name}.json', 'w'), indent=1)
print(name, ' '.join(f'{k} {v:.2f}' for k, v in res.items()))
