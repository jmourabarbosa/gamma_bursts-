import json, sys
from paper_model import readout as R, model as M
from paper_model.fit import loss
name = sys.argv[1]
th = M.FIT if name == 'old' else json.load(open(f'/home/user/fits_all/{name}/best.json'))['theta']
summ, hist = R.load_data()
res = []
for seed in (11, 12):
    out = R.evaluate(th, n_trials=2112, seed=seed, summ=summ)
    L, t = loss(out, summ, hist)
    res.append((L, out['mem'], sum(v for k, v in t.items() if k.startswith('hist')), t))
print(name, ' '.join(f'loss {L:.2f} mem {m:.3f} hist {h:.2f}' for L, m, h, _ in res))
