import sys, json
import numpy as np
from paper_model import model_ours_stp as H, readout as R
from paper_model.fit_stp import loss
R.M = H
ses, hist = R.load_data()
for spec in sys.argv[1:]:
    th = json.loads(spec)
    out = R.evaluate(H.params(t_pre=1000.0, **th), ses, n_trials=528, seed=3, sets=('all',))
    L = loss(out, ses, hist); r = out['sets']['all']
    print(spec, f"loss {L:.1f} mem {out['mem']:.2f} spont {out['spont']:.2f} | Off {r['n_off']:.2f}/{r['off_mean']:.0f}ms/{r['frac_off']:.2f} On {r['n_on']:.2f}/{r['on_mean']:.0f}ms/{r['frac_on']:.2f}", flush=True)
