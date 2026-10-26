import sys, json
import numpy as np
from paper_model import model_stp as S, readout as R
R.M = S     # read out the STP model exactly like ours
ses, hist = R.load_data()
for spec in sys.argv[1:]:
    th = json.loads(spec)
    out = R.evaluate(S.params(**th), ses, n_trials=264, seed=3, sets=('all',))
    r, d = out['sets']['all'], ses['all']
    print(spec, f"mem {out['mem']:.2f} | gain {r['gain']:.3g} | Off {r['n_off']:.2f}/trial, mean {r['off_mean']:.0f} ms, frac {r['frac_off']:.2f}"
          f" | On {r['n_on']:.2f}/trial, mean {r['on_mean']:.0f} ms, frac {r['frac_on']:.2f}", flush=True)
d = ses['all']
print('data', f"Off {d['off_per_trial'].mean():.2f}/trial, mean {d['off_mean_ms'].mean():.0f} ms, frac {d['frac_off'].mean():.2f} | On {d['on_per_trial'].mean():.2f}/trial, mean {d['on_mean_ms'].mean():.0f} ms, frac {d['frac_on'].mean():.2f}")
