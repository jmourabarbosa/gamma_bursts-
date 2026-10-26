import numpy as np
from paper_model import model_stp as S
from scipy.signal import find_peaks
for Ib in (2.0, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 5.0, 8.0):
    for amp in (5.0, 10.0, 20.0):
        s = S.simulate(S.params(I_b=Ib, cue_amp=amp), 1)
        t = s['t']; rc, ru = s['rc'][0], s['ru'][0]
        pk, _ = find_peaks(rc, height=10); pk = pk[t[pk] > 100]
        pku, _ = find_peaks(ru, height=10)
        print(f"Ib {Ib:4.1f} amp {amp:4.0f} | rest Rc {rc[200]:.2f} | PS cued {len(pk):2d} (last at {t[pk[-1]] if len(pk) else -1:5.0f} ms, peak {rc[pk].max() if len(pk) else 0:5.1f} Hz)"
              f" PS uncued {len(pku):2d} | uc-uu end {s['uc'][0,-100:].mean()-s['uu'][0,-100:].mean():.3f} | mean Rc delay {rc[(t>500)&(t<1400)].mean():.2f} Ru {ru[(t>500)&(t<1400)].mean():.2f}")
