import sys
import numpy as np
from paper_model import model_stp as S
from scipy.signal import find_peaks
no = float(sys.argv[1])
for Ib in [float(v) for v in sys.argv[2].split(",")]:
    for amp in (5.0, 20.0):
        s = S.simulate(S.params(I_b=Ib, cue_amp=amp, n_other=no), 1)
        t = s['t']; rc, ru = s['rc'][0], s['ru'][0]
        pk, _ = find_peaks(rc, height=10, distance=20); pk = pk[t[pk] > 100]
        pku, _ = find_peaks(ru, height=10, distance=20)
        isi = np.diff(t[pk]).mean() if len(pk) > 1 else 0
        print(f"Ib {Ib:4.1f} amp {amp:3.0f} | rest R {rc[250]:.2f} | cued PS {len(pk):2d} (ISI {isi:4.0f} ms, last {t[pk[-1]] if len(pk) else -1:5.0f}, peak {rc[pk].max() if len(pk) else 0:5.1f} Hz)"
              f" uncued PS {len(pku):2d} | uc-uu end {s['uc'][0,-100:].mean()-s['uu'][0,-100:].mean():.3f} | delay mean Rc {rc[(t>500)&(t<1400)].mean():.2f} Ru {ru[(t>500)&(t<1400)].mean():.2f}")
