"""Figure 2 (On/Off statistics of the 7 unit sets, model vs data) for the model chosen with PM_VARIANT.

usage: PM_VARIANT=fac python -m paper_model.fig_unit_sets_variant     (writes figures/off_vs_memory/<variant>/)
"""
import os

from . import readout as R, variant as V
from .make_figures import figure2


def main():
    ses, _ = R.load_data()
    out = R.evaluate(V.params(), ses, n_trials=2112, seed=1)
    os.makedirs(V.OUT, exist_ok=True)
    figure2(out, ses, os.path.join(V.OUT, 'fig2_unit_sets.png'))
    print(f"memory {out['mem']:.3f}")
    for lab, r in out['sets'].items():
        print(f"{lab:8s} Off {r['n_off']:.2f}/trial median {r['off_median']:.0f} ms frac {r['frac_off']:.2f} | "
              f"On {r['n_on']:.2f}/trial median {r['on_median']:.0f} ms frac {r['frac_on']:.2f}")


if __name__ == '__main__':
    main()
