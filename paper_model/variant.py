"""Model used by the Off-time analyses (off_vs_memory, single_dip, single_dip_readout), chosen with the environment
variable PM_VARIANT: 'nmda' (default, model.py with model.FIT), 'fac' (model_ours_stp.py, facilitation only with
tau_f 1.5 s, FIT_FAC) or 'os' (model.py with the attention overshoot instead of the feed-forward inhibition,
model.FIT_OVERSHOOT). Output figures go to figures/off_vs_memory/<variant>.
"""
import os

import numpy as np

from . import model as _nmda, model_ours_stp as _stp, readout as R

NAME = os.environ.get('PM_VARIANT', 'nmda')
MOD = {'nmda': _nmda, 'fac': _stp, 'os': _nmda}[NAME]
R.M = MOD
FIT = {'nmda': lambda: _nmda.FIT, 'fac': lambda: _stp.FIT_FAC,
       'os': lambda: {**_nmda.FIT_OVERSHOOT, **_nmda.OVERSHOOT}}[NAME]()
TRACE_LABEL = {'nmda': 'NMDA cued - uncued', 'fac': 'facilitation u, cued - uncued', 'os': 'NMDA cued - uncued'}[NAME]
OUT = os.path.join('figures', 'off_vs_memory', NAME)


def params(**over):
    return MOD.params(**{**FIT, **over})


simulate = MOD.simulate
memory_correct = MOD.memory_correct


def trace(sim):
    """The slow variable that holds the memory: cued minus uncued (memory_correct thresholds it at 0.05)."""
    return sim['uc'] - sim['uu'] if NAME == 'fac' else sim['sc'] - sim['su']


def simulate_input(p, m, cue, n_tr, seed):
    """Simulate with a given common input m (n_tr, n) and cue (n,) at resolution dt."""
    rng = np.random.default_rng(seed)
    dt = p['dt']
    n = m.shape[1]
    xi = _nmda._ou(rng.standard_normal((n_tr, n, 2)), np.exp(-dt / p['tau_xi']))
    rec = int(round(1 / dt))
    o = MOD._sim(m, cue, xi, np.array([float(p[k]) for k in MOD.KEYS]), rec)
    names = ('rc', 'ru', 'sc', 'su', 'uc', 'uu', 'xc', 'xu') if NAME == 'fac' else ('rc', 'ru', 'sc', 'su')
    sim = {k: o[..., i] for i, k in enumerate(names)}
    sim.update(m=m[:, rec - 1::rec], t=np.arange(n // rec) - p['t_pre'])
    return sim
