"""Default parameter set used for the working-memory simulations."""
from .network import Params

# Brunel (2000) model A, inhibition-dominated, synchronous-irregular (SI) regime with
# a ~55-60 Hz (gamma) population rhythm set by the transmission delay D, plus a
# rank-one perturbation storing one binary memory (pools A and B, 10% of E cells each).
DEFAULT = dict(N_E=10000, N_I=2500, eps=0.1, J=0.1, g=8.0, nu_ext_ratio=1.4, D=4.0,
               f=0.1, beta=0.9)

# Task timing (ms)
TASK = dict(t_pre=1000.0, cue_dur=150.0, t_delay=2000.0, cue_strength=0.1)


def default_params(**over):
    d = dict(DEFAULT)
    d.update(over)
    return Params(**d)
