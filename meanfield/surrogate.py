"""Fast surrogate of the Panichello pipeline.

The decoder is the L2 logistic regression of the full pipeline, linearised: one Newton step from
w = 0, i.e. kernel ridge regression of 4(y - 1/2) with ridge 4*lam on the z-scored counts
(meanfield.decoder.linear_confidence). Its leave-one-trial-out prediction is exact in closed form,
and all shuffled-label nulls share one eigendecomposition per time bin, so true + 20 null decoders
cost about as much as one logistic fit. For strong regularisation (lam >= 300) it reproduces the
full pipeline (ALO logistic, itself checked against exact LOO): see scripts/mf_calibrate.py.

A first version mapped an oracle signal-to-noise e(t) to z with a fitted function; the map was
strongly nonlinear and, at lam = 10, saturated at z ~ 1.8 because with 329 units > 132 trials the
shuffled decoders are as confident as the real one (null SD ~ 0.25). The linearised decoder keeps
that behaviour by construction, so it replaced the map.
"""
import numpy as np
from numba import njit

from . import recording as R
from .decoder import linear_confidence

LAM = 1000.0


@njit(cache=True)
def _label(z, valid, thr, min_off):
    n_tr, n_b = z.shape
    state = np.zeros((n_tr, n_b), np.int64)
    for k in range(n_tr):
        i = 0
        while i < n_b:
            if valid[i] and z[k, i] > 1.96:
                j = i
                m = 0.0
                while j < n_b and valid[j] and z[k, j] > 1.96:
                    m += z[k, j]
                    j += 1
                if m > thr:
                    state[k, i:j] = 1
                i = j
            else:
                i += 1
        i = 0
        while i < n_b:
            if valid[i] and z[k, i] < 0.84:
                j = i
                while j < n_b and valid[j] and z[k, j] < 0.84:
                    j += 1
                if j - i >= min_off:
                    state[k, i:j] = -1
                i = j
            else:
                i += 1
    return state


def label(z, valid, thr, min_off=5):
    return _label(np.ascontiguousarray(z), np.asarray(valid, np.bool_), float(thr), int(min_off))


@njit(cache=True)
def _null_masses(nulls, valid):
    n_sh, n_tr, n_b = nulls.shape
    tot = np.zeros((n_tr, n_b))
    tot2 = np.zeros((n_tr, n_b))
    for s in range(n_sh):
        tot += nulls[s]
        tot2 += nulls[s] ** 2
    out = []
    for s in range(n_sh):
        mu = (tot - nulls[s]) / (n_sh - 1)
        var = (tot2 - nulls[s] ** 2) / (n_sh - 1) - mu ** 2
        zs = (nulls[s] - mu) / (np.sqrt(np.maximum(var, 0.0)) + 1e-9)
        for k in range(n_tr):
            i = 0
            while i < n_b:
                if valid[i] and zs[k, i] > 1.96:
                    m = 0.0
                    while i < n_b and valid[i] and zs[k, i] > 1.96:
                        m += zs[k, i]
                        i += 1
                    out.append(m)
                else:
                    i += 1
    return np.array(out)


def label_states(conf, nulls, valid, min_off=5):
    """Same as gamma_wm.panichello.label_states (z against the shuffled null; cluster-mass threshold
    from leave-one-shuffle-out null z), compiled. Returns state, z, thr."""
    nulls = np.ascontiguousarray(nulls)
    valid = np.asarray(valid, np.bool_)
    mu, sd = nulls.mean(0), nulls.std(0) + 1e-9
    z = (conf - mu) / sd
    nm = _null_masses(nulls, valid)
    thr = float(np.percentile(nm, 95)) if len(nm) else 0.0
    return label(z, valid, thr, min_off), z, thr


def fast_pipeline(ses, lam=LAM, n_shuffle=20, seed=0):
    """Linearised decoder + shuffled null + On/Off labelling + state statistics for one session."""
    conf, nulls = linear_confidence(ses['C'], ses['y'], lam, n_shuffle, seed)
    state, z, thr = label_states(conf, nulls, ses['valid'])
    return dict(conf=conf, z=z, state=state, thr=thr, stats=R.state_stats(state, ses['count_mask']),
                acc=float((conf[:, ses['dec_mask']] > 0.5).mean()))
