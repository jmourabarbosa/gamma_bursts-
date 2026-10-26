"""Procrustes shape similarity for the recordings (split version: states from the top 20 units of half H1,
activity of half H2), as in pca_onoff.py of the data analysis."""
import glob, json, os, sys
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'wt_data', 'analysis'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mechanism as M
import procr as PR

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def one(path):
    name = os.path.basename(path)[:-4]
    f = os.path.join(OUT, name + '.json')
    if os.path.exists(f):
        return json.load(open(f))
    S = M.Session(path)
    pu = np.random.default_rng(1).permutation(S.n_units)
    H1, H2 = np.sort(pu[:S.n_units // 2]), np.sort(pu[S.n_units // 2:])
    _, st, _ = S.label(*S.top_within(H1))
    dl = (S.tsw >= M.DEL[0]) & (S.tsw <= M.DEL[1])
    Zd = S.Zt[:, S.sw][:, dl][:, :, H2]
    Zpre = S.Zt[:, S.tt < 0][:, :, H2]
    half = PR.halves(S.cue, np.random.default_rng(0))
    res, ks = PR.shape_similarity(Zd, Zpre, S.cue, st[:, dl], half)
    out = dict(session=name, n_units=int(len(H2)), k_pcs=ks, res=res)
    json.dump(out, open(f, 'w'))
    return out


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob('/home/user/panichello/*.mat'))
    if len(sys.argv) > 1:
        files = files[:int(sys.argv[1])]
    Parallel(n_jobs=2, verbose=5)(delayed(one)(p) for p in files)
