import numpy as np
from paper_model import model as M
def one_dip(L, depth, w_ff, n=200, alpha=None, seed=0):
    p = M.params(dip_depth=depth, w_ff=w_ff)
    if alpha: p.update(J_s=0.3725 * alpha, J_c=0.1137 * alpha)
    dt = p['dt']; n_t = int(1600 / dt)
    m = np.ones((n, n_t)); a = int(850 / dt); m[:, a:a + int(L / dt)] = 1 - depth
    w = int(10 / dt); m = np.array([np.convolve(np.r_[np.ones(w), r, np.ones(w)], np.ones(w) / w, 'same')[w:-w] for r in m])
    t = np.arange(n_t) * dt; cue = ((t >= 300) & (t < 350)).astype(float)
    rng = np.random.default_rng(seed)
    noise = p['sigma'] * M._ou(rng.standard_normal((n, n_t, 2)), np.exp(-dt / p['tau_n']))
    out = M._sim(m, cue, np.zeros(n_t), noise, np.array([float(p[k]) for k in M.KEYS]), 2)
    rc, ru = out[:, -100:, 0].mean(1), out[:, -100:, 1].mean(1)
    silent = out[:, int(850 + L / 2) - 5:int(850 + L / 2) + 5, 0].mean(1) < 2
    return (rc > 8).mean(), (ru > 8).mean(), silent.mean()
if __name__ == '__main__':
    for w_ff in (0.0, 0.1, 0.2, 0.4, 0.8):
        row = []
        for depth in (0.1, 0.3, 0.6):
            for L in (50, 100, 200):
                c, u, sil = one_dip(L, depth, w_ff)
                row.append(f'd{depth} L{L}: {c:.2f}/{u:.2f} sil{sil:.1f}')
        print(f'w_ff {w_ff}: ' + ' | '.join(row))
