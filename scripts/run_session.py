"""Run a session of binary working-memory trials and store the results.

Trials are run back to back: each trial starts from the membrane potentials at the end
of the previous one, so the pre-cue state is whatever the network remembered from the
previous trial. Each trial: pre-cue period, 150 ms cue to pool A or B, 2 s delay.

usage: python scripts/run_session.py --session 0 --n-trials 12 [--beta 0.9 ...]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_wm.config import TASK, default_params  # noqa: E402
from gamma_wm.network import A, B, IA, IB, NS, build_network  # noqa: E402
from gamma_wm.task import run_trial  # noqa: E402
from gamma_wm import analysis as an  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', type=int, default=0)
    ap.add_argument('--n-trials', type=int, default=12)
    ap.add_argument('--out', default='data/session')
    ap.add_argument('--net-seed', type=int, default=0)
    ap.add_argument('--n-vrec', type=int, default=50, help='V traces recorded per pool')
    for k in ['g', 'nu_ext_ratio', 'D', 'f', 'beta']:
        ap.add_argument('--' + k, type=float, default=None)
    ap.add_argument('--set', default='{}', help='JSON dict of further Params overrides')
    args = ap.parse_args()
    over = {k: getattr(args, k) for k in ['g', 'nu_ext_ratio', 'D', 'f', 'beta']
            if getattr(args, k) is not None}
    over.update(json.loads(args.set))
    p = default_params(seed=args.net_seed, **over)
    net = build_network(p)
    rng = np.random.default_rng(1000 + args.session)
    cues = rng.permutation(['A', 'B'] * (args.n_trials // 2) + ['A'] * (args.n_trials % 2))
    rec = np.r_[np.where(net.group == A)[0][:args.n_vrec], np.where(net.group == B)[0][:args.n_vrec]]
    out = f'{args.out}{args.session:02d}'
    os.makedirs(out, exist_ok=True)
    json.dump(dict(params=p.__dict__, task=TASK, cues=list(cues)), open(f'{out}/meta.json', 'w'), indent=1)
    np.savez_compressed(f'{out}/net.npz', group=net.group, xi=net.xi, vrec_idx=rec)
    V0 = None
    for k, cue in enumerate(cues):
        t0 = time.time()
        r = run_trial(net, cue, seed=10000 * args.session + k, V0=V0, rec_V_idx=rec, **TASK)
        V0 = r['V']
        pc = an.pop_rates(r, net)
        lfp = an.lfp_proxy(r, net)
        Vr = r['V_rec'].reshape(-1, int(round(1 / p.dt)), len(rec)).mean(1)  # 1 ms
        np.savez_compressed(f'{out}/trial{k:03d}.npz', cue=cue, rates=pc.astype(np.float32),
                            lfp=lfp.astype(np.float32), spk_t=r['t'].astype(np.float32),
                            spk_i=r['i'], V=Vr.astype(np.float16))
        d = slice(int(r['t_delay_on']) + 500, int(r['T']))
        print(f'session {args.session} trial {k} cue {cue}: delay A {pc[d, A].mean():.1f} '
              f'B {pc[d, B].mean():.1f} NS {pc[d, NS].mean():.1f} I_A {pc[d, IA].mean():.1f} '
              f'I_B {pc[d, IB].mean():.1f} Hz ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
