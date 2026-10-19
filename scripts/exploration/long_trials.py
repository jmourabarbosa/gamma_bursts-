"""Very long delays to test whether the memory is a true attractor (stationary) rather than a
slow transient. Saves population rates (1 ms) per trial.

usage: python scripts/exploration/long_trials.py --out data/long --delay 30000 --trials A:1 B:2 \
           --set '{...}' --noise-sd 0.45 --noise-tau 500
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from gamma_wm.network import Params, build_network, A, B, NS  # noqa: E402
from gamma_wm.task import run_trial  # noqa: E402
from gamma_wm import analysis as an  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--delay', type=float, default=30000)
    ap.add_argument('--trials', nargs='+', default=['A:1', 'B:2'])
    ap.add_argument('--set', default='{}')
    ap.add_argument('--noise-sd', type=float, default=0.45)
    ap.add_argument('--noise-tau', type=float, default=500)
    ap.add_argument('--cue-strength', type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    kw = json.loads(args.set)
    net = build_network(Params(**kw))
    for tr in args.trials:
        cue, seed = tr.split(':')
        t0 = time.time()
        r = run_trial(net, cue, t_pre=1000, cue_dur=150, t_delay=args.delay, cue_strength=args.cue_strength,
                      seed=int(seed), ext_noise_sd=args.noise_sd, ext_noise_tau=args.noise_tau)
        pc = an.pop_rates(r, net)
        np.savez_compressed(os.path.join(args.out, f'trial_{cue}_{seed}.npz'), cue=cue, rates=pc.astype(np.float32),
                            params=json.dumps(kw), noise_sd=args.noise_sd, noise_tau=args.noise_tau)
        C, U = (A, B) if cue == 'A' else (B, A)
        k = [pc[t:t + 5000, C].mean() - pc[t:t + 5000, U].mean() for t in range(1150, pc.shape[0] - 4999, 5000)]
        print(cue, seed, 'kappa per 5 s:', np.round(k, 1), f'({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
