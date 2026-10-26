"""Diagnostic: Off and On length distributions of the spiking network decoded with the top vs bottom (least
selective) neurons, k = 5, 20, 50, and all units; drawn as Fig. 1b. No fitting; not in the paper."""
import json, os, sys
import numpy as np
sys.path.insert(0, '/home/user/spk_wt')
from paper_model.make_figures import plt, save, letter, C_ON, C_OFF, INK, GREY

D = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(D, 'topbottom.json')))
SHADES = {'off': ('#16606d', C_OFF), 'on': ('#9c1215', C_ON)}   # as Fig. 1b, dark to light


def hist(lab, name):
    h = np.sum([r[lab]['h_' + name] for r in res], 0)[:30]
    h = h[0::2] + h[1::2]                                  # 40 ms bins, 0-600 ms
    tot = np.sum([r[lab]['h_' + name] for r in res])
    return h / max(tot, 1), np.mean([r[lab]['n_' + name] for r in res]), tot


fig = plt.figure(figsize=(7.2, 3.0))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35, left=0.1, right=0.99, top=0.88, bottom=0.15)
for j, k in enumerate((5, 20, 50)):
    groups = [(f'top {k}', f'Top {k}'), ('all', 'All')]
    for i, name in enumerate(('off', 'on')):
        h = fig.add_subplot(gs[i, j])
        w = 40 / len(groups)
        top = 0
        for g, (lab, txt) in enumerate(groups):
            hm, n, tot = hist(lab, name)
            if tot >= 100:      # too few states for a distribution otherwise (bottom decoders are almost never On)
                h.bar(np.arange(0, 600, 40) + g * w, hm, width=w, align='edge', color=SHADES[name][g], lw=0)
                top = max(top, hm.max())
            h.text(0.97, 0.66 - 0.17 * g, (f'{txt}, {n:.1f}/trial' if tot >= 100 else f'{txt}, ~0/trial'),
                   transform=h.transAxes, ha='right', va='top',
                   color=SHADES[name][g], fontsize=6.5)
        h.text(0.97, 0.95, f'{name.capitalize()} states', transform=h.transAxes, ha='right', va='top',
               color=SHADES[name][1], fontsize=7.5)
        top = np.ceil(top * 10) / 10
        h.set_xlim(0, 600); h.set_xticks([0, 200, 400, 600])
        h.set_ylim(0, top); h.set_yticks([0, top / 2, top], ['0', '', f'{top:g}'])
        if j == 0:
            h.set_ylabel('Fraction\nof states', labelpad=-14)
        if i == 0:
            h.tick_params(labelbottom=False)
            h.set_title(f'{k} neurons per decoder', fontsize=7.5)
        else:
            h.set_xlabel('State length (ms)')
fig.text(0.01, 0.97, f'Spiking network, top neurons vs all units ({len(res)} sessions)', fontsize=7.5)
save(fig, sys.argv[1])
