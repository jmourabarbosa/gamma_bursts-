"""Figure: after a spontaneous network-wide silence, does the cued pool restart first?
Data: figures/screens/global_silence_nmda.jsonl (globsil.py screens; 4 runs per setting)."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

INK, INK2 = '#0b0b0b', '#52514e'
COLS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6']
plt.rcParams.update({'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9, 'legend.frameon': False,
                     'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
rows = [json.loads(l) for l in open('figures/screens/global_silence_nmda.jsonl')]
pick = [(0.88, 1.0, 150.0, False, 'no memory (alpha 1.0)'), (0.86, 1.8, 150.0, False, 'alpha 1.8, tau 150'),
        (0.86, 1.85, 200.0, True, 'alpha 1.85, tau 200'), (0.86, 1.9, 200.0, True, 'alpha 1.9, tau 200'),
        (0.86, 2.0, 150.0, False, 'alpha 2.0, tau 150')]
bins = ['after_20_50', 'after_50_100', 'after_100_200', 'after_200']
labels = ['20-50', '50-100', '100-200', '>200']
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[2.2, 1]))
w = 0.16
for j, (nu, al, tn, mix, lab) in enumerate(pick):
    r = [x for x in rows if abs(x['kw']['nu_ext_ratio'] - nu) < 1e-9 and abs(x['kw']['alpha'] - al) < 1e-9
         and x['kw'].get('tau_nmda') == tn and bool(x['kw'].get('fixed_group_indegree_pools_only', False)) == mix]
    if not r:
        continue
    r = r[0]
    for i, b in enumerate(bins):
        v = r[b]
        if v:
            ax[0].bar(i + (j - 2) * w, v[0], w * 0.9, color=COLS[j], label=lab if i == 0 or not any(r[bb] for bb in bins[:i]) else None)
            ax[0].text(i + (j - 2) * w, v[0] + 0.02, str(v[1]), ha='center', fontsize=6.5, color=INK2)
    ax[1].bar(j, r['end_ok'] / 4, 0.6, color=COLS[j])
ax[0].axhline(0.5, color=INK2, ls=':', lw=0.8)
ax[0].set_xticks(range(4), labels)
ax[0].set_xlabel('duration of the spontaneous network-wide silence (ms)')
ax[0].set_ylabel('P(cued pool restarts first)')
ax[0].set_ylim(0, 1.15)
ax[0].set_title('NMDA bridges natural silences up to ~200 ms (numbers: silences)')
ax[0].legend(fontsize=7.5, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.22))
ax[1].set_xticks(range(len(pick)), [p[4].replace(', ', '\n').replace(' (', '\n(') for p in pick], fontsize=7, rotation=0)
ax[1].set_ylabel('memory correct at end of 4 s delay')
ax[1].set_ylim(0, 1.05)
ax[1].set_title('... but long silences break it')
fig.tight_layout()
fig.savefig('figures/screens/nmda_bridging.png', dpi=150, bbox_inches='tight')
print('ok')
