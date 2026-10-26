"""Histograms of On/Off states per trial and of their lengths (full range), model (sessions 1-11, correct trials,
all units) vs data. Run from the repository root: PYTHONPATH=. python spiking/fig_state_histograms.py"""
import numpy as np, csv, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from paper_model import readout as PR
d=np.load('spiking/results/sessions_dips_trials.npz'); tz=d['tz']
dl=(tz>=500)&(tz<=1400)
L={'on':[],'off':[]}; n_on=[]; n_off_dl=[]
for s in range(1,12):
    for row in d[f's{s}_state_all']:
        for st,v in (('on',1),('off',-1)):
            L[st]+= [(b-a)*10 for a,b in PR.runs(row==v)]
        n_on.append(len(PR.runs(row==1))); n_off_dl.append(sum(1 for a,b in PR.runs(row==-1) if dl[a:b].any()))
L={k:np.array(v) for k,v in L.items()}; n_on,n_off_dl=np.array(n_on),np.array(n_off_dl)
_,hd=PR.load_data()
rows=list(csv.DictReader(open('paper_model/data/behaviour_per_trial_correct_and_error.csv')))
dd=np.array([float(r['n_off']) for r in rows if r['correct']=='1' and r['n_off']!='']).astype(int)
C_ON,C_OFF='#e8474a','#3cb4c9'
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,2,figsize=(7,4.8),gridspec_kw=dict(hspace=0.55,wspace=0.3,left=0.09,right=0.98,top=0.93,bottom=0.1))
k=np.arange(0,max(dd.max(),n_on.max(),n_off_dl.max())+1)
a=ax[0,0]; a.bar(k,np.bincount(n_off_dl,minlength=len(k))/len(n_off_dl),color=C_OFF,width=0.8,label=f'Model (mean {n_off_dl.mean():.2f})')
a.step(k,np.bincount(dd,minlength=len(k))/len(dd),where='mid',color='k',lw=1,label=f'Data (mean {dd.mean():.2f})')
a.set_xlabel('Off states per trial (500–1,400 ms)'); a.set_ylabel('Fraction of trials'); a.legend(frameon=False,fontsize=7); a.set_xticks(k)
a=ax[0,1]; a.bar(k,np.bincount(n_on,minlength=len(k))/len(n_on),color=C_ON,width=0.8,label=f'Model (mean {n_on.mean():.2f})')
a.axvline(2.10,color='k',ls='--',lw=1,label='Data mean (2.10)\n(no per-trial On counts)')
a.set_ylim(0,0.5); a.set_xlabel('On states per trial (0–1,400 ms)'); a.legend(frameon=False,fontsize=7,loc='upper right'); a.set_xticks(k)
edges=np.arange(0,1420,20)
for a,st,c in ((ax[1,0],'off',C_OFF),(ax[1,1],'on',C_ON)):
    h=np.histogram(L[st],edges)[0]
    a.bar(edges[:-1]+10,h/len(L[st]),width=20,color=c,label=f'Model (n = {len(L[st])}, longest {L[st].max()} ms)')
    hdd=hd['all',st]; tot=hdd.sum()
    a.step(np.arange(30)*20+10,hdd[:30]/tot,where='mid',color='k',lw=1,label='Data (0–600 ms bins)')
    a.bar(700,hdd[30]/tot,width=180,facecolor='none',edgecolor='k',lw=1,hatch='///',label=f'Data ≥ 600 ms ({100*hdd[30]/tot:.1f}%, pooled)')
    a.axvline(600,color='#999',lw=0.5,ls=':')
    a.set_xlabel(f'{st.capitalize()} state length (ms)'); a.set_xlim(0,1410); a.legend(frameon=False,fontsize=6.5)
ax[1,0].set_ylabel('Fraction of states')
for a,l in zip(ax.ravel(),'abcd'): a.text(-0.12,1.05,l,transform=a.transAxes,fontweight='bold',fontsize=10)
fig.savefig('spiking/figures/fig_state_histograms.png',dpi=200); fig.savefig('spiking/figures/fig_state_histograms.svg')
