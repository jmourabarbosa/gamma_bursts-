import json,re
c=json.load(open('cur.json'));T=c['T'];F=c['F'];idx=c['idx']
def u(s): return len(s.encode('utf-16-le'))//2
def pos(k): return idx[k]
E=[]  # (start,end,newtext)
def rep(old,new,start=0):
    k=T.index(old,start); assert set(F[k:k+len(old)])=={'.'}; E.append((pos(k),pos(k)+u(old),new))
def ins(k,new): E.append((pos(k),pos(k),new))
# F renumber in text
rep('(Fig. 2; Supplementary Fig. 1)','(Fig. 2; Supplementary Fig. 2)')
rep('NMDA model (Supplementary Fig. 2)','NMDA model (Supplementary Fig. 3)')
# captions renumber
k=T.index('Supplementary Figure 2. The'); rep('Supplementary Figure 2.','Supplementary Figure 3.',k)
k=T.index('Supplementary Figure 1. Off'); rep('Supplementary Figure 1.','Supplementary Figure 2.',k)
# new supp fig 1 caption, before old Supp Fig 1 image paragraph
cap=("Supplementary Figure 1. Reading the model out like the data. a, One model trial through the read-out. From top to bottom, rates of the cued (black) and uncued (grey) pools, with dips of the common input in grey; Δr, their difference averaged in 100 ms windows every 10 ms; c·Δr (black) and the read-out noise η (grey); and z = c·Δr + η, with On (red) and Off (blue) states labelled with the thresholds of Panichello et al. (2024) (dashed, z = 1.96 and 0.84). "
"b, One data trial (all-unit decoder, session 211008) with its On and Off states. "
"c, Autocorrelation of the decoder z before the cue, where there is no memory signal, for three decoded populations, compared with the model's noise η (orange dashed) and with η plus 4% white noise (blue dashed). The SD of pre-cue z was 1.06 for all three populations. "
"d, Distribution of delay z (500–1,400 ms after cue onset) in the data (grey, 25 sessions) and in the model (orange). Dashed, decoding strength (the upper mean of a two-Gaussian mixture), 2.18 in the data and 2.20 in the model. "
"e, Decoding strength of the model as a function of the gain c. The data's decoding strength (horizontal line) sets c (vertical line). "
"f–h, Validation on synthetic data. The fitted model drove 330 Poisson neurons whose spikes were decoded with the data pipeline. f, Example trial with the true dips (grey), the decoder's z (black) and the read-out z = c·Δr + η (orange). g, Correlation of each read-out's delay z with the decoder's, at three decoding strengths matched to the data's random-20, all-unit and top-20 decoders. Grey, the decoder rerun on new spikes, which sets the ceiling for two noisy read-outs. h, On and Off statistics of the decoder (circles), the read-out (squares) and the read-out with jitter-matched noise (diamonds), mean ± 2 SEM over 8 synthetic sessions. "
"i, Controls on the fitted model with the all-unit read-out. Off and On states per trial, and percentage of the delay Off, for the data, the read-out used in the paper, noise only (c = 0), gain ×0.75 and ×1.25, ρ = 0 and 0.7, jitter-matched noise and the model without dips.")
k=T.index('\nSupplementary Figures\n')+len('\nSupplementary Figures\n')
ins(k,'\n'+cap+'\n')
NEWCAP_AT=pos(k)
# Methods paragraph
mold_s='The decoder z score was modelled as'; mold_e='the 10 draws were first averaged within each session.'
a=T.index(mold_s); b=T.index(mold_e,a)+len(mold_e)
meth=("To compare the model with the recordings, we converted its activity into a decoding z score and labelled it with the same criteria as the data (Supplementary Fig. 1). The decoding evidence was Δr(t), the difference between the cued and uncued pool rates averaged in 100 ms windows stepped by 10 ms, as for the neural data. The model z score was z(t) = c Δr(t) + η(t), where the gain c converts rates into z units and η is read-out noise, which stands for the variability of a decoder trained on a finite number of neurons and trials. "
"The noise η has unit variance, like the pre-cue z of the data (SD 1.06), the autocorrelation of overlapping 100 ms windows, and a constant per-trial offset whose share of the variance, ρ, equals the autocorrelation of the data's pre-cue z at 200 ms (0.35 for all neurons, 0.09 for the most selective neurons and 0.11 for random neurons; Supplementary Fig. 1c). "
"The gain c was not fitted to the On/Off statistics. For each decoded population, it was set so that the decoding strength of the model's delay z, the upper mean of a two-component Gaussian mixture of delay z values, equalled the data's (Supplementary Fig. 1d,e). "
"States were labelled as in Panichello et al. (2024). On states are runs of z > 1.96 whose summed z exceeds the 95th percentile of the same quantity in pure noise η. Off states are runs of at least 5 points (50 ms) with z < 0.84. States were counted from 0 to 1,400 ms after cue onset, and fractions of the delay spent On or Off from 500 to 1,400 ms. Model statistics were computed from 2,112 simulated trials (792 per evaluation during fitting). Data statistics are means over the 25 sessions; for random neurons, the 10 draws were first averaged within each session.\n"
"To validate this read-out, we drove 330 synthetic Poisson neurons with the pool rates of the fitted model for two cue locations, with the pools swapping roles (Supplementary Fig. 1f–h). 30% of the neurons were tuned to one of the two locations and the rest were untuned, baseline rates were lognormal with a median of 5 Hz, and all neurons shared a per-trial gain. Their spikes were decoded with the unchanged data pipeline (70 trials per cue, 8 synthetic sessions at each of three tuning strengths, matched to the decoding strength of the random-20, all-unit and top-20 decoders). "
"The read-out's z correlated with the decoder's as closely as two runs of the decoder on independent spikes, and it reproduced the number and fraction of Off states. It undercounted On states by about one per trial, and this disappeared when a white-noise component matched to the decoder's fast fluctuations was added to η. Controls on the fitted model showed that On/Off statistics change smoothly with the gain and with ρ, and that noise alone produces Off states but almost no On states (Supplementary Fig. 1i).")
E.append((pos(a),pos(b-1)+1,meth))
# Results: rewrite first sentence, add validation sentence
r1='To compare the model with the data, we turned its activity into a decoding z score: the difference between the cued and uncued pool rates in 100 ms windows, multiplied by a gain (c) and added to noise (η) whose variance and temporal correlations match those of the decoder output in the data (Methods).'
r1n=("To compare the model with the data, we read it out like the recordings (Supplementary Fig. 1; Methods). In the data, a decoder trained on the recorded neurons gives, every 10 ms, a z score of how well the cue can be read out. The model has no neurons, so we built the equivalent score from its two pools. The evidence for the cue is the difference between the cued and uncued pool rates in the same 100 ms windows. A gain converts this difference into z units, and read-out noise stands for the variability of a decoder trained on a limited sample of neurons, with the same variance and temporal correlations as the decoder output before the cue.")
rep(r1,r1n)
anc='which we call the decoding strength (Methods).'
k=T.index(anc)+len(anc)
ins(k," When synthetic neurons driven by the model were decoded with the same pipeline as the data, the decoder's z agreed with this read-out as closely as two runs of the decoder on independent spikes, and gave similar Off states (Supplementary Fig. 1f–h).")
E.sort(key=lambda e:-e[0])
R=[]
for s,e,n in E:
    if e>s: R.append({"deleteContentRange":{"range":{"startIndex":s,"endIndex":e}}})
    R.append({"insertText":{"location":{"index":s},"text":n}})
    for m in re.finditer(r'\b(?:On|Off)(?:/(?:On|Off))?\b',n):
        if n[m.start():].startswith('On the other'): continue
        p=s+u(n[:m.start()]); R.append({"updateTextStyle":{"range":{"startIndex":p,"endIndex":p+u(m.group(0))},"textStyle":{"italic":True},"fields":"italic"}})
json.dump({'rev':c['rev'],'reqs':R,'img_at':NEWCAP_AT},open('req_ro.json','w'),ensure_ascii=False)
print(len(R),'img at',NEWCAP_AT)
