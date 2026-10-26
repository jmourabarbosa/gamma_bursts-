import json,re,difflib
S='/tmp/claude-0/-home-user/51e93236-a4df-5b7b-99f9-872ebc6e17b8/scratchpad/'
c=json.load(open(S+'cur.json'));T=c['T'];F=c['F'];idx=c['idx']
def u(s): return len(s.encode('utf-16-le'))//2
E=[]
def at(anchor, text, after=True):
    hits=[m.start() for m in re.finditer(re.escape(anchor),T) if set(F[m.start():m.start()+len(anchor)])<={'.','I'}]
    assert len(hits)==1,(anchor[:50],len(hits))
    p=hits[0]+(len(anchor) if after else 0)
    E.append((idx[p-1]+1 if after else idx[p],None,text))
def rep(old,new):
    hits=[m.start() for m in re.finditer(re.escape(old),T) if set(F[m.start():m.start()+len(old)])<={'.'}]
    assert len(hits)==1,(old[:50],len(hits)); a0=hits[0]
    E.append((idx[a0],idx[a0+len(old)-1]+1,new))
# Results
at("but fewer of them (Fig. 4b).", " On about a third of trials, the decoder using the 20 most cue-selective neurons stayed Off for the whole delay (Fig. 4b), because these neurons all preferred the same cue of the decoded pair and were silent on trials of the other cue.")
rep("not tuned during Off states (Fig. 4c)","barely tuned during Off states (Fig. 4c)")
at("is therefore not evidence for activity-silent storage.", " The same held at the level of the population. In the plane of the eight cue-condition means, delay activity of both the network and the recordings moved toward the centre during Off states (Fig. 4d), and its alignment with the cued location dropped but remained above its level before the cue (Fig. 4e).")
# Caption
at("Bottom: decoding z score from 330 simulated neurons", " (black) and from the 20 most cue-selective of them (grey)")
at(" c, Cross-validated memory tuning", " Last bar, states of 600 ms or longer, which for the top 20 neurons nearly always spanned the whole delay. KS, two-sample Kolmogorov–Smirnov test.", after=False)
at("computed as in Panichello et al. (2024) (mean ± s.e.m., 330 neurons).", " d, Delay activity in the plane of the first two principal components of the eight cue-condition means, for one simulated session (left) and one recorded session (221024, right). Grey, the cue-condition means. Filled and open circles, the mean position of each cue's trials during On and Off moments, with an arrow from On to Off. Colours, cue location (inset). e, Alignment of each trial's position with its cue's direction (cosine, minus the same with shuffled cue labels) during On and Off moments and before the cue (Pre). Thin lines, sessions. Thick, mean ± s.e.m. (data, 25 sessions, black; network, 10 sessions, blue).")
# Methods: new paragraph after the spiking-network tuning paragraph
at("(10 random splits, 330 neurons of one session).", "\nDelay activity during On and Off states (Fig. 4d, e) was analysed in the same way in the recordings and in the network. In each session, neurons were split at random into two halves. On and Off states were labelled from one half, with the decoder and criteria above, and all other analyses used the other half. In the recordings, states were labelled from the 20 most cue-selective units of the labelling half. In the network, they were labelled from all of its neurons, because its 20 most selective neurons often preferred the same cue and were silent on trials of the other cue while the bump was still active. Delay activity (z-scored firing rates in 100 ms windows, 500–1,400 ms after cue onset) was averaged for each of the eight cue locations, and PCA was applied to these condition means. Axes were estimated on one half of the trials, used to project the other half, and aligned across halves by an orthogonal Procrustes rotation. Each trial's position was averaged over its On and, separately, its Off time points. Its alignment was the cosine between this position and the direction of its cue's mean in the other half of the trials, minus the same quantity with cue labels shuffled within each half (50 shuffles). Activity before the cue, projected on the same axes, served as a null (window centres from −400 ms to cue onset in the recordings and to −50 ms in the network, whose neurons respond to the cue within a few milliseconds). Ten simulated sessions were analysed.")
R=[]
for s,e,n in sorted(E,key=lambda x:-x[0]):
    if e is not None and e>s: R.append({"deleteContentRange":{"range":{"startIndex":s,"endIndex":e}}})
    R.append({"insertText":{"location":{"index":s},"text":n}})
    for m in re.finditer(r'\b(?:On|Off)(?:(?: and |/)(?:On|Off))?(?:(?:-| )states?\b)?',n):
        pp=s+u(n[:m.start()]); R.append({"updateTextStyle":{"range":{"startIndex":pp,"endIndex":pp+u(m.group(0))},"textStyle":{"italic":True},"fields":"italic"}})
open(S+'req_f4de.txt','w').write(json.dumps(R,ensure_ascii=False)); print(len(R), c['rev'][:12])
img=[m.start() for m in re.finditer('Figure 4\\. Intermittent',T)]; print('fig4 caption at', [idx[i] for i in img])
