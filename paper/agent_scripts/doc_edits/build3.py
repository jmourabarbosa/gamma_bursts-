import json
d=json.load(open('/root/.claude/projects/-home-user/51e93236-a4df-5b7b-99f9-872ebc6e17b8/tool-results/mcp-Google_Docs-read_doc-1792933867805.txt'))
c=d['content']; print(c['revisionId'])
body=c['tabs'][0]['documentTab']['body']['content']
txt=[];idx=[];fl=[]
for el in body:
    p=el.get('paragraph')
    if not p: continue
    for r in p['elements']:
        tr=r.get('textRun')
        if not tr: continue
        s=r['startIndex'];f='I' if tr.get('suggestedInsertionIds') else 'D' if tr.get('suggestedDeletionIds') else '.'
        for ch in tr['content']:
            txt.append(ch);idx.append(s);fl.append(f);s+=len(ch.encode('utf-16-le'))//2
T=''.join(txt); F=''.join(fl)
print('pending:',F.count('I'),F.count('D'))
def u16(s): return len(s.encode('utf-16-le'))//2
E=[]
def rep(old,new):
    n=T.count(old); assert n==1,(old[:50],n)
    k=T.index(old); assert set(F[k:k+len(old)])=={'.'},old[:40]
    # trim common prefix/suffix to keep suggestions minimal
    a=0
    while a<min(len(old),len(new)) and old[a]==new[a]: a+=1
    b=0
    while b<min(len(old),len(new))-a and old[-1-b]==new[-1-b]: b+=1
    # snap to word boundaries
    while a>0 and old[a-1] not in ' ': a-=1
    while b>0 and old[len(old)-b] not in ' .,': b-=1
    o=old[a:len(old)-b]; nw=new[a:len(new)-b]
    s=idx[k+a]; e=s+u16(o)
    E.append((s,e,nw))
EDITS=[
("and needs no ongoing spikes for memory maintenance","and needs no ongoing spiking during maintenance"),
("However, fluctuations in a common input, which we interpret as a fluctuating attentional signal, briefly drop and silence the network. During each silence slow NMDA currents carry the memory across,",
 "However, a common input to the network, which we interpret as an attentional signal, fluctuates and briefly drops, silencing the network. During each silence, slow NMDA currents carry the memory,"),
("We read out from the model with the same decoding and On/Off criteria as Panichello et al. (2024) and fit how often, for how long and how deeply attention lapses to the On/Off statistics of these recordings.",
 "We read out the model with the same decoding and On/Off criteria as Panichello et al. (2024), and fit how often, for how long and how deeply attention lapses to the On/Off statistics of their recordings."),
("the number and length of On and Off states while keeping the memory. Our reanalyses","the number and length of On and Off states while keeping the memory in nearly all trials. Our reanalyses"),
("partly reflects how well the recorded sample reads it out.","partly reflects how well the recorded sample reads out the memory."),
("After the cue, the cued pool stays active and decoding stays On for the whole delay (Fig. 1a, first second after the cue).",
 "Without fluctuations in its input, the cued pool stays active after the cue and decoding stays On (Fig. 1a, first second after the cue)."),
("Off states therefore do not erase the memory and NMDA gating carries it across On states, ensuring a correct memory of the cue at the end of the delay.",
 "Off states therefore do not erase the memory, because NMDA gating carries it across each silence, and the cue is still correctly remembered at the end of the delay."),
("are then labelled with the same thresholds as in Panichello et al., 2024 (red and cyan dashed lines in Fig. 1a).",
 "are then labelled with the thresholds of Panichello et al. (2024) (dashed lines in Fig. 1a)."),
("The measured fraction of Off states is therefore an upper bound","The measured fraction of time Off is therefore an upper bound"),
("Without any refitting, the model predicts quantitatively the shift between selective and random neurons","The model thus predicts quantitatively the shift between selective and random neurons"),
("The NMDA account therefore predicts that errors become more frequent when the memory is Off for long,","The NMDA account therefore predicts that errors become more frequent when the memory stays Off for long,"),
("and far shorter than the ~1.5 s of facilitation.","and far shorter than the time constants of facilitation (0.5–1.5 s)."),
("Fitted to the decoder trained on the empirical data, the model reproduced the number and length of On and Off states while keeping the memory. Unlike a model in which short-term facilitation carries the memory, it also predicted that errors grow with the time spent Off, which we validated in prefrontal recordings.",
 "Fitted to the recordings, the model reproduced the number and length of On and Off states while keeping the memory in nearly all trials. Unlike a model in which short-term facilitation carries the memory, it also predicted that errors grow with the time spent Off, as we found in the recordings, and the decline in performance pointed to a trace of about 120 ms, matching NMDA kinetics."),
("Our model is compatible with both: NMDA currents carry the memory across full silence. Importantly, the memory is held by persistent activity without any Off state while attention is sustained (Fig. 1a, first second after the cue).",
 "Our model is compatible with both, because NMDA currents can carry the memory across full silences but are not needed when activity is only reduced. While attention is sustained, the memory is held by persistent activity without any Off state (Fig. 1a, first second after the cue)."),
("by NMDA gating that decays within ~150 ms and reflects recent firing","by NMDA gating that decays with a time constant of 150 ms and reflects recent firing"),
("In sum, intermittent decoding of working memory is expected from an attractor network with slow NMDA currents and a fluctuating, attention-like input, and its statistics depend on how well the recorded neurons read out the memory, while errors increase with the time spent Off, as expected from the limited duration of the NMDA trace. These results reconcile intermittent rate coding with persistent-activity accounts and link Off states to the rhythmic but irregular nature of attentional sampling.",
 "In sum, intermittent decoding of working memory is expected from an attractor network with slow NMDA currents and a fluctuating, attention-like input. Its statistics depend on how well the recorded neurons read out the memory, and errors increase with the time spent Off, as expected from the limited duration of the NMDA trace. These results reconcile intermittent rate coding with persistent-activity accounts, identify a trace of about 120 ms that bridges Off states, and link Off states to the rhythmic but irregular nature of attentional sampling."),
("The model was fitted to all neurons only (grey bar);","The model was fitted to all neurons only (grey band);"),
]
for o,n in EDITS: rep(o,n)
E.sort(key=lambda e:-e[0]); reqs=[]
for s,e,new in E:
    if e>s: reqs.append({"deleteContentRange":{"range":{"startIndex":s,"endIndex":e}}})
    if new: reqs.append({"insertText":{"location":{"index":s},"text":new}})
json.dump(reqs,open('req3.json','w'))
print(len(E)); print(json.dumps(reqs))
