import json
c=json.load(open('cur.json'));T=c['T'];F=c['F'];idx=c['idx']
# accepted view: drop D chars, keep map to doc indices
A=''.join(ch for ch,f in zip(T,F) if f!='D'); Ai=[i for ch,f,i in zip(T,F,idx) if f!='D']
def rng(s):
    assert A.count(s)==1,(s[:50],A.count(s)); i=A.index(s); return Ai[i], Ai[i+len(s)-1]+1, T[T.index(A[i:i+1]):0]
ops=[]
def dele(s):
    a,b,_=rng(s); assert b-a==len(s) or True; ops.append((a,[{"deleteContentRange":{"range":{"startIndex":a,"endIndex":b}}}]))
def repl(old,new,ital=()):
    a,b,_=rng(old); r=[{"insertText":{"location":{"index":b},"text":new}}]
    for w in ital:
        k=new.index(w); r.append({"updateTextStyle":{"range":{"startIndex":b+k,"endIndex":b+k+len(w)},"textStyle":{"italic":True},"fields":"italic"}})
    r.append({"deleteContentRange":{"range":{"startIndex":a,"endIndex":b}}})
    ops.append((a,r))
dele('As in the recordings and the rate model (Figs. 1b and 2), On/Off statistics depended on which neurons were decoded. ')
repl('Decoders using the 20 most cue-selective neurons gave On and Off states of similar lengths to decoders using all units, but fewer of them (Fig. 4b).',
     'Both state lengths had long-tailed distributions, as in the rate model (Fig. 4b).')
dele(' On about a third of trials, the decoder using the 20 most cue-selective neurons stayed Off for the whole delay, because these neurons all preferred the same cue of the decoded pair and were silent on trials of the other cue.')
dele(' (black) and from the 20 most cue-selective of them (grey)')
repl('Off (left) and On (right) states for decoders using the 20 most cue-selective neurons (dark) or all units (light),','Off (top) and On (bottom) states,',ital=('Off','On'))
dele(' KS, two-sample Kolmogorov–Smirnov test on all states.')
dele(', using all units or the 20 most cue-selective neurons (ranked by split-half ANOVA between the two cues of each decoded pair)')
reqs=[]
for a,r in sorted(ops,key=lambda o:-o[0]): reqs+=r
# check ranges are contiguous in doc (no D chars inside)
for a,r in ops:
    for x in r:
        if 'deleteContentRange' in x:
            s,e=x['deleteContentRange']['range'].values(); i=idx.index(s); print(s,e, repr(T[i:i+(e-s)][:70]), set(F[i:i+(e-s)]))
json.dump(reqs,open('req_top20.json','w'),ensure_ascii=False); print(c['rev'])
