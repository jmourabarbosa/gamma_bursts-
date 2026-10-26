import json,re
c=json.load(open('cur.json'));T=c['T'];F=c['F'];idx=c['idx']
assert set(F)<={'.'}
def u16(s): return len(s.encode('utf-16-le'))//2
def at(s):
    assert T.count(s)==1,(s[:40],T.count(s)); return idx[T.index(s)]
ops=[]
def ital(base,new):
    return [{"updateTextStyle":{"range":{"startIndex":base+u16(new[:m.start()]),"endIndex":base+u16(new[:m.end()])},"textStyle":{"italic":True},"fields":"italic"}} for m in re.finditer(r'\b(On|Off)\b',new)]
def repl(old,new):
    a=at(old); b=a+u16(old)
    ops.append((a,[{"insertText":{"location":{"index":b},"text":new}}]+ital(b,new)+[{"deleteContentRange":{"range":{"startIndex":a,"endIndex":b}}}]))
# Results (comment 14:40)
repl("In the plane of the eight cue-condition means, delay activity of both the network and the recordings moved toward the centre during Off states (Fig. 4d), but the per-cue averages of Off activity kept the shape of the ring of cue means well above chance, whereas activity before the cue did not (Fig. 4e).",
     "Information about the cue might still be visible at the level of the population, where weak signals of many neurons add up. We therefore first applied PCA to the eight cue-condition means of delay activity, in the network and in the recordings. In the plane of the first two components, delay activity moved toward the centre during Off states (Fig. 4d). We then compared the shape of the per-cue averages of activity with the ring of cue means, using a Procrustes shape similarity that ignores size (Methods). Off activity kept the shape of the ring well above chance, whereas activity before the cue did not (Fig. 4e).")
# Methods: new heading before the read-out of the network
title="Population analyses of On and Off states"
p=at("In each session, 330 neurons (264 pyramidal")
ins="\n"+title+"\n"
r=[{"insertText":{"location":{"index":p},"text":ins}},
   {"updateParagraphStyle":{"range":{"startIndex":p+1,"endIndex":p+1+u16(title)},"paragraphStyle":{"namedStyleType":"HEADING_2"},"fields":"namedStyleType"}}]+ital(p+1,title)
ops.append((p,r))
repl("In each session, 330 neurons","In each simulated session, 330 neurons")
# merge labelling sentences
repl("On and Off states were labelled from one half, with the decoder and criteria above, and all other analyses used the other half. In both, states were labelled from all neurons of the labelling half.",
     "On and Off states were labelled from all neurons of one half, with the decoder and criteria above, and all other analyses used the other half.")
# paragraph break before the shape analysis
q=at("For Fig. 4e, each trial's activity")
ops.append((q,[{"insertText":{"location":{"index":q},"text":"\n"}},{"deleteContentRange":{"range":{"startIndex":q-1,"endIndex":q}}}]))
reqs=[]
for a,r in sorted(ops,key=lambda o:-o[0]): reqs+=r
json.dump(reqs,open('req_split.json','w'),ensure_ascii=False); print(len(reqs),c['rev'])
