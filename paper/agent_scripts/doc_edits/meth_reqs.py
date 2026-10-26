import json,re
B=json.load(open('meth_new.json')); P=json.load(open('paras.json'))
c=json.load(open('cur.json'));T=c['T'];F=c['F'];idx=c['idx']
assert set(F)<={'.'}
def u16(s): return len(s.encode('utf-16-le'))//2
S=[p['start'] for p in P if p['text']=='NMDA rate model\n'][0]
E_=[p['start'] for p in P if p['text']=='AI methodology\n'][0]
text='';pr=[];runs=[]
for cs,k in B:
    a=u16(text); t=''.join(ch for ch,_ in cs)
    pos=a
    for ch,st in cs:
        if st: runs.append((S+pos,S+pos+u16(ch),json.dumps(st,sort_keys=True)))
        pos+=u16(ch)
    text+=t+'\n'; pr.append((S+a,S+u16(text),k))
L=u16(text)
reqs=[{"insertText":{"location":{"index":S},"text":text}}]
for a,b,k in pr:
    ps={"namedStyleType":"HEADING_2" if k=='H' else "NORMAL_TEXT","alignment":"CENTER" if k=='C' else "JUSTIFIED"}
    reqs.append({"updateParagraphStyle":{"range":{"startIndex":a,"endIndex":b},"paragraphStyle":ps,"fields":"namedStyleType,alignment"}})
    if k!='H':
        reqs.append({"updateTextStyle":{"range":{"startIndex":a,"endIndex":b-1},"textStyle":{"bold":False,"italic":False,"baselineOffset":"NONE"},"fields":"bold,italic,baselineOffset"}})
# merge runs
runs.sort(); merged=[]
for a,b,s in runs:
    if merged and merged[-1][1]==a and merged[-1][2]==s: merged[-1][1]=b
    else: merged.append([a,b,s])
for a,b,s in merged:
    st=json.loads(s); fl=','.join(st)
    reqs.append({"updateTextStyle":{"range":{"startIndex":a,"endIndex":b},"textStyle":st,"fields":fl}})
reqs.append({"deleteContentRange":{"range":{"startIndex":S+L,"endIndex":E_+L}}})
# ---- lower-index edits (Discussion, Results)
def at(s):
    assert T.count(s)==1,(s[:40],T.count(s)); return idx[T.index(s)]
low=[]
def ins_after(anchor,new):
    b=at(anchor)+u16(anchor)
    r=[{"insertText":{"location":{"index":b},"text":new}}]
    for m in re.finditer(r'\b(On|Off)\b',new):
        s=b+u16(new[:m.start()]); r.append({"updateTextStyle":{"range":{"startIndex":s,"endIndex":s+u16(m.group())},"textStyle":{"italic":True},"fields":"italic"}})
    low.append((b,r))
ins_after("Persistent activity and slower silent traces are therefore not mutually exclusive",
  ", but longer silent periods are required to show that such traces matter for the voluntary maintenance of working memory")
ins_after("is therefore an upper bound on how often the memory leaves spiking activity, and comparisons of Off-state statistics across studies or brain areas need to take the decoded population into account.",
  " Population activity points the same way. In both the recordings and the spiking network, the per-cue averages of activity during Off states kept the shape of the ring of cue means (Fig. 4e), so a decoder at chance does not mean that the cue has left population activity.")
for m in re.finditer('τf',T):
    i=idx[m.start()+1]
    if i<S: low.append((i,[{"updateTextStyle":{"range":{"startIndex":i,"endIndex":i+1},"textStyle":{"baselineOffset":"SUBSCRIPT"},"fields":"baselineOffset"}}]))
for a,r in sorted(low,key=lambda o:-o[0]):
    assert a<S; reqs+=r
json.dump(reqs,open('req_meth.json','w'),ensure_ascii=False)
print(len(reqs),S,E_,L,c['rev'], [T[m.start()-20:m.start()+5] for m in re.finditer('τf',T)])
