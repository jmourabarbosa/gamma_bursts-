import json,re,difflib
S='/tmp/claude-0/-home-user/51e93236-a4df-5b7b-99f9-872ebc6e17b8/scratchpad/'
c=json.load(open(S+'cur.json'));T=c['T'];F=c['F'];idx=c['idx']
tok=lambda s: re.findall(r'\s+|[^\s]+',s)
E=[]
WHOLE=False
def rep(old,new,a0=None):
    if a0 is None:
        assert T.count(old)==1,(old[:60],T.count(old)); a0=T.index(old)
    assert set(F[a0:a0+len(old)])<={'.'},old[:50]
    if WHOLE:
        E.append((idx[a0],idx[a0+len(old)-1]+1,new)); return
    ot,nt=tok(old),tok(new); pos=[0]
    for x in ot: pos.append(pos[-1]+len(x))
    for op,i1,i2,j1,j2 in difflib.SequenceMatcher(None,ot,nt,autojunk=False).get_opcodes():
        if op=='equal': continue
        s=a0+pos[i1]; e=a0+pos[i2]
        ds=idx[s]; de=idx[e-1]+1 if e>s else ds
        E.append((ds,de,''.join(nt[j1:j2])))
for m in re.finditer(r"\(Amit & Brunel, 1997",T): rep("Amit & Brunel","Amit and Brunel",m.start()+1)
new=[x.replace('\\em ','').replace("‘Activity-silent'","“Activity-silent”") for x in json.load(open(S+'refs_fmt.json'))]
a=T.index('\nReferences\n')+len('\nReferences\n'); b=T.index('\nSupplementary Figures')
paras=[]; p=a
for line in T[a:b].split('\n'):
    paras.append((p,line)); p+=len(line)+1
def key(s):
    m=re.match(r"([A-Z][^\s,]+)[\s,].*?\((\d{4})\)",s); return (m.group(1),m.group(2)) if m else None
newd={}
for n in new:
    k=key(n)
    if k in newd: k=k+(n.split(',')[1].strip()[:6],)
    newd[k]=n
used=set()
WHOLE=True
for p0,line in paras:
    if not line.strip(): continue
    k=key(line)
    cand=[kk for kk in newd if kk[:2]==k]
    if len(cand)>1:
        cand=[kk for kk in cand if len(kk)>2 and line.split(',')[0]+',' and kk[2].replace('.','').replace(' ','')[:1]==line.split(' ')[1][0]] or cand
    if not cand:
        print('DELETE',line[:70]); E.append((idx[p0],idx[p0+len(line)]+1,'')); continue
    assert len(cand)==1,(line[:60],cand)
    used.add(cand[0]); rep(line,newd[cand[0]],p0)
print('unused',[k for k in newd if k not in used])
R=[]
for s,e,n in sorted(E,key=lambda x:-x[0]):
    if e>s: R.append({"deleteContentRange":{"range":{"startIndex":s,"endIndex":e}}})
    if n: R.append({"insertText":{"location":{"index":s},"text":n}})
json.dump(R,open(S+'req_refs.json','w')); print(len(E),len(R))
