import json,re,difflib
S='/tmp/claude-0/-home-user/51e93236-a4df-5b7b-99f9-872ebc6e17b8/scratchpad/'
c=json.load(open(S+'cur.json'));T=c['T'];F=c['F'];idx=c['idx']
def u(s): return len(s.encode('utf-16-le'))//2
tok=lambda s: re.findall(r'\s+|[^\s]+',s)
E=[]
def rep(old,new):
    hits=[m.start() for m in re.finditer(re.escape(old),T) if set(F[m.start():m.start()+len(old)])<={'.'}]
    assert len(hits)==1,(old[:60],len(hits)); a0=hits[0]
    ot,nt=tok(old),tok(new); pos=[0]
    for x in ot: pos.append(pos[-1]+len(x))
    for op,i1,i2,j1,j2 in difflib.SequenceMatcher(None,ot,nt,autojunk=False).get_opcodes():
        if op=='equal': continue
        s=a0+pos[i1]; e=a0+pos[i2]
        ds=idx[s]; de=idx[e-1]+1 if e>s else ds
        E.append((ds,de,''.join(nt[j1:j2])))
rep("and models of persistent activity typically use about 100 ms","and models of persistent activity typically use an NMDA decay time constant of about 100 ms")
rep("We modeled working memory storage in persistent activity supported by recurrent attractors (Wang, 2002; Wong and Wang, 2006).",
    "We modeled working memory with a classic attractor network, in which two cue-selective populations sustain their firing through recurrent excitation mediated by slow NMDA currents and compete through shared inhibition (Wang, 2002; Wong and Wang, 2006).")
rep("is a natural candidate. In our model, brief drops in such an attentional input silence the network.",
    "is a natural candidate. We therefore added to the network a fluctuating attentional input common to both populations, whose brief drops silence the network.")
rep("The evidence for the cue is the difference between the cued and uncued pool rates (delta r) in the same 100 ms windows. We then convert this difference into z units multiplying it with a gain c plus a read-out noise variability that we estimate from the data (Methods).",
    "The evidence for the cue is the difference between the cued and uncued pool rates (Δr) in the same 100 ms windows. We converted this difference into z units by multiplying it by a gain c and adding read-out noise, whose variance and temporal correlations we estimated from the decoder output before the cue (Methods).")
rep("In reanalyses of PFC recordings by Panichello et al, we found that","In our reanalysis of the PFC recordings of Panichello et al. (2024), we found that")
rep("read-out gain and noise (Method).","read-out gain and noise (Methods).")
rep("of increasing duration ( Methods).","of increasing duration (Methods).")
rep("covariance matrix adaptation evolution strategy (CMA-ES; Hansen and Ostermeier, 2001)","covariance matrix adaptation evolution strategy (Hansen and Ostermeier, 2001)")
new=json.load(open(S+'refs_fmt.json'))
a=T.index('\nReferences\n')+len('\nReferences\n'); b=T.index('\nSupplementary Figures')
def key(x):
    m=re.match(r"([A-Z][^\s,]+)[\s,].*?\((\d{4})\)",x); return (m.group(1),m.group(2))
p=a
for line in T[a:b].split('\n'):
    if line.strip():
        k=key(line); cand=[n for n in new if key(n)==k]
        assert set(F[p:p+len(line)])<={'.'}
        if not cand: print('DELETE',line[:60]); E.append((idx[p],idx[p+len(line)]+1,''))
        else:
            assert len(cand)==1,line; E.append((idx[p],idx[p+len(line)-1]+1,cand[0]))
    p+=len(line)+1
R=[]
for s,e,n in sorted(E,key=lambda x:-x[0]):
    if e>s: R.append({"deleteContentRange":{"range":{"startIndex":s,"endIndex":e}}})
    if n:
        R.append({"insertText":{"location":{"index":s},"text":n}})
        for m in re.finditer(r'\b(?:On|Off)(?:(?: and |/)(?:On|Off))?(?:(?:-| )states?\b)?',n):
            p=s+u(n[:m.start()]); R.append({"updateTextStyle":{"range":{"startIndex":p,"endIndex":p+u(m.group(0))},"textStyle":{"italic":True},"fields":"italic"}})
        for m in re.finditer(r'(?<=gain )c(?![A-Za-z])|(?<=into )z(?= units)',n):
            p=s+u(n[:m.start()]); R.append({"updateTextStyle":{"range":{"startIndex":p,"endIndex":p+1},"textStyle":{"italic":True},"fields":"italic"}})
json.dump(R,open(S+'req_cm.json','w'),ensure_ascii=False)
for s,e,n in sorted(E): print(s,e,repr(n[:100]))
print(len(R))
