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
rep("We modeled working memory","We modelled working memory")
rep("The model reproduces quantitatively the number","The model quantitatively reproduces the number")
rep("reads out the memory and the attractor model predicts","reads out the memory, and the attractor model predicts")
rep("first second after the cue).  With brief dips","first second after the cue). With brief dips")
rep("Importantly, when synthetic neurons driven","When synthetic neurons driven")
rep("Fixing the noise dynamics for this particular simulated trial, halving the gain","Keeping the dynamics and noise fixed, halving the gain")
rep("(grey band), the other six decoded populations","(grey band). The other six decoded populations")
rep("The model thus predicts quantitatively the shift","The model thus quantitatively predicts the shift")
rep("not on how often it went Off: frequent Off states are harmless","not on how often it went Off. Frequent Off states are harmless")
R=[]
for s,e,n in sorted(E,key=lambda x:-x[0]):
    if e>s: R.append({"deleteContentRange":{"range":{"startIndex":s,"endIndex":e}}})
    if n:
        R.append({"insertText":{"location":{"index":s},"text":n}})
        for m in re.finditer(r'\b(?:On|Off)(?:(?: and |/)(?:On|Off))?(?:(?:-| )states?\b)?',n):
            p=s+u(n[:m.start()]); R.append({"updateTextStyle":{"range":{"startIndex":p,"endIndex":p+u(m.group(0))},"textStyle":{"italic":True},"fields":"italic"}})
        for m in re.finditer(r'(?<=gain )c(?![A-Za-z])|(?<=into )z(?= units)',n):
            p=s+u(n[:m.start()]); R.append({"updateTextStyle":{"range":{"startIndex":p,"endIndex":p+1},"textStyle":{"italic":True},"fields":"italic"}})
json.dump(R,open(S+'req_cm2.json','w'),ensure_ascii=False)
for s,e,n in sorted(E): print(s,e,repr(n[:100]))
print(len(R))
