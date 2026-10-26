import json,re,difflib
S='/tmp/claude-0/-home-user/51e93236-a4df-5b7b-99f9-872ebc6e17b8/scratchpad/'
c=json.load(open(S+'cur.json'));T=c['T'];F=c['F'];idx=c['idx']
def u(s): return len(s.encode('utf-16-le'))//2
tok=lambda s: re.findall(r'\s+|[^\s]+',s)
E=[]
def rep(old,new):
    assert T.count(old)==1,(old[:60],T.count(old)); a0=T.index(old)
    assert set(F[a0:a0+len(old)])<={'.'},old[:50]
    ot,nt=tok(old),tok(new); pos=[0]
    for x in ot: pos.append(pos[-1]+len(x))
    for op,i1,i2,j1,j2 in difflib.SequenceMatcher(None,ot,nt,autojunk=False).get_opcodes():
        if op=='equal': continue
        s=a0+pos[i1]; e=a0+pos[i2]
        ds=idx[s]; de=idx[e-1]+1 if e>s else ds
        E.append((ds,de,''.join(nt[j1:j2])))
rep("Read out this way, the model alternates between On and Off states (Fig. 1a, bottom). Both length distributions peak near 100 ms and have long tails (Fig. 1b), qualitatively like those in PFC recordings (Panichello et al., 2024).",
    "Read out this way, the model alternates between On and Off states (Fig. 1a, bottom, mid gain). Both length distributions peak near 100 ms and have long tails (Fig. 1b), qualitatively like those in PFC recordings (Panichello et al., 2024). The On/Off statistics depend strongly on the gain. With the same trials and noise, halving the gain gave more Off states and almost no On states, whereas doubling it gave fewer and shorter Off states and more On states (Fig. 1a, bottom, low and high gain, and Fig. 1b).")
rep("Bottom: decoding z score, with On (red) and Off (cyan) states. Dashed lines: On threshold (z = 1.96) and Off threshold (z = 0.84), as in Panichello et al. (2024). b, Length distributions of Off (top) and On (bottom) states in the model.",
    "Bottom: decoding z score of the same trial read out with the calibrated gain c (mid gain, black), twice this gain (high gain, dark grey) and half of it (low gain, light grey), with the same read-out noise. On (red) and Off (cyan) states are those of the mid-gain read-out. Dashed lines: On threshold (z = 1.96) and Off threshold (z = 0.84), as in Panichello et al. (2024). b, Length distributions of Off (top) and On (bottom) states in the model at the three gains (dark, high; medium, mid; light, low; 2,112 simulated trials), as fractions of states in 40 ms bins, with the mean number of states per trial.")
R=[]
for s,e,n in sorted(E,key=lambda x:-x[0]):
    if e>s: R.append({"deleteContentRange":{"range":{"startIndex":s,"endIndex":e}}})
    if n:
        R.append({"insertText":{"location":{"index":s},"text":n}})
        for m in re.finditer(r'\b(?:On|Off)(?:(?: and |/)(?:On|Off))?(?:(?:-| )states?\b)?',n):
            p=s+u(n[:m.start()]); R.append({"updateTextStyle":{"range":{"startIndex":p,"endIndex":p+u(m.group(0))},"textStyle":{"italic":True},"fields":"italic"}})
        for m in re.finditer(r'(?<=gain )c(?![A-Za-z])',n):   # the gain symbol
            p=s+u(n[:m.start()]); R.append({"updateTextStyle":{"range":{"startIndex":p,"endIndex":p+1},"textStyle":{"italic":True},"fields":"italic"}})
json.dump(R,open(S+'req_f1.json','w'))
for s,e,n in E: print(s,e,repr(n))
