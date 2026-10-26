import json,re,difflib
S='/tmp/claude-0/-home-user/51e93236-a4df-5b7b-99f9-872ebc6e17b8/scratchpad/'
c=json.load(open(S+'cur.json'));T=c['T'];F=c['F'];idx=c['idx']
assert set(F)<= {'.'}
def u(s): return len(s.encode('utf-16-le'))//2
tok=lambda s: re.findall(r'\s+|[^\s]+',s)
E=[]
def rep(old,new):
    assert T.count(old)==1,(old[:60],T.count(old)); a0=T.index(old)
    ot,nt=tok(old),tok(new); pos=[0]
    for x in ot: pos.append(pos[-1]+len(x))
    for op,i1,i2,j1,j2 in difflib.SequenceMatcher(None,ot,nt,autojunk=False).get_opcodes():
        if op=='equal': continue
        s=a0+pos[i1]; e=a0+pos[i2]
        ds=idx[s]; de=idx[e-1]+1 if e>s else ds
        E.append((ds,de,''.join(nt[j1:j2])))
# Abstract
rep("This view has been challenged by large-scale prefrontal recordings in which the remembered item is decoded only intermittently on single trials, suggesting that memories survive silent periods through synaptic facilitation.",
    "This view has been challenged by large-scale prefrontal recordings in which the remembered item is decoded only intermittently on single trials, with Off states in which it cannot be decoded, suggesting that memories survive these periods through synaptic facilitation.")
rep("the model reproduces the number and duration of silent states and predicts how they change with the decoded neurons. We found that silent states are more frequent in less informative samples, validating this prediction and suggesting that measured silences overestimate how often memory leaves spiking activity.",
    "the model reproduces the number and duration of Off states and predicts how they change with the decoded neurons. We found that Off states are more frequent in less informative samples, validating this prediction and suggesting that measured Off states overestimate how often memory leaves spiking activity.")
rep("monkeys made more errors when silent states were longer and their performance","monkeys made more errors when Off states were longer and their performance")
# Intro
rep("The advance of large-scale, intracranial recordings have made it possible","The advance of large-scale, intracranial recordings has made it possible")
rep("However, fluctuations in a common input, which we interpret as a fluctuating attentional signal, briefly drop and silence the network.",
    "The synchrony of Off states across prefrontal and parietal cortex (Mozumder et al., 2026) points to a common input as their source, and attention, which waxes and wanes on sub-second time scales (Fiebelkorn and Kastner, 2019), is a natural candidate. In our model, brief drops in such an attentional input silence the network.")
rep("We read out from the model with the same decoding","We read out the model with the same decoding")
# Results
rep("This matters for interpreting Off states as evidence for activity-silent memory: a decoder trained","This matters for interpreting Off states as evidence for activity-silent memory. A decoder trained")
rep("Assuming that they are always wrong, or guess correctly up to half of the time, give 102–129 ms","Assuming that they are always wrong, or guess correctly up to half of the time, gave 102–129 ms")
# Discussion
rep("the model reproduced the number and length of On and Off states while keeping the memory. Unlike",
    "the model reproduced the number and length of On and Off states while keeping the memory in persistent activity. Unlike")
rep("especially at attended locations (Busch and VanRullen, 2010) or about 4 Hz (Landau and Fries, 2012).",
    "especially at attended locations (Busch and VanRullen, 2010), and when attention is divided between two locations, detection at each fluctuates at about 4 Hz (Landau and Fries, 2012).")
rep("A memory held by facilitation decaying on a time scale of 1.5 s should hardly","A memory held by facilitation decaying on a time scale of the order of a second should hardly")
# Methods
rep("The standard deviation of pre-cue z was 1.06, close to","The standard deviation of pre-cue z, averaged over the 25 sessions, was 1.06 for each decoded population, close to")
rep("would ignore the read-out noise inherent to a recording subsampling.","would ignore the read-out noise inherent to recording only a small sample of the neurons in the circuit.")
rep("within the range measured in cortex (Vicini et al., 1998; Wang et al., 2013).","within the range measured in cortex (Vicini et al., 1998; Wang et al., 2008, 2013).")
# move "Simulations of memory against Off time" section before "NMDA time constant fitted to performance"
sim_h='\nSimulations of memory against Off time\n'; a=T.index(sim_h)+1
b=T.index('Off time was computed as in the data.',a)+len('Off time was computed as in the data.')+1
block=T[a:b]
E.append((idx[a],idx[b-1]+1,''))
nm=T.index('\nNMDA time constant fitted to performance\n')+1
MOVE=(idx[nm],block)
R=[]
for s,e,n in sorted(E+[(MOVE[0],MOVE[0],MOVE[1])],key=lambda x:-x[0]):
    if e>s: R.append({"deleteContentRange":{"range":{"startIndex":s,"endIndex":e}}})
    if n:
        R.append({"insertText":{"location":{"index":s},"text":n}})
        for m in re.finditer(r'\b(?:On|Off)(?:(?: and |/)(?:On|Off))?(?:(?:-| )states?\b)?',n):
            if n[m.start():].startswith('On the'): continue
            p=s+u(n[:m.start()]); R.append({"updateTextStyle":{"range":{"startIndex":p,"endIndex":p+u(m.group(0))},"textStyle":{"italic":True},"fields":"italic"}})
json.dump(R,open(S+'req_review.json','w'))
print(len(E),len(R),c['rev']); print('move block starts:',repr(block[:60]),'...',repr(block[-50:]))
