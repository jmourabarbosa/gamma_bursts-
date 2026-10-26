import json,re
c=json.load(open('cur.json'));T=c['T'];F=c['F'];idx=c['idx']
def pos(anchor,after=True):
    assert T.count(anchor)==1,(anchor,T.count(anchor)); i=T.index(anchor)
    j=i+len(anchor) if after else i
    assert set(F[i:i+len(anchor)])<={'.'},anchor
    return idx[j-1]+1 if after else idx[j]
def parse(s):
    # _{x} -> subscript, ^{x} -> superscript, *x* -> italic
    out='';sty=[]
    for m in re.finditer(r'_\{([^}]*)\}|\^\{([^}]*)\}|\*([^*]*)\*|([^_^*]+|[_^*])',s):
        if m.group(1) is not None: sty.append((len(out),len(m.group(1)),'SUB')); out+=m.group(1)
        elif m.group(2) is not None: sty.append((len(out),len(m.group(2)),'SUP')); out+=m.group(2)
        elif m.group(3) is not None: sty.append((len(out),len(m.group(3)),'IT')); out+=m.group(3)
        else: out+=m.group(4)
    return out,sty
def u16(s): return len(s.encode('utf-16-le'))//2
ops=[]
def ins(at,s):
    txt,sty=parse(s); r=[{"insertText":{"location":{"index":at},"text":txt}}]
    for a,n,k in sty:
        st=at+u16(txt[:a]); en=st+u16(txt[a:a+n])
        if k=='IT': r.append({"updateTextStyle":{"range":{"startIndex":st,"endIndex":en},"textStyle":{"italic":True},"fields":"italic"}})
        else: r.append({"updateTextStyle":{"range":{"startIndex":st,"endIndex":en},"textStyle":{"baselineOffset":'SUBSCRIPT' if k=='SUB' else 'SUPERSCRIPT'},"fields":"baselineOffset"}})
    ops.append((at,r))
def dele(a,b): ops.append((a,[{"deleteContentRange":{"range":{"startIndex":a,"endIndex":b}}}]))

# Comment 1: discrete vs continuous attractor
old=' and leaves open whether its mechanism works in a network of spiking neurons,'
a=pos(old,False); b=a+len(' and leaves open whether its mechanism works in')
assert T[T.index(old):T.index(old)+b-a]==' and leaves open whether its mechanism works in'
ins(b,', with one discrete attractor for each memory. It leaves open whether its mechanism works in a continuous attractor, which can hold any location and along which the memory could drift during silences, and in')
dele(a,b)

# Comment 3: dependence on decoded neurons
ins(pos('Read out like the recordings, the network therefore alternated between On and Off states (Fig. 4a, bottom). Decoders',False)+len('Read out like the recordings, the network therefore alternated between On and Off states (Fig. 4a, bottom). '),
    'As in the recordings and the rate model (Figs. 1b and 2), *On*/*Off* statistics depended on which neurons were decoded. ')
# Comment 4: persistent code in Off states
ins(pos('remained above its level before the cue (Fig. 4e).'),
    ' Even when the decoder labelled the memory *Off*, population activity therefore still held a weak code for the cued location. *Off states* defined by a decoder thus overestimate how often the memory leaves spiking activity, as the dependence of *On*/*Off* statistics on the decoded population also suggests (Fig. 2).')
# Caption e
ins(pos('(data, 25 sessions, black; network, 10 sessions, blue).'),
    ' Alignment above Pre during *Off* moments shows that population activity still coded the cue while decoding was *Off*.')
# Comment 2: equations
eq=('\nAs in Compte et al. (2000), the membrane potential V of each neuron obeys\n'
'C_{m} dV/dt = −g_{L} (V − E_{L}) − I_{ext} − I_{NMDA} − I_{GABA} + I_{cue},\n'
'with C_{m} = 0.5 nF and g_{L} = 25 nS for pyramidal neurons, C_{m} = 0.2 nF and g_{L} = 20 nS for interneurons, E_{L} = −70 mV, a firing threshold of −50 mV, a reset potential of −60 mV and refractory periods of 2 ms (pyramidal neurons) and 1 ms (interneurons). The synaptic currents are\n'
'I_{ext} = g_{ext} s_{AMPA} (V − E_{E}),   I_{NMDA} = g_{NMDA} B(V) (V − E_{E}) Σ_{j} W_{ij} s_{NMDA,j},   I_{GABA} = g_{GABA} (V − E_{I}) Σ_{j} s_{GABA,j},\n'
'with E_{E} = 0 mV, E_{I} = −70 mV and B(V) = 1/(1 + [Mg^{2+}] exp(−0.062 V)/3.57), [Mg^{2+}] = 1 mM. The AMPA and GABA_{A} gating variables increase by 1 at each presynaptic spike and decay with τ_{AMPA} = 2 ms and τ_{GABA} = 10 ms. The NMDA gating of each pyramidal neuron obeys\n'
'dx/dt = −x/τ_{x} + Σ_{k} δ(t − t_{k}),   ds_{NMDA}/dt = −s_{NMDA}/τ_{NMDA} + α x (1 − s_{NMDA}),\n'
'with τ_{x} = 2 ms, α = 0.5 ms^{−1} and τ_{NMDA} = 150 ms. External Poisson input arrives at 1.8 kHz, with g_{ext} = 3.1 nS onto pyramidal neurons and 2.38 nS onto interneurons. Recurrent conductances are g_{NMDA} = 0.33 nS between pyramidal neurons and 0.292 nS from pyramidal neurons to interneurons, and g_{GABA} = 1.336 nS onto pyramidal neurons and 1.024 nS onto interneurons. Between pyramidal neurons with preferred locations θ_{i} and θ_{j},\n'
'W_{ij} = J^{−} + (J^{+} − J^{−}) exp(−(θ_{i} − θ_{j})^{2}/2σ^{2}),\n'
'normalised to a mean of 1, with σ = 18° and J^{+} = 3.0. For heterogeneous tuning, pyramidal neuron i received a fraction h_{i} of this input (h_{i} uniform between 0 and 1) through 1 + 0.25 cos(θ_{i} − θ_{j}) instead of W_{ij}. The external input rate to pyramidal neurons was multiplied by m_{eff}(t) = m(t) + A [m(t) − m̄(t)], where m(t) is the attentional signal, m̄(t) follows m(t) down into each dip and recovers with a time constant of 150 ms, and A = 1.')
ins(pos('briefly overshot when it recovered. Equations were integrated with Euler steps of 0.1 ms.'),eq)
reqs=[]
for at,r in sorted(ops,key=lambda o:-o[0]): reqs+=r
json.dump(reqs,open('req_cm3.json','w'),ensure_ascii=False)
print(len(reqs),c['rev'])
