import json,re
P=json.load(open('paras.json'))
c=json.load(open('cur.json'))
KEYS=('bold','italic','baselineOffset')
def chars(k):
    out=[]
    for t,s in P[k]['runs']:
        st={kk:s[kk] for kk in KEYS if kk in s and s[kk] not in (False,'NONE')}
        for ch in t: out.append((ch,st))
    assert out[-1][0]=='\n'; return out[:-1]
def txt(cs): return ''.join(ch for ch,_ in cs)
def plain(s):
    """new text: italic On/Off, **x** bold, _{x} subscript"""
    out=[];i=0
    for m in re.finditer(r'\*\*(.+?)\*\*|_\{(.+?)\}|([^*_]+|[*_])',s):
        if m.group(1): out+= [(ch,{'bold':True}) for ch in m.group(1)]
        elif m.group(2): out+= [(ch,{'baselineOffset':'SUBSCRIPT'}) for ch in m.group(2)]
        else:
            seg=m.group(3); st=[{} for _ in seg]
            for mm in re.finditer(r'\b(On|Off)\b',seg):
                for j in range(mm.start(),mm.end()): st[j]={'italic':True}
            out+=list(zip(seg,st))
    return out
def rep(cs,old,new):
    t=txt(cs); assert t.count(old)==1,(old[:50],t.count(old)); i=t.index(old)
    return cs[:i]+plain(new)+cs[i+len(old):]
def cut(cs,a,b=None):
    t=txt(cs); i=0 if a is None else t.index(a); j=len(t) if b is None else t.index(b)
    return cs[i:j]
def strip(cs):
    while cs and cs[-1][0]==' ': cs=cs[:-1]
    while cs and cs[0][0]==' ': cs=cs[1:]
    return cs
B=[]   # (chars, kind) kind: H, J, C
def H(t): B.append((plain(t),'H'))
def J(cs): B.append((strip(cs),'J'))
def C(cs): B.append((strip(cs),'C'))

# --- Data, decoders, labelling
p85=chars(85)
H('Data');           J(cut(p85,None,'For each time bin'))
H('Decoders and decoded populations'); J(cut(p85,'For each time bin'))
p98=chars(98)
p98=rep(p98,'States were labelled exactly as in Panichello et al. (2024), so that model and data are compared with identical definitions.','States were labelled exactly as in Panichello et al. (2024), in the data and in both models, so that they are compared with identical definitions.')
p98=rep(p98,'in pure noise η.','in the null (label shuffles in the data and the spiking network, read-out noise η in the rate model; see below).')
H('On/Off labelling and statistics'); J(p98)
# --- rate model
H('NMDA rate model'); J(chars(57))
for k in range(58,63): C(chars(k))
p63=chars(63)
J(cut(p63,None,'The memory was scored correct'))
J(cut(p63,'The memory was scored correct','Our main modification'))
H('Fluctuating attentional input'); J(cut(p63,'Our main modification'))
# --- read-out
H('Read-out of the rate model')
J(chars(90)); J(chars(92))
p94=chars(94); p94=rep(p94,'In Panichello et al. (2024), z was computed in 100 ms windows stepped by 10 ms, so consecutive points','Because the windows overlap, consecutive points'); J(p94)
J(chars(96))
H('Validation of the read-out on synthetic data')
p100=chars(100); p100=rep(p100,'Our read-out approximates a real decoder by a simple formula (z(t) = c Δr(t) + η(t)), so we tested','Our read-out approximates a real decoder by a simple formula, so we tested'); J(p100)
H('Fitting'); J(chars(103))
H('STP model'); p66=chars(66)
p66=rep(p66,' and held the memory in nearly all trials.','.'); J(p66)
H('Simulations of memory against Off time'); J(chars(106))
H('Behavioural analysis'); p87=chars(87)
p87=rep(p87,' For the fit of τN, trials were pooled over sessions and binned by Off time in 90 ms bins (one tenth of the delay window).',''); J(p87)
H('NMDA time constant fitted to performance'); p108=chars(108)
p108=rep(p108,'in the ten Off-time bins','in ten 90 ms bins of Off time (one tenth of the delay window)'); J(p108)
# --- spiking network
H('Spiking network'); p69=chars(69)
p69=rep(p69,' (Gaussian footprint, σ = 18°),',',')
p69=rep(p69,' that dropped at random times and briefly overshot when it recovered.',' (see below).')
J(p69)
J(chars(70)); C(chars(71)); J(chars(72)); C(chars(73)); J(chars(74)); C(chars(75)); J(chars(76)); C(chars(77))
p78=chars(78); J(cut(p78,None,'The external input rate')); J(chars(80))
H('Attentional input of the spiking network')
J(cut(p78,'The external input rate'))
p79=chars(79); p79=rep(p79,'so the overshoot decayed with a time constant of 150 ms.','so the overshoot was made slow.'); J(p79)
# --- population analyses
H('Population analyses of On and Off states in spiking network and data')
p111=chars(111); p112=chars(112)
J(cut(p111,None,'Memory tuning'))
J(plain('For the analyses of Fig. 4c–e, the neurons of each session were split at random into two halves. On and Off states were labelled from all neurons of one half, with the decoder and criteria above, and all other analyses used the neurons of the other half, with firing rates in 100 ms windows z-scored across trials.'))
t=cut(p111,'Memory tuning'); t=rep(t,' Firing rates in 100 ms windows were z-scored across trials, the recorded neurons were split at random into two halves, and On and Off states were labelled with the decoder of one half.','')
J(plain('**Single-neuron tuning.** ')+t)
t=cut(p112,'Delay activity (z-scored'); t=rep(t,'Delay activity (z-scored firing rates in 100 ms windows, 500–1,400 ms after cue onset) was averaged for each of the eight cue locations,','Delay activity (500–1,400 ms after cue onset) was averaged for each of the eight cue locations, in the recordings and in the network,')
J(plain('**PCA.** ')+t)
J(plain('**Shape similarity.** ')+chars(113))
json.dump([[[[ch,st] for ch,st in cs],k] for cs,k in B],open('meth_new.json','w'),ensure_ascii=False)
for cs,k in B: print(k, txt(cs)[:110])
