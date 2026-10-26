import json
T=open('doc_full.txt').read(); idx=json.load(open('doc_idx.json'))
def u16(s): return len(s.encode('utf-16-le'))//2
E=[]  # (anchor_old, new) replacements; anchor must be unique
def rep(old,new,ctx=None):
    hay=ctx if ctx else old
    assert T.count(hay)==1,(hay,T.count(hay))
    k=T.index(hay)+hay.index(old)
    E.append((idx[k], idx[k+len(old)-1]+u16(old[-1]) if old else idx[k], new, old))
def ins_before(anchor,new):
    assert T.count(anchor)==1,(anchor,T.count(anchor))
    k=T.index(anchor); E.append((idx[k],idx[k],new,''))
def ins_after(anchor,new):
    assert T.count(anchor)==1,(anchor,T.count(anchor))
    k=T.index(anchor)+len(anchor)-1; p=idx[k]+u16(T[k]); E.append((p,p,new,''))

# Intro
ins_after('make this activity stable (Amit & Brunel, 1997; Wang, 1999, 2001; Compte et al., 2000).',
 ' Consistent with these models, blocking NMDA receptors markedly reduces persistent firing in monkey prefrontal cortex (Wang et al., 2013).')
rep('(Mongillo et al., 2008; Stokes, 2015).',
 '(Mongillo et al., 2008; Stokes, 2015; Lundqvist et al., 2016; Mi et al., 2017; Miller et al., 2018; Masse et al., 2019; Panichello et al., 2024), and hidden memories can be reactivated after periods without decodable activity (Rose et al., 2016; Wolff et al., 2017). The two accounts rely on traces with very different time scales. NMDA receptor currents decay within tens to a few hundred milliseconds (Vicini et al., 1998), whereas synaptic facilitation decays over hundreds of milliseconds in cortical slices (Markram et al., 1998; Hempel et al., 2000; Wang et al., 2006) and is modelled with effective time constants of about 1.5 s (Mongillo et al., 2008; Mi et al., 2017).')
ins_after('in prefrontal cortex fluctuates within a trial',' (Panichello et al., 2024; Tao and Libedinsky, 2024; Mozumder et al., 2026)')
rep('drop and silences the network','drop and silence the network')
rep('On and Off states without memory loss. Our reanalyses','On and Off states while keeping the memory in at least 98% of trials, as required by the fit. Our reanalyses')
ins_after('the NMDA model predicts that errors grow with the time spent Off, as we found in the recordings',
 ". Fitted to the monkeys' performance, the decline in accuracy with Off time points to a trace of about 120 ms, which matches NMDA kinetics rather than facilitation. Panichello et al. (2024) took intermittent decoding as evidence that persistent activity is not sufficient for working memory. Our results show that this evidence has a simpler explanation within the standard persistent-activity account.")
# Results
rep('(Fig 2, cite sup)','(Fig. 2; Supplementary Fig. 1).')
rep('facilitation decays over ~1.5 s (Mongillo et al., 2008; Mi et al., 2017), as long as the whole memory period',
    'facilitation decays over 0.5–1.5 s (Markram et al., 1998; Hempel et al., 2000; Wang et al., 2006; Mongillo et al., 2008; Mi et al., 2017), up to the length of the whole memory period')
rep('NMDA model (supp).','NMDA model (Supplementary Fig. 2).')
rep('decayed too far for the cued pool to restart,','decayed too far for the cued pool to restart (about 2.5 τN),')
rep('whereas facilitation predicts no such relation.','whereas facilitation predicts no such relation for silences shorter than its own decay, which here spans the whole delay.')
rep('(β per SD of total Off time 0.48, p = 1×10⁻⁴, and 0.52, p = 0.005)','(total Off time, β = 0.48, p = 1×10⁻⁴, and β = 0.52, p = 0.005; logistic mixed model, Methods)')
# Discussion
rep('the model reproduced the number and length of On and Off states without memory loss.','the model reproduced the number and length of On and Off states while keeping the memory in 98% of trials, as required by the fit.')
ins_after('high when attention is deployed on the memorandum and drops during lapses.',
 ' Synchronous On and Off population states, whose timing is modulated by spatial attention, have been recorded directly in macaque visual cortex (Engel et al., 2016).')
rep('which are slowest for receptors containing the GluN2B (NR2B) subunit (Vicini et al., 1998)',
    'from about 50 ms for receptors containing the GluN2A subunit to several hundred milliseconds for those containing the GluN2B (NR2B) subunit (Vicini et al., 1998)')
# Methods
rep('In the single-dip test, each trial had one dip','To measure how long each model can hold the memory through a single silence (single-dip test, Fig. 3a), we simulated trials with only one dip')
rep('but a strong attentional input during the response period can restart the pool with more NMDA gating and recover the correct response with virtually no memory loss',
    'and a strong attentional input, modelling the response period, restarts the pool with more NMDA gating and so recovers the correct response')
# References
refs=[('Fiebelkorn IC, Kastner S (2019)','Engel TA, Steinmetz NA, Gieselmann MA, Thiele A, Moore T, Boahen K (2016). Selective modulation of cortical state during spatial attention. Science 354:1140–1144.\n'),
('Helfrich RF et al. (2018)','Hempel CM, Hartman KH, Wang XJ, Turrigiano GG, Nelson SB (2000). Multiple forms of short-term plasticity at excitatory synapses in rat medial prefrontal cortex. J Neurophysiol 83:3031–3041.\n'),
('Mi Y, Katkov M, Tsodyks M (2017)','Lundqvist M, Rose J, Herman P, Brincat SL, Buschman TJ, Miller EK (2016). Gamma and beta bursts underlie working memory. Neuron 90:152–164.\nMarkram H, Wang Y, Tsodyks M (1998). Differential signaling via the same axon of neocortical pyramidal neurons. Proc Natl Acad Sci USA 95:5323–5328.\nMasse NY, Yang GR, Song HF, Wang XJ, Freedman DJ (2019). Circuit mechanisms for the maintenance and manipulation of information in working memory. Nat Neurosci 22:1159–1167.\n'),
('Mongillo G, Barak O, Tsodyks M (2008)','Miller EK, Lundqvist M, Bastos AM (2018). Working Memory 2.0. Neuron 100:463–475.\n'),
('Stokes MG (2015)','Rose NS, LaRocque JJ, Riggall AC, Gosseries O, Starrett MJ, Meyering EE, Postle BR (2016). Reactivation of latent working memories with transcranial magnetic stimulation. Science 354:1136–1139.\n'),
('Wong KF, Wang XJ (2006)','Wang Y, Markram H, Goodman PH, Berger TK, Ma J, Goldman-Rakic PS (2006). Heterogeneity in the pyramidal network of the medial prefrontal cortex. Nat Neurosci 9:534–542.\nWolff MJ, Jochim J, Akyürek EG, Stokes MG (2017). Dynamic hidden states underlying working-memory-guided behavior. Nat Neurosci 20:864–871.\n')]
for a,n in refs:
    ins_before('\n'+a,'')  # check uniqueness of paragraph start
    E.pop()
    k=T.index('\n'+a)+1; E.append((idx[k],idx[k],n,''))
# Supp Fig 2 caption at end of doc (after Supp Fig 1 caption)
cap2="Supplementary Figure 2. The STP model fits the On/Off statistics about as well as the NMDA model."
cap2b=cap2+" As Fig. 2, for the STP model, in which short-term facilitation replaces NMDA gating (Methods). The model was fitted to the decoder using all neurons only (grey band), and the other six decoded populations are predictions obtained by changing only the read-out gain and noise. Black circles, data (mean ± 2 SEM over 25 sessions). Grey dots, single sessions. Cyan and red circles, model Off and On statistics."
endk=T.rindex('for the fitted model).')+len('for the fitted model).')-1
pend=idx[endk]+1
E.append((pend,pend,'\n\n'+cap2b,''))
E.sort(key=lambda e:-e[0])
reqs=[]
for s,e,new,old in E:
    if e>s: reqs.append({"deleteContentRange":{"range":{"startIndex":s,"endIndex":e}}})
    if new: reqs.append({"insertText":{"location":{"index":s},"text":new}})
json.dump(reqs,open('req_comments.json','w'),ensure_ascii=False)
print(len(E),'edits', len(reqs),'requests; supp2 at',pend)
# image paragraph index after all edits: pend+1 (the empty paragraph), since all other edits are before pend
print('img index', pend+1, 'cap2 title range', pend+2, pend+2+u16(cap2))
