import json
T=open('doc_full2.txt').read(); m=json.load(open('doc_idx2.json')); idx=m['idx']
def u16(s): return len(s.encode('utf-16-le'))//2
E=[]
def rep(old,new):
    assert T.count(old)==1,(old[:40],T.count(old)); k=T.index(old)
    E.append((idx[k], idx[k+len(old)-1]+u16(old[-1]), new))
def ins_para_before(start,new):
    a='\n'+start; assert T.count(a)==1,start; k=T.index(a)+1; E.append((idx[k],idx[k],new))
rep('The two accounts rely on traces with very different time scales. NMDA receptor currents decay within tens to a few hundred milliseconds (Vicini et al., 1998), whereas synaptic facilitation decays over hundreds of milliseconds in cortical slices (Markram et al., 1998; Hempel et al., 2000; Wang et al., 2006) and is modelled with effective time constants of about 1.5 s (Mongillo et al., 2008; Mi et al., 2017).',
 'The two accounts rely on traces with very different time scales. NMDA receptor currents decay with time constants of about 50 ms for receptors containing the GluN2A subunit and about 300 ms for those containing GluN2B (Vicini et al., 1998), and models of persistent activity typically use about 100 ms (Wang, 1999; Compte et al., 2000). Facilitation at excitatory cortical synapses decays with a time constant of about 0.5 s in slices (530 ms in somatosensory cortex, Markram et al., 1998; similar values in prefrontal cortex, Hempel et al., 2000; Wang et al., 2006), and activity-silent models use 0.6 to 1.5 s, the longer values partly absorbing slower synaptic augmentation (Barak and Tsodyks, 2007; Mongillo et al., 2008; Mi et al., 2017).')
rep('NMDA gating decays within ~150 ms, whereas facilitation decays over 0.5–1.5 s (Markram et al., 1998; Hempel et al., 2000; Wang et al., 2006; Mongillo et al., 2008; Mi et al., 2017), up to the length of the whole memory period',
    'In our model, NMDA gating decays with a time constant of 150 ms, whereas facilitation in activity-silent models decays over up to 1.5 s (Mongillo et al., 2008), about the length of the whole memory period')
rep('There several studies suporting this. First, Synchronous On and Off population states, whose timing is modulated by spatial attention, have been recorded directly in macaque visual cortex (Engel et al., 2016). Second, This links intermittent decoding to evidence that sustained attention samples rhythmically rather than continuously.',
    'Several lines of evidence support this interpretation. First, synchronous On and Off population states, whose timing is modulated by spatial attention, have been recorded directly in macaque visual cortex (Engel et al., 2016). Second, sustained attention samples the environment rhythmically rather than continuously.')
rep('the information the subject actively maintains.','the information the subject actively maintains (Kiyonaga et al., 2017; Pascucci et al., 2023).')
ins_para_before('Barbosa J et al. (2020)','Barak O, Tsodyks M (2007). Persistent activity in neural networks with dynamic synapses. PLoS Comput Biol 3:e35.\n')
ins_para_before('Landau AN, Fries P (2012)','Kiyonaga A, Scimeca JM, Bliss DP, Whitney D (2017). Serial dependence across perception, attention, and memory. Trends Cogn Sci 21:493–497.\n')
ins_para_before('Rose NS','Pascucci D, Tanrikulu ÖD, Ozkirli A, Houborg C, Ceylan G, Zerr P, Rafiei M, Kristjánsson Á (2023). Serial dependence in visual perception: a review. J Vis 23(1):9.\n')
E.sort(key=lambda e:-e[0]); reqs=[]
for s,e,new in E:
    if e>s: reqs.append({"deleteContentRange":{"range":{"startIndex":s,"endIndex":e}}})
    reqs.append({"insertText":{"location":{"index":s},"text":new}})
print(json.dumps(reqs,ensure_ascii=True))
