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
def ins_after(anchor,text):   # insert text right after the unique anchor string
    hits=[m.end() for m in re.finditer(re.escape(anchor),T)]; assert len(hits)==1,anchor[:50]
    p=hits[0]; E.append((idx[p-1]+1,idx[p-1]+1,text))
# --- comments
rep("Finally, we replicate the main findings in a ring-attractor network of spiking neurons with NMDA synapsis, demonstrating",
    "Finally, we replicate the main findings in a biophysical ring-attractor network of spiking neurons with NMDA-mediated recurrent excitation, demonstrating")
rep("Rigorously testing predictions of these models requires single-trial resolution that only recently became possible with the advance of large-scale, intracranial recordings (Urai).",
    "Rigorously testing predictions of these models requires single-trial resolution (Lundqvist et al., 2018), which only recently became possible with the advance of large-scale, intracranial recordings (Williams and Linderman, 2021; Urai et al., 2022).")
rep("modulated by a global signal and  replicate the main results of Panichello et al. Previous",
    "modulated by a global signal and replicate the main results of Panichello et al. (2024). Previous")
rep("using variation of the classic two-pool attractor network (Wang, 2002;","using a variation of the classic two-pool attractor model (Wang, 2002;")
rep("Memory was scored on the network, without read-out","Memory was scored on the model, without read-out")
rep("This is not surprising, since in our model Off are driven by attentional lapses and STP time scale (tau_f = 1500ms) is longer than NMDA (tau_N = 150ms).",
    "This is expected, because in both models Off states are driven by lapses of attention, which facilitation (τf = 1.5 s) bridges even more easily than NMDA gating (τN = 150 ms).")
# --- new Results section, into the empty paragraph after its heading
R1=("The rate model describes the activity of two populations and leaves open whether its mechanism works in a network of spiking neurons, where NMDA currents, recurrent inhibition and noise interact at the level of single cells. We therefore turned to the spiking network of Compte et al. (2000), a ring of 2,048 excitatory and 512 inhibitory integrate-and-fire neurons in which recurrent excitation is mediated by NMDA receptors and a localized bump of persistent activity holds the cued location. We used this network with minimal changes (Methods). The NMDA decay time was set to 150 ms, as in the rate model, the neurons were made heterogeneous, and all excitatory neurons received the same kind of fluctuating attentional input as the rate model. The network was not fitted to the data. We sampled 330 of its neurons, the average number of units recorded per session, and read them out with the same decoder, null and On/Off labelling as the recordings."
"\nWith a constant attentional input, the cued location was held by a stable bump of persistent activity, and decoding stayed On throughout the delay (Fig. 4a, first second after the cue). Each drop of the input silenced the bump, and decoding fell to chance. When the input returned, the bump restarted at the cued location, because the cued neurons kept more NMDA gating through the silence than the other neurons, as in the rate model. Read out like the recordings, the network therefore alternated between On and Off states (Fig. 4a, bottom). Without dips, the network held the memory in all trials and showed no Off states. With dips, it held the memory in 83% of trials."
"\nOff states had a similar number and length as in the recordings (3.0 vs 3.3 per trial, median length 98 vs 112 ms; Fig. 4b). On states were longer than in the data (median 202 vs 134 ms), and the network spent less of the delay Off (26% vs 45%). This is expected from its stronger decoding. The simulated neurons were more selective than recorded ones, and the decoding strength of the network was 5.7, against 2.2 in the data, which favours On over Off states (Fig. 2)."
"\nFinally, Panichello et al. (2024) found that neurons lose their memory tuning during Off states and concluded that synaptic changes carry the memory through these periods. We applied their cross-validated analysis to the spiking network. Held-out neurons were strongly tuned to the cued location during On states and not tuned during Off states (Fig. 4c), as in the recordings. Yet, in the network, the memory is held throughout by persistent activity bridged by NMDA currents, with fixed synapses. The loss of tuning during Off states is therefore not evidence for activity-silent storage. Together, these results show that intermittent rate coding, including the loss of tuning in Off states, emerges in a biophysical attractor network without short-term plasticity."
"\n"
"\nFigure 4. Intermittent rate coding in a spiking ring-attractor network with NMDA currents. a, Example trial. Top: spikes of the 2,048 excitatory neurons, sorted by preferred location (counts in 5 ms and 8-neuron bins); grey shading marks the cue (180°). Middle: common input (attention signal) to the excitatory neurons, constant for the first 1,000 ms after the cue and with dips afterwards, each followed by a transient overshoot. The dip-free first second is shown for illustration only; in all other simulated trials, dips occur throughout the delay. Bottom: decoding z score from 330 simulated neurons, with the decoder, null and On (red) and Off (cyan) labelling of the recordings. Dashed lines: On threshold (z = 1.96) and Off threshold (z = 0.84). b, Length distributions of Off (left) and On (right) states (all units, 12 simulated sessions of 480 trials). c, Cross-validated memory tuning of held-out simulated neurons during On (red) and Off (cyan) states, computed as in Panichello et al. (2024) (mean ± s.e.m., 330 neurons).")
p=T.index('ring-attractor network without short-term plasticity\n')+len('ring-attractor network without short-term plasticity\n')
assert T[p]=='\n' and F[p]=='.'
E.append((idx[p],idx[p],R1))
RES=(idx[p],R1)
# --- Methods
M=("\nSpiking network"
"\nWe used the spatial working-memory network of Compte et al. (2000) with minimal changes. It consists of 2,048 pyramidal neurons and 512 interneurons, modelled as leaky integrate-and-fire neurons and arranged on a ring by preferred location. Recurrent excitation is mediated by NMDA receptors with a voltage-dependent Mg²⁺ block and saturating gating. Connections between pyramidal neurons decrease with the difference in their preferred locations (Gaussian footprint, σ = 18°), and all connections involving interneurons are uniform. All neurons receive external excitatory Poisson input through AMPA receptors. Neuronal and synaptic parameters were those of Compte et al. (2000), with four changes. First, the NMDA decay time was 150 ms instead of 100 ms, as in the rate model, and the NMDA conductance between pyramidal neurons was lowered from 0.381 to 0.33 nS so that the network remained bistable with the slower decay. Second, we made the neurons heterogeneous, so that their activity would resemble recorded neurons more closely. Firing thresholds varied across neurons (SD 0.3 mV), and each pyramidal neuron received a random fraction of its recurrent excitation through a broad, weakly tuned footprint instead of the narrow one. To keep the bump stable despite this weaker tuned input, the peak of the narrow footprint was raised from 1.62 to 3.0. Third, the cue was a 50 ms current to the pyramidal neurons around the cued location, as long as in the task and stronger than in Compte et al. (2000) (500 pA), so that it reliably started a bump. Fourth, as in the rate model, the external input to all pyramidal neurons was scaled by a common attentional signal m(t) that dropped at random times and briefly overshot when it recovered. Equations were integrated with Euler steps of 0.1 ms."
"\nThe dip statistics of the spiking network were set by hand rather than fitted, and differ from those of the fitted rate model. Dips started at random times (Poisson, 7 per s instead of 10.2 per s), lasted longer (gamma distribution with the same shape, mean 150 ms instead of 90 ms) and were shallower (50% instead of 60% of the drive), with 10 ms ramps as in the rate model. The spiking network needed longer dips because its decoding was stronger. With 90 ms dips, the 100 ms decoding windows bridged most silences and Off states were rare. Shallower dips were needed because 60% dips lost the memory more often. In both models, the drive briefly overshoots when the input recovers. In the rate model, this overshoot comes from the lagging feed-forward inhibition (τff = 15 ms). In the spiking network, a fast overshoot made all pyramidal neurons fire together, which saturated their NMDA gating and erased the memory, so the overshoot decayed with a time constant of 150 ms."
"\nSessions of 480 trials (60 per cue location, 8 locations) were simulated, each with 500 ms of fixation, a 50 ms cue and a 1,500 ms delay with dips throughout. In each session, 330 neurons (264 pyramidal neurons and 66 interneurons) were sampled at random across the ring and read out exactly as the recordings, with the same windows, decoder, label-shuffle null and On/Off labelling (12 sessions, Fig. 4b). Memory tuning in On and Off states was computed as in Panichello et al. (2024). Firing rates in 100 ms windows were z-scored across trials, the recorded neurons were split at random into two halves, and On and Off states were labelled with the decoder of one half. The delay tuning of each held-out neuron was computed separately for On and Off time points and aligned to its preferred cue, and the same analysis with shuffled cue labels was subtracted to remove the bias of this alignment (10 random splits, 330 neurons of one session).")
q=T.index('the best τN was 129, 118 and 102 ms for G = 0, 1/4 and 1/2.')+len('the best τN was 129, 118 and 102 ms for G = 0, 1/4 and 1/2.')
assert T[q]=='\n'
E.append((idx[q],idx[q],M))
METH=(idx[q],M)
# --- references (new paragraphs in alphabetical order)
ins_after("Gamma and beta bursts underlie working memory. Neuron, 90:152–164.","\nLundqvist, M., Herman, P., and Miller, E. K. (2018). Working memory: delay activity, yes! Persistent activity? Maybe not. Journal of Neuroscience, 38:7013–7019.")
ins_after("neocortical pyramidal axons in vitro. Journal of Physiology, 502:131–147.","\nUrai, A. E., Doiron, B., Leifer, A. M., and Churchland, A. K. (2022). Large-scale neural recordings call for new insights to link brain and behavior. Nature Neuroscience, 25:11–19.")
ins_after("Heterogeneity in the pyramidal network of the medial prefrontal cortex. Nature Neuroscience, 9:534–542.","\nWilliams, A. H. and Linderman, S. W. (2021). Statistical neuroscience in the single trial limit. Current Opinion in Neurobiology, 70:193–205.")
R=[]
for s,e,n in sorted(E,key=lambda x:-x[0]):
    if e>s: R.append({"deleteContentRange":{"range":{"startIndex":s,"endIndex":e}}})
    if n:
        R.append({"insertText":{"location":{"index":s},"text":n}})
        if n is M:   # Methods heading style, applied before any insertion earlier in the document shifts it
            R.append({"updateParagraphStyle":{"range":{"startIndex":s+1,"endIndex":s+2},"paragraphStyle":{"namedStyleType":"HEADING_2"},"fields":"namedStyleType"}})
        for m in re.finditer(r'\b(?:On|Off)(?:(?: and |/)(?:On|Off))?(?:(?:-| )states?\b)?',n):
            if n[m.start():].startswith('On the'): continue
            pp=s+u(n[:m.start()]); R.append({"updateTextStyle":{"range":{"startIndex":pp,"endIndex":pp+u(m.group(0))},"textStyle":{"italic":True},"fields":"italic"}})
json.dump(R,open(S+'req_spk.json','w'),ensure_ascii=False)
# where the empty figure paragraph will be (for the image insert after this batch)
off=R1.index('\n\nFigure 4.')+1
print('results insert at',RES[0],'empty figure paragraph at',RES[0]+u(R1[:off]),'methods at',METH[0],'n req',len(R))
