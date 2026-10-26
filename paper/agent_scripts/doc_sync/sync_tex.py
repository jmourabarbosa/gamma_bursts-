import re,sys
S='/tmp/claude-0/-home-user/51e93236-a4df-5b7b-99f9-872ebc6e17b8/scratchpad/'
L=open(S+'doc_base.txt').read().split('\n')
KEYS={('amit',1997):'amit1997',('awh',2001):'awh2001',('barak',2007):'barak2007',('barbosa',2020):'barbosa2020',('busch',2009):'busch2009',('busch',2010):'busch2010',
('compte',2000):'compte2000',('engel',2016):'engel2016',('fiebelkorn',2019):'fiebelkorn2019',('fiebelkorn',2018):'fiebelkorn2018',('fiebelkorn',2013):'fiebelkorn2013',
('hansen',2001):'hansen2001',('helfrich',2018):'helfrich2018',('hempel',2000):'hempel2000',('kiyonaga',2017):'kiyonaga2017',('landau',2012):'landau2012',
('lundqvist',2016):'lundqvist2016',('markram',1998):'markram1998',('masse',2019):'masse2019',('mi',2017):'mi2017',('miller',2018):'miller2018',('mongillo',2008):'mongillo2008',
('mozumder',2026):'mozumder2026',('panichello',2024):'panichello2024',('panichello',2021):'panichello2021',('pascucci',2023):'pascucci2023',('rose',2016):'rose2016',
('stokes',2015):'stokes2015',('tao',2024):'tao2024',('vicini',1998):'vicini1998',('wang',1999):'wang1999',('wang',2001):'wang2001',('wang',2002):'wang2002',
('wang',2013):'wangm2013',('wang',2006):'wangy2006',('wolff',2017):'wolff2017',('wong',2006):'wong2006',('thomson',1997):'thomson1997',('wang',2008):'wangh2008',('mongillo',2025):'mongillo2025',('urai',2022):'urai2022',('williams',2021):'williams2021',('lundqvist',2018):'lundqvist2018',('barbosa',2025):'barbosa2025'}
missing=set()
def key(name,yr):
    k=KEYS.get((name.lower(),int(yr)))
    if not k: missing.add((name,yr))
    return k or 'MISSING'
AUTH=r"[A-Z][A-Za-zÀ-ÿ\-]+"
ITEM=re.compile(rf"^({AUTH})(?:\s+(?:et al\.|(?:&|and)\s+{AUTH}))?,\s*(\d{{4}}(?:,\s*\d{{4}})*)$")
def item_keys(it):
    m=ITEM.match(it.strip())
    if not m: return None
    return [key(m.group(1),y) for y in re.findall(r'\d{4}',m.group(2))]
def paren(m):
    inner=m.group(1); parts=[p.strip() for p in inner.split(';')]
    ks=[item_keys(p) for p in parts]
    if all(ks): return '\\citep{'+','.join(sum(ks,[]))+'}'
    if not any(re.search(r'\d{4}',p) for p in parts): return m.group(0)
    out=[]
    for p,k in zip(parts,ks):
        if k: out.append('\\citealp{'+','.join(k)+'}'); continue
        # "text, Author et al., 1998"
        mm=re.match(rf"^(.*?,\s*)({AUTH}(?:\s+(?:et al\.|(?:&|and)\s+{AUTH}))?,\s*\d{{4}}(?:,\s*\d{{4}})*)$",p)
        if mm and item_keys(mm.group(2)): out.append(mm.group(1)+'\\citealp{'+','.join(item_keys(mm.group(2)))+'}')
        else: out.append(p)
    return '('+'; '.join(out)+')'
def narr(m):
    return '\\citet{'+','.join(key(m.group(1),y) for y in re.findall(r'\d{4}',m.group(2)))+'}'
def conv(s):
    s=s.replace('&','&') 
    s=re.sub(r'\(([^()]*\d{4}[^()]*)\)',paren,s)
    s=re.sub(rf"({AUTH})(?:\s+(?:et al\.|and\s+{AUTH}))?\s+\((\d{{4}}(?:,\s*\d{{4}})*)\)",narr,s)
    s=s.replace('%','\\%').replace('&','\\&')
    rep=[('c·Δr','$c\\cdot\\Delta r$'),('Δr','$\\Delta r$'),('Δt','$\\Delta t$'),('·','$\\cdot$'),('ε','$\\varepsilon$'),('ξ','$\\xi$'),('ρ','$\\rho$'),('< 10⁻⁶','$<10^{-6}$'),('10⁻⁶','$10^{-6}$'),('×10⁻⁴','$\\times10^{-4}$'),('×10⁻⁵','$\\times10^{-5}$'),('τN','$\\tau_N$'),('τ','$\\tau$'),('β','$\\beta$'),('±','$\\pm$'),('−','$-$'),
         ('≥','$\\geq$'),('μ','$\\mu$'),('δ','$\\delta$'),('η','$\\eta$'),('κ','$\\kappa$'),('~','$\\sim$'),('×','$\\times$'),('’',"'"),('“','``'),('”',"''"),('\u00a0',' ')]
    for a,b in rep: s=s.replace(a,b)
    s=re.sub(r'(\d) ms\b',r'\1~ms',s); s=re.sub(r'(\d) s\b',r'\1~s',s); s=re.sub(r'(\d) Hz\b',r'\1~Hz',s)
    s=re.sub(r'(\d)–(\d)',r'\1--\2',s)
    s=re.sub(r'Fig\. (\d)',r'Fig.~\1',s); s=re.sub(r'Fig (\d)',r'Fig.~\1',s)
    s=s.replace('***p','***$p$').replace('**p','**$p$')
    s=re.sub(r'\bp ([=<]) ',r'$p \1$ ',s)
    s=re.sub(r'\$\$',' ',s)
    return s
def caption(s):
    m=re.match(r'^(?:Supplementary )?Figure \d+\.\s*(.*?\.)\s(.*)$',s)
    t,rest=m.group(1),m.group(2)
    rest=conv(rest); rest=re.sub(r'(^|\. |: |; )([a-f]), ',r'\1\\textbf{\2}, ',rest)
    return '\\textbf{'+conv(t)+'} '+rest
start=L.index('# Abstract'); meth=L.index('# Methods'); sup=L.index('# Supplementary Figures')
FIG=['\\includegraphics[width=\\textwidth]{fig1_trial_histograms.png}','\\includegraphics[width=0.92\\textwidth]{fig2_unit_sets.png}','\\includegraphics[width=\\textwidth]{fig3_behaviour_tau.png}','\\includegraphics[width=\\textwidth]{fig4_spiking.png}']
out=[]; fi=0; i=start; sec=None
while i<meth:
    l=L[i]
    if l.startswith('# '):
        h=l[2:].strip()
        if h=='Abstract': out.append('\\begin{abstract}'); sec='abs'
        else:
            if sec=='abs': out.append('\\end{abstract}\n')
            sec='x'
            out.append(('\\section{' if h in ('Introduction','Results','Discussion') else '\\subsection{')+conv(h)+'}\n')
    elif l=='[IMG]':
        cap=L[i+1]; i+=1
        out.append('\\begin{figure}[t]\n  \\centering\n  '+FIG[fi]+'\n  \\caption{'+caption(cap)+'}\n  \\label{fig:'+str(fi+1)+'}\n\\end{figure}\n'); fi+=1
    elif l.strip(): out.append(conv(l.strip())+'\n')
    i+=1
body='\n'.join(out)
supp=['\\clearpage\n\\section{Supplementary Figures}\n\\renewcommand{\\figurename}{Supplementary Figure}\n\\setcounter{figure}{0}\n']
SF=['figS_readout.png','figS_control_no_dips.png','figS_stp_unit_sets.png']; k=0
for j in range(sup,len(L)):
    if L[j].startswith('Supplementary Figure'):
        supp.append('\\begin{figure}[h]\n  \\centering\n  \\includegraphics[width=0.92\\textwidth]{'+SF[k]+'}\n  \\caption{'+caption(L[j])+'}\n  \\label{fig:S'+str(k+1)+'}\n\\end{figure}\n'); k+=1
t=open('main.tex').read()
a=t.index('\\begin{abstract}'); b=t.index('\\section{Methods}')
t=t[:a]+body+'\n'+t[b:]
a=t.index('\\clearpage\n\\section{Supplementary Figures}'); b=t.index('\\end{document}')
t=t[:a]+'\n'.join(supp)+'\n'+t[b:]
# methods targeted edits
old1=''
t=t.replace("because the network may be silent at the end of the delay, but a strong attentional input during the response period can restart the pool with more NMDA gating and recover the correct response with virtually no memory loss (not shown).","because the network may be silent at the end of the delay, and a strong attentional input, modelling the response period, restarts the pool with more NMDA gating and so recovers the correct response (not shown).")
old2="In the single-dip test, each trial had one dip of the common input"
t=t.replace(old2,"To measure how long each model can hold the memory through a single silence (single-dip test, Fig.~3a), we simulated trials with only one dip of the common input")
# author
t=re.sub(r'\\author\{.*?\}\n\\date',lambda m:"\\author{Joao Barbosa\\\\[4pt]\n\\small Cognitive Neuroimaging Unit, INSERM, CEA, CNRS, Universit\\'e Paris-Saclay, NeuroSpin center \\& Institut de neuromodulation,\\\\ \\small GHU Paris, psychiatrie et neurosciences, centre hospitalier Sainte-Anne, p\\^ole hospitalo-universitaire 15,\\\\ \\small Universit\\'e Paris Cit\\'e, Paris, France.}\n\\date",t,count=1,flags=re.S)
t=re.sub(r'\\title\{.*?\}',lambda m:'\\title{'+conv(L[0][2:])+'}',t,count=1)
open('main.tex','w').write(t)
print('missing',missing,'figs',fi,k)
