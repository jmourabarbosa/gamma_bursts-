import json,re
c=json.load(open('cur.json'));T=c['T'];F=c['F'];idx=c['idx']
keep=[k for k,f in enumerate(F) if f!='D']; A=''.join(T[k] for k in keep)
ops=[]
def u16(s): return len(s.encode('utf-16-le'))//2
def repl(old,new):
    assert A.count(old)==1,(old[:40],A.count(old)); i=A.index(old)
    ks=keep[i:i+len(old)]; assert ks[-1]-ks[0]==len(old)-1, 'D inside range'
    a=idx[ks[0]]; b=idx[ks[-1]]+u16(T[ks[-1]])
    r=[{"insertText":{"location":{"index":b},"text":new}}]
    for m in re.finditer(r'\b(On|Off)\b',new):
        s=b+u16(new[:m.start()]); r.append({"updateTextStyle":{"range":{"startIndex":s,"endIndex":s+u16(m.group())},"textStyle":{"italic":True},"fields":"italic"}})
    r.append({"deleteContentRange":{"range":{"startIndex":a,"endIndex":b}}})
    ops.append((a,r))
repl("kept the shape of the ring of cue means, less than during On states but well above chance, whereas",
     "kept the shape of the ring of cue means well above chance, whereas")
repl("(1 − Procrustes disparity). Dashes, chance (shuffled cue labels).",
     "(1 − Procrustes disparity divided by the disparity with shuffled cue labels; 0, chance, dashed line).")
repl("In the recordings, states were labelled from the 20 most cue-selective units of the labelling half. In the network, they were labelled from all of its neurons, because its 20 most selective neurons often preferred the same cue and were silent on trials of the other cue while the bump was still active.",
     "In both, states were labelled from all neurons of the labelling half.")
repl("These two components were used only","The first two components were used only")
repl("in the space of the principal components of the cue means that held 90% of their variance (2–6 components). Shape similarity was 1 minus the Procrustes disparity, averaged over the two halves of the trials. Chance was estimated by shuffling cue labels within each half (50 shuffles).",
     "in the seven-dimensional space spanned by the cue means. The disparity was averaged over the two halves of the trials and divided by its mean with cue labels shuffled within each half (50 shuffles). Shape similarity was 1 minus this ratio, so that 0 is chance and 1 an identical shape.")
reqs=[]
for a,r in sorted(ops,key=lambda o:-o[0]): reqs+=r
json.dump(reqs,open('req_procr2.json','w'),ensure_ascii=False); print(len(reqs),c['rev'])
d=json.load(open('/root/.claude/projects/-home-user/51e93236-a4df-5b7b-99f9-872ebc6e17b8/tool-results/mcp-Google_Docs-read_doc-1793025193938.txt'))
for el in d['content']['tabs'][0]['documentTab']['body']['content']:
  p=el.get('paragraph')
  if p:
    for r in p['elements']:
      if 'inlineObjectElement' in r and 20000<r['startIndex']<24000: print('IMG',r['startIndex'])
