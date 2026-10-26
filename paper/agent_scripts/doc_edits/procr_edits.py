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
repl("its alignment with the cued location dropped but remained above its level before the cue (Fig. 4e).",
     "the per-cue averages of Off activity kept the shape of the ring of cue means, less than during On states but well above chance, whereas activity before the cue did not (Fig. 4e).")
repl("Alignment of each trial's position with its cue's direction (cosine, minus the same with shuffled cue labels) during On and Off moments and before the cue (Pre).",
     "Shape similarity of the eight per-cue averages of activity during On and Off moments and before the cue (Pre) to the ring of cue means (1 − Procrustes disparity). Dashes, chance (shuffled cue labels).")
repl("Alignment above Pre during Off moments","Similarity above chance during Off moments")
repl("Each trial's position was averaged over its On and, separately, its Off time points. Its alignment was the cosine between this position and the direction of its cue's mean in the other half of the trials, minus the same quantity with cue labels shuffled within each half (50 shuffles). Activity before the cue, projected on the same axes, served as a null (window centres from −400 ms to cue onset in the recordings and to −50 ms in the network, whose neurons respond to the cue within a few milliseconds).",
     "These two components were used only to display activity (Fig. 4d). For Fig. 4e, each trial's activity was averaged over its On and, separately, its Off time points, and over its time points before the cue (window centres from −400 ms to cue onset in the recordings and to −50 ms in the network, whose neurons respond to the cue within a few milliseconds). These averages were averaged again for each cue location and compared with the eight cue means of the other half of the trials by Procrustes analysis (translation, rotation and uniform scaling), in the space of the principal components of the cue means that held 90% of their variance (2–6 components). Shape similarity was 1 minus the Procrustes disparity, averaged over the two halves of the trials. Chance was estimated by shuffling cue labels within each half (50 shuffles).")
reqs=[]
for a,r in sorted(ops,key=lambda o:-o[0]): reqs+=r
json.dump(reqs,open('req_procr.json','w'),ensure_ascii=False); print(len(reqs),c['rev'])
import subprocess
d=json.load(open('/root/.claude/projects/-home-user/51e93236-a4df-5b7b-99f9-872ebc6e17b8/tool-results/mcp-Google_Docs-read_doc-1793023008059.txt'))
for el in d['content']['tabs'][0]['documentTab']['body']['content']:
  p=el.get('paragraph')
  if p:
    for r in p['elements']:
      if 'inlineObjectElement' in r and 21000<r['startIndex']<23000: print('IMG',r['startIndex'])
