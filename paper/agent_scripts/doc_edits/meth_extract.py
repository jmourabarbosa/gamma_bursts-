import json
d=json.load(open('doc_full_latest.json'))
body=d['content']['tabs'][0]['documentTab']['body']['content']
paras=[]
for el in body:
    p=el.get('paragraph')
    if not p: continue
    runs=[]
    for r in p['elements']:
        tr=r.get('textRun')
        if not tr: runs.append(('￼',{})); continue
        runs.append((tr['content'],tr.get('textStyle',{})))
    paras.append(dict(start=el['startIndex'],end=el['endIndex'],ps=p['paragraphStyle'],runs=runs,text=''.join(x for x,_ in runs)))
json.dump(paras,open('paras.json','w'),ensure_ascii=False)
i0=[k for k,p in enumerate(paras) if p['text'].strip()=='NMDA rate model'][0]
i1=[k for k,p in enumerate(paras) if p['text'].strip()=='References'][0]
for k in range(i0,i1):
    p=paras[k]; ps=p['ps']
    sty=set()
    for t,s in p['runs']:
        for key in ('bold','italic','baselineOffset','fontSize','weightedFontFamily','foregroundColor'):
            if key in s: sty.add(key+'='+str(s[key])[:30])
    print(k,p['start'],ps.get('namedStyleType'),ps.get('alignment',''),repr(p['text'][:60]),sorted(sty)[:6])
