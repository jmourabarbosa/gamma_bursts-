import json,sys
d=json.load(open(sys.argv[1]))
c=d['content']
body=c['tabs'][0]['documentTab']['body']['content']
txt=[];idx=[];fl=[]
for el in body:
    p=el.get('paragraph')
    if not p: continue
    for r in p['elements']:
        tr=r.get('textRun')
        if not tr: continue
        s=r['startIndex'];f='I' if tr.get('suggestedInsertionIds') else 'D' if tr.get('suggestedDeletionIds') else ('i' if tr.get('suggestedTextStyleChanges') else '.')
        for ch in tr['content']:
            txt.append(ch);idx.append(s);fl.append(f);s+=len(ch.encode('utf-16-le'))//2
T=''.join(txt);F=''.join(fl)
json.dump({'T':T,'F':F,'idx':idx,'rev':c['revisionId']},open('cur.json','w'))
print(c['revisionId'],'I',F.count('I'),'D',F.count('D'),'style',F.count('i'))
