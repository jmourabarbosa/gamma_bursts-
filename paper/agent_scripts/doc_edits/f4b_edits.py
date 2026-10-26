import json,re,difflib
exec(open('spk_edits.py').read().split('# --- comments')[0])
rep("the network therefore alternated between On and Off states (Fig. 4a, bottom). ",
    "the network therefore alternated between On and Off states (Fig. 4a, bottom). Decoders using the 20 most cue-selective neurons gave On and Off states of similar lengths to decoders using all units, but fewer of them (Fig. 4b). ")
rep("b, Length distributions of Off (left) and On (right) states (all units, 12 simulated sessions of 480 trials).",
    "b, Length distributions of Off (left) and On (right) states for decoders using the 20 most cue-selective neurons (dark) or all units (light), with the mean number of states per trial (4 simulated sessions of 480 trials).")
rep("(12 sessions, Fig. 4b).","(4 sessions, Fig. 4b), using all units or the 20 most cue-selective neurons (ranked by split-half ANOVA between the two cues of each decoded pair).")
R=[]
for s,e,n in sorted(E,key=lambda x:-x[0]):
    if e>s: R.append({"deleteContentRange":{"range":{"startIndex":s,"endIndex":e}}})
    if n:
        R.append({"insertText":{"location":{"index":s},"text":n}})
        for m in re.finditer(r'\b(?:On|Off)(?:(?: and |/)(?:On|Off))?(?:(?:-| )states?\b)?',n):
            pp=s+u(n[:m.start()]); R.append({"updateTextStyle":{"range":{"startIndex":pp,"endIndex":pp+u(m.group(0))},"textStyle":{"italic":True},"fields":"italic"}})
print(json.dumps(R,ensure_ascii=False))
for s,e,n in sorted(E): print(s,e,repr(n[:90]))
