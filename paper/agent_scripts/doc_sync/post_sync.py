import re
S='/tmp/claude-0/-home-user/51e93236-a4df-5b7b-99f9-872ebc6e17b8/scratchpad/'
p='main.tex'; t=open(p).read(); prev=open(S+'main_before.tex').read()
old=[l for l in t.split('\n') if l.startswith('\\small Cognitive Neuroimaging')][0]
new=[l for l in prev.split('\n') if l.startswith('\\small Cognitive Neuroimaging')][0]
t=t.replace(old,new)
t=re.sub(r'\\bibliographystyle\{[^}]*\}','\\\\bibliographystyle{apalike}',t)
t=t.replace("gives every 10~ms a z score","gives every 10~ms a $z$ score")
t=t.replace("decoding z score of the same trial read out with the calibrated gain c (mid gain","decoding $z$ score of the same trial read out with the calibrated gain $c$ (mid gain")
t=t.replace("\\begin{figure}[h]\n  \\centering\n  \\includegraphics[width=0.92\\textwidth]{figS_readout","\\begin{figure}[p]\n  \\centering\n  \\includegraphics[width=0.92\\textwidth]{figS_readout")
b=t.index('\\begin{abstract}') if '\\begin{abstract}' in t else t.index('\\maketitle')
m=t.index('\\section{Methods}'); bb=t.index('\\bibliographystyle')
pat=re.compile(r"(?<![\\{A-Za-z/])(On and Off states|Off and On states|On and Off|Off and On|On/Off states|On/Off|Off-state|Off states|On states|Off state|On state|Off|On)(?![A-Za-z}/])")
def f(mm):
    s=mm.string; i=mm.start()
    if s[i:].startswith(('On the','On average','On each','On a ','On this','On every')): return mm.group(0)
    return '\\emph{'+mm.group(0)+'}'
t=t[:b]+pat.sub(f,t[b:m])+t[m:bb]+pat.sub(f,t[bb:])
t=t.replace("\\emph{On and \\emph{Off} states}","\\emph{On and Off states}").replace("\\emph{On/\\emph{Off}","\\emph{On/Off")
a=t.index('\\caption{\\textbf{Reading the model out like the data.}'); e=t.index('\\label',a)
cap=t[a:e]
cap=re.sub(r'(?<=\. )([b-i])(–|--)([b-i]), ',lambda m:'\\textbf{%s--%s}, '%(m.group(1),m.group(3)),cap)
cap=re.sub(r'(?<=[.)] )([b-i]), (?=[A-Z\\])',lambda m:'\\textbf{%s}, '%m.group(1),cap)
cap=cap.replace('. h, i, mean','. \\textbf{h}, \\textbf{i}, mean')
t=t[:a]+cap+t[e:]
open(p,'w').write(t)
