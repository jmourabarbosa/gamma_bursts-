import re,subprocess
p='main.tex'; t=open(p).read()
head=subprocess.run(['git','show','HEAD:paper/main.tex'],capture_output=True,text=True).stdout
def R(a,b):
    global t; assert t.count(a)==1,(a,t.count(a)); t=t.replace(a,b)
# affiliation wrap from HEAD
old=[l for l in t.split('\n') if l.startswith('\\small Cognitive Neuroimaging')][0]
new=[l for l in head.split('\n') if l.startswith('\\small Cognitive Neuroimaging')][0]
R(old,new)
R("The standard deviation of pre-cue $z$ was 1.06, close to","The standard deviation of pre-cue $z$, averaged over the 25 sessions, was 1.06 for each decoded population, close to")
R("within the range measured in cortex \\citep{vicini1998,wangm2013}.","within the range measured in cortex \\citep{vicini1998,wangh2008,wangm2013}.")
t=re.sub(r"inherent to a recording subsampling\.","inherent to recording only a small sample of the neurons in the circuit.",t)
R("gives, every 10~ms, a z score","gives, every 10~ms, a $z$ score")
t=t.replace("\\begin{figure}[h]\n  \\centering\n  \\includegraphics[width=\\textwidth]{figures/figS_readout","\\begin{figure}[p]\n  \\centering\n  \\includegraphics[width=\\textwidth]{figures/figS_readout")
# italics
b=t.index('\\begin{abstract}') if '\\begin{abstract}' in t else t.index('\\maketitle')
pat=re.compile(r"(?<![\\{A-Za-z])(On and Off states|Off and On states|On and Off|Off and On|On/Off states|On/Off|Off-state|Off states|On states|Off state|On state|Off|On)(?![A-Za-z}])")
def f(m):
    s=m.string; i=m.start()
    if s[i:].startswith(('On the','On average','On each','On a ','On this','On every')): return m.group(0)
    return '\\emph{'+m.group(0)+'}'
body=pat.sub(f,t[b:])
t=t[:b]+body
open(p,'w').write(t)
