p='main.tex'; t=open(p).read()
anc='which we call the decoding strength (Methods).'
assert t.count(anc)==1
k=t.index(anc)+len(anc)
t=t[:k]+r" When synthetic neurons driven by the model were decoded with the same pipeline as the data, the decoder's $z$ agreed with this read-out as closely as two runs of the decoder on independent spikes, and gave similar \emph{Off states} (Supplementary Fig.~1f--h)."+t[k:]
a=t.index('labelling and statistics}'); a=t.index('}',a)+1
b=t.index('\\bibliographystyle')
meth=r"""

To compare the model with the recordings, we converted its activity into a decoding $z$ score and labelled it with the same criteria as the data (Supplementary Fig.~1).
The decoding evidence was $\Delta r(t)$, the difference between the cued and uncued pool rates averaged in 100~ms windows stepped by 10~ms, as for the neural data.
The model $z$ score was $z(t) = c\, \Delta r(t) + \eta(t)$, where the gain $c$ converts rates into $z$ units and $\eta$ is read-out noise, which stands for the variability of a decoder trained on a finite number of neurons and trials.
The noise $\eta$ has unit variance, like the pre-cue $z$ of the data (SD 1.06), the autocorrelation of overlapping 100~ms windows, and a constant per-trial offset whose share of the variance, $\rho$, equals the autocorrelation of the data's pre-cue $z$ at 200~ms (0.35 for all neurons, 0.09 for the most selective neurons and 0.11 for random neurons; Supplementary Fig.~1c).
The gain $c$ was not fitted to the \emph{On}/\emph{Off} statistics. For each decoded population, it was set so that the decoding strength of the model's delay $z$, the upper mean of a two-component Gaussian mixture of delay $z$ values, equalled the data's (Supplementary Fig.~1d,e).
States were labelled as in \citet{panichello2024}.
\emph{On states} are runs of $z > 1.96$ whose summed $z$ exceeds the 95th percentile of the same quantity in pure noise $\eta$.
\emph{Off states} are runs of at least 5 points (50~ms) with $z < 0.84$.
States were counted from 0 to 1,400~ms after cue onset, and fractions of the delay spent \emph{On} or \emph{Off} from 500 to 1,400~ms.
Model statistics were computed from 2,112 simulated trials (792 per evaluation during fitting).
Data statistics are means over the 25 sessions; for random neurons, the 10 draws were first averaged within each session.

To validate this read-out, we drove 330 synthetic Poisson neurons with the pool rates of the fitted model for two cue locations, with the pools swapping roles (Supplementary Fig.~1f--h).
30\% of the neurons were tuned to one of the two locations and the rest were untuned, baseline rates were lognormal with a median of 5~Hz, and all neurons shared a per-trial gain.
Their spikes were decoded with the unchanged data pipeline (70 trials per cue, 8 synthetic sessions at each of three tuning strengths, matched to the decoding strength of the random-20, all-unit and top-20 decoders).
The read-out's $z$ correlated with the decoder's as closely as two runs of the decoder on independent spikes, and it reproduced the number and fraction of \emph{Off states}.
It undercounted \emph{On states} by about one per trial, and this disappeared when a white-noise component matched to the decoder's fast fluctuations was added to $\eta$.
Controls on the fitted model showed that \emph{On}/\emph{Off} statistics change smoothly with the gain and with $\rho$, and that noise alone produces \emph{Off states} but almost no \emph{On states} (Supplementary Fig.~1i).

"""
t=t[:a]+meth+t[b:]
fig=r"""\begin{figure}[p]
  \centering
  \includegraphics[width=0.92\textwidth]{figS_readout.png}
  \caption{\textbf{Reading the model out like the data.} \textbf{a}, One model trial through the read-out. From top to bottom, rates of the cued (black) and uncued (grey) pools, with dips of the common input in grey; $\Delta r$, their difference averaged in 100~ms windows every 10~ms; $c\cdot\Delta r$ (black) and the read-out noise $\eta$ (grey); and $z = c\cdot\Delta r + \eta$, with \emph{On} (red) and \emph{Off} (blue) states labelled with the thresholds of \citet{panichello2024} (dashed, $z$ = 1.96 and 0.84). \textbf{b}, One data trial (all-unit decoder, session 211008) with its \emph{On} and \emph{Off} states. \textbf{c}, Autocorrelation of the decoder $z$ before the cue, where there is no memory signal, for three decoded populations, compared with the model's noise $\eta$ (orange dashed) and with $\eta$ plus 4\% white noise (blue dashed). The SD of pre-cue $z$ was 1.06 for all three populations. \textbf{d}, Distribution of delay $z$ (500--1,400~ms after cue onset) in the data (grey, 25 sessions) and in the model (orange). Dashed, decoding strength (the upper mean of a two-Gaussian mixture), 2.18 in the data and 2.20 in the model. \textbf{e}, Decoding strength of the model as a function of the gain $c$. The data's decoding strength (horizontal line) sets $c$ (vertical line). \textbf{f--h}, Validation on synthetic data. The fitted model drove 330 Poisson neurons whose spikes were decoded with the data pipeline. \textbf{f}, Example trial with the true dips (grey), the decoder's $z$ (black) and the read-out $z = c\cdot\Delta r + \eta$ (orange). \textbf{g}, Correlation of each read-out's delay $z$ with the decoder's, at three decoding strengths matched to the data's random-20, all-unit and top-20 decoders. Grey, the decoder rerun on new spikes, which sets the ceiling for two noisy read-outs. \textbf{h}, \emph{On} and \emph{Off} statistics of the decoder (circles), the read-out (squares) and the read-out with jitter-matched noise (diamonds), mean $\pm$ 2 SEM over 8 synthetic sessions. \textbf{i}, Controls on the fitted model with the all-unit read-out. \emph{Off} and \emph{On} states per trial, and percentage of the delay \emph{Off}, for the data, the read-out used in the paper, noise only ($c = 0$), gain $\times$0.75 and $\times$1.25, $\rho$ = 0 and 0.7, jitter-matched noise and the model without dips.}
  \label{fig:S0}
\end{figure}

"""
k=t.index('\\setcounter{figure}{0}'); k=t.index('\n',k)+1
t=t[:k]+'\n'+fig+t[k:]
open(p,'w').write(t)
print('done')
