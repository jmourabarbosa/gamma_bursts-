#!/bin/sh
# CCG / Manhattan-distance analysis (Panichello et al. 2024 Fig. 5) on sessions 0-5, committing after each session
cd "$(dirname "$0")/.."
for s in 0 1 2 3 4 5; do
  [ -f spiking/results/ccg/session_$s.json ] && grep -q "\"off\"" spiking/results/ccg/session_$s.json && continue
  python -m spiking.ccg --sessions $s || exit 1
  git add -A spiking && git commit -qm "Spiking ring: CCG analysis, session $s

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N9S9icRkWWDhXiDqvtAFL2" && git push -q origin claude/gallant-volta-bkjojq
done
echo CCG_DONE
