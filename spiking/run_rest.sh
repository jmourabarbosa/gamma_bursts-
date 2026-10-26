#!/bin/sh
# remaining simulations after the sessions (each result committed as it finishes)
cd "$(dirname "$0")/.."
commit() { git add -A spiking && git commit -qm "Spiking ring: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N9S9icRkWWDhXiDqvtAFL2" && git push -q origin claude/gallant-volta-bkjojq; }
python -m spiking.run stability && commit "stability test (no dips, 2 s pre-cue)"
python -m spiking.run single_dip && commit "single-dip test"
python -m spiking.run nodips --n_sessions 6 && commit "no-dip control"
python -m spiking.validate_brian2 && commit "Brian2 cross-check"
python -m spiking.figures && commit "figures with all runs"
echo ALL_DONE
