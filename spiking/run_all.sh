#!/bin/sh
# full set of simulations for the figures (about 2.5 h on 4 cores)
set -e
cd "$(dirname "$0")/.."
python -m spiking.run sessions
python -m spiking.run example
python -m spiking.run nodips --n_sessions 6
python -m spiking.run single_dip
python -m spiking.run stability
python -m spiking.figures
