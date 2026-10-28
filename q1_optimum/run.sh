#!/bin/sh
# Q1 runs: every dataset, base configuration, all four methods (200 series each). From the repo root:
#   sh q1_optimum/run.sh
set -eu
export PYTHONPATH=.
for ds in m5w wtw m5w-104 wtw-104 m5w-80 wtw-80 m4w tm m5d wtd m4d; do
    uv run python -m common.run "$ds" base S0 S1 S2 X --n 200
done
uv run python -m q1_optimum.analyze m5w wtw m5w-104 wtw-104 m5w-80 wtw-80 m4w tm m5d wtd m4d
