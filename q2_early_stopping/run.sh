#!/bin/sh
# Q2 runs: daily and weekly real data, holidays with day windows, additive and multiplicative; S0, S1 and X.
# (base S0 / S1 / X are shared with Q1.) From the repo root:
#   sh q2_early_stopping/run.sh
set -eu
export PYTHONPATH=.
for ds in m5w wtw m5d wtd; do
    for mode in "" "-mult"; do
        for cfg in base hol0 hol3 hol7 hol15; do
            uv run python -m common.run "$ds" "$cfg$mode" S0 S1 X --n 200
        done
    done
done
uv run python -m q2_early_stopping.analyze m5w wtw m5d wtd
