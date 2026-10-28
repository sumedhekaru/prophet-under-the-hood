#!/bin/sh
# Q3 runs: weekly datasets, stock Prophet (S0), no holidays / day windows / week-aligned windows. From the repo root:
#   sh q3_weekly_holidays/run.sh
set -eu
export PYTHONPATH=.
for ds in m5w wtw; do
    for cfg in base hol0 hol3 hol7 hol15 wk0 wk1 hol0-major hol3-major hol7-major wk0-major wk1-major; do
        uv run python -m common.run "$ds" "$cfg" S0 --n 200
    done
done
uv run python -m q3_weekly_holidays.analyze m5w wtw
