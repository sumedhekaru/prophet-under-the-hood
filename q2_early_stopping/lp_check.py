"""Q2 check: is the exact fit's worse forecast overfitting (higher lp) or a worse stationary point (lower lp)?

    uv run python -m q2_early_stopping.lp_check m5w wtw m5d wtd
"""
from __future__ import annotations

import sys

import numpy as np

from common.report import load_run

CONFIGS = ('base', 'hol7', 'hol15', 'base-mult', 'hol7-mult', 'hol15-mult')


def main() -> None:
    print('| dataset | configuration | lp X − S0: median / min | X below S0 | X below S1 | X fell back to Stan |')
    print('|---|---|---|---:|---:|---:|')
    for ds in sys.argv[1:]:
        for cfg in CONFIGS:
            runs = {m: load_run(ds, cfg, m) for m in ('S0', 'S1', 'X')}
            lp = {m: np.array([r['lp'] for r in run['rows']]) for m, run in runs.items()}
            stan = np.mean([not r.get('exact', True) for r in runs['X']['rows']])
            d0, d1 = lp['X'] - lp['S0'], lp['X'] - lp['S1']
            print(f'| {ds} | {cfg} | {np.median(d0):.3g} / {d0.min():.3g} | {np.mean(d0 < -1e-6):.0%} | '
                  f'{np.mean(d1 < -1e-6):.0%} | {stan:.0%} |')


if __name__ == '__main__':
    main()
