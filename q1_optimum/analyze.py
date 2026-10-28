"""Q1: how far Prophet's default fit (S0) is from the optimum of its own model (S1: L-BFGS run to convergence;
X: prophet-batch's exact MAP), and whether that changes the forecast.

    uv run python -m q1_optimum.analyze m5d m5w wtd wtw m4w m4d tm m5w-104 wtw-104 m5w-80 wtw-80 [--config base]

Per dataset: the optimizer S0 used (Newton below 100 rows); Δlp of S1 (L-BFGS to convergence), S2 (Newton) and X
(exact MAP) over S0; how often X is below the better stock reference; the forecast change from S0 to X as a share of
the series' mean |y| (median, p90, share of series ≥ 1%); holdout MASE of all four (mean; X - S0 paired, ± 2 SE).
"""
from __future__ import annotations

import argparse

import numpy as np

from common.report import env_line, load_run, paired, per_series_mase, quantiles


def change(a: dict, b: dict, s: dict) -> float:
    """Mean |forecast(b) - forecast(a)| over the holdout, as a share of the series' mean |y| in training."""
    scale = np.mean(np.abs(s['y_train']))
    return float(np.mean(np.abs(b['yhat_test'] - a['yhat_test'])) / scale) if scale > 0 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='+')
    ap.add_argument('--config', default='base')
    a = ap.parse_args()
    print(f'Configuration: {a.config}. Δlp in log-posterior units (Stan lp__ scale); change = mean |Δ forecast| / '
          'mean |y|.\n')
    print('| dataset | series | rows | S0 optimizer | Δlp S1 − S0: median / p90 / max | Δlp S2 − S0: median / max '
          '| Δlp X − S0: median / p90 / max | X < max(S1, S2) | forecast change S0 → X: median / p90 | ≥ 1% '
          '| MASE S0 / S1 / S2 / X | X − S0 (± 2 SE) |')
    print('|---|---:|---:|---|---|---|---|---:|---|---:|---|---|')
    env = None
    for ds in a.datasets:
        s0, s1, s2, x = (load_run(ds, a.config, m) for m in ('S0', 'S1', 'S2', 'X'))
        env = env or env_line(s0)
        rows = np.array([len(s['y_train']) for s in s0['series']])
        lp0, lp1, lp2, lpx = (np.array([r['lp'] for r in run['rows']]) for run in (s0, s1, s2, x))
        best = np.maximum(lp1, lp2)
        below = np.mean(lpx < best - 1e-6 * np.abs(best))
        ch = np.array([change(r0, rx, s) for r0, rx, s in zip(s0['rows'], x['rows'], s0['series'])])
        m0, m1, m2, mx = (per_series_mase(run) for run in (s0, s1, s2, x))
        dm, se, _ = paired(mx, m0)
        opt = 'Newton' if rows.max() < 100 else 'L-BFGS' if rows.min() >= 100 else 'mixed'
        q1, q2, qx = quantiles(lp1 - lp0), quantiles(lp2 - lp0, (0.5, 1.0)), quantiles(lpx - lp0)
        qc = quantiles(ch, (0.5, 0.9))
        print(f'| {ds} | {len(rows)} | {int(np.median(rows))} | {opt} | {q1[0]:.2g} / {q1[1]:.2g} / {q1[2]:.2g} | '
              f'{q2[0]:.2g} / {q2[1]:.2g} | {qx[0]:.2g} / {qx[1]:.2g} / {qx[2]:.2g} | {below:.0%} | '
              f'{qc[0]:.2%} / {qc[1]:.2%} | {np.nanmean(ch >= 0.01):.0%} | {np.nanmean(m0):.3f} / {np.nanmean(m1):.3f}'
              f' / {np.nanmean(m2):.3f} / {np.nanmean(mx):.3f} | {dm:+.4f} ± {2 * se:.4f} |')
    print(f'\n{env}')


if __name__ == '__main__':
    main()
