"""Q2: when does Prophet's early stop (S0) forecast better than the exact optimum of its model (X)?

    uv run python -m q2_early_stopping.analyze m5d m5w wtd wtw

Per dataset and configuration (base; hol0 / hol3 / hol7 / hol15: country holidays with day windows; '-mult':
multiplicative seasonality):
  r          median rank(holiday columns over the training rows) / T (0 without holidays)
  one-row    median share of nonempty holiday columns with exactly one nonzero training row
  MASE       holdout, S0 / S1 / X (mean), and X - S0 paired (± 2 SE): positive = the exact optimum forecasts worse
  holiday    the same on holdout rows within ± 7 days of a holiday date (daily) or holiday weeks (weekly)
  |β| ratio  median over series of L1(holiday β) and L1(yearly β), X / S0: how much larger the exact fit's terms are
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import holidays_for, load
from common.model import make_model
from common.report import env_line, load_run, mase, paired

CONFIGS = ('base', 'hol0', 'hol3', 'hol7', 'hol15')


def holiday_rows(ds: np.ndarray, country: str, freq: str) -> np.ndarray:
    ds = pd.DatetimeIndex(ds)
    dates = pd.DatetimeIndex(holidays_for(country, range(ds.year.min() - 1, ds.year.max() + 2))['ds'])
    if freq == 'W':
        return np.array([((dates >= d) & (dates < d + pd.Timedelta(days=7))).any() for d in ds])
    near = np.zeros(len(ds), bool)
    for d in dates:
        near |= np.abs((ds - d).days) <= 7
    return near


def structure(config: str, s) -> tuple[float, float, np.ndarray, np.ndarray]:
    """r = rank(holiday block) / T, the one-row share, and the holiday / yearly column masks of the feature matrix."""
    m = make_model(config, s)
    inputs = m.preprocess(s.train)
    X = np.asarray(inputs.X, float)
    cols = m.train_component_cols
    hol = cols['holidays'].to_numpy(bool) if 'holidays' in cols else np.zeros(X.shape[1], bool)
    yearly = cols['yearly'].to_numpy(bool) if 'yearly' in cols else np.zeros(X.shape[1], bool)
    if not hol.any():
        return 0.0, np.nan, hol, yearly
    H = X[:, hol]
    nz = (H != 0).sum(axis=0)
    rows = (H != 0).any(axis=1)
    r = np.linalg.matrix_rank(H[rows]) / len(X) if rows.any() else 0.0
    return float(r), float(np.mean(nz[nz > 0] == 1)) if (nz > 0).any() else np.nan, hol, yearly


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='+')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--modes', nargs='+', default=['', '-mult'])
    a = ap.parse_args()
    env = None
    for ds in a.datasets:
        series = {s.sid: s for s in load(ds, n=a.n)}
        print(f'\n**{ds}**\n')
        print('| configuration | r | one-row | MASE S0 / S1 / X | X − S0 (± 2 SE) | holiday rows: S0 / X '
              '| X − S0 (± 2 SE) | L1 β X / S0: holiday, yearly |')
        print('|---|---:|---:|---|---|---|---|---|')
        for mode in a.modes:
            for base in CONFIGS:
                cfg = base + mode
                s0, s1, x = (load_run(ds, cfg, m) for m in ('S0', 'S1', 'X'))
                env = env or env_line(s0)
                ser = [series[s['sid']] for s in s0['series']]
                masks = [holiday_rows(s['test_ds'], s['country'], ser[0].freq) for s in s0['series']]
                m = {}
                for key, run in (('S0', s0), ('S1', s1), ('X', x)):
                    m[key] = np.array([mase(s['y_test'], r['yhat_test'], s['y_train'], s['season'])
                                       for s, r in zip(run['series'], run['rows'])])
                    m[key + 'h'] = np.array([mase(s['y_test'][k], r['yhat_test'][k], s['y_train'], s['season'])
                                             for s, r, k in zip(run['series'], run['rows'], masks)])
                d, se, _ = paired(m['X'], m['S0'])
                dh, seh, _ = paired(m['Xh'], m['S0h'])
                st = [structure(cfg, s) for s in ser]
                r = np.median([t[0] for t in st])
                one = np.nanmedian([t[1] for t in st]) if base != 'base' else np.nan
                ratios_h, ratios_y = [], []
                for (_, _, hol, yearly), r0, rx in zip(st, s0['rows'], x['rows']):
                    b0, bx = r0['params']['beta'], rx['params']['beta']
                    if hol.any() and np.abs(b0[hol]).sum() > 0:
                        ratios_h.append(np.abs(bx[hol]).sum() / np.abs(b0[hol]).sum())
                    if yearly.any() and np.abs(b0[yearly]).sum() > 0:
                        ratios_y.append(np.abs(bx[yearly]).sum() / np.abs(b0[yearly]).sum())
                rh = f'{np.median(ratios_h):.2f}' if ratios_h else '—'
                ry = f'{np.median(ratios_y):.2f}' if ratios_y else '—'
                print(f'| {cfg} | {r:.3f} | ' + ('—' if np.isnan(one) else f'{one:.0%}')
                      + f' | {np.nanmean(m["S0"]):.3f} / {np.nanmean(m["S1"]):.3f} / {np.nanmean(m["X"]):.3f} | '
                      f'{d:+.4f} ± {2 * se:.4f} | {np.nanmean(m["S0h"]):.3f} / {np.nanmean(m["Xh"]):.3f} | '
                      f'{dh:+.4f} ± {2 * seh:.4f} | {rh}, {ry} |')
    print(f'\n{env}')


if __name__ == '__main__':
    main()
