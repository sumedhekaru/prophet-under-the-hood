"""Q3: do holiday windows given in days carry into the forecast on weekly data? Stock Prophet (S0) only.

    uv run python -m q3_weekly_holidays.analyze m5w wtw

Per dataset and configuration (base: no holidays; hol0 / hol3 / hol7 / hol15: Prophet's day windows; wk0 / wk1: the
same holidays snapped to their week, ± 0 / 1 week; '-major': only New Year, Christmas, Easter, US Thanksgiving):
  untrained  share of holdout holiday weeks whose active holiday columns all had no training row (so their
             coefficients are at the prior mean, 0, and the forecast gets no holiday effect there)
  effect     mean |holiday component| on holiday weeks, holdout vs training, as a share of mean |y|
  MASE       holdout, all weeks and holiday weeks only; paired vs base, ± 2 SE
A holiday week is a week that contains one of the country's holiday dates (the same weeks for every configuration).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import holidays_for, load
from common.model import MAJOR, make_model
from common.report import env_line, load_run, mase, paired

PAIRS = (('wk0', 'hol3'), ('wk1', 'hol7'), ('wk1', 'hol15'), ('wk0-major', 'hol3-major'),
         ('wk1-major', 'hol7-major'))
CONFIGS = ('base', 'hol0', 'hol3', 'hol7', 'hol15', 'wk0', 'wk1', 'hol0-major', 'hol3-major', 'hol7-major', 'wk0-major',
           'wk1-major')


def holiday_weeks(ds: np.ndarray, country: str, major: bool = False) -> np.ndarray:
    """True for weekly ds (week starts) whose week contains one of the country's (major) holiday dates."""
    ds = pd.DatetimeIndex(ds)
    table = holidays_for(country, range(ds.year.min() - 1, ds.year.max() + 2))
    if major:
        table = table[table['holiday'].isin(MAJOR[country])]
    dates = pd.DatetimeIndex(table['ds'])
    return np.array([((dates >= d) & (dates < d + pd.Timedelta(days=7))).any() for d in ds])


def untrained_share(config: str, s) -> float:
    """Share of holdout holiday weeks whose active holiday columns were all zero on every training row."""
    m = make_model(config, s)
    ds = pd.concat([s.train['ds'], s.test['ds']], ignore_index=True)
    features, _, _ = m.make_holiday_features(ds, m.holidays)
    F = features.to_numpy(float) != 0
    n = len(s.train)
    trained = F[:n].any(axis=0)
    active = F[n:]
    rows = active.any(axis=1)
    if not rows.any():
        return np.nan
    return float(np.mean([not trained[a].any() for a in active[rows]]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='+')
    ap.add_argument('--n', type=int, default=200)
    a = ap.parse_args()
    env = None
    for ds in a.datasets:
        series = {s.sid: s for s in load(ds, n=a.n)}
        base = load_run(ds, 'base', 'S0')
        env = env or env_line(base)
        mask_sets = {major: [holiday_weeks(s['test_ds'], s['country'], major) for s in base['series']]
                     for major in (False, True)}
        masks = mask_sets[False]
        m_base = np.array([mase(s['y_test'], r['yhat_test'], s['y_train'], s['season'])
                           for s, r in zip(base['series'], base['rows'])])
        h_bases = {major: np.array([mase(s['y_test'][k], r['yhat_test'][k], s['y_train'], s['season'])
                                    for s, r, k in zip(base['series'], base['rows'], ks)])
                   for major, ks in mask_sets.items()}
        print(f'\n**{ds}** ({len(masks)} series, {np.mean([k.sum() for k in masks]):.1f} holiday weeks of 52 in the '
              'holdout on average)\n')
        print(f'Major holidays only: {np.mean([k.sum() for k in mask_sets[True]]):.1f} holiday weeks of 52; '
              '"-major" rows are scored on those weeks, the others on all holiday weeks.\n')
        print('| configuration | untrained holdout holiday weeks | effect, holdout / training | MASE all '
              '| vs base (± 2 SE) | MASE holiday weeks | vs base (± 2 SE) |')
        print('|---|---:|---|---:|---|---:|---|')
        for cfg in CONFIGS:
            run = load_run(ds, cfg, 'S0')
            major = cfg.endswith('-major')
            masks, h_base = mask_sets[major], h_bases[major]
            m_all = np.array([mase(s['y_test'], r['yhat_test'], s['y_train'], s['season'])
                              for s, r in zip(run['series'], run['rows'])])
            m_hol = np.array([mase(s['y_test'][k], r['yhat_test'][k], s['y_train'], s['season'])
                              for s, r, k in zip(run['series'], run['rows'], masks)])
            d_all, se_all, _ = paired(m_all, m_base)
            d_hol, se_hol, _ = paired(m_hol, h_base)
            if cfg == 'base':
                unt, eff = '—', '—'
            else:
                shares = [untrained_share(cfg, series[s['sid']]) for s in run['series']]
                unt = '—' if np.all(np.isnan(shares)) else f'{np.nanmean(shares):.0%}'
                e_test, e_train = [], []
                for s, r, k in zip(run['series'], run['rows'], masks):
                    scale = np.mean(np.abs(s['y_train']))
                    k_train = holiday_weeks(series[s['sid']].train['ds'].to_numpy(), s['country'], major)
                    e_test.append(np.mean(np.abs(r['hol_test'][k])) / scale)
                    e_train.append(np.mean(np.abs(r['hol_train'][k_train])) / scale)
                eff = f'{np.nanmean(e_test):.1%} / {np.nanmean(e_train):.1%}'
            print(f'| {cfg} | {unt} | {eff} | {np.nanmean(m_all):.3f} | '
                  + ('—' if cfg == 'base' else f'{d_all:+.4f} ± {2 * se_all:.4f}')
                  + f' | {np.nanmean(m_hol):.3f} | '
                  + ('— |' if cfg == 'base' else f'{d_hol:+.4f} ± {2 * se_hol:.4f} |'))
        print('\nWeek-aligned vs day windows, paired (holiday-week MASE; negative = week-aligned better):\n')
        print('| week-aligned | day windows | Δ holiday weeks (± 2 SE) | Δ all weeks (± 2 SE) |')
        print('|---|---|---|---|')
        for wk, day in PAIRS:
            major = wk.endswith('-major')
            ks = mask_sets[major]
            runs = [load_run(ds, c, 'S0') for c in (wk, day)]
            h = [np.array([mase(s['y_test'][k], r['yhat_test'][k], s['y_train'], s['season'])
                           for s, r, k in zip(run['series'], run['rows'], ks)]) for run in runs]
            w = [np.array([mase(s['y_test'], r['yhat_test'], s['y_train'], s['season'])
                           for s, r in zip(run['series'], run['rows'])]) for run in runs]
            dh, seh, _ = paired(h[0], h[1])
            dw, sew, _ = paired(w[0], w[1])
            print(f'| {wk} | {day} | {dh:+.4f} ± {2 * seh:.4f} | {dw:+.4f} ± {2 * sew:.4f} |')
    print(f'\n{env}')


if __name__ == '__main__':
    main()
