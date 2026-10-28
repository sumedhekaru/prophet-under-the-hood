"""Datasets as lists of `Series`: training and holdout frames (ds, y), plus the country for holidays.

Dataset ids (see docs/evidence-plan.md):
  m5d    M5 items summed over the 10 stores, daily (1,941 days, 2011-01-29 to 2016-05-22), US
  m5w    the same, summed to Walmart weeks (Saturday to Friday, ds = the Saturday): 277 weeks
  wtd    Wikipedia web traffic, all-agents pages of en / ja / de / fr / ru Wikipedia with mean >= 100 views a day,
         daily (2015-07-01 to 2022-06-30); country from the language (en -> US)
  wtw    the same pages summed to weeks (Monday to Sunday, ds = the Monday; partial first and last weeks dropped)
  m4w, m4d, tm   M4 Weekly, M4 Daily, Tourism Monthly (dates not meaningful: Q1 only)
Suffix '-104' or '-80' (weekly ids): train on the first 104 or 80 weeks, hold out the next 52.
Otherwise the holdout is the last 52 weeks / 365 days / 24 months. Samples are seeded; series that are all zero in
training are dropped.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from common.download import DATA

CACHE = DATA / 'cache'
WT_PROJECTS = {'en': 'US', 'ja': 'JP', 'de': 'DE', 'fr': 'FR', 'ru': 'RU'}


@dataclasses.dataclass
class Series:
    sid: str
    train: pd.DataFrame  # ds, y
    test: pd.DataFrame   # ds, y
    country: str | None
    freq: str            # 'D', 'W' or 'M'

    @property
    def season(self) -> int:
        """Seasonal period for MASE."""
        return {'D': 7, 'W': 52, 'M': 12}[self.freq]


def _m5_wide() -> pd.DataFrame:
    cache = CACHE / f'm5-item-pandas{pd.__version__.split(".")[0]}.pkl'
    if cache.exists():
        return pd.read_pickle(cache)
    sales = pd.read_csv(DATA / 'm5' / 'sales_train_evaluation.csv')
    day_cols = [c for c in sales.columns if c.startswith('d_')]
    wide = sales.groupby('item_id')[day_cols].sum()
    dates = pd.to_datetime(pd.read_csv(DATA / 'm5' / 'calendar.csv', usecols=['date'])['date'])
    wide.columns = pd.DatetimeIndex(dates.iloc[[int(c[2:]) - 1 for c in day_cols]].to_numpy())
    wide = wide.astype(float)
    CACHE.mkdir(parents=True, exist_ok=True)
    wide.to_pickle(cache)
    return wide


def _wt_candidates() -> pd.DataFrame:
    """All pages that pass the filter, with their daily values (cached: parsing the 1.3 GB file takes minutes)."""
    cache = CACHE / f'wt-candidates-pandas{pd.__version__.split(".")[0]}.pkl'
    if cache.exists():
        return pd.read_pickle(cache)
    path = DATA / 'wt' / 'web_traffic_extended_dataset_without_missing_values.tsf'
    names, countries, rows, start = [], [], [], None
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('@data'):
                break
        for line in f:
            # name:start:article:project:access:agent:values (the article may contain ':')
            head, values = line.rsplit(':', 1)
            name, rest = head.split(':', 1)
            stamp, rest = rest[:19], rest[20:]
            article, project, access, agent = rest.rsplit(':', 3)
            lang = project.split('.')[0]
            if (agent != 'all-agents' or access != 'all-access' or lang not in WT_PROJECTS
                    or not project.endswith('wikipedia.org')):
                continue
            y = np.array(values.split(','), dtype=float)
            if y.mean() < 100:
                continue
            start = start or stamp
            names.append(f'{lang}:{article}')
            countries.append(WT_PROJECTS[lang])
            rows.append(y)
    dates = pd.date_range(pd.Timestamp(start[:10]), periods=len(rows[0]), freq='D')
    wide = pd.DataFrame(np.vstack(rows), index=names, columns=dates)
    wide.insert(0, 'country', countries)
    CACHE.mkdir(parents=True, exist_ok=True)
    wide.to_pickle(cache)
    return wide


def _to_weeks(wide: pd.DataFrame, first_weekday: int) -> pd.DataFrame:
    """Sum daily columns into complete weeks starting on first_weekday (0 = Monday); ds = the first day."""
    dates = pd.DatetimeIndex(wide.columns)
    start = dates - pd.to_timedelta((dates.weekday - first_weekday) % 7, unit='D')
    weekly = wide.T.groupby(start).agg(['sum', 'count'])
    full = weekly.xs('count', axis=1, level=1).iloc[:, 0] == 7
    return weekly.xs('sum', axis=1, level=1)[full].T


def _sample(index: pd.Index, n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    return sorted(index[rng.choice(len(index), size=min(n, len(index)), replace=False)])


def _split(sid: str, ds: pd.DatetimeIndex, y: np.ndarray, h: int, country: str | None, freq: str,
           n_train: int | None = None) -> Series:
    if n_train is None:
        n_train = len(y) - h
    train = pd.DataFrame({'ds': ds[:n_train], 'y': y[:n_train]})
    test = pd.DataFrame({'ds': ds[n_train:n_train + h], 'y': y[n_train:n_train + h]})
    return Series(sid, train, test, country, freq)


def _m4(name: str, n: int, seed: int) -> list[Series]:
    freq, h = {'m4w': ('W', 13), 'm4d': ('D', 14)}[name]
    label = {'m4w': 'Weekly', 'm4d': 'Daily'}[name]
    train = pd.read_csv(DATA / 'm4' / f'{label}-train.csv', index_col=0)
    test = pd.read_csv(DATA / 'm4' / f'{label}-test.csv', index_col=0)
    out = []
    for sid in _sample(train.index, n, seed):
        x = train.loc[sid].dropna().to_numpy(float)
        t = test.loc[sid].dropna().to_numpy(float)
        ds = pd.date_range('2000-01-03', periods=len(x) + len(t), freq='W-MON' if freq == 'W' else 'D')
        out.append(_split(sid, ds, np.concatenate([x, t]), len(t), None, freq))
    return out


def _tourism(n: int, seed: int) -> list[Series]:
    path = DATA / 'tourism' / 'tourism_monthly_dataset.tsf'
    rows = {}
    with open(path, encoding='latin-1') as f:  # the header has a non-UTF-8 dash
        for line in f:
            if line.startswith('@data'):
                break
        for line in f:
            name, stamp, values = line.strip().split(':', 2)
            rows[name] = (pd.Timestamp(stamp[:10]), np.array(values.split(','), dtype=float))
    out = []
    for sid in _sample(pd.Index(list(rows)), n, seed):
        start, y = rows[sid]
        out.append(_split(sid, pd.date_range(start, periods=len(y), freq='MS'), y, 24, None, 'M'))
    return out


def load(name: str, n: int = 200, seed: int = 0) -> list[Series]:
    """A seeded sample of n series of dataset `name` (see the module docstring)."""
    base, _, cut = name.partition('-')
    n_train = int(cut) if cut else None
    if base in ('m4w', 'm4d'):
        return _m4(base, n, seed)
    if base == 'tm':
        return _tourism(n, seed)
    if base in ('m5d', 'm5w'):
        wide = _m5_wide()
        country = 'US'
        if base == 'm5w':
            wide = _to_weeks(wide, first_weekday=5)
    elif base in ('wtd', 'wtw'):
        cand = _wt_candidates()
        countries = cand.pop('country')
        wide = cand if base == 'wtd' else _to_weeks(cand, first_weekday=0)
    else:
        raise ValueError(f'unknown dataset {name!r}')
    freq = 'D' if base.endswith('d') else 'W'
    h = 365 if freq == 'D' else 52
    out = []
    ds = pd.DatetimeIndex(wide.columns)
    for sid in _sample(wide.index, n * 2, seed):
        y = wide.loc[sid].to_numpy(float)
        s = _split(str(sid), ds, y, h, country if base.startswith('m5') else countries[sid], freq, n_train)
        if (s.train['y'] != 0).any():
            out.append(s)
        if len(out) == n:
            break
    return out


def holidays_for(country: str, years: range, window: int = 0) -> pd.DataFrame:
    """Prophet's own holiday table for a country (prophet.make_holidays), with a symmetric window in days."""
    from prophet.make_holidays import make_holidays_df

    h = make_holidays_df(year_list=list(years), country=country)
    return h.assign(lower_window=-window, upper_window=window)


def holidays_by_week(holidays: pd.DataFrame, ds: pd.Series, weeks: int = 0) -> pd.DataFrame:
    """Holidays snapped to the weekly ds that contains them, in plain pandas (Q3's week-aligned windows).

    Each occurrence moves to the start of its week (ds holds week starts), and the window becomes `weeks` whole weeks
    on each side, so the same coefficients are used every year.
    """
    starts = pd.DatetimeIndex(sorted(pd.to_datetime(ds).unique()))
    step = starts[1] - starts[0]
    when = pd.to_datetime(holidays['ds'])
    week = starts[0] + ((when - starts[0]) // step) * step
    out = pd.DataFrame({'holiday': holidays['holiday'].to_numpy(), 'ds': week.to_numpy(),
                        'lower_window': -weeks * step.days, 'upper_window': weeks * step.days})
    return out.drop_duplicates(['holiday', 'ds']).reset_index(drop=True)
