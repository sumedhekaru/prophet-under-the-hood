"""Model configurations, fitting methods, and the log posterior of a fitted Prophet model.

Configurations (`make_model(config, series)`):
  base        Prophet's defaults
  hol0 / hol3 / hol7 / hol15   + the country's holidays (Prophet's table), windows of ± 0 / 3 / 7 / 15 days
  wk0 / wk1   + the same holidays snapped to their week, window ± 0 / 1 week (weekly data; `holidays_by_week`)
  any of these + '-mult': seasonality_mode='multiplicative'; + '-yearly': yearly_seasonality=True;
  + '-major': only the major holidays (MAJOR below)
Methods (`fit(model, df, method)`):
  S0  Prophet.fit(df): what users run (Newton below 100 rows, L-BFGS from 100)
  S1  L-BFGS run to convergence: iter=100000, absolute tolerances 1e-12 (relative ones off); 1e-10, then 1e-8 if the
      line search fails first
  S2  Newton
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet

from common.data import Series, holidays_by_week, holidays_for

for _name in ('cmdstanpy', 'prophet'):
    logging.getLogger(_name).setLevel(logging.ERROR)

# S1: at 1e-12 cmdstan's line search can fail at the optimum (exit code 1), and Prophet then retries Newton with
# arguments Newton does not take; so S1 tries looser tolerances in turn and records the one that finished.
S1_TOLERANCES = (1e-12, 1e-10, 1e-8)


def s1_kwargs(tol: float) -> dict[str, Any]:
    return dict(algorithm='lbfgs', iter=100000, tol_obj=tol, tol_rel_obj=1, tol_grad=tol, tol_rel_grad=1,
                tol_param=tol)


METHODS: dict[str, dict[str, Any]] = {'S0': {}, 'S2': {'algorithm': 'Newton'}}

# '-major': New Year, Christmas, Easter where it is a public holiday, and US Thanksgiving (names as in Prophet's table).
MAJOR: dict[str, list[str]] = {
    'US': ["New Year's Day", 'Thanksgiving Day', 'Christmas Day'],
    'DE': ["New Year's Day", 'Good Friday', 'Easter Monday', 'Christmas Day', 'Second Day of Christmas'],
    'FR': ["New Year's Day", 'Easter Monday', 'Christmas Day'],
    'JP': ["New Year's Day"],
    'RU': ['New Year Holidays', 'Christmas Day'],
}


def make_model(config: str, series: Series) -> Prophet:
    base, *flags = config.split('-')
    kw: dict[str, Any] = dict(uncertainty_samples=0)
    if 'mult' in flags:
        kw['seasonality_mode'] = 'multiplicative'
    if 'yearly' in flags:
        kw['yearly_seasonality'] = True
    if base != 'base':
        if series.country is None:
            raise ValueError(f'{config}: series {series.sid} has no country')
        years = range(series.train['ds'].dt.year.min() - 1, series.test['ds'].dt.year.max() + 2)
        table = holidays_for(series.country, years)
        if 'major' in flags:
            table = table[table['holiday'].isin(MAJOR[series.country])].reset_index(drop=True)
        if base.startswith('hol'):
            window = int(base[3:])
            kw['holidays'] = table.assign(lower_window=-window, upper_window=window)
        elif base.startswith('wk'):
            kw['holidays'] = holidays_by_week(table, pd.concat([series.train['ds'], series.test['ds']]),
                                              int(base[2:]))
        else:
            raise ValueError(f'unknown configuration {config!r}')
    return Prophet(**kw)


def fit(config: str, series: Series, method: str) -> tuple[Prophet, float | None]:
    """A fitted model, and for S1 the tolerance it finished with."""
    if method != 'S1':
        return make_model(config, series).fit(series.train, **METHODS[method]), None
    for tol in S1_TOLERANCES:
        try:
            return make_model(config, series).fit(series.train, **s1_kwargs(tol)), tol
        except ValueError:
            continue
    raise RuntimeError(f'S1 failed at every tolerance for {series.sid}')


def design(model: Prophet) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Seasonal / holiday feature matrix X of the training rows, additive and multiplicative masks, prior scales."""
    features, prior_scales, component_cols, _ = model.make_all_seasonality_features(model.history)
    X = features.to_numpy(float)
    s_a = component_cols['additive_terms'].to_numpy(float)
    s_m = component_cols['multiplicative_terms'].to_numpy(float)
    return X, s_a, s_m, np.asarray(prior_scales, float)


def log_posterior(model: Prophet, params: dict[str, np.ndarray] | None = None) -> float:
    """Prophet's log posterior (Stan's lp__: constants dropped) at params (default: the model's own fit).

    Written from prophet.stan (linear trend): k, m ~ N(0, 5); delta ~ Laplace(0, tau); sigma_obs ~ N(0, 0.5);
    beta ~ N(0, sigmas); y_scaled ~ N(trend * (1 + X (beta * s_m)) + X (beta * s_a), sigma_obs).
    """
    p = params or model.params
    k, m = float(np.ravel(p['k'])[0]), float(np.ravel(p['m'])[0])
    delta = np.ravel(p['delta']).astype(float)
    beta = np.ravel(p['beta']).astype(float)
    sigma = float(np.ravel(p['sigma_obs'])[0])
    X, s_a, s_m, sigmas = design(model)
    t = model.history['t'].to_numpy(float)
    y = model.history['y_scaled'].to_numpy(float)
    tau = model.changepoint_prior_scale
    trend = model.piecewise_linear(t, delta, k, m, model.changepoints_t)
    yhat = trend * (1 + X @ (beta * s_m)) + X @ (beta * s_a)
    rss = float(np.sum((y - yhat) ** 2))
    return (-(k ** 2 + m ** 2) / 50 - np.sum(np.abs(delta)) / tau - 0.5 * np.sum((beta / sigmas) ** 2)
            - sigma ** 2 / 0.5 - len(y) * np.log(sigma) - rss / (2 * sigma ** 2))
