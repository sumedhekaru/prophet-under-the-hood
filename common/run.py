"""Fit a dataset sample with one configuration and method; save per-series results for the analysis scripts.

    uv run python -m common.run m5w base S0 S1 X [--n 200] [--jobs 10]

Methods S0 / S1 / S2 are stock Prophet (common.model); X is prophet-batch's exact MAP (fit_many). Per series it saves:
sid, the log posterior (common.model.log_posterior; Stan's lp__ too for S methods), fit seconds, the in-sample and
holdout yhat, the holdout's holiday component, and the fitted parameters. Output: data/results/<dataset>/<config>-
<method>.pkl (not committed). Existing outputs are kept unless --force.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.data import Series, load
from common.download import DATA

RESULTS = DATA / 'results'


def _quiet() -> None:
    import logging

    warnings.simplefilter('ignore')
    for name in ('cmdstanpy', 'prophet', 'prophet_batch'):
        logging.getLogger(name).setLevel(logging.ERROR)


def _summary(model: Any, s: Series, seconds: float) -> dict[str, Any]:
    from common.model import log_posterior

    future = pd.concat([s.train[['ds']], s.test[['ds']]], ignore_index=True)
    fc = model.predict(future)
    n = len(s.train)
    hol = fc['holidays'].to_numpy(float) if 'holidays' in fc else np.zeros(len(fc))
    params = {k: np.ravel(v).astype(float) for k, v in model.params.items()}
    out = dict(sid=s.sid, lp=log_posterior(model), seconds=seconds, yhat_train=fc['yhat'].to_numpy(float)[:n],
               yhat_test=fc['yhat'].to_numpy(float)[n:], hol_test=hol[n:], hol_train=hol[:n], params=params,
               y_scale=float(model.y_scale), n_holiday_cols=_n_holiday_cols(model))
    if 'lp__' in model.params:
        out['lp_stan'] = float(np.ravel(model.params['lp__'])[0])
    return out


def _n_holiday_cols(model: Any) -> int:
    cols = getattr(model, 'train_component_cols', None)
    return int(cols['holidays'].sum()) if cols is not None and 'holidays' in cols else 0


def _fit_stock(args: tuple[Series, str, str]) -> dict[str, Any]:
    _quiet()
    from common.model import fit

    s, config, method = args
    t0 = time.perf_counter()
    model, tol = fit(config, s, method)
    out = _summary(model, s, time.perf_counter() - t0)
    out['tol'] = tol
    return out


def _fit_exact(series: list[Series], config: str) -> list[dict[str, Any]]:
    _quiet()
    import prophet_batch
    from prophet_batch import batch

    from common.model import make_model

    # prophet-batch >= 0.1.4 fits wide-holiday series with Stan instead; here we want the exact MAP for all.
    if hasattr(batch, '_HOLIDAY_RANK_SHARE'):
        batch._HOLIDAY_RANK_SHARE = float('inf')
    by_sid = {s.sid: s for s in series}
    long = pd.concat([s.train.assign(unique_id=s.sid) for s in series], ignore_index=True)
    t0 = time.perf_counter()
    models = prophet_batch.fit_many(long, make_model=lambda sid: make_model(config, by_sid[sid]))
    per = (time.perf_counter() - t0) / len(series)
    out = []
    for s in series:
        m = models[s.sid]
        r = _summary(m, s, per)
        r['exact'] = m.stan_fit is None  # False: prophet-batch fell back to Prophet.fit for this series
        out.append(r)
    return out


def run(dataset: str, config: str, method: str, n: int, jobs: int, force: bool = False) -> Path:
    path = RESULTS / dataset / f'{config}-{method}.pkl'
    if path.exists() and not force:
        return path
    series = load(dataset, n=n)
    t0 = time.perf_counter()
    if method == 'X':
        rows = _fit_exact(series, config)
    else:
        with ProcessPoolExecutor(jobs) as pool:
            rows = list(pool.map(_fit_stock, [(s, config, method) for s in series], chunksize=4))
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(dataset=dataset, config=config, method=method, n=len(series), wall=time.perf_counter() - t0,
                versions=_versions())
    with open(path, 'wb') as f:
        pickle.dump(dict(meta=meta, rows=rows,
                         series=[dict(sid=s.sid, y_train=s.train['y'].to_numpy(float),
                                      y_test=s.test['y'].to_numpy(float), test_ds=s.test['ds'].to_numpy(),
                                      season=s.season, country=s.country) for s in series]), f)
    print(f'{dataset} {config} {method}: {len(series)} series, {meta["wall"]:.0f} s', flush=True)
    return path


def _versions() -> dict[str, str]:
    import platform
    import subprocess

    import prophet
    import prophet_batch

    commit = subprocess.run(['git', '-C', str(Path(__file__).parents[1]), 'describe', '--always', '--dirty'],
                            capture_output=True, text=True).stdout.strip()
    return dict(prophet=prophet.__version__, prophet_batch=prophet_batch.__version__, numpy=np.__version__,
                pandas=pd.__version__, python=platform.python_version(), machine=platform.platform(), commit=commit)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('config')
    ap.add_argument('methods', nargs='+')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    for method in a.methods:
        run(a.dataset, a.config, method, a.n, a.jobs, a.force)


if __name__ == '__main__':
    main()
