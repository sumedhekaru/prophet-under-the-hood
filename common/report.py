"""Shared analysis helpers: load run outputs, holdout metrics, paired differences."""
from __future__ import annotations

import pickle
from typing import Any

import numpy as np

from common.run import RESULTS


def load_run(dataset: str, config: str, method: str) -> dict[str, Any]:
    with open(RESULTS / dataset / f'{config}-{method}.pkl', 'rb') as f:
        return pickle.load(f)


def mase(y: np.ndarray, f: np.ndarray, train: np.ndarray, m: int) -> float:
    """M4's MASE: mean |y - f| over the seasonal-naive in-sample error with period m (NaN if undefined)."""
    if len(train) <= m:
        return np.nan
    scale = np.mean(np.abs(train[m:] - train[:-m]))
    return float(np.mean(np.abs(y - f)) / scale) if scale > 0 else np.nan


def per_series_mase(run: dict[str, Any], mask: str | None = None) -> np.ndarray:
    """Holdout MASE per series; mask='holiday' keeps only holdout rows with a nonzero holiday component in `mask_run`."""
    out = []
    for s, r in zip(run['series'], run['rows']):
        assert s['sid'] == r['sid']
        out.append(mase(s['y_test'], r['yhat_test'], s['y_train'], s['season']))
    return np.array(out)


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    """Mean of a - b over series where both are finite, its standard error, and the count."""
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan, int(ok.sum())


def quantiles(x: np.ndarray, qs: tuple[float, ...] = (0.5, 0.9, 1.0)) -> list[float]:
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, q)) for q in qs] if len(x) else [np.nan] * len(qs)


def env_line(run: dict[str, Any]) -> str:
    v = run['meta']['versions']
    return (f"prophet {v['prophet']}, prophet-batch {v['prophet_batch']}, numpy {v['numpy']}, pandas {v['pandas']}, "
            f"Python {v['python']}, {v['machine']}, repo {v['commit']}")
