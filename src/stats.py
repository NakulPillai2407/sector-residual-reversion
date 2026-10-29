"""Performance statistics and inference (Newey-West, stationary bootstrap, Fama-MacBeth)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------- basic metrics
def sharpe(r: pd.Series | np.ndarray, ann: int = 252) -> float:
    x = np.asarray(r, dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(ann)) if sd > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def nw_mean(x: pd.Series | np.ndarray, lags: int) -> tuple[float, float, float]:
    """Mean, Newey-West (Bartlett, ``lags``) standard error and t-stat of the mean."""
    y = np.asarray(x, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < lags + 5:
        return float("nan"), float("nan"), float("nan")
    res = sm.OLS(y, np.ones(len(y))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params[0]), float(res.bse[0]), float(res.tvalues[0])


def alpha_beta(r: pd.Series, bench: pd.Series, lags: int, ann: int = 252) -> dict[str, float]:
    """Regress strategy daily return on benchmark daily return with Newey-West errors."""
    df = pd.concat([r.rename("y"), bench.rename("x")], axis=1).dropna()
    res = sm.OLS(df["y"], sm.add_constant(df["x"])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return dict(alpha_ann=float(res.params["const"] * ann), alpha_t=float(res.tvalues["const"]),
                beta=float(res.params["x"]), beta_t=float(res.tvalues["x"]))


def min_detectable_sharpe(years: float) -> float:
    """Approximate 5% two-sided minimum detectable annual Sharpe: 1.96 / sqrt(years)."""
    return float(1.96 / np.sqrt(years))


# ------------------------------------------------------------ stationary bootstrap
def stationary_bootstrap_indices(T: int, B: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix (B x T), geometric blocks with mean ``mean_block``."""
    p = 1.0 / mean_block
    starts = rng.integers(0, T, size=(B, T))
    restart = rng.random((B, T)) < p
    restart[:, 0] = True
    pos = np.arange(T)
    last = np.maximum.accumulate(np.where(restart, pos, 0), axis=1)
    base = np.take_along_axis(starts, last, axis=1)
    return (base + (pos - last)) % T


def bootstrap_sharpe_ci(r: pd.Series | np.ndarray, rng: np.random.Generator, B: int, mean_block: float,
                        level: float = 0.95, ann: int = 252, chunk: int = 1000) -> tuple[float, float, np.ndarray]:
    x = np.asarray(r, dtype=float)
    x = x[~np.isnan(x)]
    T = len(x)
    out = []
    for i in range(0, B, chunk):
        idx = stationary_bootstrap_indices(T, min(chunk, B - i), mean_block, rng)
        s = x[idx]
        sd = s.std(axis=1, ddof=1)
        out.append(np.where(sd > 0, s.mean(axis=1) / np.where(sd > 0, sd, 1) * np.sqrt(ann), np.nan))
    dist = np.concatenate(out)
    a = (1 - level) / 2
    return float(np.nanquantile(dist, a)), float(np.nanquantile(dist, 1 - a)), dist


def paired_bootstrap_diff_ci(a: pd.Series, b: pd.Series, rng: np.random.Generator, B: int, mean_block: float,
                             level: float = 0.95, ann: int = 252, chunk: int = 1000) -> tuple[float, float, float]:
    """CI for Sharpe(a) - Sharpe(b), resampling the same dates for both series."""
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    xa, xb = df["a"].to_numpy(), df["b"].to_numpy()
    T = len(df)
    diffs = []
    for i in range(0, B, chunk):
        idx = stationary_bootstrap_indices(T, min(chunk, B - i), mean_block, rng)
        sa, sb = xa[idx], xb[idx]
        diffs.append(sa.mean(1) / sa.std(1, ddof=1) * np.sqrt(ann) - sb.mean(1) / sb.std(1, ddof=1) * np.sqrt(ann))
    d = np.concatenate(diffs)
    al = (1 - level) / 2
    return sharpe(xa, ann) - sharpe(xb, ann), float(np.quantile(d, al)), float(np.quantile(d, 1 - al))


# ----------------------------------------------------------------- Fama-MacBeth
def daily_cross_sectional_slopes(x: pd.DataFrame, y: pd.DataFrame, mask: pd.DataFrame, min_names: int = 20) -> pd.Series:
    """Per-row OLS slope of y on x (with intercept) over masked, jointly valid entries. Vectorised over rows."""
    valid = mask & x.notna() & y.notna()
    xv = x.where(valid)
    yv = y.where(valid)
    n = valid.sum(axis=1).astype(float)
    sx, sy = xv.sum(axis=1), yv.sum(axis=1)
    sxx, sxy = (xv * xv).sum(axis=1), (xv * yv).sum(axis=1)
    cov = sxy - sx * sy / n
    var = sxx - sx * sx / n
    slope = cov / var
    return slope.where((n >= min_names) & (var > 0))


def fama_macbeth(slopes: pd.Series, lags: int) -> dict[str, float]:
    s = slopes.dropna()
    m, se, t = nw_mean(s, lags)
    return dict(mean_coef=m, nw_se=se, nw_t=t, n_days=int(len(s)))
