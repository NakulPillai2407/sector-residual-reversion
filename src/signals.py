"""Beta-adjusted residuals and z-scores. Everything at date t uses information available at the close of t.

* ``beta_t``: OLS slope (with intercept) of stock close-to-close return on its benchmark return over the
  60 sessions t-60..t-1 (>= 50 valid pairs required). The slope is applied WITHOUT an intercept.
* ``e_t = r_stock,t - beta_t * r_bench,t``.
* ``z_t = e_t / std(e_{t-60..t-1})`` (>= 50 valid values required); the denominator uses the stored historical
  residual series, each element computed with the beta known at its own date.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_io import Panels


@dataclass
class MarketData:
    stocks: Panels             # columns: tradable stock tickers
    etfs: Panels               # columns: sector ETFs + benchmark
    membership: pd.DataFrame   # bool, same index/columns as stocks.close
    etf_of: pd.Series          # ticker -> sector ETF
    gics_of: pd.Series         # ticker -> GICS sector


@dataclass
class Signals:
    beta: pd.DataFrame
    resid: pd.DataFrame
    std: pd.DataFrame
    z: pd.DataFrame
    bench_ret: pd.DataFrame


def returns_cc(adj_close: pd.DataFrame) -> pd.DataFrame:
    return adj_close.pct_change(fill_method=None)


def returns_oo(adj_open: pd.DataFrame) -> pd.DataFrame:
    """Open-to-open return ending at the stamp date: O_d / O_{d-1} - 1."""
    return adj_open.pct_change(fill_method=None)


def bench_matrix(etf_ret: pd.DataFrame, etf_of: pd.Series, columns: pd.Index) -> pd.DataFrame:
    """Per-stock benchmark return matrix (each stock's own ETF series)."""
    sel = etf_of.reindex(columns)
    out = etf_ret.reindex(columns=sel.fillna("__none__").to_numpy())
    out.columns = columns
    return out


def rolling_beta(r: pd.DataFrame, b: pd.DataFrame, window: int, min_obs: int) -> pd.DataFrame:
    """Slope of r on b over rows t-window..t-1 (shifted by one so row t never sees day t)."""
    valid = r.notna() & b.notna()
    x, y = b.where(valid), r.where(valid)
    roll = lambda d: d.rolling(window, min_periods=1).sum()  # noqa: E731
    n = roll(valid.astype(float))
    sx, sy, sxx, sxy = roll(x), roll(y), roll(x * x), roll(x * y)
    var = sxx - sx * sx / n
    beta = (sxy - sx * sy / n) / var
    beta = beta.where((n >= min_obs) & (var > 1e-18))
    return beta.shift(1)


def compute_signals(md: MarketData, cfg: dict, etf_of: pd.Series | None = None) -> Signals:
    s = cfg["signal"]
    etf_of = md.etf_of if etf_of is None else etf_of
    r = returns_cc(md.stocks.adj_close)
    b = bench_matrix(returns_cc(md.etfs.adj_close), etf_of, r.columns)
    beta = rolling_beta(r, b, s["beta_window"], s["min_beta_obs"])
    e = r - beta * b
    std = e.shift(1).rolling(s["z_window"], min_periods=s["min_z_obs"]).std()
    z = e / std.where(std > 0)
    return Signals(beta=beta, resid=e, std=std, z=z, bench_ret=b)


def eligibility(md: MarketData, sig: Signals, cfg: dict) -> pd.DataFrame:
    """Point-in-time eligibility at signal time t."""
    start = pd.Timestamp(cfg["dates"]["signal_start"])
    elig = md.membership & (md.stocks.close >= cfg["filters"]["min_price"]) & sig.z.notna() & sig.beta.notna()
    elig.loc[elig.index < start] = False
    return elig
