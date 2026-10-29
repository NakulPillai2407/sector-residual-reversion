"""Data-quality checks. Nothing is fixed silently: every finding is returned as a table."""
from __future__ import annotations

import pandas as pd


def check_prices(px: pd.DataFrame, sessions: pd.DatetimeIndex, etf_like: set[str], move_threshold: float = 0.5) -> dict[str, pd.DataFrame]:
    """``px`` long frame (date, ticker, OHLC, Adj Close, Volume, Splits). Restricted to ``sessions`` window by caller."""
    out: dict[str, pd.DataFrame] = {}

    dup = px[px.duplicated(["ticker", "date"], keep=False)]
    out["duplicate_dates"] = dup[["ticker", "date"]].drop_duplicates()

    bad = px[(px[["Open", "High", "Low", "Close", "Adj Close"]].isna().any(axis=1))
             | (px[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any(axis=1)]
    out["nan_or_nonpositive_prices"] = bad.groupby("ticker").size().rename("n_rows").reset_index()

    miss_rows = []
    sess = pd.DatetimeIndex(sessions)
    for t, g in px.groupby("ticker"):
        d = pd.DatetimeIndex(g["date"].unique())
        lo, hi = d.min(), d.max()
        expected = sess[(sess >= lo) & (sess <= hi)]
        missing = expected.difference(d)
        extra = d.difference(sess)
        if len(missing) or len(extra):
            miss_rows.append(dict(ticker=t, first=lo, last=hi, n_missing_sessions=len(missing), n_extra_dates=len(extra),
                                  is_etf_or_benchmark=t in etf_like))
    out["missing_sessions"] = pd.DataFrame(miss_rows, columns=["ticker", "first", "last", "n_missing_sessions", "n_extra_dates", "is_etf_or_benchmark"])

    pxs = px.sort_values(["ticker", "date"]).copy()
    pxs["ret"] = pxs.groupby("ticker")["Adj Close"].pct_change(fill_method=None)
    big = pxs[pxs["ret"].abs() > move_threshold]
    out["moves_gt_50pct"] = big[["ticker", "date", "ret", "Close", "Adj Close", "Stock Splits"]].assign(
        split_same_day=lambda d: d["Stock Splits"].fillna(0) > 0).reset_index(drop=True)
    return out
