"""Transaction-cost and borrow model.

* Stock turnover cost: ``stock_bps`` per side on sum|dw| of the stock book.
* ETF turnover cost: ``etf_bps`` per side on sum|dw| of the hedge book.
* Borrow: ``borrow_bps_per_year`` / days_per_year per day on short notional (stocks and any short ETF hedge).
Turnover is charged on the CHANGE in target weights; unchanged positions cost nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    stock_bps: float = 5.0
    etf_bps: float = 1.0
    borrow_bps_per_year: float = 50.0
    days_per_year: int = 252


def turnover(weights: pd.DataFrame) -> pd.Series:
    """sum_i |w_t,i - w_{t-1,i}| ; the first row is measured against an empty book."""
    return weights.sub(weights.shift(1).fillna(0.0)).abs().sum(axis=1)


def short_notional(*books: pd.DataFrame) -> pd.Series:
    out = None
    for b in books:
        s = (-b.clip(upper=0.0)).sum(axis=1)
        out = s if out is None else out.add(s, fill_value=0.0)
    return out


def cost_series(turn_stock: pd.Series, turn_etf: pd.Series, short: pd.Series, cm: CostModel) -> pd.Series:
    return (cm.stock_bps * 1e-4 * turn_stock + cm.etf_bps * 1e-4 * turn_etf
            + cm.borrow_bps_per_year * 1e-4 / cm.days_per_year * short)


def breakeven_bps(gross: pd.Series, turn_stock: pd.Series, turn_etf: pd.Series, short: pd.Series, cm: CostModel) -> float:
    """Per-side stock cost (bps) at which mean net return is zero, holding ETF cost and borrow fixed."""
    fixed = cm.etf_bps * 1e-4 * turn_etf.mean() + cm.borrow_bps_per_year * 1e-4 / cm.days_per_year * short.mean()
    ts = turn_stock.mean()
    return float((gross.mean() - fixed) / ts * 1e4) if ts > 0 else float("nan")
