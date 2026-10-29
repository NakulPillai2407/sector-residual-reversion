"""Price I/O with a physical in-sample / holdout split.

Raw prices are split at download time into ``prices_is.parquet`` (<= in-sample end) and
``prices_ho.parquet`` (> in-sample end). Everything before Stage 3 loads the in-sample file only;
the holdout loader refuses to run unless ``results/HOLDOUT_LOCK`` exists (written by the Stage 3 script).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import PROC_DIR, RESULTS_DIR

HOLDOUT_LOCK = RESULTS_DIR / "HOLDOUT_LOCK"


class HoldoutAccessError(RuntimeError):
    pass


def split_and_save_prices(raw: pd.DataFrame, is_end: str, proc_dir: Path = PROC_DIR) -> tuple[Path, Path]:
    """Mechanically split; no inspection of the holdout part."""
    proc_dir.mkdir(parents=True, exist_ok=True)
    cut = pd.Timestamp(is_end)
    p_is, p_ho = proc_dir / "prices_is.parquet", proc_dir / "prices_ho.parquet"
    raw[raw["date"] <= cut].to_parquet(p_is, index=False)
    raw[raw["date"] > cut].to_parquet(p_ho, index=False)
    return p_is, p_ho


def load_prices(include_holdout: bool = False, proc_dir: Path = PROC_DIR) -> pd.DataFrame:
    frames = [pd.read_parquet(proc_dir / "prices_is.parquet")]
    if include_holdout:
        if not HOLDOUT_LOCK.exists():
            raise HoldoutAccessError("Holdout prices may only be loaded by the Stage 3 script (results/HOLDOUT_LOCK missing).")
        frames.append(pd.read_parquet(proc_dir / "prices_ho.parquet"))
    return pd.concat(frames, ignore_index=True)


@dataclass
class Panels:
    """Wide (date x ticker) price panels on the official session calendar."""

    close: pd.DataFrame       # Yahoo 'Close' (split-adjusted, not dividend-adjusted): used for the $5 filter
    adj_close: pd.DataFrame   # split+dividend adjusted
    adj_open: pd.DataFrame    # Open * (Adj Close / Close)
    volume: pd.DataFrame

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index


def build_panels(px: pd.DataFrame, sessions: pd.DatetimeIndex) -> Panels:
    px = px.copy()
    px["adj_open"] = px["Open"] * (px["Adj Close"] / px["Close"])
    idx = pd.DatetimeIndex(sessions)

    def wide(col: str) -> pd.DataFrame:
        w = px.pivot(index="date", columns="ticker", values=col)
        return w.reindex(idx)

    return Panels(close=wide("Close"), adj_close=wide("Adj Close"), adj_open=wide("adj_open"), volume=wide("Volume"))
