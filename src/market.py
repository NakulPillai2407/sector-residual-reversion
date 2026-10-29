"""Assemble :class:`MarketData` from the cached, processed files."""
from __future__ import annotations

import pandas as pd

from .config import PROC_DIR
from .data_io import build_panels, load_prices
from .providers import DataProvider, YahooWikipediaProvider
from .signals import MarketData


def load_market_data(cfg: dict, include_holdout: bool = False, provider: DataProvider | None = None
                     ) -> tuple[MarketData, pd.Series]:
    """Returns (MarketData, annualised risk-free series in decimal). In-sample only unless ``include_holdout``
    (which additionally requires results/HOLDOUT_LOCK, enforced in :func:`load_prices`)."""
    provider = provider or YahooWikipediaProvider(cfg)
    px = load_prices(include_holdout=include_holdout)
    last = cfg["dates"]["end"] if include_holdout else cfg["split"]["in_sample"][1]
    sessions = provider.trading_calendar(cfg["dates"]["data_start"], last)

    special = set(cfg["etfs"]) | {cfg["benchmark"], cfg["tbill_ticker"]}
    sectors = pd.read_csv(PROC_DIR / "sector_map.csv")
    ok = sectors[sectors["status"] == "ok"].set_index("ticker")
    stock_tk = sorted(t for t in px["ticker"].unique() if t in ok.index and t not in special)

    stocks = build_panels(px[px["ticker"].isin(stock_tk)], sessions)
    etf_cols = list(cfg["etfs"]) + [cfg["benchmark"]]
    etfs = build_panels(px[px["ticker"].isin(etf_cols)], sessions)
    etfs = type(etfs)(*(getattr(etfs, f)[etf_cols] for f in ("close", "adj_close", "adj_open", "volume")))
    stocks = type(stocks)(*(getattr(stocks, f).reindex(columns=stock_tk) for f in ("close", "adj_close", "adj_open", "volume")))

    mem = pd.read_parquet(PROC_DIR / "membership.parquet").reindex(index=sessions, columns=stock_tk).fillna(False).astype(bool)
    md = MarketData(stocks=stocks, etfs=etfs, membership=mem, etf_of=ok.loc[stock_tk, "etf"], gics_of=ok.loc[stock_tk, "gics_sector"])

    irx = px[px["ticker"] == cfg["tbill_ticker"]].set_index("date")["Close"] / 100.0
    rf = irx.reindex(sessions).ffill().rename("rf_annual")
    return md, rf
