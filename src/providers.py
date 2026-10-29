"""Data access layer.

All data access in the project goes through :class:`DataProvider`, so a Bloomberg/WRDS
provider can replace :class:`YahooWikipediaProvider` without touching signals/backtest.

Price conventions (documented, per the pre-registration):
* ``auto_adjust=False``. Yahoo's ``Close`` is split-adjusted but NOT dividend-adjusted
  (true as-traded prices are not available from Yahoo); ``Adj Close`` is split+dividend adjusted.
* Close-to-close returns use ``Adj Close``.
* Open-to-open returns use adjusted opens: ``adj_open = Open * (Adj Close / Close)``.
* The $5 price filter uses the (split-adjusted, dividend-unadjusted) ``Close``.
"""
from __future__ import annotations

import io
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import requests

from .config import RAW_DIR

log = logging.getLogger(__name__)

PRICE_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]


def to_yahoo_symbol(sym: str) -> str:
    """Wikipedia uses BRK.B, Yahoo uses BRK-B."""
    return str(sym).strip().replace(".", "-").upper()


class DataProvider(ABC):
    """Interface every data source must implement."""

    @abstractmethod
    def current_constituents(self) -> pd.DataFrame:
        """Columns: ticker, name, sector (GICS), date_added."""

    @abstractmethod
    def index_changes(self) -> pd.DataFrame:
        """Columns: date, added, removed, added_name, removed_name, reason (tickers Yahoo-normalised, NaN if none)."""

    @abstractmethod
    def prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Long frame: date, ticker, Open, High, Low, Close, Adj Close, Volume, Dividends, Stock Splits.
        ``end`` is INCLUSIVE. Tickers with no data are simply absent."""

    @abstractmethod
    def sector_labels(self, tickers: list[str]) -> pd.DataFrame:
        """Columns: ticker, raw_sector (provider-native or NaN), source."""

    @abstractmethod
    def risk_free(self, start: str, end: str) -> pd.Series:
        """Annualised risk-free rate in decimal (e.g. 0.05), indexed by date."""

    @abstractmethod
    def trading_calendar(self, start: str, end: str) -> pd.DatetimeIndex:
        """Official exchange sessions (NYSE), inclusive."""


class YahooWikipediaProvider(DataProvider):
    def __init__(self, cfg: dict, raw_dir: Path = RAW_DIR):
        self.cfg = cfg
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {"User-Agent": cfg["data"]["user_agent"]}

    # ---------------------------------------------------------------- wikipedia
    def _get_html(self, url: str, cache_name: str) -> str:
        cache = self.raw_dir / cache_name
        if cache.exists():
            return cache.read_text()
        r = requests.get(url, headers=self.headers, timeout=60)
        r.raise_for_status()
        cache.write_text(r.text)
        return r.text

    def current_constituents(self) -> pd.DataFrame:
        html = self._get_html(self.cfg["data"]["wikipedia_constituents_url"], "wikipedia_constituents.html")
        tbl = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})[0]
        out = pd.DataFrame(
            {
                "ticker": tbl["Symbol"].map(to_yahoo_symbol),
                "wiki_symbol": tbl["Symbol"],
                "name": tbl["Security"],
                "sector": tbl["GICS Sector"],
                "date_added": pd.to_datetime(tbl["Date added"], errors="coerce"),
            }
        )
        return out

    def index_changes(self) -> pd.DataFrame:
        html = self._get_html(self.cfg["data"]["wikipedia_changes_url"], "wikipedia_changes.html")
        tbl = pd.read_html(io.StringIO(html))[0]
        tbl.columns = ["date", "added", "added_name", "removed", "removed_name", "reason", "refs"]
        tbl = tbl.iloc[1:].copy()  # first row repeats the header
        tbl["date"] = pd.to_datetime(tbl["date"], format="%B %d, %Y")
        for c in ("added", "removed"):
            tbl[c] = tbl[c].map(lambda s: to_yahoo_symbol(s) if isinstance(s, str) else float("nan"))
        return tbl.drop(columns="refs").sort_values("date", ascending=False, kind="stable").reset_index(drop=True)

    # -------------------------------------------------------------------- yahoo
    def prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        bs = int(self.cfg["data"]["yahoo_batch_size"])
        frames: list[pd.DataFrame] = []

        def _unpack(d: pd.DataFrame, tks: list[str]) -> None:
            if d is None or d.empty:
                return
            for t in tks:
                if t not in d.columns.get_level_values(0):
                    continue
                s = d[t].dropna(how="all")
                if s.empty or s["Close"].dropna().empty:
                    continue
                s = s.reindex(columns=PRICE_FIELDS)
                s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
                s.index.name = "date"
                s.insert(0, "ticker", t)
                frames.append(s.reset_index())

        for i in range(0, len(tickers), bs):
            batch = tickers[i : i + bs]
            for attempt in range(3):
                try:
                    d = yf.download(batch, start=start, end=end_excl, auto_adjust=False, actions=True,
                                    group_by="ticker", progress=False, threads=True)
                    _unpack(d, batch)
                    break
                except Exception as exc:  # network hiccup / rate limit
                    log.warning("batch %d attempt %d failed: %s", i, attempt, exc)
                    time.sleep(5 * (attempt + 1))
            log.info("downloaded %d/%d tickers", min(i + bs, len(tickers)), len(tickers))
        have = {f["ticker"].iat[0] for f in frames}
        missing = [t for t in tickers if t not in have]
        for t in missing:  # one individual retry for tickers the batch call dropped
            try:
                d = yf.download(t, start=start, end=end_excl, auto_adjust=False, actions=True,
                                group_by="ticker", progress=False, threads=False)
                _unpack(d if isinstance(d.columns, pd.MultiIndex) else pd.concat({t: d}, axis=1), [t])
            except Exception as exc:
                log.warning("retry failed for %s: %s", t, exc)
        if not frames:
            return pd.DataFrame(columns=["date", "ticker", *PRICE_FIELDS])
        return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)

    def sector_labels(self, tickers: list[str]) -> pd.DataFrame:
        import yfinance as yf

        cache = self.raw_dir / "yahoo_sector_info.json"
        store: dict[str, str | None] = json.loads(cache.read_text()) if cache.exists() else {}
        for t in tickers:
            if t in store:
                continue
            try:
                store[t] = yf.Ticker(t).info.get("sector")
            except Exception:
                store[t] = None
            time.sleep(0.2)
        cache.write_text(json.dumps(store, indent=1, sort_keys=True))
        return pd.DataFrame({"ticker": tickers, "raw_sector": [store.get(t) for t in tickers], "source": "yahoo.info"})

    def risk_free(self, start: str, end: str) -> pd.Series:
        px = self.prices([self.cfg["tbill_ticker"]], start, end)
        s = px.set_index("date")["Close"] / 100.0
        s.name = "rf_annual"
        return s

    def trading_calendar(self, start: str, end: str) -> pd.DatetimeIndex:
        import pandas_market_calendars as mcal

        sched = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
        return pd.DatetimeIndex(sched.index).tz_localize(None).normalize()
