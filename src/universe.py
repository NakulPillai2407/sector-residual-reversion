"""Point-in-time S&P 500 membership, sector labels and coverage reporting."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .providers import DataProvider

log = logging.getLogger(__name__)


def reconstruct_membership(
    current_tickers: list[str], changes: pd.DataFrame, trading_days: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Work backwards from today's list using the additions/removals table.

    Convention: a change with effective date D takes effect for the session on D
    (a stock added on D is a member on D; a stock removed on D is not). Changes dated on
    non-sessions apply from the next session.

    Returns (membership bool panel date x ticker, issues log).
    """
    start = trading_days[0]
    state = set(current_tickers)
    issues: list[dict[str, Any]] = []
    ch = changes.sort_values("date", ascending=False, kind="stable")

    def _events(row: pd.Series) -> list[tuple[str, int]]:
        ev = []
        if isinstance(row["added"], str):
            ev.append((row["added"], +1))
        if isinstance(row["removed"], str):
            ev.append((row["removed"], -1))
        return ev

    # Undo every change that happened after the first panel day (state at `start`).
    for _, row in ch[ch["date"] > start].iterrows():
        for tk, sign in _events(row):
            if sign == +1:  # undo an addition
                if tk not in state:
                    issues.append(dict(date=row["date"], ticker=tk, issue="undo-add: ticker not in membership", row=row["reason"]))
                state.discard(tk)
            else:  # undo a removal
                if tk in state:
                    issues.append(dict(date=row["date"], ticker=tk, issue="undo-remove: ticker already in membership", row=row["reason"]))
                state.add(tk)
        if isinstance(row["added"], str) and isinstance(row["removed"], str) and str(row["added_name"]) == str(row["removed_name"]):
            issues.append(dict(date=row["date"], ticker=row["added"], issue="possible symbol change (same security name)", row=row["reason"]))

    fwd = ch[ch["date"] > start].sort_values("date", kind="stable")
    all_tickers = sorted(state | set(fwd["added"].dropna()) | set(fwd["removed"].dropna()))
    col = {t: i for i, t in enumerate(all_tickers)}
    panel = np.zeros((len(trading_days), len(all_tickers)), dtype=bool)
    events = [(r["date"], tk, s) for _, r in fwd.iterrows() for tk, s in _events(r)]
    ei = 0
    for di, d in enumerate(trading_days):
        while ei < len(events) and events[ei][0] <= d:
            _, tk, sign = events[ei]
            if sign == +1:
                state.add(tk)
            else:
                state.discard(tk)
            ei += 1
        for tk in state:
            panel[di, col[tk]] = True
    # events dated after the last trading day are not applied; state should match the current list only if none exist
    final_gap = set(current_tickers) ^ state
    if ei == len(events) and final_gap:
        for tk in sorted(final_gap):
            issues.append(dict(date=pd.NaT, ticker=tk, issue="forward replay does not reproduce current list", row=""))
    out = pd.DataFrame(panel, index=trading_days, columns=all_tickers)
    out = out.loc[:, out.any(axis=0)]
    out.index.name = "date"
    return out, pd.DataFrame(issues, columns=["date", "ticker", "issue", "row"])


def assign_sectors(
    provider: DataProvider, cfg: dict[str, Any], current: pd.DataFrame, tickers: list[str]
) -> pd.DataFrame:
    """Sector label for every ticker ever in the panel (current labels applied across the whole sample)."""
    s2e: dict[str, str] = cfg["sector_to_etf"]
    ymap: dict[str, str] = cfg["yahoo_sector_to_gics"]
    cur = current.set_index("ticker")["sector"].to_dict()
    rows = []
    removed = [t for t in tickers if t not in cur]
    yl = provider.sector_labels(removed).set_index("ticker")["raw_sector"].to_dict() if removed else {}
    for t in tickers:
        if t in cur and cur[t] in s2e:
            rows.append(dict(ticker=t, raw_sector=cur[t], gics_sector=cur[t], etf=s2e[cur[t]], source="wikipedia.current", status="ok"))
        elif t in cur:
            rows.append(dict(ticker=t, raw_sector=cur[t], gics_sector=None, etf=None, source="wikipedia.current", status="excluded: unmapped sector"))
        else:
            raw = yl.get(t)
            g = ymap.get(raw) if raw else None
            if g is None:
                rows.append(dict(ticker=t, raw_sector=raw, gics_sector=None, etf=None, source="yahoo.info",
                                 status="excluded: no sector label" if raw is None else f"excluded: unmapped Yahoo sector {raw!r}"))
            else:
                rows.append(dict(ticker=t, raw_sector=raw, gics_sector=g, etf=s2e[g], source="yahoo.info->mapping", status="ok"))
    return pd.DataFrame(rows)


def coverage_by_year(
    membership: pd.DataFrame,
    has_price: pd.DataFrame | None,
    sector_ok: pd.Series,
    years: list[int],
) -> pd.DataFrame:
    """Per calendar year averages over sessions: (a) #members, (b) share with price data,
    (c) share with sector label, (d) share usable for trading (price AND sector).
    ``has_price`` None => price-dependent columns are NaN (deferred)."""
    rows = []
    sec = sector_ok.reindex(membership.columns).fillna(False).to_numpy()
    for y in years:
        m = membership[membership.index.year == y]
        if m.empty:
            continue
        n = m.sum(axis=1).to_numpy(dtype=float)
        sec_share = (m.to_numpy() & sec).sum(axis=1) / n
        row = dict(year=y, avg_members=n.mean(), share_sector_label=sec_share.mean(), n_sessions=len(m))
        if has_price is None:
            row.update(share_price_data=np.nan, share_usable=np.nan)
        else:
            hp = has_price.reindex(index=m.index, columns=m.columns).fillna(False).to_numpy()
            row["share_price_data"] = ((m.to_numpy() & hp).sum(axis=1) / n).mean()
            row["share_usable"] = ((m.to_numpy() & hp & sec).sum(axis=1) / n).mean()
        rows.append(row)
    return pd.DataFrame(rows)
