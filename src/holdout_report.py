"""Data-quality and coverage addendum for a (holdout) window; parametrised by window so it can be rehearsed in-sample."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .quality import check_prices
from .universe import coverage_by_year


def addendum(px: pd.DataFrame, sessions: pd.DatetimeIndex, membership: pd.DataFrame, sectors: pd.DataFrame, cfg: dict,
             window: tuple[pd.Timestamp, pd.Timestamp], out_dir: Path, title: str) -> pd.DataFrame:
    """Price checks restricted to ``window``, the full coverage-by-year table (all years), and a text report."""
    lo, hi = window
    special = set(cfg["etfs"]) | {cfg["benchmark"], cfg["tbill_ticker"]}
    px = px[px["ticker"] != cfg["tbill_ticker"]]
    sess_w = sessions[(sessions >= lo) & (sessions <= hi)]
    q = check_prices(px[(px["date"] >= lo) & (px["date"] <= hi)], sess_w, special)
    moves = check_prices(px, sessions, special)["moves_gt_50pct"]  # returns need the prior session, so use full history
    moves = moves[(moves["date"] >= lo) & (moves["date"] <= hi)]
    q["moves_gt_50pct"] = moves
    for k, v in q.items():
        v.to_csv(out_dir / f"dq_{title}_{k}.csv", index=False)

    close = px.pivot(index="date", columns="ticker", values="Close").reindex(sessions)
    has_price = close.notna().reindex(columns=membership.columns, fill_value=False)
    sector_ok = sectors.set_index("ticker")["status"].eq("ok")
    cov = coverage_by_year(membership.loc[membership.index.isin(sessions)], has_price, sector_ok, sorted(set(sessions.year)))
    cov["flag_usable_below_90pct"] = cov["share_usable"] < 0.90
    cov.to_csv(out_dir / "coverage_by_year.csv", index=False)

    mem_w = membership.loc[(membership.index >= lo) & (membership.index <= hi)]
    ever = mem_w.columns[mem_w.any(axis=0)]
    have = set(px["ticker"])
    no_px = sorted(t for t in ever if t not in have)
    no_sec = sorted(t for t in ever if t in have and not sector_ok.get(t, False))
    L = [f"# Data quality & coverage ({title} addendum, Stage 3)\n",
         f"Window {lo.date()} to {hi.date()}.\n", "## Coverage by calendar year (all years, average over sessions)\n",
         "| year | (a) PIT members | (b) with price data | (c) with sector label | (d) usable for trading | flag |", "|---|---|---|---|---|---|"]
    for _, r in cov.iterrows():
        L.append(f"| {int(r['year'])} | {r['avg_members']:.1f} | {r['share_price_data']:.1%} | {r['share_sector_label']:.1%} | {r['share_usable']:.1%} | "
                 f"{'**<90%**' if r['flag_usable_below_90pct'] else ''} |")
    L += ["", f"## Members during the window excluded from trading", f"- No downloadable prices ({len(no_px)}): {', '.join(no_px) or 'none'}",
          f"- Priced but no usable sector label ({len(no_sec)}): {', '.join(no_sec) or 'none'}", "",
          "## Price checks inside the window",
          f"- Duplicate (ticker,date) rows: **{len(q['duplicate_dates'])}**",
          f"- Tickers with NaN/zero/negative OHLC rows: **{len(q['nan_or_nonpositive_prices'])}**",
          f"- Tickers with missing NYSE sessions inside their history / non-session dates: **{len(q['missing_sessions'])}** "
          f"(ETFs/benchmark: {', '.join(q['missing_sessions'].query('is_etf_or_benchmark')['ticker']) or 'none'})",
          f"- Single-day |Adj Close return| > 50%: **{len(moves)}** rows ({int(moves['split_same_day'].sum()) if len(moves) else 0} on a split day). Kept as downloaded."]
    if len(moves):
        L += ["", moves.to_string(index=False)]
    (out_dir / f"data_quality_{title}.md").write_text("\n".join(L) + "\n")
    return cov
