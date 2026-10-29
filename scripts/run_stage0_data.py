"""Stage 0: download + cache data, rebuild point-in-time membership, quality + coverage reports.

Holdout discipline: raw prices are split on download into in-sample / holdout parquet files. All diagnostics here
read the in-sample file ONLY. Coverage columns that need holdout-period prices are deferred to Stage 3.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PROC_DIR, RAW_DIR, RESULTS_DIR, ROOT, load_config, sha256_file  # noqa: E402
from src.data_io import load_prices, split_and_save_prices  # noqa: E402
from src.providers import YahooWikipediaProvider  # noqa: E402
from src.quality import check_prices  # noqa: E402
from src.universe import assign_sectors, coverage_by_year, reconstruct_membership  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("stage0")


def main() -> None:
    cfg = load_config()
    prov = YahooWikipediaProvider(cfg)
    for d in (RAW_DIR, PROC_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    start, end = cfg["dates"]["data_start"], cfg["dates"]["end"]
    is_end = cfg["split"]["in_sample"][1]

    # ---- universe ---------------------------------------------------------------------------------
    current = prov.current_constituents()
    changes = prov.index_changes()
    sessions = prov.trading_calendar(start, end)
    membership, issues = reconstruct_membership(current["ticker"].tolist(), changes, sessions)
    membership.to_parquet(PROC_DIR / "membership.parquet")
    issues.to_csv(RESULTS_DIR / "membership_reconstruction_issues.csv", index=False)
    changes.to_csv(RAW_DIR / "index_changes.csv", index=False)
    current.to_csv(RAW_DIR / "current_constituents.csv", index=False)
    log.info("membership panel %s, %d reconstruction issues", membership.shape, len(issues))

    # ---- sectors ----------------------------------------------------------------------------------
    members_ever = list(membership.columns)
    sectors = assign_sectors(prov, cfg, current, members_ever)
    sectors.to_csv(PROC_DIR / "sector_map.csv", index=False)
    pd.DataFrame({"yahoo_sector": list(cfg["yahoo_sector_to_gics"]), "gics_sector": list(cfg["yahoo_sector_to_gics"].values())}
                 ).to_csv(PROC_DIR / "yahoo_to_gics_mapping.csv", index=False)

    # ---- prices -----------------------------------------------------------------------------------
    raw_path = RAW_DIR / "prices_raw.parquet"
    if raw_path.exists():
        raw = pd.read_parquet(raw_path)
    else:
        tickers = sorted(set(members_ever) | set(cfg["etfs"]) | {cfg["benchmark"], cfg["tbill_ticker"]})
        raw = prov.prices(tickers, start, end)
        raw.to_parquet(raw_path, index=False)
    split_and_save_prices(raw, is_end)
    del raw  # nothing below may touch holdout rows

    px = load_prices(include_holdout=False)
    log.info("in-sample price rows: %d, tickers: %d", len(px), px["ticker"].nunique())

    # ---- quality checks (in-sample window only) ---------------------------------------------------
    is_sessions = sessions[sessions <= pd.Timestamp(is_end)]
    special = set(cfg["etfs"]) | {cfg["benchmark"], cfg["tbill_ticker"]}
    # ^IRX is a yield (can print ~0 or negative), so return-based flags are meaningless for it; it is checked for sessions only.
    q = check_prices(px[px["ticker"] != cfg["tbill_ticker"]], is_sessions, special)
    for k, v in q.items():
        v.to_csv(RESULTS_DIR / f"dq_{k}.csv", index=False)

    have_t = set(px["ticker"])
    expected_special = [t for t in cfg["etfs"] + [cfg["benchmark"], cfg["tbill_ticker"]] if t not in have_t]
    if expected_special:
        raise SystemExit(f"Missing ETF/benchmark price data: {expected_special}")

    # ticker status
    first = px.groupby("ticker")["date"].min()
    last = px.groupby("ticker")["date"].max()
    ever = pd.Series(members_ever)
    in_is = membership[membership.index <= pd.Timestamp(is_end)].any(axis=0)
    st = sectors.set_index("ticker")
    ticker_table = pd.DataFrame({"ticker": members_ever})
    ticker_table["member_in_insample"] = ticker_table["ticker"].map(in_is)
    ticker_table["has_price_data_insample"] = ticker_table["ticker"].isin(have_t)
    ticker_table["sector_status"] = ticker_table["ticker"].map(st["status"])
    ticker_table["gics_sector"] = ticker_table["ticker"].map(st["gics_sector"])
    ticker_table["first_price"] = ticker_table["ticker"].map(first)
    ticker_table["last_price"] = ticker_table["ticker"].map(last)
    ticker_table["is_current_constituent"] = ticker_table["ticker"].isin(set(current["ticker"]))
    ticker_table.to_csv(PROC_DIR / "ticker_status.csv", index=False)

    # ---- coverage by year -------------------------------------------------------------------------
    close = px.pivot(index="date", columns="ticker", values="Close").reindex(is_sessions)
    has_price = close.notna().reindex(columns=membership.columns, fill_value=False)
    sector_ok = sectors.set_index("ticker")["status"].eq("ok")
    years_all = sorted(set(membership.index.year))
    is_years = [y for y in years_all if y <= pd.Timestamp(is_end).year]
    cov_is = coverage_by_year(membership.loc[membership.index <= pd.Timestamp(is_end)], has_price, sector_ok, is_years)
    cov_ho = coverage_by_year(membership.loc[membership.index > pd.Timestamp(is_end)], None, sector_ok,
                              [y for y in years_all if y > pd.Timestamp(is_end).year])
    cov = pd.concat([cov_is, cov_ho], ignore_index=True)
    cov["flag_usable_below_90pct"] = cov["share_usable"] < 0.90
    cov.to_csv(RESULTS_DIR / "coverage_by_year.csv", index=False)

    # sample statistics used to calibrate simulations (in-sample only, no signals)
    cal = calibration_stats(px, membership, sectors, cfg, is_end)
    (PROC_DIR / "sim_calibration.json").write_text(json.dumps(cal, indent=1))

    write_quality_md(cfg, q, cov, ticker_table, sectors, issues, current, is_end, sessions, px)

    # ---- manifest ---------------------------------------------------------------------------------
    man = {}
    for d in (RAW_DIR, PROC_DIR):
        for f in sorted(d.glob("*")):
            if f.is_file():
                man[str(f.relative_to(ROOT))] = sha256_file(f)
    (ROOT / "data" / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    log.info("Stage 0 complete.")


def calibration_stats(px: pd.DataFrame, membership: pd.DataFrame, sectors: pd.DataFrame, cfg: dict, is_end: str) -> dict:
    """Loose summary statistics (counts, vols, beta distribution) for the simulator. In-sample only; no signals."""
    adj = px.pivot(index="date", columns="ticker", values="Adj Close")
    ret = adj.pct_change(fill_method=None)
    ret = ret[(ret.index >= cfg["dates"]["signal_start"]) & (ret.index <= is_end)]
    s2e = cfg["sector_to_etf"]
    ok = sectors[sectors["status"] == "ok"].set_index("ticker")
    mem = membership[(membership.index >= cfg["dates"]["signal_start"]) & (membership.index <= is_end)]
    avg_members = mem.mean(axis=0)  # fraction of days a member
    out: dict = {"n_days": int(len(ret)), "sectors": {}}
    betas, idio = [], []
    for sec, etf in s2e.items():
        tk = [t for t in ok.index[ok["gics_sector"] == sec] if t in ret.columns and avg_members.get(t, 0) > 0.5]
        e = ret[etf]
        sub = ret[tk]
        cov = sub.apply(lambda s: s.cov(e))
        b = cov / e.var()
        resid = sub - np.outer(e.fillna(0).to_numpy(), b.to_numpy())
        isd = resid.std()
        betas += b.dropna().tolist()
        idio += isd.dropna().tolist()
        out["sectors"][etf] = dict(n_stocks=len(tk), etf_daily_vol=float(e.std()), beta_mean=float(b.mean()), beta_sd=float(b.std()),
                                   idio_vol_median=float(isd.median()), idio_vol_sd=float(isd.std()))
    out["beta_mean"], out["beta_sd"] = float(np.mean(betas)), float(np.std(betas))
    out["idio_vol_median"], out["idio_vol_sd"] = float(np.median(idio)), float(np.std(idio))
    spy = ret[cfg["benchmark"]]
    out["etf_corr_with_spy_mean"] = float(np.mean([ret[e].corr(spy) for e in cfg["etfs"]]))
    out["spy_daily_vol"] = float(spy.std())
    return out


def write_quality_md(cfg, q, cov, tt, sectors, issues, current, is_end, sessions, px) -> None:
    L: list[str] = []
    L.append("# Data quality & coverage (Stage 0)\n")
    L.append(f"Generated by `scripts/run_stage0_data.py`. Diagnostics cover **{cfg['dates']['data_start']} to {is_end} only**; "
             "holdout-period prices were split off at download and were not inspected (holdout checks run in Stage 3).\n")
    L.append("## Sources and conventions\n")
    L.append("- Prices: Yahoo Finance via `yfinance` (`auto_adjust=False`). `Close` is **split-adjusted but not dividend-adjusted** "
             "(true as-traded prices are not available from Yahoo); the $5 filter uses it, so stocks that later split may look cheaper "
             "than they were at the time. Returns use `Adj Close`; open-to-open uses `adj_open = Open*AdjClose/Close`.")
    L.append(f"- Universe: current S&P 500 list from Wikipedia ({len(current)} tickers) rolled back with the additions/removals table "
             f"({cfg['data']['wikipedia_changes_url']}; the table now lives on the 'Historical components' page, see DEVIATIONS.md). "
             "A change takes effect for the session on its effective date.")
    L.append(f"- Risk-free series for benchmark excess returns: `{cfg['tbill_ticker']}` (Yahoo 13-week T-bill yield).\n")
    L.append("## Survivorship bias\n")
    L.append("Survivorship bias is **reduced, not eliminated**. Removed stocks (acquired, bankrupt, shrunk) are far less likely than "
             "survivors to have Yahoo price history or a Yahoo sector label, so the usable universe is tilted towards survivors. "
             "The coverage table below quantifies this by year.\n")
    L.append("## Coverage by calendar year (average over sessions)\n")
    c = cov.copy()
    for col in ("share_sector_label", "share_price_data", "share_usable"):
        c[col] = c[col].map(lambda v: "deferred to Stage 3" if pd.isna(v) else f"{v:.1%}")
    c["avg_members"] = c["avg_members"].map(lambda v: f"{v:.1f}")
    L.append("| year | (a) PIT members | (b) with price data | (c) with sector label | (d) usable for trading | flag |")
    L.append("|---|---|---|---|---|---|")
    for _, r in c.iterrows():
        L.append(f"| {int(r['year'])} | {r['avg_members']} | {r['share_price_data']} | {r['share_sector_label']} | {r['share_usable']} | "
                 f"{'**<90%**' if r['flag_usable_below_90pct'] else ''} |")
    L.append("\n(b) and (d) for holdout years need holdout prices and are filled in by Stage 3.\n")
    L.append("## Excluded / problem tickers (point-in-time members during the in-sample window)\n")
    ins = tt[tt["member_in_insample"]]
    nop = ins[~ins["has_price_data_insample"]]
    nosec = ins[ins["has_price_data_insample"] & (ins["sector_status"] != "ok")]
    L.append(f"- Members in-sample: **{len(ins)}** tickers. With price data: {int(ins['has_price_data_insample'].sum())}. "
             f"No downloadable prices (excluded, logged): **{len(nop)}**. Priced but no usable sector label (excluded): **{len(nosec)}**.")
    L.append(f"- Full list: `data/processed/ticker_status.csv`; sector sources/mapping: `data/processed/sector_map.csv`, `yahoo_to_gics_mapping.csv`.")
    rem = ins[~ins["is_current_constituent"]]
    L.append(f"- Of the {len(rem)} in-sample members that are no longer in the index, {int((rem['has_price_data_insample'] & (rem['sector_status']=='ok')).sum())} "
             f"are usable ({(rem['has_price_data_insample'] & (rem['sector_status']=='ok')).mean():.0%}), versus "
             f"{((ins[ins['is_current_constituent']]['has_price_data_insample']) & (ins[ins['is_current_constituent']]['sector_status']=='ok')).mean():.0%} of survivors.")
    L.append("\nNo-price tickers:\n")
    L.append(", ".join(sorted(nop["ticker"])) or "none")
    L.append("\nPriced but excluded for sector:\n")
    L.append(", ".join(f"{r.ticker} ({r.sector_status})" for r in nosec.itertuples()) or "none")
    L.append(f"\n## Membership reconstruction issues ({len(issues)})\n")
    L.append("Logged, not silently fixed. See `results/membership_reconstruction_issues.csv`.\n")
    if len(issues):
        L.append(issues.head(25).to_markdown(index=False) if hasattr(issues, "to_markdown") else issues.head(25).to_string(index=False))
    L.append("\n## Price checks (in-sample window)\n")
    nonpos = q["nan_or_nonpositive_prices"]
    ms = q["missing_sessions"]
    mv = q["moves_gt_50pct"]
    L.append(f"- Duplicate (ticker,date) rows: **{len(q['duplicate_dates'])}**.")
    L.append(f"- Tickers with NaN/zero/negative OHLC rows: **{len(nonpos)}** (total rows {int(nonpos['n_rows'].sum()) if len(nonpos) else 0}). `results/dq_nan_or_nonpositive_prices.csv`.")
    L.append(f"- Tickers with missing NYSE sessions inside their price history, or dates that are not NYSE sessions: **{len(ms)}**. `results/dq_missing_sessions.csv`.")
    etf_ms = ms[ms["is_etf_or_benchmark"]]
    L.append(f"  - ETFs/benchmark among them: {', '.join(etf_ms['ticker']) if len(etf_ms) else 'none'}.")
    L.append(f"- Single-day |Adj Close return| > 50%: **{len(mv)}** rows ({int(mv['split_same_day'].sum()) if len(mv) else 0} coincide with a split flag). `results/dq_moves_gt_50pct.csv`. "
             "Nothing was removed or winsorised; flagged observations are used as downloaded.")
    if len(mv):
        L.append("\n" + mv.head(25).to_string(index=False))
    (RESULTS_DIR / "data_quality.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
