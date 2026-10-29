"""Compile results/results_pack.md from the CSV/JSON outputs of the stages."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import RESULTS_DIR, ROOT

COLS = ["strategy", "cost_scenario", "ann_return", "ann_vol", "sharpe", "sharpe_ci_lo", "sharpe_ci_hi", "max_drawdown",
        "turnover_daily", "avg_long", "avg_short", "hit_rate", "breakeven_bps_per_side", "beta_spy", "alpha_ann", "alpha_nw_t"]
NAMES = {"ann_return": "ann ret", "ann_vol": "ann vol", "sharpe": "Sharpe", "sharpe_ci_lo": "CI lo", "sharpe_ci_hi": "CI hi",
         "max_drawdown": "MDD", "turnover_daily": "turnover/day", "avg_long": "avg L", "avg_short": "avg S", "hit_rate": "hit",
         "breakeven_bps_per_side": "breakeven bp", "beta_spy": "beta SPY", "alpha_ann": "alpha ann", "alpha_nw_t": "alpha NW t",
         "cost_scenario": "costs"}


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def _tab(tabs: pd.DataFrame, groups: list[str]) -> str:
    t = tabs[tabs["group"].isin(groups)][COLS].rename(columns=NAMES)
    return _md(t)


def _period_section(tag: str, label: str) -> str:
    f = RESULTS_DIR / f"{tag}_strategy_tables.csv"
    if not f.exists():
        return f"_{label}: not run._\n"
    tabs = pd.read_csv(f)
    fm = pd.read_csv(RESULTS_DIR / f"{tag}_fama_macbeth.csv")
    dec = json.loads((RESULTS_DIR / f"{tag}_decision_metrics.json").read_text())
    diag = json.loads((RESULTS_DIR / f"{tag}_primary_diagnostics.json").read_text())
    out = [f"### {label} ({dec['start']} to {dec['end']}, {dec['n_days']} sessions)\n"]
    out.append("All P&L is stamped by exit date. Sharpe = mean/std x sqrt(252) of daily returns including flat days, no risk-free subtraction "
               "(dollar-neutral book). CI = stationary bootstrap (mean block 10, 10,000 resamples, 95%). Alpha/beta: daily strategy return on SPY "
               "open-to-open return, Newey-West 5 lags. `breakeven bp` = per-side stock cost at which mean net return is zero.\n")
    out.append("**Primary spec (open_open, sector-ETF hedge, |z|>=1, 1-day hold)**\n")
    out.append(_tab(tabs, ["primary"]))
    out.append(f"\nDiagnostics: {diag['days_with_position']} of {diag['period_days']} sessions in this period held a position. Empty-leg counts "
               f"(no trade that day) cover ALL signal dates in the data loaded for the run, i.e. warm-up + in-sample + (for the holdout run) holdout: "
               f"{diag['empty_leg_days']} empty-leg day(s) of {diag['signal_days']} (no long leg: {diag['no_long_days']}, no short leg: {diag['no_short_days']}). "
               f"Position-days with a missing price (0 return assumed): {diag['nan_position_days']}.\n")
    out.append("**Timing variants** (`same_close` is NON-TRADABLE: it enters at the close that generated the signal and is an upper bound only)\n")
    out.append(_tab(tabs, ["variants"]))
    out.append("\n**Pre-registered robustness (non-gating)**\n")
    out.append(_tab(tabs, ["robustness"]))
    out.append("\n**Benchmarks** (SPY and the equal-weight sector ETFs: Sharpe is on EXCESS returns over `^IRX`/252; the unhedged signal is a dollar-neutral stock book with no ETF leg)\n")
    out.append(_tab(tabs, ["benchmark", "benchmark_signal"]))
    out.append("\n**EXPLORATORY -- not used to choose anything; many comparisons, expect false positives**\n")
    out.append(_tab(tabs, ["exploratory"]))
    out.append("\n**Fama-MacBeth** (daily cross-sectional OLS of next-day residual on today's residual, with intercept; mean slope, Newey-West 5-lag t)\n")
    out.append(_md(fm[["regression", "sector", "mean_coef", "nw_se", "nw_t", "n_days"]]))
    out.append("\n**Decision-rule inputs**\n")
    keys = ["fm_mean_coef", "fm_nw_t", "H1a", "net5_sharpe", "net5_sharpe_ci_lo", "net5_sharpe_ci_hi", "H1b_i_net_ci_above_zero", "alpha_vs_spy_ann",
            "alpha_nw_t", "H1b_ii_alpha_t_gt_2", "H1b", "gross_sharpe", "gross_sharpe_ci_lo", "gross_sharpe_ci_hi", "sharpe_diff_vs_spy_buy_hold",
            "sharpe_diff_ci_lo", "sharpe_diff_ci_hi", "breakeven_bps_per_side", "min_detectable_sharpe"]
    out.append(_md(pd.DataFrame({"metric": keys, "value": [dec[k] for k in keys]})))
    out.append("")
    figs = sorted((RESULTS_DIR / "figures").glob(f"{tag}_fig*.png"))
    for f_ in figs:
        out.append(f"![{f_.stem}](figures/{f_.name})")
    out.append("")
    return "\n".join(out)


def build_pack() -> Path:
    L: list[str] = ["# Results pack: ETF-hedged residual reversion (Holdout Research, strategy 03)\n"]
    ho = RESULTS_DIR / "ho_decision_metrics.json"
    if ho.exists():
        d = json.loads(ho.read_text())
        L.append("## 0. Verdict (holdout, run once)\n")
        L.append(f"**VERDICT: {d['verdict']}**  (H1a: {'holds' if d['H1a'] else 'fails'}; H1b: {'holds' if d['H1b'] else 'fails'})\n")
        L.append(f"- H1a: holdout FM mean coefficient {d['fm_mean_coef']:.4f}, Newey-West t = {d['fm_nw_t']:.2f} (needs negative and t < -2).")
        L.append(f"- H1b(i): net Sharpe at 5 bp/side = {d['net5_sharpe']:.2f}, 95% stationary-bootstrap CI [{d['net5_sharpe_ci_lo']:.2f}, {d['net5_sharpe_ci_hi']:.2f}] (needs CI entirely above 0).")
        L.append(f"- H1b(ii): alpha vs SPY = {d['alpha_vs_spy_ann']:.3f}/yr, NW t = {d['alpha_nw_t']:.2f} (needs t > 2); beta {d['beta_vs_spy']:.3f}.")
        L.append(f"- Gross Sharpe {d['gross_sharpe']:.2f}, CI [{d['gross_sharpe_ci_lo']:.2f}, {d['gross_sharpe_ci_hi']:.2f}].")
        L.append(f"- Non-gating: Sharpe difference vs SPY buy-and-hold (excess over T-bill) = {d['sharpe_diff_vs_spy_buy_hold']:.2f}, paired-bootstrap 95% CI "
                 f"[{d['sharpe_diff_ci_lo']:.2f}, {d['sharpe_diff_ci_hi']:.2f}]; breakeven cost = {d['breakeven_bps_per_side']:.2f} bp/side; "
                 f"minimum detectable Sharpe for this holdout length = {d['min_detectable_sharpe']:.2f}.\n")
        if d["verdict"] != "GROSS-ONLY":
            L.append(f"Relative to the pre-registered prior (a detectable gross effect that does not survive costs): the verdict is {d['verdict']}. "
                     f"The gross effect was not detectable either (FM t = {d['fm_nw_t']:.2f}; gross Sharpe CI includes 0). Because the minimum detectable "
                     f"Sharpe is {d['min_detectable_sharpe']:.2f}, a true gross Sharpe below that level could not have been detected in this holdout.\n")
        L.append("Net result: " + ("net Sharpe is NEGATIVE at 5 bp per side; the strategy lost money after costs.\n" if d["net5_sharpe"] < 0 else "net Sharpe is positive at 5 bp per side.\n"))
    else:
        L.append("## 0. Verdict\n\n**Holdout not run yet.** The sections below contain data, simulation-validation and in-sample results only.\n")

    L.append("## 1. Data summary and coverage\n")
    for name in ("data_quality.md", "data_quality_holdout.md"):
        p = RESULTS_DIR / name
        if p.exists():
            txt = p.read_text().replace("# Data quality & coverage (Stage 0)", "### Stage 0 report").replace("# Data quality & coverage (holdout addendum, Stage 3)", "### Holdout addendum (Stage 3)")
            L.append(txt)
    L.append("\nSurvivorship bias is **reduced, not eliminated**: removed stocks are less likely to have Yahoo prices or sector labels than survivors, so the tested universe over-represents survivors, most strongly in the early years.\n")

    L.append("## 2. Simulation validation (Stage 1)\n")
    c = pd.read_csv(RESULTS_DIR / "stage1_checks.csv")
    L.append(_md(c))
    ex = json.loads((RESULTS_DIR / "stage1_extra.json").read_text())
    L.append("\nExtra simulation statistics: " + ", ".join(f"{k}={v:.3f}" for k, v in ex.items()) + "\n")
    pl = pd.read_csv(RESULTS_DIR / "stage1_placebo.csv")
    L.append(f"Placebo (real in-sample data, each stock hedged against a random wrong ETF; {len(pl)} shuffles): median FM coef {pl['fm_coef'].median():.4f} "
             f"(median NW t {pl['fm_t'].median():.2f}), median gross Sharpe {pl['gross_sharpe'].median():.2f}, median net-5bp Sharpe {pl['net_sharpe'].median():.2f}; "
             f"shuffles flagged significant: {((pl['fm_t'] < -2) | (pl['net_ci_lo'] > 0)).mean():.0%}. "
             "Note that a wrong-ETF hedge still removes most market beta, so any generic (non-sector) short-term reversal would also show up here.\n")

    L.append("## 3. Performance tables\n")
    L.append(_period_section("is", "In-sample"))
    L.append(_period_section("ho", "Holdout"))

    L.append("## 4. Deviations, exclusions and assumptions\n")
    L.append("Full dated log: `DEVIATIONS.md`.\n")
    dev = (ROOT / "DEVIATIONS.md").read_text()
    L.append(dev.split("---", 1)[-1].strip() + "\n")
    L.append("Standing assumptions and exclusions:\n")
    L.append("- Borrow cost is a flat 50 bp/yr on all short notional; real borrow varies by name and is much higher for hard-to-borrow stocks.")
    L.append("- Sector labels are CURRENT labels applied to the whole sample (e.g. Visa/Mastercard/PayPal were Information Technology until March 2023). Removed tickers are labelled via Yahoo `info` mapped with an explicit table; those without a label are excluded.")
    L.append("- Removed/delisted tickers without Yahoo price history are excluded (logged), so removed stocks are only partially covered.")
    L.append("- Betas are estimated on close-to-close returns and applied to open-to-open holds (and the ETF hedge uses the same betas).")
    L.append("- The $5 filter uses Yahoo `Close`, which is split-adjusted but not dividend-adjusted; true as-traded prices are unavailable.")
    L.append("- No delisting returns; a held name with a missing price contributes 0 return.")
    L.append("- Costs: 5 bp/side stock, 1 bp/side ETF on target-weight changes; no market impact, no capacity limits, no intraday drift modelling; fills are assumed at the open.\n")

    L.append("## 5. Limitations\n")
    L.append("- A free-data, long-short daily strategy with ~1.7x daily gross turnover is dominated by trading costs; conclusions about net performance depend almost entirely on the assumed cost levels and fills at the open.")
    L.append("- Yahoo adjusted data can contain errors; flagged >50% moves were kept as downloaded.")
    L.append("- The point-in-time universe is reconstructed from a Wikipedia table that may be incomplete (see reconstruction issues); coverage of removed stocks is partial.")
    L.append("- The holdout is short (about 2.75 years) so only large Sharpe ratios are detectable (see minimum detectable Sharpe); a non-significant result is not evidence of absence.")
    L.append("- Exploratory variants and per-sector results involve many comparisons and were never used to change the primary specification.")
    L.append("- Results are for large-cap US stocks 2018-2026 and may not generalise.\n")

    p = RESULTS_DIR / "results_pack.md"
    p.write_text("\n".join(L))
    return p
