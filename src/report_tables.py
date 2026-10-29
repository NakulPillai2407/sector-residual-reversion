"""Tables, figures and the mechanical decision rule. The SAME code produces the Stage 2 (in-sample) and Stage 3 (holdout) outputs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import costs as C
from .backtest import BacktestResult, Backtester, Spec, cost_models
from .config import RESULTS_DIR
from .signals import MarketData, eligibility, returns_oo
from .stats import (alpha_beta, bootstrap_sharpe_ci, fama_macbeth, max_drawdown, min_detectable_sharpe, nw_mean,
                    paired_bootstrap_diff_ci, sharpe)

NON_TRADABLE = {"same_close"}
SCEN = {"gross": None}  # filled from cost models


@dataclass
class Period:
    tag: str
    start: pd.Timestamp
    end: pd.Timestamp

    def mask(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return (idx >= self.start) & (idx <= self.end)

    @property
    def years(self) -> float:
        return float(np.busday_count(self.start.date(), self.end.date()) / 252.0)


def spec_label(spec: Spec) -> str:
    return spec.name + ("  [NON-TRADABLE UPPER BOUND]" if spec.timing in NON_TRADABLE else "")


class PeriodReport:
    def __init__(self, md: MarketData, rf: pd.Series, cfg: dict, period: Period, seed_offset: int = 0):
        self.md, self.cfg, self.period = md, cfg, period
        self.bt = Backtester(md, cfg)
        self.cms = cost_models(cfg)
        self.inf = cfg["inference"]
        self.rng = np.random.default_rng(int(cfg["seed"]) + 20_000 + seed_offset)
        self.idx = md.stocks.close.index
        self.m = period.mask(self.idx)
        spy = returns_oo(md.etfs.adj_open[[cfg["benchmark"]]])[cfg["benchmark"]]
        self.spy_oo = spy
        self.rf_daily = (rf.shift(1) / cfg["costs"]["days_per_year"]).reindex(self.idx).fillna(0.0)
        self.runs: dict[str, BacktestResult] = {}

    # ------------------------------------------------------------------ helpers
    def run(self, spec: Spec) -> BacktestResult:
        if spec.name not in self.runs:
            self.runs[spec.name] = self.bt.run(spec)
        return self.runs[spec.name]

    def series(self, res: BacktestResult) -> dict[str, pd.Series]:
        out = {"gross": res.daily["gross"]}
        for k, cm in self.cms.items():
            out[k] = res.net(cm)
        return {k: v[self.m] for k, v in out.items()}

    def _metrics(self, r: pd.Series, daily: pd.DataFrame | None, cm: C.CostModel | None, excess: pd.Series | None = None) -> dict:
        ann = self.inf["ann_factor"]
        basis = r if excess is None else excess
        lo, hi, _ = bootstrap_sharpe_ci(basis, self.rng, self.inf["bootstrap_resamples"], self.inf["bootstrap_mean_block"], self.inf["ci_level"], ann)
        mean, se, t = nw_mean(basis, self.inf["nw_lags"])
        ab = alpha_beta(r, self.spy_oo[self.m], self.inf["nw_lags"], ann)
        row = dict(ann_return=r.mean() * ann, ann_vol=r.std(ddof=1) * np.sqrt(ann), sharpe=sharpe(basis, ann),
                   sharpe_ci_lo=lo, sharpe_ci_hi=hi, mean_nw_t=t, max_drawdown=max_drawdown(r),
                   beta_spy=ab["beta"], alpha_ann=ab["alpha_ann"], alpha_nw_t=ab["alpha_t"],
                   sharpe_basis="excess over T-bill" if excess is not None else "raw (dollar-neutral)")
        if daily is not None:
            d = daily[self.m]
            act = d["active"].to_numpy()
            row.update(turnover_daily=d["turn_stock"].mean(), turnover_annual=d["turn_stock"].mean() * ann,
                       etf_turnover_daily=d["turn_etf"].mean(), avg_long=d["n_long"][act].mean() if act.any() else np.nan,
                       avg_short=d["n_short"][act].mean() if act.any() else np.nan, active_day_share=act.mean(),
                       hit_rate=(r[act] > 0).mean() if act.any() else np.nan)
            if cm is not None:
                row["breakeven_bps_per_side"] = C.breakeven_bps(d["gross"], d["turn_stock"], d["turn_etf"], d["short"], cm)
        return row

    def strategy_rows(self, spec: Spec, group: str, scen: tuple[str, ...] = ("gross", "net_2bps", "net_5bps", "net_10bps")) -> pd.DataFrame:
        res = self.run(spec)
        s = self.series(res)
        rows = []
        for k in scen:
            r = self._metrics(s[k], res.daily, self.cms.get("net_5bps"))
            r.update(group=group, strategy=spec_label(spec), cost_scenario=k, n_days=int(self.m.sum()))
            rows.append(r)
        return pd.DataFrame(rows)

    def benchmark_rows(self) -> pd.DataFrame:
        etf_oo = returns_oo(self.md.etfs.adj_open[list(self.cfg["etfs"])])
        defs = {"SPY buy-and-hold (open-to-open)": self.spy_oo,
                "Equal-weight 11 sector ETFs (daily rebalanced, open-to-open)": etf_oo.mean(axis=1)}
        rows = []
        for name, r in defs.items():
            r = r[self.m]
            ex = r - self.rf_daily[self.m]
            row = self._metrics(r, None, None, excess=ex)
            row.update(group="benchmark", strategy=name, cost_scenario="n/a", n_days=int(self.m.sum()))
            rows.append(row)
        return pd.DataFrame(rows)

    # --------------------------------------------------------------- strategy sets
    def specs(self) -> dict[str, list[Spec]]:
        ro, ex = self.cfg["robustness"], self.cfg["exploratory"]
        s = {
            "primary": [Spec("PRIMARY open_open, sector-ETF hedge, |z|>=1, 1-day hold")],
            "variants": [Spec("close_close_lag", timing="close_close_lag"), Spec("same_close", timing="same_close")],
            "robustness": [Spec("(i) SPY-only hedge", hedge="spy"),
                           Spec("(ii) excl. Visa/Mastercard/PayPal", exclude_tickers=tuple(ro["exclude_tickers"])),
                           Spec(f"(iii) excl. top {ro['exclude_top_n_adv']} by {ro['adv_window']}d avg dollar volume",
                                exclude_top_adv=ro["exclude_top_n_adv"], adv_window=ro["adv_window"])],
            "benchmark_signal": [Spec("Unhedged signal (stock book only, no ETF leg)", hedge="none")],
        }
        expl = [Spec(f"EXPLORATORY threshold {t:g}", threshold=t) for t in ex["thresholds"]]
        expl += [Spec(f"EXPLORATORY hold {h}d (overlapping sleeves)", hold=h) for h in ex["holds"]]
        expl += [Spec(f"EXPLORATORY continuous -z clipped +/-{ex['continuous_clip']:g}", mode="continuous", clip=ex["continuous_clip"], threshold=0.0),
                 Spec(f"EXPLORATORY excl. |z|>{ex['exclude_abs_z_above']:g}", abs_z_cap=ex["exclude_abs_z_above"])]
        expl += [Spec(f"EXPLORATORY sector {g}", sector=g) for g in self.cfg["sector_to_etf"]]
        s["exploratory"] = expl
        return s

    def all_strategy_tables(self) -> pd.DataFrame:
        parts = []
        sp = self.specs()
        for grp in ("primary", "variants", "robustness", "benchmark_signal"):
            for spec in sp[grp]:
                parts.append(self.strategy_rows(spec, grp))
        parts.append(self.benchmark_rows())
        for spec in sp["exploratory"]:
            parts.append(self.strategy_rows(spec, "exploratory", scen=("gross", "net_5bps")))
        # first-half / second-half of the PRIMARY spec (exploratory)
        prim = self.run(sp["primary"][0])
        s = self.series(prim)
        idx = s["gross"].index
        mid = idx[len(idx) // 2]
        for lab, sel in (("first half", idx < mid), ("second half", idx >= mid)):
            rows = []
            for k in ("gross", "net_5bps"):
                r = s[k][sel]
                d = prim.daily[self.m][sel]
                act = d["active"].to_numpy()
                ab = alpha_beta(r, self.spy_oo[self.m][sel], self.inf["nw_lags"])
                lo, hi, _ = bootstrap_sharpe_ci(r, self.rng, self.inf["bootstrap_resamples"], self.inf["bootstrap_mean_block"])
                rows.append(dict(group="exploratory", strategy=f"EXPLORATORY primary spec, {lab} ({idx[sel][0].date()} to {idx[sel][-1].date()})",
                                 cost_scenario=k, n_days=int(sel.sum()), ann_return=r.mean() * 252, ann_vol=r.std() * np.sqrt(252), sharpe=sharpe(r),
                                 sharpe_ci_lo=lo, sharpe_ci_hi=hi, mean_nw_t=nw_mean(r, 5)[2], max_drawdown=max_drawdown(r), beta_spy=ab["beta"],
                                 alpha_ann=ab["alpha_ann"], alpha_nw_t=ab["alpha_t"], turnover_daily=d["turn_stock"].mean(),
                                 hit_rate=(r[act] > 0).mean() if act.any() else np.nan, sharpe_basis="raw (dollar-neutral)"))
            parts.append(pd.DataFrame(rows))
        return pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------ Fama-MacBeth
    def fm_tables(self) -> pd.DataFrame:
        rows = []
        for kind, lab in (("cc", "close-to-close e(t+1) on e(t)  [primary]"), ("oo", "next-day open-to-open residual (open t+1 -> t+2) on e(t)  [secondary]")):
            for sector in [None, *self.cfg["sector_to_etf"]]:
                # Per-sector cross-sections are small (e.g. Communication Services); 10 names minimum there, 20 overall.
                sl = self.bt.fm_slopes(kind, sector=sector, min_names=10 if sector else 20)
                sl = sl[self.m].dropna()
                fm = fama_macbeth(sl, self.inf["nw_lags"])
                rows.append(dict(regression=lab, sector=sector or "ALL", **fm))
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- decision rule
    def decision(self) -> dict:
        prim = self.run(self.specs()["primary"][0])
        s = self.series(prim)
        fm_all = fama_macbeth(self.bt.fm_slopes("cc")[self.m].dropna(), self.inf["nw_lags"])
        h1a = bool(fm_all["mean_coef"] < 0 and fm_all["nw_t"] < -2)
        net, gross = s["net_5bps"], s["gross"]
        nlo, nhi, _ = bootstrap_sharpe_ci(net, self.rng, self.inf["bootstrap_resamples"], self.inf["bootstrap_mean_block"])
        glo, ghi, _ = bootstrap_sharpe_ci(gross, self.rng, self.inf["bootstrap_resamples"], self.inf["bootstrap_mean_block"])
        ab = alpha_beta(net, self.spy_oo[self.m], self.inf["nw_lags"])
        h1b_i, h1b_ii = bool(nlo > 0), bool(ab["alpha_t"] > 2)
        h1b = h1b_i and h1b_ii
        verdict = "SUPPORTED" if (h1a and h1b) else ("GROSS-ONLY" if (h1a and glo > 0) else "NOT SUPPORTED")
        spy_ex = (self.spy_oo - self.rf_daily)[self.m]
        diff, dlo, dhi = paired_bootstrap_diff_ci(net, spy_ex, self.rng, self.inf["bootstrap_resamples"], self.inf["bootstrap_mean_block"])
        be = C.breakeven_bps(prim.daily["gross"][self.m], prim.daily["turn_stock"][self.m], prim.daily["turn_etf"][self.m],
                             prim.daily["short"][self.m], self.cms["net_5bps"])
        n_days = int(self.m.sum())
        return dict(
            period=self.period.tag, start=str(self.period.start.date()), end=str(self.period.end.date()), n_days=n_days,
            fm_mean_coef=fm_all["mean_coef"], fm_nw_t=fm_all["nw_t"], H1a=h1a,
            net5_sharpe=sharpe(net), net5_sharpe_ci_lo=nlo, net5_sharpe_ci_hi=nhi, H1b_i_net_ci_above_zero=h1b_i,
            alpha_vs_spy_ann=ab["alpha_ann"], alpha_nw_t=ab["alpha_t"], beta_vs_spy=ab["beta"], H1b_ii_alpha_t_gt_2=h1b_ii, H1b=h1b,
            gross_sharpe=sharpe(gross), gross_sharpe_ci_lo=glo, gross_sharpe_ci_hi=ghi,
            sharpe_diff_vs_spy_buy_hold=diff, sharpe_diff_ci_lo=dlo, sharpe_diff_ci_hi=dhi,
            breakeven_bps_per_side=be, min_detectable_sharpe=min_detectable_sharpe(n_days / 252.0),
            verdict=verdict,
        )

    def primary_diagnostics(self) -> dict:
        prim = self.run(self.specs()["primary"][0])
        d = prim.daily[self.m]
        out = dict(prim.diag)
        out.update(period_days=int(self.m.sum()), days_with_position=int(d["active"].sum()),
                   avg_stock_turnover_daily=float(d["turn_stock"].mean()))
        return out


# ------------------------------------------------------------------------ figures
def make_figures(rep: PeriodReport, out_dir: Path, prefix: str) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "font.size": 9, "axes.titlesize": 10})
    c_gross, c_net, c_spy, c_ref = "#1b6ca8", "#d1495b", "#555555", "#8a8a8a"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    prim = rep.run(rep.specs()["primary"][0])
    s = rep.series(prim)
    d = prim.daily[rep.m]
    tag = rep.period.tag

    # 1. cumulative gross vs net vs SPY
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for k, lab, col in (("gross", "Strategy gross", c_gross), ("net_5bps", "Strategy net (5 bp/side)", c_net)):
        ax.plot((1 + s[k]).cumprod(), label=lab, color=col, lw=1.4)
    ax.plot((1 + rep.spy_oo[rep.m]).cumprod(), label="SPY buy-and-hold (open-to-open)", color=c_spy, lw=1.1, ls="--")
    ax.axhline(1, color=c_ref, lw=0.8)
    ax.set_title(f"Cumulative growth of 1 ({tag}: {rep.period.start.date()} to {rep.period.end.date()})")
    ax.legend(frameon=False)
    p = out_dir / f"{prefix}_fig1_cumulative.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    # 2. rolling 1-year Sharpe
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for k, lab, col in (("gross", "gross", c_gross), ("net_5bps", "net 5 bp", c_net)):
        r = s[k]
        ax.plot(r.rolling(252).mean() / r.rolling(252).std() * np.sqrt(252), label=lab, color=col, lw=1.3)
    sp = rep.spy_oo[rep.m]
    ax.plot(sp.rolling(252).mean() / sp.rolling(252).std() * np.sqrt(252), label="SPY (raw)", color=c_spy, lw=1, ls="--")
    ax.axhline(0, color=c_ref, lw=0.8)
    ax.set_title("Rolling 1-year Sharpe (252 sessions)"); ax.legend(frameon=False)
    p = out_dir / f"{prefix}_fig2_rolling_sharpe.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    # 3. cost sensitivity
    grid = np.arange(0, 15.5, 0.5)
    base = rep.cms["net_5bps"]
    nets = [sharpe(prim.daily["gross"][rep.m] - C.cost_series(d["turn_stock"], d["turn_etf"], d["short"], C.CostModel(b, base.etf_bps, base.borrow_bps_per_year, base.days_per_year))) for b in grid]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(grid, nets, color=c_net, lw=1.6)
    ax.axhline(0, color=c_ref, lw=0.8)
    for b in (2, 5, 10):
        ax.axvline(b, color=c_ref, lw=0.6, ls=":")
    be = C.breakeven_bps(d["gross"], d["turn_stock"], d["turn_etf"], d["short"], base)
    ax.set_title(f"Net Sharpe vs stock cost per side (ETF 1 bp, borrow 50 bp/yr fixed); breakeven = {be:.1f} bp")
    ax.set_xlabel("bps per side on stock turnover"); ax.set_ylabel("net Sharpe")
    p = out_dir / f"{prefix}_fig3_cost_sensitivity.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    # 4. distribution of z
    sig = rep.bt.signals("sector")
    elig = eligibility(rep.md, sig, rep.cfg)
    z = sig.z.where(elig)[rep.m].to_numpy().ravel()
    z = z[~np.isnan(z)]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(np.clip(z, -8, 8), bins=160, color=c_gross, alpha=0.85)
    for x in (-1, 1):
        ax.axvline(x, color=c_net, lw=1, ls="--")
    ax.set_yscale("log"); ax.set_title(f"Distribution of z (eligible stock-days, clipped at +/-8); share |z|>=1: {(np.abs(z) >= 1).mean():.1%}")
    p = out_dir / f"{prefix}_fig4_z_distribution.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    # 5. turnover over time
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.plot(d["turn_stock"], color=c_gross, lw=0.5, alpha=0.5, label="daily")
    ax.plot(d["turn_stock"].rolling(60).mean(), color=c_net, lw=1.4, label="60-day mean")
    ax.set_title("Stock-book turnover per day (sum |dw|; 2.0 = full replacement of a 1.0-gross book)"); ax.legend(frameon=False)
    p = out_dir / f"{prefix}_fig5_turnover.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    # 6. active names
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.plot(d["n_long"], color=c_gross, lw=0.8, label="longs")
    ax.plot(d["n_short"], color=c_net, lw=0.8, label="shorts")
    ax.set_title("Number of active long / short names (0 = leg empty, no position)"); ax.legend(frameon=False)
    p = out_dir / f"{prefix}_fig6_active_names.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)
    return paths


def run_report(md: MarketData, rf: pd.Series, cfg: dict, period: Period, prefix: str, out_dir: Path = RESULTS_DIR) -> dict:
    """Produce every table/figure for one period. Identical for Stage 2 (in-sample) and Stage 3 (holdout)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = PeriodReport(md, rf, cfg, period)
    tabs = rep.all_strategy_tables()
    tabs.to_csv(out_dir / f"{prefix}_strategy_tables.csv", index=False)
    fm = rep.fm_tables()
    fm.to_csv(out_dir / f"{prefix}_fama_macbeth.csv", index=False)
    dec = rep.decision()
    (out_dir / f"{prefix}_decision_metrics.json").write_text(json.dumps(dec, indent=1, default=str))
    diag = rep.primary_diagnostics()
    (out_dir / f"{prefix}_primary_diagnostics.json").write_text(json.dumps(diag, indent=1))
    figs = make_figures(rep, out_dir / "figures", prefix)
    return dict(tables=tabs, fm=fm, decision=dec, diag=diag, figs=figs)
