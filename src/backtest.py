"""Dollar-neutral, ETF-hedged reversal backtest.

Conventions
-----------
* A signal formed at the close of session ``t`` produces target weights ``W_t``.
* Every P&L observation is stamped with its EXIT date ``d`` and equals ``A_d . ret_d`` where ``ret_d`` is the
  return realised ending at ``d`` (open-to-open for ``open_open``, close-to-close otherwise) and
  ``A_d = mean_{k<hold} W_{d - lag - k}``. ``lag = 2`` for open_open / close_close_lag (signal t, enter t+1, exit t+2),
  ``lag = 1`` for same_close (NON-TRADABLE upper bound: enter at the close that generated the signal).
* Turnover = sum|A_d - A_{d-1}| (entry-side stamping), so a position that stays signalled is not charged twice.
* Missing prices on a held name contribute 0 return (counted in ``nan_pos``).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import costs as C
from .signals import MarketData, Signals, bench_matrix, compute_signals, eligibility, returns_cc, returns_oo
from .stats import daily_cross_sectional_slopes

TIMING = {"open_open": (2, "oo"), "close_close_lag": (2, "cc"), "same_close": (1, "cc")}


@dataclass(frozen=True)
class Spec:
    name: str
    threshold: float = 1.0
    timing: str = "open_open"
    hold: int = 1
    hedge: str = "sector"            # 'sector' | 'spy' | 'none'
    mode: str = "threshold"          # 'threshold' | 'continuous'
    clip: float = 3.0
    abs_z_cap: float | None = None
    exclude_tickers: tuple[str, ...] = ()
    exclude_top_adv: int = 0
    adv_window: int = 60
    sector: str | None = None        # restrict the book to one GICS sector
    placebo_etf_of: object = field(default=None, compare=False, hash=False, repr=False)


@dataclass
class BacktestResult:
    spec: Spec
    daily: pd.DataFrame              # gross, turn_stock, turn_etf, short, n_long, n_short, active, nan_pos
    weights: pd.DataFrame            # stock target weights W_t (signal-dated)
    diag: dict[str, float]

    def net(self, cm: C.CostModel) -> pd.Series:
        d = self.daily
        return d["gross"] - C.cost_series(d["turn_stock"], d["turn_etf"], d["short"], cm)


def build_weights(z: pd.DataFrame, elig: pd.DataFrame, threshold: float, leg_gross: float, mode: str = "threshold",
                  clip: float = 3.0, abs_z_cap: float | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    zz = z.where(elig)
    if abs_z_cap is not None:
        zz = zz.where(zz.abs() <= abs_z_cap)
    if mode == "threshold":
        lg = (zz <= -threshold).astype(float)
        sh = (zz >= threshold).astype(float)
    elif mode == "continuous":
        zc = zz.clip(-clip, clip)
        lg = (-zc).clip(lower=0.0).fillna(0.0)
        sh = zc.clip(lower=0.0).fillna(0.0)
    else:
        raise ValueError(mode)
    sl, ss = lg.sum(axis=1), sh.sum(axis=1)
    both = (sl > 0) & (ss > 0)
    wl = lg.div(sl.where(sl > 0), axis=0) * leg_gross
    ws = -sh.div(ss.where(ss > 0), axis=0) * leg_gross
    w = (wl.fillna(0.0) + ws.fillna(0.0)).where(both, 0.0)
    has_universe = elig.any(axis=1)
    diag = dict(signal_days=int(has_universe.sum()), empty_leg_days=int((has_universe & ~both).sum()),
                no_long_days=int((has_universe & (sl == 0)).sum()), no_short_days=int((has_universe & (ss == 0)).sum()))
    return w, diag


def hedge_weights(w: pd.DataFrame, beta: pd.DataFrame, etf_of: pd.Series, etf_cols: list[str]) -> pd.DataFrame:
    """ETF_k weight = - sum_{i in k} w_i * beta_i."""
    contrib = (w * beta).fillna(0.0)
    grp = contrib.T.groupby(etf_of.reindex(w.columns).to_numpy()).sum().T
    return (-grp).reindex(columns=etf_cols).fillna(0.0)


class Backtester:
    """Caches signals and return panels so many specs can be run cheaply on one dataset."""

    def __init__(self, md: MarketData, cfg: dict):
        self.md, self.cfg = md, cfg
        self.etf_cols = list(md.etfs.adj_close.columns)
        self._sig: dict[str, Signals] = {}
        self.ret_s = {"cc": returns_cc(md.stocks.adj_close), "oo": returns_oo(md.stocks.adj_open)}
        self.ret_e = {"cc": returns_cc(md.etfs.adj_close), "oo": returns_oo(md.etfs.adj_open)}
        self._adv = (md.stocks.close * md.stocks.volume)

    def signals(self, bench: str = "sector", etf_of: pd.Series | None = None) -> Signals:
        """Cached by benchmark kind; a custom ``etf_of`` (placebo) is computed fresh and never cached."""
        if etf_of is not None:
            return compute_signals(self.md, self.cfg, etf_of)
        if bench not in self._sig:
            m = pd.Series(self.cfg["benchmark"], index=self.md.etf_of.index) if bench == "spy" else None
            self._sig[bench] = compute_signals(self.md, self.cfg, m)
        return self._sig[bench]

    def fm_slopes(self, kind: str = "cc", sector: str | None = None, min_names: int = 20,
                  etf_of: pd.Series | None = None) -> pd.Series:
        """Daily Fama-MacBeth slope of next-day residual on today's residual, stamped with the DEPENDENT date.

        kind='cc': e_{t+1} (close-to-close, own beta_{t+1}); kind='oo': next-interval open-to-open residual
        (open t+1 -> open t+2) using beta_t, i.e. consistent with the trade timing."""
        sig = self.signals(etf_of=etf_of) if etf_of is not None else self.signals("sector")
        elig = eligibility(self.md, sig, self.cfg)
        if sector:
            elig = elig & (self.md.gics_of.reindex(elig.columns) == sector).to_numpy()[None, :]
        if kind == "cc":
            y, shift = sig.resid.shift(-1), 1
        else:
            b_oo = bench_matrix(self.ret_e["oo"], self.md.etf_of if etf_of is None else etf_of, sig.beta.columns)
            y, shift = (self.ret_s["oo"] - sig.beta * b_oo).shift(-2), 2
        return daily_cross_sectional_slopes(sig.resid, y, elig, min_names).shift(shift)

    def _etf_map(self, spec: Spec) -> pd.Series:
        if spec.placebo_etf_of is not None:
            return spec.placebo_etf_of
        if spec.hedge == "spy":
            return pd.Series(self.cfg["benchmark"], index=self.md.etf_of.index)
        return self.md.etf_of

    def run(self, spec: Spec) -> BacktestResult:
        cfg, md = self.cfg, self.md
        etf_of = self._etf_map(spec)
        if spec.placebo_etf_of is not None:
            sig = self.signals(etf_of=spec.placebo_etf_of)
        else:
            sig = self.signals("spy" if spec.hedge == "spy" else "sector")
        elig = eligibility(md, sig, cfg)
        if spec.exclude_tickers:
            elig.loc[:, [t for t in spec.exclude_tickers if t in elig.columns]] = False
        if spec.exclude_top_adv:
            adv = self._adv.rolling(spec.adv_window, min_periods=spec.adv_window // 2).mean().where(md.membership)
            elig &= ~(adv.rank(axis=1, ascending=False) <= spec.exclude_top_adv)
        if spec.sector:
            elig = elig & (md.gics_of.reindex(elig.columns) == spec.sector).to_numpy()[None, :]
        w, diag = build_weights(sig.z, elig, spec.threshold, cfg["signal"]["leg_gross"], spec.mode, spec.clip, spec.abs_z_cap)
        w_etf = (hedge_weights(w, sig.beta, etf_of, self.etf_cols) if spec.hedge != "none"
                 else pd.DataFrame(0.0, index=w.index, columns=self.etf_cols))

        lag, rt = TIMING[spec.timing]
        daily = pnl_from_weights(w, w_etf, self.ret_s[rt], self.ret_e[rt], lag, spec.hold)
        nan_pos = daily["nan_pos"]
        diag["nan_position_days"] = int((nan_pos > 0).sum())
        diag["nan_positions"] = int(nan_pos.sum())
        return BacktestResult(spec, daily, w, diag)


def pnl_from_weights(w: pd.DataFrame, w_etf: pd.DataFrame, rs: pd.DataFrame, re_: pd.DataFrame,
                     lag: int, hold: int) -> pd.DataFrame:
    """Signal-dated target weights -> exit-date-stamped P&L, turnover and exposure (see module docstring)."""
    if lag < 1:
        raise ValueError("trades must happen strictly after the signal (lag >= 1)")
    agg = lambda W: sum(W.shift(lag + k) for k in range(hold)).fillna(0.0) / hold  # noqa: E731
    a_s, a_e = agg(w), agg(w_etf)
    gross = (a_s * rs.fillna(0.0)).sum(axis=1) + (a_e * re_.fillna(0.0)).sum(axis=1)
    nan_pos = ((a_s != 0) & rs.isna()).sum(axis=1) + ((a_e != 0) & re_.isna()).sum(axis=1)
    return pd.DataFrame({
        "gross": gross,
        "turn_stock": C.turnover(a_s),
        "turn_etf": C.turnover(a_e),
        "short": C.short_notional(a_s, a_e),
        "n_long": (a_s > 0).sum(axis=1),
        "n_short": (a_s < 0).sum(axis=1),
        "active": (a_s != 0).any(axis=1),
        "nan_pos": nan_pos,
    })


def cost_models(cfg: dict) -> dict[str, C.CostModel]:
    c = cfg["costs"]
    base = dict(etf_bps=c["etf_bps_per_side"], borrow_bps_per_year=c["borrow_bps_per_year"], days_per_year=c["days_per_year"])
    out = {"net_5bps": C.CostModel(stock_bps=c["stock_bps_per_side"], **base)}
    for b in c["sensitivities_bps"]:
        out[f"net_{b:g}bps"] = C.CostModel(stock_bps=b, **base)
    return out
