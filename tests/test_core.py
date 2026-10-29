"""Unit tests incl. look-ahead tests (Stage 1d)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import costs as C
from src.backtest import Backtester, Spec, build_weights, hedge_weights, pnl_from_weights
from src.config import load_config
from src.data_io import HoldoutAccessError, Panels, load_prices
from src.signals import MarketData, compute_signals, eligibility, rolling_beta
from src.simulate import simulate_market
from src.stats import (alpha_beta, daily_cross_sectional_slopes, nw_mean, sharpe,
                       stationary_bootstrap_indices)
from src.universe import reconstruct_membership

CFG = load_config()
CAL = {
    "spy_daily_vol": 0.01, "etf_corr_with_spy_mean": 0.8,
    "sectors": {e: dict(n_stocks=12, etf_daily_vol=0.012, beta_mean=1.0, beta_sd=0.3, idio_vol_median=0.015, idio_vol_sd=0.005)
                for e in CFG["etfs"]},
}


@pytest.fixture(scope="module")
def sim_md() -> MarketData:
    sessions = pd.bdate_range("2018-06-01", periods=420)
    return simulate_market(CAL, CFG, -0.10, sessions, np.random.default_rng(1))


def _truncate(md: MarketData, last: pd.Timestamp) -> MarketData:
    cut = lambda p: Panels(*(getattr(p, f).loc[:last] for f in ("close", "adj_close", "adj_open", "volume")))  # noqa: E731
    return MarketData(cut(md.stocks), cut(md.etfs), md.membership.loc[:last], md.etf_of, md.gics_of)


# ------------------------------------------------------------------ look-ahead
def test_beta_and_std_at_t_use_data_through_t_minus_1(sim_md):
    k = 300
    t = sim_md.stocks.adj_close.index[k]
    base = compute_signals(sim_md, CFG)
    # corrupt every price from day k onward (stocks and ETFs, close and open)
    rng = np.random.default_rng(0)
    bad = MarketData(
        stocks=Panels(*(getattr(sim_md.stocks, f).copy() for f in ("close", "adj_close", "adj_open", "volume"))),
        etfs=Panels(*(getattr(sim_md.etfs, f).copy() for f in ("close", "adj_close", "adj_open", "volume"))),
        membership=sim_md.membership, etf_of=sim_md.etf_of, gics_of=sim_md.gics_of)
    for p in (bad.stocks, bad.etfs):
        p.adj_close.iloc[k:] *= rng.uniform(0.5, 1.5, p.adj_close.iloc[k:].shape)
    alt = compute_signals(bad, CFG)
    pd.testing.assert_frame_equal(base.beta.loc[:t], alt.beta.loc[:t])          # beta_t: data through t-1
    pd.testing.assert_frame_equal(base.std.loc[:t], alt.std.loc[:t])            # rolling std at t: through t-1
    pd.testing.assert_frame_equal(base.resid.iloc[:k], alt.resid.iloc[:k])
    assert not np.allclose(base.z.loc[t].dropna(), alt.z.loc[t].reindex(base.z.loc[t].dropna().index).dropna(), equal_nan=True)  # z_t does use day t


def test_pipeline_pnl_independent_of_future_data(sim_md):
    full = Backtester(sim_md, CFG).run(Spec("p")).daily
    d0 = sim_md.stocks.adj_close.index[350]
    part = Backtester(_truncate(sim_md, d0), CFG).run(Spec("p")).daily
    cols = ["gross", "turn_stock", "turn_etf", "short", "n_long", "n_short"]
    pd.testing.assert_frame_equal(full.loc[:d0, cols], part.loc[:d0, cols], check_exact=False, rtol=1e-9, atol=1e-12)


def test_membership_is_point_in_time():
    days = pd.bdate_range("2020-01-01", periods=20)
    ch = pd.DataFrame({"date": [days[10], days[5]], "added": ["NEW", float("nan")], "removed": ["OLD", "GONE"],
                       "added_name": ["n", "x"], "removed_name": ["o", "g"], "reason": ["", ""]})
    mem, issues = reconstruct_membership(["NEW", "KEEP"], ch, days)
    assert not mem.loc[days[9], "NEW"] and mem.loc[days[10], "NEW"]           # added effective on days[10]
    assert mem.loc[days[9], "OLD"] and not mem.loc[days[10], "OLD"]           # removed effective on days[10]
    assert mem.loc[days[4], "GONE"] and not mem.loc[days[5], "GONE"]
    assert mem["KEEP"].all() and len(issues) == 0


def test_pnl_timing_strictly_after_signal():
    idx = pd.bdate_range("2020-01-01", periods=10)
    w = pd.DataFrame(0.0, index=idx, columns=["A", "B"]); w.iloc[3] = [0.5, -0.5]
    ret = pd.DataFrame(np.arange(20, dtype=float).reshape(10, 2) / 100 + 0.01, index=idx, columns=["A", "B"])
    etf = pd.DataFrame(0.0, index=idx, columns=["E"])
    out = pnl_from_weights(w, etf.copy(), ret, etf, lag=2, hold=1)
    assert out["gross"].iloc[:5].abs().sum() == 0                              # nothing before signal+2
    assert out["gross"].iloc[5] == pytest.approx((0.5 * ret.iloc[5, 0] - 0.5 * ret.iloc[5, 1]))
    assert out["gross"].iloc[6:].abs().sum() == 0
    out1 = pnl_from_weights(w, etf.copy(), ret, etf, lag=1, hold=1)             # same_close (non-tradable)
    assert out1["gross"].iloc[4] == pytest.approx((0.5 * ret.iloc[4, 0] - 0.5 * ret.iloc[4, 1]))
    with pytest.raises(ValueError):
        pnl_from_weights(w, etf.copy(), ret, etf, lag=0, hold=1)


# ----------------------------------------------------------------------- costs
def test_no_turnover_cost_when_weights_unchanged():
    idx = pd.bdate_range("2020-01-01", periods=12)
    w = pd.DataFrame(0.0, index=idx, columns=["A", "B"]); w.iloc[2:] = [0.5, -0.5]
    ret = pd.DataFrame(0.0, index=idx, columns=["A", "B"]); etf = pd.DataFrame(0.0, index=idx, columns=["E"])
    d = pnl_from_weights(w, etf.copy(), ret, etf, lag=1, hold=1)
    cm = C.CostModel(stock_bps=5, etf_bps=1, borrow_bps_per_year=0.0)
    cost = C.cost_series(d["turn_stock"], d["turn_etf"], d["short"], cm)
    assert cost.iloc[:3].sum() == 0 and cost.iloc[4:].sum() == 0                # only the entry day costs anything
    assert cost.iloc[3] == pytest.approx(5e-4 * 1.0)                            # sum|dw| = 1.0 at 5bp
    borrow_only = C.cost_series(d["turn_stock"] * 0, d["turn_etf"] * 0, d["short"], C.CostModel())
    assert borrow_only.iloc[6] == pytest.approx(50e-4 / 252 * 0.5)               # borrow on 0.5 short notional


def test_breakeven_is_zero_net():
    rng = np.random.default_rng(0)
    g = pd.Series(rng.normal(2e-4, 1e-3, 500)); ts = pd.Series(rng.uniform(0.5, 1.5, 500)); te = ts * 0.3; sh = pd.Series(0.5, index=g.index)
    cm = C.CostModel()
    be = C.breakeven_bps(g, ts, te, sh, cm)
    net = g - C.cost_series(ts, te, sh, C.CostModel(stock_bps=be))
    assert net.mean() == pytest.approx(0, abs=1e-12)


# --------------------------------------------------------------------- weights
def test_weights_dollar_neutral_and_hedge(sim_md):
    bt = Backtester(sim_md, CFG)
    sig = bt.signals("sector")
    elig = eligibility(sim_md, sig, CFG)
    w, diag = build_weights(sig.z, elig, 1.0, 0.5)
    active = w.abs().sum(axis=1) > 0
    assert active.sum() > 50
    assert np.allclose(w[active].sum(axis=1), 0, atol=1e-12)                    # stock book sums to zero
    assert np.allclose(w[active].clip(lower=0).sum(axis=1), 0.5) and np.allclose(w[active].clip(upper=0).sum(axis=1), -0.5)
    assert (w[w > 0].notna().sum(axis=1)[active] > 0).all()
    h = hedge_weights(w, sig.beta, sim_md.etf_of, list(sim_md.etfs.adj_close.columns))
    d = w.index[active][5]
    for k in CFG["etfs"]:
        cols = sim_md.etf_of.index[sim_md.etf_of == k]
        assert h.loc[d, k] == pytest.approx(-(w.loc[d, cols] * sig.beta.loc[d, cols].fillna(0)).sum())
    assert h.loc[d, CFG["benchmark"]] == 0


def test_empty_leg_means_no_position():
    idx = pd.bdate_range("2020-01-01", periods=3)
    z = pd.DataFrame([[-2.0, -1.5, 0.1], [-2.0, 1.5, 0.1], [0.1, 0.2, 0.3]], index=idx, columns=list("ABC"))
    w, diag = build_weights(z, pd.DataFrame(True, index=idx, columns=z.columns), 1.0, 0.5)
    assert w.iloc[0].abs().sum() == 0 and w.iloc[2].abs().sum() == 0 and w.iloc[1].abs().sum() == pytest.approx(1.0)
    assert diag["empty_leg_days"] == 2


# ------------------------------------------------------------------- statistics
def test_rolling_beta_matches_ols():
    rng = np.random.default_rng(3)
    b = pd.DataFrame(rng.normal(0, 0.01, (200, 1)), columns=["x"])
    r = 1.3 * b + pd.DataFrame(rng.normal(0, 0.005, (200, 1)), columns=["x"])
    beta = rolling_beta(r, b, 60, 50)
    k = 150
    ref = np.polyfit(b["x"].iloc[k - 60:k], r["x"].iloc[k - 60:k], 1)[0]
    assert beta["x"].iloc[k] == pytest.approx(ref, rel=1e-8)
    assert beta["x"].iloc[:50].isna().all()


def test_fama_macbeth_slope_recovery():
    rng = np.random.default_rng(4)
    x = pd.DataFrame(rng.normal(size=(300, 200)))
    y = -0.1 * x + pd.DataFrame(rng.normal(size=(300, 200)))
    s = daily_cross_sectional_slopes(x, y, pd.DataFrame(True, index=x.index, columns=x.columns))
    assert s.mean() == pytest.approx(-0.1, abs=0.01)


def test_stationary_bootstrap_block_length():
    idx = stationary_bootstrap_indices(1000, 400, 10, np.random.default_rng(0))
    assert idx.min() >= 0 and idx.max() < 1000
    jumps = (np.diff(idx, axis=1) % 1000 != 1).mean()                           # restart probability ~ 1/10 (minus chance continuation)
    assert 0.07 < jumps < 0.12


def test_nw_and_alpha_beta():
    rng = np.random.default_rng(5)
    m = pd.Series(rng.normal(0, 0.01, 2000))
    r = 0.5 * m + 0.0003 + pd.Series(rng.normal(0, 0.004, 2000))
    ab = alpha_beta(r, m, 5)
    assert ab["beta"] == pytest.approx(0.5, abs=0.05) and ab["alpha_ann"] == pytest.approx(0.0003 * 252, abs=0.04)
    mean, se, t = nw_mean(pd.Series(rng.normal(0.1, 1, 1000)), 5)
    assert t > 2
    assert sharpe(np.array([0.01, 0.03, 0.01, 0.03]), 252) == pytest.approx(0.02 / np.std([0.01, 0.03, 0.01, 0.03], ddof=1) * np.sqrt(252))


def test_holdout_prices_guarded(tmp_path, monkeypatch):
    import src.data_io as dio
    monkeypatch.setattr(dio, "HOLDOUT_LOCK", tmp_path / "HOLDOUT_LOCK")
    with pytest.raises(HoldoutAccessError):
        load_prices(include_holdout=True)
