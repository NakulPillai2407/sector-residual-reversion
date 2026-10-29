"""One-factor-per-sector simulated return panels for Stage 1 validation.

Calibrated loosely (sector stock counts, ETF vols, beta and idiosyncratic-vol distributions) to Stage 0
in-sample summary statistics only -- never to signals. Residuals follow an exact AR(1) with parameter ``phi``.

r_i,t = beta_i * f_k(i),t + eps_i,t,   eps_i,t = phi * eps_i,t-1 + u_i,t
open_t = close_{t-1} * (1 + overnight_share * r_t + gap_noise)   (so open-to-open != close-to-close)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import Backtester, Spec, cost_models
from .data_io import Panels
from .signals import MarketData, eligibility
from .stats import bootstrap_sharpe_ci, fama_macbeth, nw_mean, sharpe


def simulate_market(cal: dict, cfg: dict, phi: float, sessions: pd.DatetimeIndex, rng: np.random.Generator) -> MarketData:
    T = len(sessions)
    sim = cfg["sim"]
    lam, gap_sd = sim["overnight_share"], sim["gap_noise_sd"]
    etfs = list(cfg["etfs"])
    rho = float(np.clip(cal["etf_corr_with_spy_mean"], 0.2, 0.95))
    zm = rng.standard_normal(T)
    spy_r = cal["spy_daily_vol"] * zm
    f = np.empty((T, len(etfs)))
    for j, e in enumerate(etfs):
        f[:, j] = cal["sectors"][e]["etf_daily_vol"] * (rho * zm + np.sqrt(1 - rho**2) * rng.standard_normal(T))

    tickers, etf_of, gics_of, betas, vols = [], [], [], [], []
    e2s = {v: k for k, v in cfg["sector_to_etf"].items()}
    for j, e in enumerate(etfs):
        c = cal["sectors"][e]
        n = int(c["n_stocks"])
        b = np.clip(rng.normal(c["beta_mean"], max(c["beta_sd"], 0.05), n), 0.1, 2.5)
        med = max(c["idio_vol_median"], 1e-3)
        ls = np.sqrt(np.log(1 + (max(c["idio_vol_sd"], 1e-4) / med) ** 2))
        v = med * np.exp(rng.normal(0, ls, n))
        for i in range(n):
            tickers.append(f"S{e}{i:03d}")
            etf_of.append(e)
            gics_of.append(e2s[e])
        betas += list(b)
        vols += list(v)
    N = len(tickers)
    betas, vols = np.array(betas), np.array(vols)
    fidx = np.array([etfs.index(e) for e in etf_of])
    u = rng.standard_normal((T, N)) * vols * np.sqrt(1 - phi**2)
    eps = np.empty((T, N))
    eps[0] = rng.standard_normal(N) * vols
    for t in range(1, T):
        eps[t] = phi * eps[t - 1] + u[t]
    r = np.clip(betas * f[:, fidx] + eps, -0.5, 1.0)

    def to_prices(ret: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        close = 100.0 * np.cumprod(1 + ret, axis=0)
        gap = lam * ret + rng.normal(0, gap_sd, ret.shape)
        opn = np.vstack([close[:1], close[:-1] * (1 + gap[1:])])
        return close, opn

    sc, so = to_prices(r)
    ec, eo = to_prices(np.column_stack([f, spy_r]))
    etf_cols = etfs + [cfg["benchmark"]]

    def panel(close: np.ndarray, opn: np.ndarray, cols: list[str]) -> Panels:
        c = pd.DataFrame(close, index=sessions, columns=cols)
        return Panels(close=c, adj_close=c.copy(), adj_open=pd.DataFrame(opn, index=sessions, columns=cols),
                      volume=pd.DataFrame(1e6, index=sessions, columns=cols))

    return MarketData(
        stocks=panel(sc, so, tickers), etfs=panel(ec, eo, etf_cols),
        membership=pd.DataFrame(True, index=sessions, columns=tickers),
        etf_of=pd.Series(etf_of, index=tickers), gics_of=pd.Series(gics_of, index=tickers),
    )


PRIMARY = Spec(name="primary")


def evaluate_world(md: MarketData, cfg: dict, rng: np.random.Generator, n_boot: int | None = None) -> dict:
    """Run FM + primary backtest on one simulated panel and return summary numbers."""
    inf = cfg["inference"]
    B = n_boot or inf["bootstrap_resamples"]
    bt = Backtester(md, cfg)
    fm = fama_macbeth(bt.fm_slopes("cc"), inf["nw_lags"])
    res = bt.run(PRIMARY)
    mask = res.daily.index >= pd.Timestamp(cfg["dates"]["signal_start"]) + pd.Timedelta(days=3)
    gross = res.daily["gross"][mask]
    net = res.net(cost_models(cfg)["net_5bps"])[mask]
    glo, ghi, _ = bootstrap_sharpe_ci(gross, rng, B, inf["bootstrap_mean_block"], inf["ci_level"], inf["ann_factor"])
    nlo, nhi, _ = bootstrap_sharpe_ci(net, rng, B, inf["bootstrap_mean_block"], inf["ci_level"], inf["ann_factor"])
    return dict(fm_coef=fm["mean_coef"], fm_t=fm["nw_t"], gross_mean_daily=float(gross.mean()),
                gross_sharpe=sharpe(gross), net_sharpe=sharpe(net), gross_ci_lo=glo, gross_ci_hi=ghi,
                net_ci_lo=nlo, net_ci_hi=nhi, avg_n_long=float(res.daily["n_long"][mask].mean()))


def sim_job(args: tuple) -> dict:
    """Top-level (picklable) worker: one simulated world."""
    cfg, cal, phi, sessions, seed_offset, world = args
    rng_sim = np.random.default_rng(int(cfg["seed"]) + 1000 + seed_offset)
    md = simulate_market(cal, cfg, phi, sessions, rng_sim)
    out = evaluate_world(md, cfg, np.random.default_rng(int(cfg["seed"]) + 5000 + seed_offset))
    out.update(world=world, sim=seed_offset)
    return out
