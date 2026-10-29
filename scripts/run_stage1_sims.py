"""Stage 1: simulation validation. Must pass before any real signal is computed (Stage 2 checks the marker).

(a) planted world, (b) null world, (c) placebo on real IN-SAMPLE data (wrong-ETF hedges), (d) unit tests.
"""
from __future__ import annotations

import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtest import Backtester, Spec, cost_models  # noqa: E402
from src.config import PROC_DIR, RESULTS_DIR, ROOT, load_config  # noqa: E402
from src.market import load_market_data  # noqa: E402
from src.simulate import sim_job  # noqa: E402
from src.stats import bootstrap_sharpe_ci, fama_macbeth, sharpe  # noqa: E402

MARKER = RESULTS_DIR / "STAGE1_PASSED"


def placebo(cfg: dict) -> pd.DataFrame:
    md, _ = load_market_data(cfg, include_holdout=False)
    bt = Backtester(md, cfg)
    inf = cfg["inference"]
    etfs = np.array(cfg["etfs"])
    cm = cost_models(cfg)["net_5bps"]
    rows = []
    for k in range(cfg["sim"]["placebo_shuffles"]):
        rng = np.random.default_rng(int(cfg["seed"]) + 9000 + k)
        wrong = pd.Series([rng.choice(etfs[etfs != e]) for e in md.etf_of], index=md.etf_of.index)
        fm = fama_macbeth(bt.fm_slopes("cc", etf_of=wrong), inf["nw_lags"])
        res = bt.run(Spec(name=f"placebo{k}", placebo_etf_of=wrong))
        start = pd.Timestamp(cfg["dates"]["signal_start"])
        mask = res.daily.index >= start
        gross, net = res.daily["gross"][mask], res.net(cm)[mask]
        glo, ghi, _ = bootstrap_sharpe_ci(gross, rng, inf["bootstrap_resamples"], inf["bootstrap_mean_block"])
        nlo, nhi, _ = bootstrap_sharpe_ci(net, rng, inf["bootstrap_resamples"], inf["bootstrap_mean_block"])
        rows.append(dict(shuffle=k, fm_coef=fm["mean_coef"], fm_t=fm["nw_t"], gross_sharpe=sharpe(gross), net_sharpe=sharpe(net),
                         gross_ci_lo=glo, gross_ci_hi=ghi, net_ci_lo=nlo, net_ci_hi=nhi))
        print(f"placebo {k}: FM t={fm['nw_t']:.2f} net Sharpe={sharpe(net):.2f} CI=({nlo:.2f},{nhi:.2f})", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    sc = cfg["sim"]
    cal = json.loads((PROC_DIR / "sim_calibration.json").read_text())
    mem = pd.read_parquet(PROC_DIR / "membership.parquet")
    sessions = mem.index[mem.index <= pd.Timestamp(cfg["split"]["in_sample"][1])]
    MARKER.unlink(missing_ok=True)

    jobs = [(cfg, cal, sc["planted_phi"], sessions, i, "planted") for i in range(sc["n_sims"])] + \
           [(cfg, cal, sc["null_phi"], sessions, 10_000 + i, "null") for i in range(sc["n_sims"])]
    with Pool(10) as pool:
        out = pool.map(sim_job, jobs, chunksize=4)
    df = pd.DataFrame(out)
    df.to_csv(RESULTS_DIR / "stage1_sim_runs.csv", index=False)

    checks: list[dict] = []
    pl, nl = df[df.world == "planted"], df[df.world == "null"]
    rec = pl["fm_coef"].mean()
    checks.append(dict(check="1a planted: mean FM coef within +/-0.02 of -0.10", value=f"{rec:.4f}", passed=bool(abs(rec - sc["planted_phi"]) <= sc["recovery_tol"])))
    checks.append(dict(check="1a planted: mean gross daily return > 0", value=f"{pl['gross_mean_daily'].mean():.2e} (positive in {(pl['gross_mean_daily'] > 0).mean():.0%} of sims)",
                       passed=bool(pl["gross_mean_daily"].mean() > 0)))
    fp_fm = (nl["fm_t"].abs() > 2).mean()
    checks.append(dict(check="1b null: share of sims with FM |t|>2 <= 10%", value=f"{fp_fm:.1%}", passed=bool(fp_fm <= sc["max_false_positive_rate"])))
    fp_net = (nl["net_ci_lo"] > 0).mean()
    checks.append(dict(check="1b null: share of sims with net-Sharpe CI entirely above 0 <= 10%", value=f"{fp_net:.1%}", passed=bool(fp_net <= sc["max_false_positive_rate"])))
    extra = dict(null_gross_ci_above_zero=float((nl["gross_ci_lo"] > 0).mean()), null_mean_fm=float(nl["fm_coef"].mean()),
                 planted_mean_gross_sharpe=float(pl["gross_sharpe"].mean()), planted_mean_net_sharpe=float(pl["net_sharpe"].mean()),
                 planted_gross_ci_above_zero=float((pl["gross_ci_lo"] > 0).mean()), planted_fm_sd=float(pl["fm_coef"].std()))

    pdf = placebo(cfg)
    pdf.to_csv(RESULTS_DIR / "stage1_placebo.csv", index=False)
    fail = ((pdf["fm_t"] < -2) | (pdf["net_ci_lo"] > 0)).mean()
    checks.append(dict(check="1c placebo (real in-sample data, wrong-ETF hedge): share of shuffles flagged <= 10%", value=f"{fail:.0%} (median FM t={pdf['fm_t'].median():.2f}, median net Sharpe={pdf['net_sharpe'].median():.2f})",
                       passed=bool(fail <= sc["max_false_positive_rate"])))

    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
    checks.append(dict(check="1d unit tests (incl. look-ahead tests)", value=r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:], passed=r.returncode == 0))

    res = pd.DataFrame(checks)
    res.to_csv(RESULTS_DIR / "stage1_checks.csv", index=False)
    (RESULTS_DIR / "stage1_extra.json").write_text(json.dumps(extra, indent=1))
    print(res.to_string(index=False))
    print(extra)
    if res["passed"].all():
        MARKER.write_text("all Stage 1 checks passed\n")
        print("STAGE 1: PASS")
    else:
        print("STAGE 1: FAIL -- STOP. Do not proceed to Stage 2.")
        sys.exit(1)


if __name__ == "__main__":
    main()
