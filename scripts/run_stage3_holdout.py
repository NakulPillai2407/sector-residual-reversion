"""Stage 3: the one-shot holdout run. Requires --confirm; writes results/HOLDOUT_LOCK; refuses to run twice.

Gates (all must pass BEFORE the lock is written): no existing lock, --confirm, Stage 1 marker, FREEZE.json hashes match the
files on disk, git tag `prereg-frozen` points at HEAD and the working tree is clean.
`--rehearsal` runs the identical code path on IN-SAMPLE data only (last 2.75 years of the in-sample window) into
results/rehearsal/, never touching the lock or holdout prices.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PROC_DIR, RESULTS_DIR, ROOT, load_config, sha256_file  # noqa: E402
from src.data_io import load_prices  # noqa: E402
from src.holdout_report import addendum  # noqa: E402
from src.market import load_market_data  # noqa: E402
from src.report_tables import Period, run_report  # noqa: E402
from src.results_pack import build_pack  # noqa: E402


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def check_gates(confirm: bool, results_dir: Path, root: Path = ROOT) -> list[str]:
    errs: list[str] = []
    if (results_dir / "HOLDOUT_LOCK").exists():
        errs.append("HOLDOUT_LOCK exists: the holdout has already been run (or a run was started). Refusing to run again.")
    if not confirm:
        errs.append("--confirm not given.")
    if not (results_dir / "STAGE1_PASSED").exists():
        errs.append("Stage 1 marker missing.")
    fz = results_dir / "FREEZE.json"
    if not fz.exists():
        errs.append("results/FREEZE.json missing: run scripts/freeze_prereg.py first.")
    else:
        frozen = json.loads(fz.read_text())
        for k, h in frozen.items():
            if sha256_file(root / k) != h:
                errs.append(f"hash mismatch for {k} vs FREEZE.json (pre-registration changed after freeze).")
        try:
            tag_sha = subprocess.run(["git", "rev-parse", "prereg-frozen^{commit}"], cwd=root, capture_output=True, text=True).stdout.strip()
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout.strip()
            if not tag_sha or tag_sha != head:
                errs.append("git tag `prereg-frozen` missing or not at HEAD.")
            if dirty:
                errs.append("git working tree is not clean (uncommitted changes after the freeze).")
        except Exception as exc:  # pragma: no cover
            errs.append(f"git check failed: {exc}")
    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="required to run the one-shot holdout")
    ap.add_argument("--rehearsal", action="store_true", help="dry run of the identical code path on in-sample data only")
    args = ap.parse_args(argv)
    cfg = load_config()

    if args.rehearsal:
        out = RESULTS_DIR / "rehearsal"
        out.mkdir(parents=True, exist_ok=True)
        md, rf = load_market_data(cfg, include_holdout=False)
        end = pd.Timestamp(cfg["split"]["in_sample"][1])
        period = Period("rehearsal", end - pd.Timedelta(days=int(2.75 * 365)), end)
        px = load_prices(include_holdout=False)
        mem = pd.read_parquet(PROC_DIR / "membership.parquet")
        sectors = pd.read_csv(PROC_DIR / "sector_map.csv")
        sessions = md.stocks.close.index
        addendum(px, sessions, mem.loc[mem.index.isin(sessions)], sectors, cfg, (period.start, period.end), out, "rehearsal")
        res = run_report(md, rf, cfg, period, "rehearsal", out)
        print("REHEARSAL (in-sample data, not a result):", res["decision"]["verdict"])
        return 0

    errs = check_gates(args.confirm, RESULTS_DIR)
    if errs:
        print("REFUSING TO RUN THE HOLDOUT:\n  - " + "\n  - ".join(errs))
        return 2

    lock = RESULTS_DIR / "HOLDOUT_LOCK"
    lock.write_text(json.dumps({"started": dt.datetime.now().isoformat(), "git": _git("rev-parse", "HEAD"),
                                "freeze": json.loads((RESULTS_DIR / "FREEZE.json").read_text())}, indent=1))
    print("HOLDOUT_LOCK written; running the pre-registered primary spec ONCE.")

    a, b = (pd.Timestamp(x) for x in cfg["split"]["holdout"])
    md, rf = load_market_data(cfg, include_holdout=True)
    px = load_prices(include_holdout=True)
    mem = pd.read_parquet(PROC_DIR / "membership.parquet")
    sectors = pd.read_csv(PROC_DIR / "sector_map.csv")
    sessions = md.stocks.close.index
    addendum(px, sessions, mem.loc[mem.index.isin(sessions)], sectors, cfg, (a, b), RESULTS_DIR, "holdout")
    res = run_report(md, rf, cfg, Period("holdout", a, b), "ho", RESULTS_DIR)
    d = res["decision"]
    build_pack()
    print("=" * 78)
    print(f"VERDICT: {d['verdict']}   (H1a={d['H1a']}, H1b={d['H1b']})")
    print(f"FM coef {d['fm_mean_coef']:.4f}  NW t {d['fm_nw_t']:.2f} | net5 Sharpe {d['net5_sharpe']:.2f} "
          f"CI [{d['net5_sharpe_ci_lo']:.2f}, {d['net5_sharpe_ci_hi']:.2f}] | alpha t {d['alpha_nw_t']:.2f} | gross Sharpe {d['gross_sharpe']:.2f} "
          f"CI [{d['gross_sharpe_ci_lo']:.2f}, {d['gross_sharpe_ci_hi']:.2f}]")
    print(f"Sharpe diff vs SPY {d['sharpe_diff_vs_spy_buy_hold']:.2f} [{d['sharpe_diff_ci_lo']:.2f}, {d['sharpe_diff_ci_hi']:.2f}] | "
          f"breakeven {d['breakeven_bps_per_side']:.2f} bp/side | min detectable Sharpe {d['min_detectable_sharpe']:.2f}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
