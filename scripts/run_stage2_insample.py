"""Stage 2: in-sample analysis (2018-10-01 .. 2023-12-31 by P&L stamp date). Holdout prices are never loaded."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RESULTS_DIR, load_config  # noqa: E402
from src.market import load_market_data  # noqa: E402
from src.report_tables import Period, run_report  # noqa: E402

STAGE1 = RESULTS_DIR / "STAGE1_PASSED"


def main() -> None:
    if not STAGE1.exists():
        raise SystemExit("Stage 1 has not passed (results/STAGE1_PASSED missing). Refusing to compute real signals.")
    cfg = load_config()
    md, rf = load_market_data(cfg, include_holdout=False)
    a, b = cfg["split"]["in_sample"]
    out = run_report(md, rf, cfg, Period("in_sample", pd.Timestamp(a), pd.Timestamp(b)), "is")
    print(json.dumps(out["decision"], indent=1, default=str))
    print(out["diag"])


if __name__ == "__main__":
    main()
