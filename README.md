# Sector Residual Reversion

A pre-registered test of whether the part of a large US stock's daily move that its sector ETF does not explain reverses the next day, and whether that is big enough to trade after costs. It is part of [Holdout Research](https://nakulpillai2407.github.io/), an independent strategy lab where ideas are tested once on held-out data and written up honestly, including the ones that fail.

**Result: not supported.** On the holdout (2024-01-01 to 2026-09-30, 689 sessions) the reversal was not detectable, the strategy lost money before costs, and it lost far more after them.

## What it tests

For each S&P 500 member and day, the residual is the stock's return minus beta times its sector ETF's return (60-day rolling beta, using data through the previous day). The signal z is the residual divided by its trailing 60-day standard deviation. The strategy goes long stocks with z of -1 or lower and short stocks with z of +1 or higher, with the two legs at 0.5 gross each, plus an ETF hedge that cancels the net sector beta. The signal is read at the close of day t, entered at the open of t+1 and exited at the open of t+2.

Membership is point in time. Costs are 5 bp per side on stocks (2 and 10 bp as sensitivities), 1 bp per side on the ETF hedge and a flat 50 bp a year borrow.

The decision rule was fixed before the build. H1a needs a holdout Fama-MacBeth coefficient that is negative with a Newey-West t below -2. H1b needs a net Sharpe at 5 bp whose 95% bootstrap interval is entirely above zero, and an alpha against SPY with t above 2. SUPPORTED needs both. GROSS-ONLY needs H1a plus a gross Sharpe interval above zero. Anything else is NOT SUPPORTED.

## Results

All numbers are from `results/ho_decision_metrics.json`, `results/is_decision_metrics.json` and the `*_strategy_tables.csv` files.

| Primary specification | In-sample (2018-10-01 to 2023-12-31) | Holdout (2024-01-01 to 2026-09-30) |
|---|---|---|
| Fama-MacBeth coefficient (Newey-West t) | -0.0069 (-1.43) | -0.0040 (-0.66) |
| Gross Sharpe (95% CI) | 0.35 (-0.48, 1.20) | -0.24 (-1.47, 0.96) |
| Net Sharpe at 5 bp (95% CI) | -3.05 (-4.07, -2.15) | -4.02 (-5.39, -2.75) |
| Net annual return at 5 bp | -21.0% | -25.5% |
| Alpha vs SPY, annual (Newey-West t), 5 bp | -21.6% (-7.78) | -26.0% (-6.50) |
| Breakeven cost per side | 0.18 bp | -0.71 bp |
| Turnover (stock book, per day) | 1.73 | 1.77 |
| Minimum detectable Sharpe | 0.86 | 1.19 |

Verdict from the pre-registered rule: **NOT SUPPORTED**. The strategy turns over the book about 1.77 times a day, so costs alone take roughly 22% a year at 5 bp per side, and there is no gross edge to pay for them. The holdout is short, so a true gross Sharpe below about 1.2 could not have been detected. The honest reading is no detectable effect and a clear cost failure, not proof that no residual reversal exists.

Robustness variants (SPY-only hedge, excluding Visa, Mastercard and PayPal, excluding the ten largest names, unhedged) and exploratory variants (other thresholds and holding periods) are in `results/results_pack.md`. None changes the verdict. The exploratory ones were never used to choose anything.

## How the work is organised

| Path | What is in it |
|---|---|
| `config.yaml` | Every pre-registered parameter, in one file |
| `DEVIATIONS.md` | Dated log of changes and interpretations after pre-registration (none changed a parameter or the decision rule) |
| `src/` | Signal, backtest, costs, statistics, simulation, data providers |
| `scripts/` | One script per stage: data, simulation checks, in-sample, freeze, one-shot holdout |
| `tests/` | Unit and look-ahead tests |
| `notebooks/main.ipynb` | Narrates each stage from the `src` modules |
| `results/` | Decision metrics, strategy tables, Fama-MacBeth tables, data-quality reports, figures and `results_pack.md`, the source of every number in the report |
| `data/processed/` | Small derived files: membership table, sector map, ticker status |
| `reports/` | The written report (Word and PDF) |

## How to run

```
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python scripts/run_stage0_data.py        # download, point-in-time universe, quality and coverage checks
python scripts/run_stage1_sims.py        # simulation validation; writes results/STAGE1_PASSED only if every check passes
python scripts/run_stage2_insample.py    # in-sample analysis (refuses without STAGE1_PASSED)
python scripts/freeze_prereg.py          # writes FREEZE.json, commits and tags prereg-frozen (needs a git repo)
python scripts/run_stage3_holdout.py --confirm   # one-shot holdout; refuses to run twice
python -m pytest tests
```

Stage 3 refuses to run if `results/HOLDOUT_LOCK` already exists, and that file ships in this repo, so delete it (and `results/FREEZE.json`) in your own copy before rerunning. Pinned versions are in `requirements.txt`, with a full freeze in `requirements-full.txt`. Data access goes only through `src/providers.py::DataProvider`, so a Bloomberg or WRDS provider could replace the free sources without touching the signal or backtest code. That re-run has not been done.

## What is not in this repo

Raw downloads (Yahoo Finance prices and the Wikipedia pages) and the processed price files are left out, because they are vendor data and large. `scripts/run_stage0_data.py` downloads them again. Yahoo data can be revised, so a fresh download may differ slightly from what I used. `data/MANIFEST.json` records the file hashes of my copy.

## Limits

- Survivorship bias is reduced, not removed. Removed index members often have no Yahoo prices or sector label, so usable coverage of point-in-time members was 86.0% in 2018 and 99.0% in 2026 (`results/coverage_by_year.csv`).
- Current sector labels are applied to the whole sample, and there are no delisting returns.
- Fills are assumed at the open, borrow is a flat 50 bp a year, and there is no market impact or capacity estimate.
- The holdout is about 2.75 years, so power is low.
- Betas come from close-to-close returns but are applied to open-to-open holds.
- Yahoo prices can contain errors. Moves above 50% were kept as downloaded.

## A note on dates

The pre-registration dates are my own record. This repo was published by upload, without the local git history, so the freeze commit and `prereg-frozen` tag are not here. `FREEZE.json` holds hashes of the config, membership table and manifest, and `HOLDOUT_LOCK` marks the one holdout run, but neither can prove when anything was written. Treat the order of events as something I am telling you, not something this repo verifies.

## Author

Nakul Pillai, MSc Financial Technology, Imperial College London. MIT licence (see `LICENSE`).
