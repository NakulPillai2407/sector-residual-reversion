# DEVIATIONS

Dated log of every change made after pre-registration. Entries below the line.

---

## 2026-10-27: Source location of the S&P 500 changes table (pre-Stage 1; no parameter change)
The spec names the "Selected changes" table on Wikipedia's "List of S&P 500 companies". On the date of data collection that
page no longer contains the table (only the constituents table); the same additions/removals data (Effective Date, Added,
Removed, Reason) now lives on "Historical components of the S&P 500". `config.yaml: data.wikipedia_changes_url` points there.
Method (working backwards from the current list) is unchanged. Not a modelling change; logged for transparency.

## 2026-10-27: Implementation interpretations fixed before any signal was computed (no parameter change)
Where the spec was silent, these were fixed BEFORE Stage 1 and are part of the pre-registration:
1. beta_t = OLS slope of stock on benchmark return WITH intercept over t-60..t-1; the intercept is then dropped from the residual (e_t = r - beta*r_bench).
2. The z denominator uses the stored historical residual series (each e_s computed with the beta known at s).
3. Min-price filter uses Close_t (the latest close known at signal time t). Yahoo `Close` is split- but not dividend-adjusted.
4. Every P&L/FM observation is stamped by its EXIT / dependent-variable date; the in-sample covers stamps <= 2023-12-31 and the
   holdout stamps >= 2024-01-01. So the first holdout trades use signals from the last in-sample days, and no in-sample
   statistic ever uses a holdout price.
5. Weights are target weights; intra-holding drift is ignored in turnover. Borrow is charged on all short notional (stocks and any short ETF hedge).
6. Held names with a missing price on entry/exit contribute 0 return (counted and reported); no delisting returns exist in the data.
7. Placebo test: each stock hedged against a uniformly random WRONG ETF (20 independent shuffles); a shuffle "fails" if its FM NW t < -2 or its net-5bp Sharpe CI lies entirely above 0; the placebo passes if <= 10% of shuffles fail.
8. Simulation: planted residual AR(1) is on close-to-close residuals; opens = prior close * (1 + overnight_share*r + noise), so open-to-open is not trivially equal to close-to-close.
9. Hit rate = share of ACTIVE days with positive net return. Sharpe uses all session days in the period (flat days count as zero).
10. Benchmarks are measured open-to-open; SPY and sector-ETF excess returns use ^IRX/252 of the previous session.

## 2026-10-27: Per-sector Fama-MacBeth minimum cross-section (reporting code only; config.yaml untouched; found during Stage 2)
The per-sector FM table used the same hard-coded minimum of 20 names per daily cross-section as the overall regression.
That left Communication Services with zero usable days and Energy with 369 (small sectors, 50-day beta + 50-day z warm-up),
so the table was undefined for them. Per-sector rows now use a minimum of 10 names; the OVERALL regression (the one that
enters H1a) still uses 20 and is unchanged (its cross-sections are never that small). This changes only the exploratory
per-sector FM table's coverage, was made because of a NaN table and not because of any estimate, and does not touch the
primary spec, the decision rule or config.yaml. Stage 2 was re-run after the change; the primary results are identical.
No parameter change is proposed on the basis of Stage 2.

## 2026-10-27: Post-holdout reporting fix (no effect on any estimate or on the verdict)
After Stage 3, `primary_diagnostics()` was found to count empty-leg days over every signal date in the loaded data (warm-up +
in-sample + holdout) rather than only the holdout window, so the diagnostics line in the results pack was ambiguous. Only the
WORDING in `src/results_pack.py` was changed to say what is counted; the numbers are unchanged. The Stage 2 (in-sample) run
had 0 empty-leg days, and the Stage 3 run reported 1 over the full sample, so exactly one empty-leg day (no long leg, no trade)
fell in the holdout. The holdout was run exactly once (HOLDOUT_LOCK present); nothing was re-run.
