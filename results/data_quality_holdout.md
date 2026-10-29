# Data quality & coverage (holdout addendum, Stage 3)

Window 2024-01-01 to 2026-09-30.

## Coverage by calendar year (all years, average over sessions)

| year | (a) PIT members | (b) with price data | (c) with sector label | (d) usable for trading | flag |
|---|---|---|---|---|---|
| 2018 | 508.0 | 86.0% | 87.6% | 86.0% | **<90%** |
| 2019 | 506.4 | 88.0% | 89.2% | 88.0% | **<90%** |
| 2020 | 506.0 | 90.3% | 91.1% | 90.3% |  |
| 2021 | 506.0 | 91.8% | 92.6% | 91.8% |  |
| 2022 | 504.7 | 93.8% | 94.7% | 93.8% |  |
| 2023 | 504.0 | 95.6% | 96.5% | 95.6% |  |
| 2024 | 504.0 | 96.5% | 97.3% | 96.5% |  |
| 2025 | 504.0 | 97.7% | 98.3% | 97.7% |  |
| 2026 | 504.0 | 99.0% | 99.6% | 99.0% |  |

## Members during the window excluded from trading
- No downloadable prices (21): ANSS, AVB, CDAY, CMA, CTLT, CTRA, DAY, DFS, EA, FLT, HES, HOLX, IPG, JNPR, K, MRO, PXD, QRVO, SATS, WBA, WBD
- Priced but no usable sector label (0): none

## Price checks inside the window
- Duplicate (ticker,date) rows: **0**
- Tickers with NaN/zero/negative OHLC rows: **0**
- Tickers with missing NYSE sessions inside their history / non-session dates: **1** (ETFs/benchmark: none)
- Single-day |Adj Close return| > 50%: **20** rows (0 on a split day). Kept as downloaded.

ticker       date       ret      Close  Adj Close  Stock Splits  split_same_day
   AAP 2025-05-22  0.570425  49.169998  47.716698           0.0           False
    BE 2024-11-15  0.591867  21.139999  21.139999           0.0           False
  ECHO 2025-08-26  0.702477  50.869999  50.869999           0.0           False
    GL 2024-04-11 -0.531402  49.169998  48.114178           0.0           False
  LUMN 2024-08-06  0.930502   5.000000   5.000000           0.0           False
  MRNA 2026-08-19  1.769695 174.380005 174.380005           0.0           False
  NKTR 2025-06-24  1.562893  24.450001  24.450001           0.0           False
  NKTR 2026-02-10  0.510656  56.000000  56.000000           0.0           False
  SBNY 2025-12-09  0.559524   1.310000   1.310000           0.0           False
  SBNY 2026-08-19 -0.562500   0.140000   0.140000           0.0           False
  SBNY 2026-09-25  0.666667   0.300000   0.300000           0.0           False
   STI 2024-03-11  1.258065  87.500000  87.500000           0.0           False
   STI 2024-03-18  0.563218 136.000000 136.000000           0.0           False
   STI 2024-06-18 -0.643523  34.400002  34.400002           0.0           False
   STI 2024-12-27  0.685096  35.049999  35.049999           0.0           False
   STI 2025-05-08  0.680851   7.900000   7.900000           0.0           False
   STI 2025-09-03  1.137195   7.010000   7.010000           0.0           False
   STI 2025-10-13  2.938980  21.625000  21.625000           0.0           False
   STI 2026-06-04  3.505952  22.709999  22.709999           0.0           False
   STI 2026-06-05  0.572876  35.720001  35.720001           0.0           False
