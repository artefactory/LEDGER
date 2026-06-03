# LLM KPI extraction benchmark — summary

- Tolerance: ±1.0%
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Test-set scope: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/test_set_reports.txt` (494 reports, 13265 ground-truth KPIs)
- Predictions root: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/output/nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-FP8/raw`
- Reports loaded: 978 (ok=897, failed=81, error=0)

## Overall

| metric | value |
| --- | --- |
| n_predictions | 12028 |
| n_ground_truth | 13265 |
| matched | 9041 |
| wrong | 2127 |
| missing | 2097 |
| extra | 860 |
| recall (matched/gt) | 0.6816 |
| precision (matched/(matched+wrong)) | 0.8095 |

## Per KPI

| kpi | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cash_incl_restricted | 349 | 52 | 32 | 265 | 19 | 0.149 | 0.619 | 0.000 |
| short_term_borrowings | 261 | 76 | 45 | 140 | 62 | 0.291 | 0.628 | 0.000 |
| long_term_debt_total | 331 | 157 | 105 | 69 | 114 | 0.474 | 0.599 | 0.000 |
| rd_expense | 308 | 156 | 24 | 128 | 26 | 0.506 | 0.867 | 0.000 |
| gross_profit | 434 | 238 | 97 | 99 | 41 | 0.548 | 0.710 | 0.000 |
| long_term_debt_current | 322 | 187 | 42 | 93 | 53 | 0.581 | 0.817 | 0.000 |
| long_term_debt_noncurrent | 300 | 175 | 40 | 85 | 94 | 0.583 | 0.814 | 0.000 |
| cost_of_revenue | 477 | 280 | 113 | 84 | 14 | 0.587 | 0.712 | 0.000 |
| stockholders_equity_incl_nci | 258 | 152 | 24 | 82 | 26 | 0.589 | 0.864 | 0.000 |
| dividends_paid | 327 | 202 | 37 | 88 | 29 | 0.618 | 0.845 | 0.000 |
| sga_expense | 489 | 309 | 104 | 76 | 2 | 0.632 | 0.748 | 0.000 |
| inventory | 467 | 303 | 39 | 125 | 4 | 0.649 | 0.886 | 0.000 |
| accounts_receivable | 451 | 303 | 79 | 69 | 35 | 0.672 | 0.793 | 0.000 |
| interest_expense | 410 | 282 | 74 | 54 | 76 | 0.688 | 0.792 | 0.000 |
| accounts_payable | 459 | 316 | 76 | 67 | 33 | 0.688 | 0.806 | 0.000 |
| operating_income | 480 | 331 | 97 | 52 | 12 | 0.690 | 0.773 | 0.000 |
| net_income | 487 | 341 | 134 | 12 | 7 | 0.700 | 0.718 | 0.000 |
| capex | 490 | 345 | 93 | 52 | 4 | 0.704 | 0.788 | 0.000 |
| depreciation_amortization | 492 | 347 | 99 | 46 | 1 | 0.705 | 0.778 | 0.000 |
| income_tax_expense | 484 | 342 | 80 | 62 | 1 | 0.707 | 0.810 | 0.000 |
| shares_outstanding | 397 | 284 | 88 | 25 | 94 | 0.715 | 0.763 | 0.000 |
| eps_basic | 435 | 321 | 69 | 45 | 46 | 0.738 | 0.823 | 0.000 |
| revenue | 494 | 369 | 107 | 18 | 0 | 0.747 | 0.775 | 0.000 |
| eps_diluted | 425 | 330 | 67 | 28 | 48 | 0.776 | 0.831 | 0.000 |
| total_liabilities | 494 | 384 | 68 | 42 | 0 | 0.777 | 0.850 | 0.000 |
| investing_cash_flow | 491 | 402 | 50 | 39 | 3 | 0.819 | 0.889 | 0.000 |
| stockholders_equity | 485 | 398 | 59 | 28 | 9 | 0.821 | 0.871 | 0.000 |
| operating_cash_flow | 494 | 407 | 59 | 28 | 0 | 0.824 | 0.873 | 0.000 |
| financing_cash_flow | 494 | 411 | 42 | 41 | 0 | 0.832 | 0.907 | 0.000 |
| cash_and_equivalents | 486 | 413 | 36 | 37 | 7 | 0.850 | 0.920 | 0.000 |
| total_assets | 494 | 428 | 48 | 18 | 0 | 0.866 | 0.899 | 0.000 |

## Per year

| year | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | 2492 | 1584 | 351 | 557 | 125 | 0.636 | 0.819 | 0.000 |
| 2017 | 2054 | 1366 | 415 | 273 | 155 | 0.665 | 0.767 | 0.000 |
| 2021 | 2387 | 1650 | 406 | 331 | 160 | 0.691 | 0.803 | 0.000 |
| 2018 | 1984 | 1375 | 297 | 312 | 132 | 0.693 | 0.822 | 0.000 |
| 2019 | 2091 | 1462 | 319 | 310 | 131 | 0.699 | 0.821 | 0.000 |
| 2020 | 2257 | 1604 | 339 | 314 | 157 | 0.711 | 0.826 | 0.000 |

## Per ground-truth source

| source | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 50 | 24 | 24 | 2 | 0 | 0.480 | 0.500 | 0.004 |
| edgar + alphavantage | 5264 | 3070 | 1006 | 1188 | 0 | 0.583 | 0.753 | 0.000 |
| edgar | 7951 | 5947 | 1097 | 907 | 0 | 0.748 | 0.844 | 0.000 |
