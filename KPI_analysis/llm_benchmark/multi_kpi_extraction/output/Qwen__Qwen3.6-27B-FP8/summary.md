# LLM KPI extraction benchmark — summary

- Tolerance: ±1.0%
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Test-set scope: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/test_set_reports.txt` (494 reports, 13455 ground-truth KPIs)
- Predictions root: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/multi_kpi_extraction/output/Qwen__Qwen3.6-27B-FP8/raw`
- Reports loaded: 494 (ok=494, failed=0, error=0)

## Overall

| metric | value |
| --- | --- |
| n_predictions | 12959 |
| n_ground_truth | 13455 |
| matched | 10664 |
| wrong | 1460 |
| missing | 1331 |
| extra | 835 |
| recall (matched/gt) | 0.7926 |
| precision (matched/(matched+wrong)) | 0.8796 |

## Per KPI

| kpi | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_term_borrowings | 301 | 85 | 54 | 162 | 17 | 0.282 | 0.612 | 0.000 |
| cash_incl_restricted | 349 | 110 | 6 | 233 | 25 | 0.315 | 0.948 | 0.000 |
| long_term_debt_total | 351 | 199 | 118 | 34 | 117 | 0.567 | 0.628 | 0.000 |
| gross_profit | 446 | 276 | 109 | 61 | 46 | 0.619 | 0.717 | 0.000 |
| rd_expense | 315 | 205 | 14 | 96 | 17 | 0.651 | 0.936 | 0.000 |
| cost_of_revenue | 487 | 324 | 105 | 58 | 7 | 0.665 | 0.755 | 0.000 |
| sga_expense | 491 | 337 | 95 | 59 | 0 | 0.686 | 0.780 | 0.000 |
| depreciation_amortization | 492 | 354 | 112 | 26 | 1 | 0.720 | 0.760 | 0.000 |
| stockholders_equity_incl_nci | 258 | 186 | 1 | 71 | 19 | 0.721 | 0.995 | 0.000 |
| inventory | 468 | 338 | 27 | 103 | 8 | 0.722 | 0.926 | 0.000 |
| dividends_paid | 346 | 256 | 19 | 71 | 22 | 0.740 | 0.931 | 0.000 |
| accounts_receivable | 459 | 341 | 71 | 47 | 28 | 0.743 | 0.828 | 0.000 |
| long_term_debt_current | 340 | 253 | 45 | 42 | 60 | 0.744 | 0.849 | 0.000 |
| operating_income | 483 | 371 | 90 | 22 | 7 | 0.768 | 0.805 | 0.000 |
| capex | 490 | 398 | 67 | 25 | 4 | 0.812 | 0.856 | 0.000 |
| accounts_payable | 460 | 374 | 42 | 44 | 34 | 0.813 | 0.899 | 0.000 |
| income_tax_expense | 486 | 401 | 44 | 41 | 7 | 0.825 | 0.901 | 0.000 |
| shares_outstanding | 419 | 347 | 62 | 10 | 71 | 0.828 | 0.848 | 0.000 |
| interest_expense | 426 | 354 | 46 | 26 | 68 | 0.831 | 0.885 | 0.000 |
| revenue | 494 | 417 | 77 | 0 | 0 | 0.844 | 0.844 | 0.000 |
| eps_basic | 435 | 391 | 33 | 11 | 56 | 0.899 | 0.922 | 0.000 |
| net_income | 493 | 446 | 47 | 0 | 1 | 0.905 | 0.905 | 0.000 |
| long_term_debt_noncurrent | 300 | 272 | 23 | 5 | 140 | 0.907 | 0.922 | 0.000 |
| eps_diluted | 425 | 392 | 33 | 0 | 63 | 0.922 | 0.922 | 0.000 |
| total_liabilities | 494 | 458 | 18 | 18 | 0 | 0.927 | 0.962 | 0.000 |
| investing_cash_flow | 491 | 457 | 16 | 18 | 3 | 0.931 | 0.966 | 0.000 |
| stockholders_equity | 486 | 454 | 28 | 4 | 8 | 0.934 | 0.942 | 0.000 |
| cash_and_equivalents | 488 | 459 | 11 | 18 | 6 | 0.941 | 0.977 | 0.000 |
| financing_cash_flow | 494 | 465 | 11 | 18 | 0 | 0.941 | 0.977 | 0.000 |
| operating_cash_flow | 494 | 467 | 20 | 7 | 0 | 0.945 | 0.959 | 0.000 |
| total_assets | 494 | 477 | 16 | 1 | 0 | 0.966 | 0.968 | 0.000 |

## Per year

| year | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018 | 2003 | 1514 | 269 | 220 | 140 | 0.756 | 0.849 | 0.000 |
| 2017 | 2087 | 1581 | 285 | 221 | 134 | 0.758 | 0.847 | 0.000 |
| 2022 | 2537 | 2024 | 251 | 262 | 125 | 0.798 | 0.890 | 0.000 |
| 2019 | 2115 | 1717 | 215 | 183 | 139 | 0.812 | 0.889 | 0.000 |
| 2020 | 2289 | 1859 | 222 | 208 | 152 | 0.812 | 0.893 | 0.000 |
| 2021 | 2424 | 1969 | 218 | 237 | 145 | 0.812 | 0.900 | 0.000 |

## Per ground-truth source

| source | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 50 | 35 | 14 | 1 | 0 | 0.700 | 0.714 | 0.000 |
| edgar + alphavantage | 6586 | 4651 | 970 | 965 | 0 | 0.706 | 0.827 | 0.000 |
| edgar | 6819 | 5978 | 476 | 365 | 0 | 0.877 | 0.926 | 0.000 |
