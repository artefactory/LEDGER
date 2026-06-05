# LLM KPI extraction benchmark — summary

- Tolerance: ±0.1%
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Test-set scope: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/test_set_reports.txt` (494 reports, 13455 ground-truth KPIs)
- Predictions root: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/multi_kpi_extraction/output/Qwen__Qwen3.6-27B-FP8/raw`
- Reports loaded: 494 (ok=494, failed=0, error=0)

## Overall

| metric | value |
| --- | --- |
| n_predictions | 12959 |
| n_ground_truth | 13455 |
| matched | 10384 |
| wrong | 1740 |
| missing | 1331 |
| extra | 835 |
| recall (matched/gt) | 0.7718 |
| precision (matched/(matched+wrong)) | 0.8565 |

## Per KPI

| kpi | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_term_borrowings | 301 | 81 | 58 | 162 | 17 | 0.269 | 0.583 | 0.000 |
| cash_incl_restricted | 349 | 109 | 7 | 233 | 25 | 0.312 | 0.940 | 0.000 |
| long_term_debt_total | 351 | 172 | 145 | 34 | 117 | 0.490 | 0.543 | 0.000 |
| gross_profit | 446 | 265 | 120 | 61 | 46 | 0.594 | 0.688 | 0.000 |
| cost_of_revenue | 487 | 302 | 127 | 58 | 7 | 0.620 | 0.704 | 0.000 |
| rd_expense | 315 | 201 | 18 | 96 | 17 | 0.638 | 0.918 | 0.000 |
| sga_expense | 491 | 315 | 117 | 59 | 0 | 0.642 | 0.729 | 0.000 |
| depreciation_amortization | 492 | 340 | 126 | 26 | 1 | 0.691 | 0.730 | 0.000 |
| stockholders_equity_incl_nci | 258 | 183 | 4 | 71 | 19 | 0.709 | 0.979 | 0.000 |
| inventory | 468 | 333 | 32 | 103 | 8 | 0.712 | 0.912 | 0.000 |
| accounts_receivable | 459 | 334 | 78 | 47 | 28 | 0.728 | 0.811 | 0.000 |
| dividends_paid | 346 | 253 | 22 | 71 | 22 | 0.731 | 0.920 | 0.000 |
| long_term_debt_current | 340 | 249 | 49 | 42 | 60 | 0.732 | 0.836 | 0.000 |
| operating_income | 483 | 354 | 107 | 22 | 7 | 0.733 | 0.768 | 0.000 |
| shares_outstanding | 419 | 316 | 93 | 10 | 71 | 0.754 | 0.773 | 0.000 |
| capex | 490 | 393 | 72 | 25 | 4 | 0.802 | 0.845 | 0.000 |
| income_tax_expense | 486 | 392 | 53 | 41 | 7 | 0.807 | 0.881 | 0.000 |
| accounts_payable | 460 | 372 | 44 | 44 | 34 | 0.809 | 0.894 | 0.000 |
| interest_expense | 426 | 349 | 51 | 26 | 68 | 0.819 | 0.873 | 0.000 |
| revenue | 494 | 407 | 87 | 0 | 0 | 0.824 | 0.824 | 0.000 |
| net_income | 493 | 440 | 53 | 0 | 1 | 0.892 | 0.892 | 0.000 |
| eps_basic | 435 | 389 | 35 | 11 | 56 | 0.894 | 0.917 | 0.000 |
| long_term_debt_noncurrent | 300 | 269 | 26 | 5 | 140 | 0.897 | 0.912 | 0.000 |
| total_liabilities | 494 | 446 | 30 | 18 | 0 | 0.903 | 0.937 | 0.000 |
| stockholders_equity | 486 | 439 | 43 | 4 | 8 | 0.903 | 0.911 | 0.000 |
| investing_cash_flow | 491 | 450 | 23 | 18 | 3 | 0.916 | 0.951 | 0.000 |
| eps_diluted | 425 | 390 | 35 | 0 | 63 | 0.918 | 0.918 | 0.000 |
| operating_cash_flow | 494 | 458 | 29 | 7 | 0 | 0.927 | 0.940 | 0.000 |
| financing_cash_flow | 494 | 460 | 16 | 18 | 0 | 0.931 | 0.966 | 0.000 |
| cash_and_equivalents | 488 | 457 | 13 | 18 | 6 | 0.936 | 0.972 | 0.000 |
| total_assets | 494 | 466 | 27 | 1 | 0 | 0.943 | 0.945 | 0.000 |

## Per year

| year | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2017 | 2087 | 1498 | 368 | 221 | 134 | 0.718 | 0.803 | 0.000 |
| 2018 | 2003 | 1479 | 304 | 220 | 140 | 0.738 | 0.830 | 0.000 |
| 2022 | 2537 | 1987 | 288 | 262 | 125 | 0.783 | 0.873 | 0.000 |
| 2019 | 2115 | 1674 | 258 | 183 | 139 | 0.791 | 0.866 | 0.000 |
| 2021 | 2424 | 1923 | 264 | 237 | 145 | 0.793 | 0.879 | 0.000 |
| 2020 | 2289 | 1823 | 258 | 208 | 152 | 0.796 | 0.876 | 0.000 |

## Per ground-truth source

| source | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 50 | 31 | 18 | 1 | 0 | 0.620 | 0.633 | 0.000 |
| edgar + alphavantage | 6586 | 4506 | 1115 | 965 | 0 | 0.684 | 0.802 | 0.000 |
| edgar | 6819 | 5847 | 607 | 365 | 0 | 0.857 | 0.906 | 0.000 |
