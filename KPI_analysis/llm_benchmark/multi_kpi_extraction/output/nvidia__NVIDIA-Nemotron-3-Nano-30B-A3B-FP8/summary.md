# LLM KPI extraction benchmark — summary

- Tolerance: ±1.0%
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Test-set scope: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/test_set_reports.txt` (494 reports, 13455 ground-truth KPIs)
- Predictions root: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/multi_kpi_extraction/output/nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-FP8/raw`
- Reports loaded: 978 (ok=897, failed=81, error=0)

## Overall

| metric | value |
| --- | --- |
| n_predictions | 12028 |
| n_ground_truth | 13455 |
| matched | 9096 |
| wrong | 2203 |
| missing | 2156 |
| extra | 729 |
| recall (matched/gt) | 0.6760 |
| precision (matched/(matched+wrong)) | 0.8050 |

## Per KPI

| kpi | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cash_incl_restricted | 349 | 52 | 32 | 265 | 19 | 0.149 | 0.619 | 0.000 |
| short_term_borrowings | 301 | 76 | 59 | 166 | 48 | 0.252 | 0.563 | 0.000 |
| long_term_debt_total | 351 | 159 | 121 | 71 | 96 | 0.453 | 0.568 | 0.002 |
| rd_expense | 315 | 156 | 24 | 135 | 26 | 0.495 | 0.867 | 0.000 |
| gross_profit | 446 | 242 | 101 | 103 | 33 | 0.543 | 0.706 | 0.000 |
| long_term_debt_current | 340 | 193 | 53 | 94 | 36 | 0.568 | 0.785 | 0.000 |
| cost_of_revenue | 487 | 282 | 121 | 84 | 4 | 0.579 | 0.700 | 0.000 |
| long_term_debt_noncurrent | 300 | 175 | 40 | 85 | 94 | 0.583 | 0.814 | 0.000 |
| stockholders_equity_incl_nci | 258 | 152 | 24 | 82 | 26 | 0.589 | 0.864 | 0.000 |
| dividends_paid | 346 | 205 | 37 | 104 | 26 | 0.592 | 0.847 | 0.000 |
| sga_expense | 491 | 310 | 105 | 76 | 0 | 0.631 | 0.747 | 0.000 |
| inventory | 468 | 303 | 40 | 125 | 3 | 0.647 | 0.883 | 0.000 |
| accounts_receivable | 459 | 306 | 84 | 69 | 27 | 0.667 | 0.785 | 0.000 |
| accounts_payable | 460 | 316 | 77 | 67 | 32 | 0.687 | 0.804 | 0.000 |
| operating_income | 483 | 333 | 98 | 52 | 9 | 0.689 | 0.773 | 0.000 |
| interest_expense | 426 | 296 | 76 | 54 | 60 | 0.695 | 0.796 | 0.000 |
| net_income | 493 | 345 | 136 | 12 | 1 | 0.700 | 0.717 | 0.000 |
| income_tax_expense | 486 | 342 | 80 | 64 | 1 | 0.704 | 0.810 | 0.000 |
| capex | 490 | 345 | 93 | 52 | 4 | 0.704 | 0.788 | 0.000 |
| depreciation_amortization | 492 | 347 | 99 | 46 | 1 | 0.705 | 0.778 | 0.000 |
| shares_outstanding | 419 | 296 | 97 | 26 | 73 | 0.706 | 0.753 | 0.000 |
| eps_basic | 435 | 321 | 69 | 45 | 46 | 0.738 | 0.823 | 0.000 |
| revenue | 494 | 369 | 107 | 18 | 0 | 0.747 | 0.775 | 0.000 |
| eps_diluted | 425 | 330 | 67 | 28 | 48 | 0.776 | 0.831 | 0.000 |
| total_liabilities | 494 | 384 | 68 | 42 | 0 | 0.777 | 0.850 | 0.000 |
| investing_cash_flow | 491 | 402 | 50 | 39 | 3 | 0.819 | 0.889 | 0.000 |
| stockholders_equity | 486 | 399 | 59 | 28 | 8 | 0.821 | 0.871 | 0.000 |
| operating_cash_flow | 494 | 407 | 59 | 28 | 0 | 0.824 | 0.873 | 0.000 |
| financing_cash_flow | 494 | 411 | 42 | 41 | 0 | 0.832 | 0.907 | 0.000 |
| cash_and_equivalents | 488 | 414 | 37 | 37 | 5 | 0.848 | 0.918 | 0.000 |
| total_assets | 494 | 428 | 48 | 18 | 0 | 0.866 | 0.899 | 0.000 |

## Per year

| year | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | 2537 | 1593 | 371 | 573 | 96 | 0.628 | 0.811 | 0.000 |
| 2017 | 2087 | 1375 | 428 | 284 | 133 | 0.659 | 0.763 | 0.000 |
| 2021 | 2424 | 1659 | 423 | 342 | 134 | 0.684 | 0.797 | 0.000 |
| 2018 | 2003 | 1384 | 301 | 318 | 119 | 0.691 | 0.821 | 0.000 |
| 2019 | 2115 | 1471 | 327 | 317 | 114 | 0.696 | 0.818 | 0.000 |
| 2020 | 2289 | 1614 | 353 | 322 | 133 | 0.705 | 0.821 | 0.000 |

## Per ground-truth source

| source | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 50 | 24 | 24 | 2 | 0 | 0.480 | 0.500 | 0.004 |
| edgar + alphavantage | 6586 | 3943 | 1301 | 1342 | 0 | 0.599 | 0.752 | 0.000 |
| edgar | 6819 | 5129 | 878 | 812 | 0 | 0.752 | 0.854 | 0.000 |
