# LLM KPI extraction benchmark — summary

- Tolerance: ±0.1%
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Test-set scope: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/test_set_reports.txt` (494 reports, 13455 ground-truth KPIs)
- Predictions root: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/multi_kpi_extraction/output/nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-FP8/raw`
- Reports loaded: 978 (ok=897, failed=81, error=0)

## Overall

| metric | value |
| --- | --- |
| n_predictions | 12028 |
| n_ground_truth | 13455 |
| matched | 8811 |
| wrong | 2488 |
| missing | 2156 |
| extra | 729 |
| recall (matched/gt) | 0.6548 |
| precision (matched/(matched+wrong)) | 0.7798 |

## Per KPI

| kpi | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cash_incl_restricted | 349 | 51 | 33 | 265 | 19 | 0.146 | 0.607 | 0.000 |
| short_term_borrowings | 301 | 72 | 63 | 166 | 48 | 0.239 | 0.533 | 0.000 |
| long_term_debt_total | 351 | 135 | 145 | 71 | 96 | 0.385 | 0.482 | 0.002 |
| rd_expense | 315 | 148 | 32 | 135 | 26 | 0.470 | 0.822 | 0.000 |
| gross_profit | 446 | 232 | 111 | 103 | 33 | 0.520 | 0.676 | 0.000 |
| cost_of_revenue | 487 | 264 | 139 | 84 | 4 | 0.542 | 0.655 | 0.000 |
| long_term_debt_current | 340 | 193 | 53 | 94 | 36 | 0.568 | 0.785 | 0.000 |
| stockholders_equity_incl_nci | 258 | 148 | 28 | 82 | 26 | 0.574 | 0.841 | 0.000 |
| long_term_debt_noncurrent | 300 | 173 | 42 | 85 | 94 | 0.577 | 0.805 | 0.000 |
| shares_outstanding | 419 | 243 | 150 | 26 | 73 | 0.580 | 0.618 | 0.000 |
| dividends_paid | 346 | 201 | 41 | 104 | 26 | 0.581 | 0.831 | 0.000 |
| sga_expense | 491 | 292 | 123 | 76 | 0 | 0.595 | 0.704 | 0.000 |
| inventory | 468 | 299 | 44 | 125 | 3 | 0.639 | 0.872 | 0.000 |
| accounts_receivable | 459 | 300 | 90 | 69 | 27 | 0.654 | 0.769 | 0.000 |
| operating_income | 483 | 318 | 113 | 52 | 9 | 0.658 | 0.738 | 0.000 |
| net_income | 493 | 336 | 145 | 12 | 1 | 0.682 | 0.699 | 0.000 |
| accounts_payable | 460 | 314 | 79 | 67 | 32 | 0.683 | 0.799 | 0.000 |
| depreciation_amortization | 492 | 336 | 110 | 46 | 1 | 0.683 | 0.753 | 0.000 |
| interest_expense | 426 | 292 | 80 | 54 | 60 | 0.685 | 0.785 | 0.000 |
| capex | 490 | 336 | 102 | 52 | 4 | 0.686 | 0.767 | 0.000 |
| income_tax_expense | 486 | 336 | 86 | 64 | 1 | 0.691 | 0.796 | 0.000 |
| revenue | 494 | 357 | 119 | 18 | 0 | 0.723 | 0.750 | 0.000 |
| eps_basic | 435 | 318 | 72 | 45 | 46 | 0.731 | 0.815 | 0.000 |
| total_liabilities | 494 | 374 | 78 | 42 | 0 | 0.757 | 0.827 | 0.000 |
| eps_diluted | 425 | 325 | 72 | 28 | 48 | 0.765 | 0.819 | 0.000 |
| stockholders_equity | 486 | 385 | 73 | 28 | 8 | 0.792 | 0.841 | 0.000 |
| operating_cash_flow | 494 | 397 | 69 | 28 | 0 | 0.804 | 0.852 | 0.000 |
| investing_cash_flow | 491 | 396 | 56 | 39 | 3 | 0.807 | 0.876 | 0.000 |
| financing_cash_flow | 494 | 406 | 47 | 41 | 0 | 0.822 | 0.896 | 0.000 |
| cash_and_equivalents | 488 | 413 | 38 | 37 | 5 | 0.846 | 0.916 | 0.000 |
| total_assets | 494 | 421 | 55 | 18 | 0 | 0.852 | 0.884 | 0.000 |

## Per year

| year | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | 2537 | 1558 | 406 | 573 | 96 | 0.614 | 0.793 | 0.000 |
| 2017 | 2087 | 1294 | 509 | 284 | 133 | 0.620 | 0.718 | 0.000 |
| 2021 | 2424 | 1612 | 470 | 342 | 134 | 0.665 | 0.774 | 0.000 |
| 2018 | 2003 | 1343 | 342 | 318 | 119 | 0.670 | 0.797 | 0.000 |
| 2019 | 2115 | 1425 | 373 | 317 | 114 | 0.674 | 0.793 | 0.000 |
| 2020 | 2289 | 1579 | 388 | 322 | 133 | 0.690 | 0.803 | 0.000 |

## Per ground-truth source

| source | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 50 | 22 | 26 | 2 | 0 | 0.440 | 0.458 | 0.004 |
| edgar + alphavantage | 6586 | 3819 | 1425 | 1342 | 0 | 0.580 | 0.728 | 0.000 |
| edgar | 6819 | 4970 | 1037 | 812 | 0 | 0.729 | 0.827 | 0.000 |
