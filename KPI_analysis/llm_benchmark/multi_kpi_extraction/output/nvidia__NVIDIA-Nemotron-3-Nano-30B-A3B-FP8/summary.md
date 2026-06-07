# LLM KPI extraction benchmark — summary

- Tolerance: ±0.1%
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Test-set scope: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/test_set_reports.txt` (494 reports, 13519 ground-truth KPIs)
- Predictions root: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/multi_kpi_extraction/output/nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-FP8/raw`
- Reports loaded: 978 (ok=897, failed=81, error=0)

## Overall

| metric | value |
| --- | --- |
| n_predictions | 12028 |
| n_ground_truth | 13519 |
| matched | 8831 |
| wrong | 2513 |
| missing | 2175 |
| extra | 684 |
| recall (matched/gt) | 0.6532 |
| precision (matched/(matched+wrong)) | 0.7785 |

## Per KPI

| kpi | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cash_incl_restricted | 349 | 51 | 33 | 265 | 19 | 0.146 | 0.607 | 0.000 |
| short_term_borrowings | 312 | 73 | 65 | 174 | 45 | 0.234 | 0.529 | 0.000 |
| long_term_debt_total | 369 | 142 | 154 | 73 | 80 | 0.385 | 0.480 | 0.003 |
| rd_expense | 317 | 148 | 32 | 137 | 26 | 0.467 | 0.822 | 0.000 |
| gross_profit | 446 | 232 | 111 | 103 | 33 | 0.520 | 0.676 | 0.000 |
| cost_of_revenue | 487 | 264 | 139 | 84 | 4 | 0.542 | 0.655 | 0.000 |
| long_term_debt_current | 347 | 193 | 59 | 95 | 30 | 0.556 | 0.766 | 0.000 |
| dividends_paid | 355 | 202 | 43 | 110 | 23 | 0.569 | 0.824 | 0.000 |
| stockholders_equity_incl_nci | 258 | 148 | 28 | 82 | 26 | 0.574 | 0.841 | 0.000 |
| shares_outstanding | 425 | 244 | 155 | 26 | 67 | 0.574 | 0.612 | 0.000 |
| long_term_debt_noncurrent | 300 | 173 | 42 | 85 | 94 | 0.577 | 0.805 | 0.000 |
| sga_expense | 491 | 292 | 123 | 76 | 0 | 0.595 | 0.704 | 0.000 |
| inventory | 468 | 299 | 44 | 125 | 3 | 0.639 | 0.872 | 0.000 |
| accounts_receivable | 459 | 300 | 90 | 69 | 27 | 0.654 | 0.769 | 0.000 |
| operating_income | 483 | 318 | 113 | 52 | 9 | 0.658 | 0.738 | 0.000 |
| net_income | 493 | 336 | 145 | 12 | 1 | 0.682 | 0.699 | 0.000 |
| depreciation_amortization | 492 | 336 | 110 | 46 | 1 | 0.683 | 0.753 | 0.000 |
| accounts_payable | 466 | 319 | 80 | 67 | 26 | 0.685 | 0.799 | 0.000 |
| capex | 490 | 336 | 102 | 52 | 4 | 0.686 | 0.767 | 0.000 |
| interest_expense | 431 | 297 | 80 | 54 | 55 | 0.689 | 0.788 | 0.000 |
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
| 2022 | 2548 | 1561 | 410 | 577 | 89 | 0.613 | 0.792 | 0.000 |
| 2017 | 2099 | 1297 | 513 | 289 | 126 | 0.618 | 0.717 | 0.000 |
| 2021 | 2433 | 1617 | 473 | 343 | 126 | 0.665 | 0.774 | 0.000 |
| 2018 | 2013 | 1345 | 348 | 320 | 111 | 0.668 | 0.794 | 0.000 |
| 2019 | 2126 | 1428 | 377 | 321 | 107 | 0.672 | 0.791 | 0.000 |
| 2020 | 2300 | 1583 | 392 | 325 | 125 | 0.688 | 0.802 | 0.000 |

## Per ground-truth source

| source | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 50 | 22 | 26 | 2 | 0 | 0.440 | 0.458 | 0.004 |
| edgar + alphavantage | 7131 | 4246 | 1502 | 1383 | 0 | 0.595 | 0.739 | 0.000 |
| edgar | 6338 | 4563 | 985 | 790 | 0 | 0.720 | 0.822 | 0.000 |
