# LLM KPI extraction benchmark — summary

- Tolerance: ±1.0%
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Test-set scope: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/test_set_reports.txt` (494 reports, 13455 ground-truth KPIs)
- Predictions root: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/multi_kpi_extraction/output/openai__gpt-oss-20b/raw`
- Reports loaded: 978 (ok=973, failed=5, error=0)

## Overall

| metric | value |
| --- | --- |
| n_predictions | 13343 |
| n_ground_truth | 13455 |
| matched | 9168 |
| wrong | 2903 |
| missing | 1384 |
| extra | 1272 |
| recall (matched/gt) | 0.6814 |
| precision (matched/(matched+wrong)) | 0.7595 |

## Per KPI

| kpi | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_term_borrowings | 301 | 99 | 134 | 68 | 135 | 0.329 | 0.425 | 0.379 |
| cash_incl_restricted | 349 | 143 | 53 | 153 | 77 | 0.410 | 0.730 | 0.000 |
| long_term_debt_total | 351 | 163 | 164 | 24 | 119 | 0.464 | 0.498 | 0.010 |
| shares_outstanding | 419 | 202 | 198 | 19 | 74 | 0.482 | 0.505 | 0.008 |
| rd_expense | 315 | 154 | 59 | 102 | 46 | 0.489 | 0.723 | 0.000 |
| gross_profit | 446 | 258 | 106 | 82 | 46 | 0.578 | 0.709 | 0.000 |
| long_term_debt_current | 340 | 202 | 91 | 47 | 85 | 0.594 | 0.689 | 0.000 |
| interest_expense | 426 | 254 | 142 | 30 | 67 | 0.596 | 0.641 | 0.000 |
| cost_of_revenue | 487 | 303 | 103 | 81 | 5 | 0.622 | 0.746 | 0.000 |
| capex | 490 | 307 | 156 | 27 | 4 | 0.627 | 0.663 | 0.000 |
| sga_expense | 491 | 311 | 113 | 67 | 2 | 0.633 | 0.733 | 0.000 |
| long_term_debt_noncurrent | 300 | 191 | 67 | 42 | 130 | 0.637 | 0.740 | 0.000 |
| income_tax_expense | 486 | 310 | 132 | 44 | 2 | 0.638 | 0.701 | 0.000 |
| total_liabilities | 494 | 317 | 117 | 60 | 0 | 0.642 | 0.730 | 0.000 |
| stockholders_equity_incl_nci | 258 | 170 | 31 | 57 | 138 | 0.659 | 0.846 | 0.000 |
| dividends_paid | 346 | 228 | 79 | 39 | 128 | 0.659 | 0.743 | 0.000 |
| operating_income | 483 | 330 | 117 | 36 | 7 | 0.683 | 0.738 | 0.000 |
| inventory | 468 | 322 | 60 | 86 | 8 | 0.688 | 0.843 | 0.000 |
| depreciation_amortization | 492 | 341 | 127 | 24 | 1 | 0.693 | 0.729 | 0.000 |
| accounts_payable | 460 | 319 | 100 | 41 | 34 | 0.693 | 0.761 | 0.000 |
| accounts_receivable | 459 | 322 | 94 | 43 | 27 | 0.702 | 0.774 | 0.000 |
| net_income | 493 | 361 | 128 | 4 | 1 | 0.732 | 0.738 | 0.000 |
| stockholders_equity | 486 | 385 | 73 | 28 | 8 | 0.792 | 0.841 | 0.000 |
| revenue | 494 | 393 | 79 | 22 | 0 | 0.796 | 0.833 | 0.000 |
| total_assets | 494 | 399 | 48 | 47 | 0 | 0.808 | 0.893 | 0.000 |
| eps_basic | 435 | 360 | 63 | 12 | 57 | 0.828 | 0.851 | 0.000 |
| investing_cash_flow | 491 | 408 | 56 | 27 | 3 | 0.831 | 0.879 | 0.000 |
| financing_cash_flow | 494 | 418 | 50 | 26 | 0 | 0.846 | 0.893 | 0.000 |
| cash_and_equivalents | 488 | 414 | 52 | 22 | 6 | 0.848 | 0.888 | 0.000 |
| eps_diluted | 425 | 361 | 57 | 7 | 62 | 0.849 | 0.864 | 0.000 |
| operating_cash_flow | 494 | 423 | 54 | 17 | 0 | 0.856 | 0.887 | 0.000 |

## Per year

| year | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | 2537 | 1641 | 439 | 457 | 175 | 0.647 | 0.789 | 0.000 |
| 2017 | 2087 | 1376 | 504 | 207 | 205 | 0.659 | 0.732 | 0.000 |
| 2018 | 2003 | 1349 | 492 | 162 | 217 | 0.673 | 0.733 | 0.000 |
| 2021 | 2424 | 1694 | 498 | 232 | 226 | 0.699 | 0.773 | 0.000 |
| 2020 | 2289 | 1607 | 510 | 172 | 230 | 0.702 | 0.759 | 0.000 |
| 2019 | 2115 | 1501 | 460 | 154 | 219 | 0.710 | 0.765 | 0.000 |

## Per ground-truth source

| source | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 50 | 21 | 20 | 9 | 0 | 0.420 | 0.512 | 0.004 |
| edgar + alphavantage | 6586 | 3982 | 1659 | 945 | 0 | 0.605 | 0.706 | 0.000 |
| edgar | 6819 | 5165 | 1224 | 430 | 0 | 0.757 | 0.808 | 0.000 |
