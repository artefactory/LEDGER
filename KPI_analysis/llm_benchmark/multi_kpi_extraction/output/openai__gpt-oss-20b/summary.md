# LLM KPI extraction benchmark — summary

- Tolerance: ±1.0%
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Test-set scope: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/test_set_reports.txt` (494 reports, 13265 ground-truth KPIs)
- Predictions root: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/output/openai__gpt-oss-20b/raw`
- Reports loaded: 978 (ok=973, failed=5, error=0)

## Overall

| metric | value |
| --- | --- |
| n_predictions | 13343 |
| n_ground_truth | 13265 |
| matched | 9113 |
| wrong | 2788 |
| missing | 1364 |
| extra | 1442 |
| recall (matched/gt) | 0.6870 |
| precision (matched/(matched+wrong)) | 0.7657 |

## Per KPI

| kpi | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_term_borrowings | 261 | 98 | 101 | 62 | 169 | 0.375 | 0.492 | 0.083 |
| cash_incl_restricted | 349 | 143 | 53 | 153 | 77 | 0.410 | 0.730 | 0.000 |
| long_term_debt_total | 331 | 156 | 151 | 24 | 139 | 0.471 | 0.508 | 0.009 |
| rd_expense | 308 | 152 | 59 | 97 | 48 | 0.494 | 0.720 | 0.000 |
| shares_outstanding | 397 | 197 | 182 | 18 | 95 | 0.496 | 0.520 | 0.007 |
| gross_profit | 434 | 256 | 97 | 81 | 57 | 0.590 | 0.725 | 0.000 |
| interest_expense | 410 | 242 | 138 | 30 | 83 | 0.590 | 0.637 | 0.000 |
| long_term_debt_current | 322 | 195 | 80 | 47 | 103 | 0.606 | 0.709 | 0.000 |
| capex | 490 | 307 | 156 | 27 | 4 | 0.627 | 0.663 | 0.000 |
| cost_of_revenue | 477 | 301 | 95 | 81 | 15 | 0.631 | 0.760 | 0.000 |
| sga_expense | 489 | 310 | 112 | 67 | 4 | 0.634 | 0.735 | 0.000 |
| long_term_debt_noncurrent | 300 | 191 | 67 | 42 | 130 | 0.637 | 0.740 | 0.000 |
| income_tax_expense | 484 | 310 | 131 | 43 | 3 | 0.640 | 0.703 | 0.000 |
| total_liabilities | 494 | 317 | 117 | 60 | 0 | 0.642 | 0.730 | 0.000 |
| stockholders_equity_incl_nci | 258 | 170 | 31 | 57 | 138 | 0.659 | 0.846 | 0.000 |
| dividends_paid | 327 | 224 | 70 | 33 | 141 | 0.685 | 0.762 | 0.000 |
| operating_income | 480 | 330 | 114 | 36 | 10 | 0.688 | 0.743 | 0.000 |
| inventory | 467 | 322 | 59 | 86 | 9 | 0.690 | 0.845 | 0.000 |
| depreciation_amortization | 492 | 341 | 127 | 24 | 1 | 0.693 | 0.729 | 0.000 |
| accounts_payable | 459 | 319 | 99 | 41 | 35 | 0.695 | 0.763 | 0.000 |
| accounts_receivable | 451 | 318 | 90 | 43 | 35 | 0.705 | 0.779 | 0.000 |
| net_income | 487 | 356 | 127 | 4 | 7 | 0.731 | 0.737 | 0.000 |
| stockholders_equity | 485 | 384 | 73 | 28 | 9 | 0.792 | 0.840 | 0.000 |
| revenue | 494 | 393 | 79 | 22 | 0 | 0.796 | 0.833 | 0.000 |
| total_assets | 494 | 399 | 48 | 47 | 0 | 0.808 | 0.893 | 0.000 |
| eps_basic | 435 | 360 | 63 | 12 | 57 | 0.828 | 0.851 | 0.000 |
| investing_cash_flow | 491 | 408 | 56 | 27 | 3 | 0.831 | 0.879 | 0.000 |
| financing_cash_flow | 494 | 418 | 50 | 26 | 0 | 0.846 | 0.893 | 0.000 |
| cash_and_equivalents | 486 | 412 | 52 | 22 | 8 | 0.848 | 0.888 | 0.000 |
| eps_diluted | 425 | 361 | 57 | 7 | 62 | 0.849 | 0.864 | 0.000 |
| operating_cash_flow | 494 | 423 | 54 | 17 | 0 | 0.856 | 0.887 | 0.000 |

## Per year

| year | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | 2492 | 1628 | 416 | 448 | 211 | 0.653 | 0.796 | 0.000 |
| 2017 | 2054 | 1363 | 489 | 202 | 233 | 0.664 | 0.736 | 0.000 |
| 2018 | 1984 | 1343 | 483 | 158 | 232 | 0.677 | 0.735 | 0.000 |
| 2020 | 2257 | 1595 | 490 | 172 | 262 | 0.707 | 0.765 | 0.000 |
| 2021 | 2387 | 1688 | 469 | 230 | 261 | 0.707 | 0.783 | 0.000 |
| 2019 | 2091 | 1496 | 441 | 154 | 243 | 0.715 | 0.772 | 0.000 |

## Per ground-truth source

| source | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 50 | 21 | 20 | 9 | 0 | 0.420 | 0.512 | 0.004 |
| edgar + alphavantage | 5264 | 3045 | 1334 | 885 | 0 | 0.578 | 0.695 | 0.000 |
| edgar | 7951 | 6047 | 1434 | 470 | 0 | 0.761 | 0.808 | 0.000 |
