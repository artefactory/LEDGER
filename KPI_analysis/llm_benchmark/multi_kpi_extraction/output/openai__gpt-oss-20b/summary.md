# LLM KPI extraction benchmark — summary

- Tolerance: ±0.1%
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Test-set scope: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/test_set_reports.txt` (494 reports, 13455 ground-truth KPIs)
- Predictions root: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/multi_kpi_extraction/output/openai__gpt-oss-20b/raw`
- Reports loaded: 978 (ok=973, failed=5, error=0)

## Overall

| metric | value |
| --- | --- |
| n_predictions | 13343 |
| n_ground_truth | 13455 |
| matched | 8833 |
| wrong | 3238 |
| missing | 1384 |
| extra | 1272 |
| recall (matched/gt) | 0.6565 |
| precision (matched/(matched+wrong)) | 0.7318 |

## Per KPI

| kpi | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_term_borrowings | 301 | 92 | 141 | 68 | 135 | 0.306 | 0.395 | 0.379 |
| shares_outstanding | 419 | 142 | 258 | 19 | 74 | 0.339 | 0.355 | 0.008 |
| long_term_debt_total | 351 | 129 | 198 | 24 | 119 | 0.368 | 0.394 | 0.010 |
| cash_incl_restricted | 349 | 139 | 57 | 153 | 77 | 0.398 | 0.709 | 0.000 |
| rd_expense | 315 | 152 | 61 | 102 | 46 | 0.483 | 0.714 | 0.000 |
| gross_profit | 446 | 248 | 116 | 82 | 46 | 0.556 | 0.681 | 0.000 |
| long_term_debt_noncurrent | 300 | 174 | 84 | 42 | 130 | 0.580 | 0.674 | 0.000 |
| cost_of_revenue | 487 | 284 | 122 | 81 | 5 | 0.583 | 0.700 | 0.000 |
| interest_expense | 426 | 249 | 147 | 30 | 67 | 0.585 | 0.629 | 0.000 |
| long_term_debt_current | 340 | 200 | 93 | 47 | 85 | 0.588 | 0.683 | 0.000 |
| capex | 490 | 292 | 171 | 27 | 4 | 0.596 | 0.631 | 0.000 |
| sga_expense | 491 | 295 | 129 | 67 | 2 | 0.601 | 0.696 | 0.000 |
| income_tax_expense | 486 | 303 | 139 | 44 | 2 | 0.623 | 0.686 | 0.000 |
| total_liabilities | 494 | 313 | 121 | 60 | 0 | 0.634 | 0.721 | 0.000 |
| stockholders_equity_incl_nci | 258 | 164 | 37 | 57 | 138 | 0.636 | 0.816 | 0.000 |
| dividends_paid | 346 | 225 | 82 | 39 | 128 | 0.650 | 0.733 | 0.000 |
| operating_income | 483 | 317 | 130 | 36 | 7 | 0.656 | 0.709 | 0.000 |
| depreciation_amortization | 492 | 327 | 141 | 24 | 1 | 0.665 | 0.699 | 0.000 |
| inventory | 468 | 319 | 63 | 86 | 8 | 0.682 | 0.835 | 0.000 |
| accounts_receivable | 459 | 315 | 101 | 43 | 27 | 0.686 | 0.757 | 0.000 |
| net_income | 493 | 339 | 150 | 4 | 1 | 0.688 | 0.693 | 0.000 |
| accounts_payable | 460 | 317 | 102 | 41 | 34 | 0.689 | 0.757 | 0.000 |
| stockholders_equity | 486 | 365 | 93 | 28 | 8 | 0.751 | 0.797 | 0.000 |
| revenue | 494 | 381 | 91 | 22 | 0 | 0.771 | 0.807 | 0.000 |
| total_assets | 494 | 395 | 52 | 47 | 0 | 0.800 | 0.884 | 0.000 |
| eps_basic | 435 | 355 | 68 | 12 | 57 | 0.816 | 0.839 | 0.000 |
| investing_cash_flow | 491 | 402 | 62 | 27 | 3 | 0.819 | 0.866 | 0.000 |
| operating_cash_flow | 494 | 414 | 63 | 17 | 0 | 0.838 | 0.868 | 0.000 |
| financing_cash_flow | 494 | 415 | 53 | 26 | 0 | 0.840 | 0.887 | 0.000 |
| eps_diluted | 425 | 358 | 60 | 7 | 62 | 0.842 | 0.856 | 0.000 |
| cash_and_equivalents | 488 | 413 | 53 | 22 | 6 | 0.846 | 0.886 | 0.000 |

## Per year

| year | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2017 | 2087 | 1289 | 591 | 207 | 205 | 0.618 | 0.686 | 0.000 |
| 2022 | 2537 | 1598 | 482 | 457 | 175 | 0.630 | 0.768 | 0.000 |
| 2018 | 2003 | 1301 | 540 | 162 | 217 | 0.650 | 0.707 | 0.000 |
| 2021 | 2424 | 1639 | 553 | 232 | 226 | 0.676 | 0.748 | 0.000 |
| 2020 | 2289 | 1556 | 561 | 172 | 230 | 0.680 | 0.735 | 0.000 |
| 2019 | 2115 | 1450 | 511 | 154 | 219 | 0.686 | 0.739 | 0.000 |

## Per ground-truth source

| source | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 50 | 18 | 23 | 9 | 0 | 0.360 | 0.439 | 0.004 |
| edgar + alphavantage | 6586 | 3845 | 1796 | 945 | 0 | 0.584 | 0.682 | 0.000 |
| edgar | 6819 | 4970 | 1419 | 430 | 0 | 0.729 | 0.778 | 0.000 |
