# LLM KPI extraction benchmark — summary

- Tolerance: ±1.0%
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Test-set scope: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/test_set_reports.txt` (494 reports, 13265 ground-truth KPIs)
- Predictions root: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/output/Qwen__Qwen3.6-27B-FP8/raw`
- Reports loaded: 494 (ok=494, failed=0, error=0)

## Overall

| metric | value |
| --- | --- |
| n_predictions | 12959 |
| n_ground_truth | 13265 |
| matched | 10603 |
| wrong | 1384 |
| missing | 1278 |
| extra | 972 |
| recall (matched/gt) | 0.7993 |
| precision (matched/(matched+wrong)) | 0.8845 |

## Per KPI

| kpi | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cash_incl_restricted | 349 | 110 | 6 | 233 | 25 | 0.315 | 0.948 | 0.000 |
| short_term_borrowings | 261 | 85 | 42 | 134 | 29 | 0.326 | 0.669 | 0.000 |
| long_term_debt_total | 331 | 197 | 100 | 34 | 137 | 0.595 | 0.663 | 0.000 |
| gross_profit | 434 | 272 | 102 | 60 | 57 | 0.627 | 0.727 | 0.000 |
| rd_expense | 308 | 205 | 14 | 89 | 17 | 0.666 | 0.936 | 0.000 |
| cost_of_revenue | 477 | 322 | 97 | 58 | 17 | 0.675 | 0.768 | 0.000 |
| sga_expense | 489 | 335 | 95 | 59 | 2 | 0.685 | 0.779 | 0.000 |
| depreciation_amortization | 492 | 354 | 112 | 26 | 1 | 0.720 | 0.760 | 0.000 |
| stockholders_equity_incl_nci | 258 | 186 | 1 | 71 | 19 | 0.721 | 0.995 | 0.000 |
| inventory | 467 | 338 | 26 | 103 | 9 | 0.724 | 0.929 | 0.000 |
| accounts_receivable | 451 | 337 | 67 | 47 | 36 | 0.747 | 0.834 | 0.000 |
| long_term_debt_current | 322 | 245 | 36 | 41 | 77 | 0.761 | 0.872 | 0.000 |
| operating_income | 480 | 369 | 89 | 22 | 10 | 0.769 | 0.806 | 0.000 |
| dividends_paid | 327 | 256 | 15 | 56 | 26 | 0.783 | 0.945 | 0.000 |
| capex | 490 | 398 | 67 | 25 | 4 | 0.812 | 0.856 | 0.000 |
| accounts_payable | 459 | 374 | 41 | 44 | 35 | 0.815 | 0.901 | 0.000 |
| interest_expense | 410 | 338 | 46 | 26 | 84 | 0.824 | 0.880 | 0.000 |
| income_tax_expense | 484 | 401 | 43 | 40 | 8 | 0.829 | 0.903 | 0.000 |
| shares_outstanding | 397 | 335 | 52 | 10 | 93 | 0.844 | 0.866 | 0.000 |
| revenue | 494 | 417 | 77 | 0 | 0 | 0.844 | 0.844 | 0.000 |
| eps_basic | 435 | 391 | 33 | 11 | 56 | 0.899 | 0.922 | 0.000 |
| net_income | 487 | 440 | 47 | 0 | 7 | 0.903 | 0.903 | 0.000 |
| long_term_debt_noncurrent | 300 | 272 | 23 | 5 | 140 | 0.907 | 0.922 | 0.000 |
| eps_diluted | 425 | 392 | 33 | 0 | 63 | 0.922 | 0.922 | 0.000 |
| total_liabilities | 494 | 458 | 18 | 18 | 0 | 0.927 | 0.962 | 0.000 |
| investing_cash_flow | 491 | 457 | 16 | 18 | 3 | 0.931 | 0.966 | 0.000 |
| stockholders_equity | 485 | 453 | 28 | 4 | 9 | 0.934 | 0.942 | 0.000 |
| cash_and_equivalents | 486 | 457 | 11 | 18 | 8 | 0.940 | 0.976 | 0.000 |
| financing_cash_flow | 494 | 465 | 11 | 18 | 0 | 0.941 | 0.977 | 0.000 |
| operating_cash_flow | 494 | 467 | 20 | 7 | 0 | 0.945 | 0.959 | 0.000 |
| total_assets | 494 | 477 | 16 | 1 | 0 | 0.966 | 0.968 | 0.000 |

## Per year

| year | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018 | 1984 | 1507 | 263 | 214 | 153 | 0.760 | 0.851 | 0.000 |
| 2017 | 2054 | 1572 | 274 | 208 | 154 | 0.765 | 0.852 | 0.000 |
| 2022 | 2492 | 2011 | 232 | 249 | 157 | 0.807 | 0.897 | 0.000 |
| 2019 | 2091 | 1708 | 205 | 178 | 158 | 0.817 | 0.893 | 0.000 |
| 2020 | 2257 | 1847 | 209 | 201 | 177 | 0.818 | 0.898 | 0.000 |
| 2021 | 2387 | 1958 | 201 | 228 | 173 | 0.820 | 0.907 | 0.000 |

## Per ground-truth source

| source | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edgar + alphavantage | 5264 | 3587 | 807 | 870 | 0 | 0.681 | 0.816 | 0.000 |
| yfinance | 50 | 35 | 14 | 1 | 0 | 0.700 | 0.714 | 0.000 |
| edgar | 7951 | 6981 | 563 | 407 | 0 | 0.878 | 0.925 | 0.000 |
