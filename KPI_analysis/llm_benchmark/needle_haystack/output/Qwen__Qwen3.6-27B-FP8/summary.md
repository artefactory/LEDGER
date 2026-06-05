# Needle-in-a-haystack KPI benchmark — Qwen/Qwen3.6-27B-FP8

- Responses: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/output/Qwen__Qwen3.6-27B-FP8/responses.jsonl`
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Match tolerance: ±0.05% (strict: ±0.050%)
- Queries scored: 10000

## Headline

| metric | value |
| --- | --- |
| queries_scored | 10000 |
| eval_n (matched+wrong+not_found) | 10000 |
| matched | 9137 |
| wrong | 630 |
| not_found | 233 |
| no_response | 0 |
| skipped | 0 |
| accuracy | 0.9137 |
| accuracy_strict | 0.9137 |
| attempt_rate | 0.9767 |
| precision_when_found | 0.9355 |
| median_abs_rel_error | 0.0000 |

## Wrong-answer diagnostics

How the `wrong` answers break down (systematic failure modes):

| bucket | count |
| --- | --- |
| scope_factor | 517 |
| other | 86 |
| sign_error | 8 |
| year_shift(-1) | 6 |
| scale_error(x1e+3) | 5 |
| scale_error(x1e-9) | 3 |
| year_shift(+2) | 2 |
| year_shift(+1) | 2 |
| scale_error(x1e-3) | 1 |

## Per KPI

| kpi | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cash_incl_restricted | 174 | 174 | 88 | 3 | 83 | 0.506 | 0.506 | 0.967 | 0.000 |
| long_term_debt_total | 260 | 260 | 162 | 91 | 7 | 0.623 | 0.623 | 0.640 | 0.000 |
| short_term_borrowings | 123 | 123 | 82 | 11 | 30 | 0.667 | 0.667 | 0.882 | 0.000 |
| depreciation_amortization | 427 | 427 | 348 | 75 | 4 | 0.815 | 0.815 | 0.823 | 0.000 |
| shares_outstanding | 344 | 344 | 295 | 45 | 4 | 0.858 | 0.858 | 0.868 | 0.000 |
| sga_expense | 358 | 358 | 316 | 34 | 8 | 0.883 | 0.883 | 0.903 | 0.000 |
| accounts_receivable | 354 | 354 | 314 | 37 | 3 | 0.887 | 0.887 | 0.895 | 0.000 |
| dividends_paid | 207 | 207 | 185 | 16 | 6 | 0.894 | 0.894 | 0.920 | 0.000 |
| capex | 369 | 369 | 330 | 33 | 6 | 0.894 | 0.894 | 0.909 | 0.000 |
| cost_of_revenue | 298 | 298 | 270 | 24 | 4 | 0.906 | 0.906 | 0.918 | 0.000 |
| interest_expense | 301 | 301 | 273 | 24 | 4 | 0.907 | 0.907 | 0.919 | 0.000 |
| net_income | 459 | 459 | 420 | 35 | 4 | 0.915 | 0.915 | 0.923 | 0.000 |
| long_term_debt_current | 228 | 228 | 209 | 15 | 4 | 0.917 | 0.917 | 0.933 | 0.000 |
| accounts_payable | 357 | 357 | 333 | 21 | 3 | 0.933 | 0.933 | 0.941 | 0.000 |
| revenue | 422 | 422 | 394 | 23 | 5 | 0.934 | 0.934 | 0.945 | 0.000 |
| operating_income | 366 | 366 | 345 | 16 | 5 | 0.943 | 0.943 | 0.956 | 0.000 |
| rd_expense | 194 | 194 | 183 | 8 | 3 | 0.943 | 0.943 | 0.958 | 0.000 |
| gross_profit | 259 | 259 | 246 | 10 | 3 | 0.950 | 0.950 | 0.961 | 0.000 |
| stockholders_equity_incl_nci | 227 | 227 | 216 | 5 | 6 | 0.952 | 0.952 | 0.977 | 0.000 |
| eps_basic | 105 | 105 | 100 | 3 | 2 | 0.952 | 0.952 | 0.971 | 0.000 |
| stockholders_equity | 436 | 436 | 416 | 14 | 6 | 0.954 | 0.954 | 0.967 | 0.000 |
| inventory | 340 | 340 | 325 | 9 | 6 | 0.956 | 0.956 | 0.973 | 0.000 |
| eps_diluted | 118 | 118 | 113 | 3 | 2 | 0.958 | 0.958 | 0.974 | 0.000 |
| income_tax_expense | 396 | 396 | 380 | 16 | 0 | 0.960 | 0.960 | 0.960 | 0.000 |
| total_liabilities | 326 | 326 | 313 | 9 | 4 | 0.960 | 0.960 | 0.972 | 0.000 |
| operating_cash_flow | 463 | 463 | 447 | 11 | 5 | 0.965 | 0.965 | 0.976 | 0.000 |
| investing_cash_flow | 444 | 444 | 430 | 10 | 4 | 0.968 | 0.968 | 0.977 | 0.000 |
| total_assets | 467 | 467 | 453 | 11 | 3 | 0.970 | 0.970 | 0.976 | 0.000 |
| cash_and_equivalents | 456 | 456 | 444 | 8 | 4 | 0.974 | 0.974 | 0.982 | 0.000 |
| financing_cash_flow | 446 | 446 | 436 | 6 | 4 | 0.978 | 0.978 | 0.986 | 0.000 |
| long_term_debt_noncurrent | 276 | 276 | 271 | 4 | 1 | 0.982 | 0.982 | 0.985 | 0.000 |

## Per fiscal year

| year | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2017 | 1477 | 1477 | 1304 | 144 | 29 | 0.883 | 0.883 | 0.901 | 0.000 |
| 2019 | 1610 | 1610 | 1462 | 97 | 51 | 0.908 | 0.908 | 0.938 | 0.000 |
| 2021 | 1841 | 1841 | 1685 | 114 | 42 | 0.915 | 0.915 | 0.937 | 0.000 |
| 2018 | 1433 | 1433 | 1317 | 85 | 31 | 0.919 | 0.919 | 0.939 | 0.000 |
| 2020 | 1755 | 1755 | 1613 | 103 | 39 | 0.919 | 0.919 | 0.940 | 0.000 |
| 2022 | 1884 | 1884 | 1756 | 87 | 41 | 0.932 | 0.932 | 0.953 | 0.000 |

## Per ground-truth source

| source | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 17 | 17 | 12 | 5 | 0 | 0.706 | 0.706 | 0.706 | 0.000 |
| edgar + alphavantage | 4398 | 4398 | 3903 | 333 | 162 | 0.887 | 0.887 | 0.921 | 0.000 |
| edgar | 5585 | 5585 | 5222 | 292 | 71 | 0.935 | 0.935 | 0.947 | 0.000 |

## Per unit class

| unit_class | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shares | 344 | 344 | 295 | 45 | 4 | 0.858 | 0.858 | 0.868 | 0.000 |
| monetary | 9433 | 9433 | 8629 | 579 | 225 | 0.915 | 0.915 | 0.937 | 0.000 |
| per_share | 223 | 223 | 213 | 6 | 4 | 0.955 | 0.955 | 0.973 | 0.000 |
