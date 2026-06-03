# Needle-in-a-haystack KPI benchmark — Qwen/Qwen3.6-27B-FP8

- Responses: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/output/Qwen__Qwen3.6-27B-FP8/responses.jsonl`
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Match tolerance: ±1.00% (strict: ±0.050%)
- Queries scored: 10000

## Headline

| metric | value |
| --- | --- |
| queries_scored | 10000 |
| eval_n (matched+wrong+not_found) | 10000 |
| matched | 9357 |
| wrong | 410 |
| not_found | 233 |
| no_response | 0 |
| skipped | 0 |
| accuracy | 0.9357 |
| accuracy_strict | 0.9137 |
| attempt_rate | 0.9767 |
| precision_when_found | 0.9580 |
| median_abs_rel_error | 0.0000 |

## Wrong-answer diagnostics

How the `wrong` answers break down (systematic failure modes):

| bucket | count |
| --- | --- |
| scope_factor | 285 |
| other | 85 |
| year_shift(-1) | 10 |
| year_shift(+1) | 9 |
| sign_error | 8 |
| scale_error(x1e+3) | 5 |
| year_shift(+2) | 4 |
| scale_error(x1e-9) | 3 |
| scale_error(x1e-3) | 1 |

## Per KPI

| kpi | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cash_incl_restricted | 174 | 174 | 89 | 2 | 83 | 0.511 | 0.506 | 0.978 | 0.000 |
| long_term_debt_total | 260 | 260 | 179 | 74 | 7 | 0.688 | 0.623 | 0.708 | 0.000 |
| short_term_borrowings | 123 | 123 | 85 | 8 | 30 | 0.691 | 0.667 | 0.914 | 0.000 |
| depreciation_amortization | 427 | 427 | 359 | 64 | 4 | 0.841 | 0.815 | 0.849 | 0.000 |
| accounts_receivable | 354 | 354 | 319 | 32 | 3 | 0.901 | 0.887 | 0.909 | 0.000 |
| capex | 369 | 369 | 334 | 29 | 6 | 0.905 | 0.894 | 0.920 | 0.000 |
| dividends_paid | 207 | 207 | 188 | 13 | 6 | 0.908 | 0.894 | 0.935 | 0.000 |
| interest_expense | 301 | 301 | 275 | 22 | 4 | 0.914 | 0.907 | 0.926 | 0.000 |
| shares_outstanding | 344 | 344 | 315 | 25 | 4 | 0.916 | 0.858 | 0.926 | 0.000 |
| long_term_debt_current | 228 | 228 | 210 | 14 | 4 | 0.921 | 0.917 | 0.938 | 0.000 |
| accounts_payable | 357 | 357 | 335 | 19 | 3 | 0.938 | 0.933 | 0.946 | 0.000 |
| sga_expense | 358 | 358 | 336 | 14 | 8 | 0.939 | 0.883 | 0.960 | 0.000 |
| net_income | 459 | 459 | 434 | 21 | 4 | 0.946 | 0.915 | 0.954 | 0.000 |
| revenue | 422 | 422 | 404 | 13 | 5 | 0.957 | 0.934 | 0.969 | 0.000 |
| rd_expense | 194 | 194 | 186 | 5 | 3 | 0.959 | 0.943 | 0.974 | 0.000 |
| eps_basic | 105 | 105 | 101 | 2 | 2 | 0.962 | 0.952 | 0.981 | 0.000 |
| eps_diluted | 118 | 118 | 114 | 2 | 2 | 0.966 | 0.958 | 0.983 | 0.000 |
| cost_of_revenue | 298 | 298 | 288 | 6 | 4 | 0.966 | 0.906 | 0.980 | 0.000 |
| inventory | 340 | 340 | 329 | 5 | 6 | 0.968 | 0.956 | 0.985 | 0.000 |
| stockholders_equity_incl_nci | 227 | 227 | 220 | 1 | 6 | 0.969 | 0.952 | 0.995 | 0.000 |
| stockholders_equity | 436 | 436 | 424 | 6 | 6 | 0.972 | 0.954 | 0.986 | 0.000 |
| total_liabilities | 326 | 326 | 318 | 4 | 4 | 0.975 | 0.960 | 0.988 | 0.000 |
| cash_and_equivalents | 456 | 456 | 446 | 6 | 4 | 0.978 | 0.974 | 0.987 | 0.000 |
| operating_income | 366 | 366 | 358 | 3 | 5 | 0.978 | 0.943 | 0.992 | 0.000 |
| income_tax_expense | 396 | 396 | 388 | 8 | 0 | 0.980 | 0.960 | 0.980 | 0.000 |
| operating_cash_flow | 463 | 463 | 455 | 3 | 5 | 0.983 | 0.965 | 0.993 | 0.000 |
| investing_cash_flow | 444 | 444 | 437 | 3 | 4 | 0.984 | 0.968 | 0.993 | 0.000 |
| gross_profit | 259 | 259 | 255 | 1 | 3 | 0.985 | 0.950 | 0.996 | 0.000 |
| long_term_debt_noncurrent | 276 | 276 | 272 | 3 | 1 | 0.986 | 0.982 | 0.989 | 0.000 |
| financing_cash_flow | 446 | 446 | 441 | 1 | 4 | 0.989 | 0.978 | 0.998 | 0.000 |
| total_assets | 467 | 467 | 463 | 1 | 3 | 0.991 | 0.970 | 0.998 | 0.000 |

## Per fiscal year

| year | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | 1610 | 1610 | 1497 | 62 | 51 | 0.930 | 0.908 | 0.960 | 0.000 |
| 2021 | 1841 | 1841 | 1718 | 81 | 42 | 0.933 | 0.915 | 0.955 | 0.000 |
| 2017 | 1477 | 1477 | 1379 | 69 | 29 | 0.934 | 0.883 | 0.952 | 0.000 |
| 2020 | 1755 | 1755 | 1641 | 75 | 39 | 0.935 | 0.919 | 0.956 | 0.000 |
| 2018 | 1433 | 1433 | 1341 | 61 | 31 | 0.936 | 0.919 | 0.956 | 0.000 |
| 2022 | 1884 | 1884 | 1781 | 62 | 41 | 0.945 | 0.932 | 0.966 | 0.000 |

## Per ground-truth source

| source | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 17 | 17 | 12 | 5 | 0 | 0.706 | 0.706 | 0.706 | 0.000 |
| edgar + alphavantage | 3434 | 3434 | 3072 | 209 | 153 | 0.895 | 0.872 | 0.936 | 0.000 |
| edgar | 6549 | 6549 | 6273 | 196 | 80 | 0.958 | 0.936 | 0.970 | 0.000 |

## Per unit class

| unit_class | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shares | 344 | 344 | 315 | 25 | 4 | 0.916 | 0.858 | 0.926 | 0.000 |
| monetary | 9433 | 9433 | 8827 | 381 | 225 | 0.936 | 0.915 | 0.959 | 0.000 |
| per_share | 223 | 223 | 215 | 4 | 4 | 0.964 | 0.955 | 0.982 | 0.000 |
