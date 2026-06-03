# Needle-in-a-haystack KPI benchmark — mistralai/Ministral-3-14B-Instruct-2512

- Responses: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/output/mistralai__Ministral-3-14B-Instruct-2512/responses.jsonl`
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Match tolerance: ±1.00% (strict: ±0.050%)
- Queries scored: 10000

## Headline

| metric | value |
| --- | --- |
| queries_scored | 10000 |
| eval_n (matched+wrong+not_found) | 10000 |
| matched | 9072 |
| wrong | 854 |
| not_found | 74 |
| no_response | 0 |
| skipped | 0 |
| accuracy | 0.9072 |
| accuracy_strict | 0.8791 |
| attempt_rate | 0.9926 |
| precision_when_found | 0.9140 |
| median_abs_rel_error | 0.0000 |

## Wrong-answer diagnostics

How the `wrong` answers break down (systematic failure modes):

| bucket | count |
| --- | --- |
| scope_factor | 366 |
| other | 163 |
| scale_error(x1e-6) | 86 |
| scale_error(x1e-3) | 73 |
| sign_error | 68 |
| scale_error(x1e+3) | 51 |
| year_shift(-1) | 25 |
| year_shift(+1) | 13 |
| year_shift(+2) | 4 |
| scale_error(x1e-9) | 3 |
| year_shift(-2) | 2 |

## Per KPI

| kpi | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_term_debt_total | 260 | 260 | 154 | 106 | 0 | 0.592 | 0.527 | 0.592 | 0.000 |
| short_term_borrowings | 123 | 123 | 92 | 31 | 0 | 0.748 | 0.715 | 0.748 | 0.000 |
| cash_incl_restricted | 174 | 174 | 138 | 27 | 9 | 0.793 | 0.776 | 0.836 | 0.000 |
| interest_expense | 301 | 301 | 242 | 58 | 1 | 0.804 | 0.774 | 0.807 | 0.000 |
| depreciation_amortization | 427 | 427 | 358 | 65 | 4 | 0.838 | 0.813 | 0.846 | 0.000 |
| long_term_debt_current | 228 | 228 | 194 | 33 | 1 | 0.851 | 0.842 | 0.855 | 0.000 |
| income_tax_expense | 396 | 396 | 338 | 57 | 1 | 0.854 | 0.833 | 0.856 | 0.000 |
| investing_cash_flow | 444 | 444 | 385 | 56 | 3 | 0.867 | 0.854 | 0.873 | 0.000 |
| net_income | 459 | 459 | 401 | 54 | 4 | 0.874 | 0.843 | 0.881 | 0.000 |
| dividends_paid | 207 | 207 | 183 | 22 | 2 | 0.884 | 0.855 | 0.893 | 0.000 |
| shares_outstanding | 344 | 344 | 305 | 36 | 3 | 0.887 | 0.828 | 0.894 | 0.000 |
| accounts_receivable | 354 | 354 | 315 | 38 | 1 | 0.890 | 0.811 | 0.892 | 0.000 |
| capex | 369 | 369 | 331 | 37 | 1 | 0.897 | 0.875 | 0.899 | 0.000 |
| financing_cash_flow | 446 | 446 | 407 | 36 | 3 | 0.913 | 0.892 | 0.919 | 0.000 |
| operating_income | 366 | 366 | 343 | 20 | 3 | 0.937 | 0.902 | 0.945 | 0.000 |
| sga_expense | 358 | 358 | 336 | 20 | 2 | 0.939 | 0.880 | 0.944 | 0.000 |
| accounts_payable | 357 | 357 | 336 | 19 | 2 | 0.941 | 0.936 | 0.946 | 0.000 |
| total_liabilities | 326 | 326 | 307 | 16 | 3 | 0.942 | 0.923 | 0.950 | 0.000 |
| long_term_debt_noncurrent | 276 | 276 | 260 | 16 | 0 | 0.942 | 0.928 | 0.942 | 0.000 |
| cost_of_revenue | 298 | 298 | 283 | 13 | 2 | 0.950 | 0.886 | 0.956 | 0.000 |
| revenue | 422 | 422 | 403 | 17 | 2 | 0.955 | 0.931 | 0.960 | 0.000 |
| total_assets | 467 | 467 | 446 | 19 | 2 | 0.955 | 0.934 | 0.959 | 0.000 |
| gross_profit | 259 | 259 | 248 | 9 | 2 | 0.958 | 0.923 | 0.965 | 0.000 |
| stockholders_equity | 436 | 436 | 418 | 14 | 4 | 0.959 | 0.931 | 0.968 | 0.000 |
| eps_basic | 105 | 105 | 101 | 3 | 1 | 0.962 | 0.952 | 0.971 | 0.000 |
| operating_cash_flow | 463 | 463 | 447 | 12 | 4 | 0.965 | 0.942 | 0.974 | 0.000 |
| eps_diluted | 118 | 118 | 114 | 2 | 2 | 0.966 | 0.958 | 0.983 | 0.000 |
| inventory | 340 | 340 | 329 | 6 | 5 | 0.968 | 0.956 | 0.982 | 0.000 |
| stockholders_equity_incl_nci | 227 | 227 | 220 | 6 | 1 | 0.969 | 0.947 | 0.973 | 0.000 |
| cash_and_equivalents | 456 | 456 | 447 | 5 | 4 | 0.980 | 0.976 | 0.989 | 0.000 |
| rd_expense | 194 | 194 | 191 | 1 | 2 | 0.985 | 0.954 | 0.995 | 0.000 |

## Per fiscal year

| year | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | 1755 | 1755 | 1577 | 164 | 14 | 0.899 | 0.878 | 0.906 | 0.000 |
| 2017 | 1477 | 1477 | 1328 | 141 | 8 | 0.899 | 0.842 | 0.904 | 0.000 |
| 2021 | 1841 | 1841 | 1666 | 165 | 10 | 0.905 | 0.879 | 0.910 | 0.000 |
| 2019 | 1610 | 1610 | 1462 | 129 | 19 | 0.908 | 0.880 | 0.919 | 0.000 |
| 2018 | 1433 | 1433 | 1302 | 125 | 6 | 0.909 | 0.883 | 0.912 | 0.000 |
| 2022 | 1884 | 1884 | 1737 | 130 | 17 | 0.922 | 0.906 | 0.930 | 0.000 |

## Per ground-truth source

| source | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 17 | 17 | 9 | 8 | 0 | 0.529 | 0.529 | 0.529 | 0.000 |
| edgar + alphavantage | 3434 | 3434 | 2949 | 421 | 64 | 0.859 | 0.831 | 0.875 | 0.000 |
| edgar | 6549 | 6549 | 6114 | 425 | 10 | 0.934 | 0.905 | 0.935 | 0.000 |

## Per unit class

| unit_class | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shares | 344 | 344 | 305 | 36 | 3 | 0.887 | 0.828 | 0.894 | 0.000 |
| monetary | 9433 | 9433 | 8552 | 813 | 68 | 0.907 | 0.879 | 0.913 | 0.000 |
| per_share | 223 | 223 | 215 | 5 | 3 | 0.964 | 0.955 | 0.977 | 0.000 |
