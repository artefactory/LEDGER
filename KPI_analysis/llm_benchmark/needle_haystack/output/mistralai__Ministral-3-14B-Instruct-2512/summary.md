# Needle-in-a-haystack KPI benchmark — mistralai/Ministral-3-14B-Instruct-2512

- Responses: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/output/mistralai__Ministral-3-14B-Instruct-2512/responses.jsonl`
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Match tolerance: ±0.05% (strict: ±0.050%)
- Queries scored: 10000

## Headline

| metric | value |
| --- | --- |
| queries_scored | 10000 |
| eval_n (matched+wrong+not_found) | 10000 |
| matched | 8791 |
| wrong | 1135 |
| not_found | 74 |
| no_response | 0 |
| skipped | 0 |
| accuracy | 0.8791 |
| accuracy_strict | 0.8791 |
| attempt_rate | 0.9926 |
| precision_when_found | 0.8857 |
| median_abs_rel_error | 0.0000 |

## Wrong-answer diagnostics

How the `wrong` answers break down (systematic failure modes):

| bucket | count |
| --- | --- |
| scope_factor | 665 |
| other | 166 |
| scale_error(x1e-6) | 85 |
| scale_error(x1e-3) | 73 |
| sign_error | 68 |
| scale_error(x1e+3) | 51 |
| year_shift(-1) | 18 |
| year_shift(+1) | 3 |
| scale_error(x1e-9) | 3 |
| year_shift(+2) | 2 |
| year_shift(-2) | 1 |

## Per KPI

| kpi | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_term_debt_total | 260 | 260 | 137 | 123 | 0 | 0.527 | 0.527 | 0.527 | 0.000 |
| short_term_borrowings | 123 | 123 | 88 | 35 | 0 | 0.715 | 0.715 | 0.715 | 0.000 |
| interest_expense | 301 | 301 | 233 | 67 | 1 | 0.774 | 0.774 | 0.777 | 0.000 |
| cash_incl_restricted | 174 | 174 | 135 | 30 | 9 | 0.776 | 0.776 | 0.818 | 0.000 |
| accounts_receivable | 354 | 354 | 287 | 66 | 1 | 0.811 | 0.811 | 0.813 | 0.000 |
| depreciation_amortization | 427 | 427 | 347 | 76 | 4 | 0.813 | 0.813 | 0.820 | 0.000 |
| shares_outstanding | 344 | 344 | 285 | 56 | 3 | 0.828 | 0.828 | 0.836 | 0.000 |
| income_tax_expense | 396 | 396 | 330 | 65 | 1 | 0.833 | 0.833 | 0.835 | 0.000 |
| long_term_debt_current | 228 | 228 | 192 | 35 | 1 | 0.842 | 0.842 | 0.846 | 0.000 |
| net_income | 459 | 459 | 387 | 68 | 4 | 0.843 | 0.843 | 0.851 | 0.000 |
| investing_cash_flow | 444 | 444 | 379 | 62 | 3 | 0.854 | 0.854 | 0.859 | 0.000 |
| dividends_paid | 207 | 207 | 177 | 28 | 2 | 0.855 | 0.855 | 0.863 | 0.000 |
| capex | 369 | 369 | 323 | 45 | 1 | 0.875 | 0.875 | 0.878 | 0.000 |
| sga_expense | 358 | 358 | 315 | 41 | 2 | 0.880 | 0.880 | 0.885 | 0.000 |
| cost_of_revenue | 298 | 298 | 264 | 32 | 2 | 0.886 | 0.886 | 0.892 | 0.000 |
| financing_cash_flow | 446 | 446 | 398 | 45 | 3 | 0.892 | 0.892 | 0.898 | 0.000 |
| operating_income | 366 | 366 | 330 | 33 | 3 | 0.902 | 0.902 | 0.909 | 0.000 |
| gross_profit | 259 | 259 | 239 | 18 | 2 | 0.923 | 0.923 | 0.930 | 0.000 |
| total_liabilities | 326 | 326 | 301 | 22 | 3 | 0.923 | 0.923 | 0.932 | 0.000 |
| long_term_debt_noncurrent | 276 | 276 | 256 | 20 | 0 | 0.928 | 0.928 | 0.928 | 0.000 |
| stockholders_equity | 436 | 436 | 406 | 26 | 4 | 0.931 | 0.931 | 0.940 | 0.000 |
| revenue | 422 | 422 | 393 | 27 | 2 | 0.931 | 0.931 | 0.936 | 0.000 |
| total_assets | 467 | 467 | 436 | 29 | 2 | 0.934 | 0.934 | 0.938 | 0.000 |
| accounts_payable | 357 | 357 | 334 | 21 | 2 | 0.936 | 0.936 | 0.941 | 0.000 |
| operating_cash_flow | 463 | 463 | 436 | 23 | 4 | 0.942 | 0.942 | 0.950 | 0.000 |
| stockholders_equity_incl_nci | 227 | 227 | 215 | 11 | 1 | 0.947 | 0.947 | 0.951 | 0.000 |
| eps_basic | 105 | 105 | 100 | 4 | 1 | 0.952 | 0.952 | 0.962 | 0.000 |
| rd_expense | 194 | 194 | 185 | 7 | 2 | 0.954 | 0.954 | 0.964 | 0.000 |
| inventory | 340 | 340 | 325 | 10 | 5 | 0.956 | 0.956 | 0.970 | 0.000 |
| eps_diluted | 118 | 118 | 113 | 3 | 2 | 0.958 | 0.958 | 0.974 | 0.000 |
| cash_and_equivalents | 456 | 456 | 445 | 7 | 4 | 0.976 | 0.976 | 0.985 | 0.000 |

## Per fiscal year

| year | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2017 | 1477 | 1477 | 1243 | 226 | 8 | 0.842 | 0.842 | 0.846 | 0.000 |
| 2020 | 1755 | 1755 | 1541 | 200 | 14 | 0.878 | 0.878 | 0.885 | 0.000 |
| 2021 | 1841 | 1841 | 1618 | 213 | 10 | 0.879 | 0.879 | 0.884 | 0.000 |
| 2019 | 1610 | 1610 | 1417 | 174 | 19 | 0.880 | 0.880 | 0.891 | 0.000 |
| 2018 | 1433 | 1433 | 1266 | 161 | 6 | 0.883 | 0.883 | 0.887 | 0.000 |
| 2022 | 1884 | 1884 | 1706 | 161 | 17 | 0.906 | 0.906 | 0.914 | 0.000 |

## Per ground-truth source

| source | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 17 | 17 | 9 | 8 | 0 | 0.529 | 0.529 | 0.529 | 0.000 |
| edgar + alphavantage | 4398 | 4398 | 3738 | 594 | 66 | 0.850 | 0.850 | 0.863 | 0.000 |
| edgar | 5585 | 5585 | 5044 | 533 | 8 | 0.903 | 0.903 | 0.904 | 0.000 |

## Per unit class

| unit_class | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shares | 344 | 344 | 285 | 56 | 3 | 0.828 | 0.828 | 0.836 | 0.000 |
| monetary | 9433 | 9433 | 8293 | 1072 | 68 | 0.879 | 0.879 | 0.886 | 0.000 |
| per_share | 223 | 223 | 213 | 7 | 3 | 0.955 | 0.955 | 0.968 | 0.000 |
