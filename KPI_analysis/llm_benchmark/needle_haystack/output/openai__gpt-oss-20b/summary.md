# Needle-in-a-haystack KPI benchmark — openai/gpt-oss-20b

- Responses: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/output/openai__gpt-oss-20b/responses.jsonl`
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Match tolerance: ±1.00% (strict: ±0.050%)
- Queries scored: 10000

## Headline

| metric | value |
| --- | --- |
| queries_scored | 10000 |
| eval_n (matched+wrong+not_found) | 9996 |
| matched | 8786 |
| wrong | 1038 |
| not_found | 172 |
| no_response | 4 |
| skipped | 0 |
| accuracy | 0.8790 |
| accuracy_strict | 0.8528 |
| attempt_rate | 0.9828 |
| precision_when_found | 0.8943 |
| median_abs_rel_error | 0.0000 |

## Wrong-answer diagnostics

How the `wrong` answers break down (systematic failure modes):

| bucket | count |
| --- | --- |
| scope_factor | 416 |
| other | 185 |
| scale_error(x1e+3) | 134 |
| sign_error | 102 |
| scale_error(x1e-3) | 92 |
| scale_error(x1e-6) | 51 |
| year_shift(-1) | 30 |
| year_shift(+1) | 12 |
| scale_error(x1e+6) | 6 |
| year_shift(+2) | 5 |
| scale_error(x1e-9) | 3 |
| year_shift(-2) | 2 |

## Per KPI

| kpi | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_term_debt_total | 260 | 260 | 156 | 100 | 4 | 0.600 | 0.512 | 0.609 | 0.000 |
| cash_incl_restricted | 174 | 174 | 121 | 10 | 43 | 0.695 | 0.672 | 0.924 | 0.000 |
| shares_outstanding | 344 | 343 | 239 | 100 | 4 | 0.697 | 0.636 | 0.705 | 0.000 |
| short_term_borrowings | 123 | 123 | 87 | 23 | 13 | 0.707 | 0.683 | 0.791 | 0.000 |
| interest_expense | 301 | 300 | 228 | 69 | 3 | 0.760 | 0.747 | 0.768 | 0.000 |
| capex | 369 | 369 | 300 | 62 | 7 | 0.813 | 0.802 | 0.829 | 0.000 |
| dividends_paid | 207 | 207 | 170 | 34 | 3 | 0.821 | 0.812 | 0.833 | 0.000 |
| depreciation_amortization | 427 | 427 | 353 | 69 | 5 | 0.827 | 0.803 | 0.836 | 0.000 |
| income_tax_expense | 396 | 396 | 338 | 57 | 1 | 0.854 | 0.843 | 0.856 | 0.000 |
| long_term_debt_noncurrent | 276 | 275 | 240 | 33 | 2 | 0.873 | 0.833 | 0.879 | 0.000 |
| accounts_receivable | 354 | 354 | 309 | 43 | 2 | 0.873 | 0.853 | 0.878 | 0.000 |
| cash_and_equivalents | 456 | 456 | 399 | 40 | 17 | 0.875 | 0.862 | 0.909 | 0.000 |
| investing_cash_flow | 444 | 444 | 389 | 53 | 2 | 0.876 | 0.860 | 0.880 | 0.000 |
| long_term_debt_current | 228 | 228 | 202 | 22 | 4 | 0.886 | 0.877 | 0.902 | 0.000 |
| net_income | 459 | 458 | 410 | 42 | 6 | 0.895 | 0.867 | 0.907 | 0.000 |
| accounts_payable | 357 | 357 | 320 | 31 | 6 | 0.896 | 0.891 | 0.912 | 0.000 |
| stockholders_equity_incl_nci | 227 | 227 | 205 | 16 | 6 | 0.903 | 0.859 | 0.928 | 0.000 |
| eps_basic | 105 | 105 | 95 | 8 | 2 | 0.905 | 0.895 | 0.922 | 0.000 |
| stockholders_equity | 436 | 436 | 397 | 33 | 6 | 0.911 | 0.876 | 0.923 | 0.000 |
| financing_cash_flow | 446 | 446 | 409 | 33 | 4 | 0.917 | 0.910 | 0.925 | 0.000 |
| rd_expense | 194 | 194 | 178 | 13 | 3 | 0.918 | 0.902 | 0.932 | 0.000 |
| sga_expense | 358 | 358 | 331 | 24 | 3 | 0.925 | 0.866 | 0.932 | 0.000 |
| eps_diluted | 118 | 118 | 111 | 6 | 1 | 0.941 | 0.924 | 0.949 | 0.000 |
| operating_cash_flow | 463 | 463 | 436 | 23 | 4 | 0.942 | 0.924 | 0.950 | 0.000 |
| total_liabilities | 326 | 326 | 307 | 16 | 3 | 0.942 | 0.923 | 0.950 | 0.000 |
| inventory | 340 | 340 | 321 | 15 | 4 | 0.944 | 0.932 | 0.955 | 0.000 |
| operating_income | 366 | 366 | 346 | 17 | 3 | 0.945 | 0.907 | 0.953 | 0.000 |
| cost_of_revenue | 298 | 298 | 283 | 12 | 3 | 0.950 | 0.886 | 0.959 | 0.000 |
| revenue | 422 | 422 | 401 | 18 | 3 | 0.950 | 0.919 | 0.957 | 0.000 |
| gross_profit | 259 | 259 | 251 | 5 | 3 | 0.969 | 0.934 | 0.980 | 0.000 |
| total_assets | 467 | 467 | 454 | 11 | 2 | 0.972 | 0.951 | 0.976 | 0.000 |

## Per fiscal year

| year | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2017 | 1477 | 1477 | 1282 | 176 | 19 | 0.868 | 0.812 | 0.879 | 0.000 |
| 2019 | 1610 | 1610 | 1401 | 172 | 37 | 0.870 | 0.842 | 0.891 | 0.000 |
| 2018 | 1433 | 1432 | 1248 | 164 | 20 | 0.872 | 0.852 | 0.884 | 0.000 |
| 2020 | 1755 | 1754 | 1537 | 194 | 23 | 0.876 | 0.858 | 0.888 | 0.000 |
| 2021 | 1841 | 1841 | 1627 | 176 | 38 | 0.884 | 0.864 | 0.902 | 0.000 |
| 2022 | 1884 | 1882 | 1691 | 156 | 35 | 0.899 | 0.879 | 0.916 | 0.000 |

## Per ground-truth source

| source | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 17 | 17 | 11 | 6 | 0 | 0.647 | 0.647 | 0.647 | 0.000 |
| edgar + alphavantage | 3434 | 3433 | 2836 | 494 | 103 | 0.826 | 0.802 | 0.852 | 0.000 |
| edgar | 6549 | 6546 | 5939 | 538 | 69 | 0.907 | 0.880 | 0.917 | 0.000 |

## Per unit class

| unit_class | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shares | 344 | 343 | 239 | 100 | 4 | 0.697 | 0.636 | 0.705 | 0.000 |
| monetary | 9433 | 9430 | 8341 | 924 | 165 | 0.885 | 0.859 | 0.900 | 0.000 |
| per_share | 223 | 223 | 206 | 14 | 3 | 0.924 | 0.910 | 0.936 | 0.000 |
