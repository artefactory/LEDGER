# Needle-in-a-haystack KPI benchmark — openai/gpt-oss-20b

- Responses: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/output/openai__gpt-oss-20b/responses.jsonl`
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Match tolerance: ±0.05% (strict: ±0.050%)
- Queries scored: 10000

## Headline

| metric | value |
| --- | --- |
| queries_scored | 10000 |
| eval_n (matched+wrong+not_found) | 9996 |
| matched | 8525 |
| wrong | 1299 |
| not_found | 172 |
| no_response | 4 |
| skipped | 0 |
| accuracy | 0.8528 |
| accuracy_strict | 0.8528 |
| attempt_rate | 0.9828 |
| precision_when_found | 0.8678 |
| median_abs_rel_error | 0.0000 |

## Wrong-answer diagnostics

How the `wrong` answers break down (systematic failure modes):

| bucket | count |
| --- | --- |
| scope_factor | 692 |
| other | 208 |
| scale_error(x1e+3) | 130 |
| sign_error | 103 |
| scale_error(x1e-3) | 84 |
| scale_error(x1e-6) | 48 |
| year_shift(-1) | 19 |
| year_shift(+1) | 8 |
| scale_error(x1e-9) | 3 |
| scale_error(x1e+6) | 3 |
| year_shift(+2) | 1 |

## Per KPI

| kpi | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_term_debt_total | 260 | 260 | 133 | 123 | 4 | 0.512 | 0.512 | 0.520 | 0.000 |
| shares_outstanding | 344 | 343 | 218 | 121 | 4 | 0.636 | 0.636 | 0.643 | 0.000 |
| cash_incl_restricted | 174 | 174 | 117 | 14 | 43 | 0.672 | 0.672 | 0.893 | 0.000 |
| short_term_borrowings | 123 | 123 | 84 | 26 | 13 | 0.683 | 0.683 | 0.764 | 0.000 |
| interest_expense | 301 | 300 | 224 | 73 | 3 | 0.747 | 0.747 | 0.754 | 0.000 |
| capex | 369 | 369 | 296 | 66 | 7 | 0.802 | 0.802 | 0.818 | 0.000 |
| depreciation_amortization | 427 | 427 | 343 | 79 | 5 | 0.803 | 0.803 | 0.813 | 0.000 |
| dividends_paid | 207 | 207 | 168 | 36 | 3 | 0.812 | 0.812 | 0.824 | 0.000 |
| long_term_debt_noncurrent | 276 | 275 | 229 | 44 | 2 | 0.833 | 0.833 | 0.839 | 0.000 |
| income_tax_expense | 396 | 396 | 334 | 61 | 1 | 0.843 | 0.843 | 0.846 | 0.000 |
| accounts_receivable | 354 | 354 | 302 | 50 | 2 | 0.853 | 0.853 | 0.858 | 0.000 |
| stockholders_equity_incl_nci | 227 | 227 | 195 | 26 | 6 | 0.859 | 0.859 | 0.882 | 0.000 |
| investing_cash_flow | 444 | 444 | 382 | 60 | 2 | 0.860 | 0.860 | 0.864 | 0.000 |
| cash_and_equivalents | 456 | 456 | 393 | 46 | 17 | 0.862 | 0.862 | 0.895 | 0.000 |
| sga_expense | 358 | 358 | 310 | 45 | 3 | 0.866 | 0.866 | 0.873 | 0.000 |
| net_income | 459 | 458 | 397 | 55 | 6 | 0.867 | 0.867 | 0.878 | 0.000 |
| stockholders_equity | 436 | 436 | 382 | 48 | 6 | 0.876 | 0.876 | 0.888 | 0.000 |
| long_term_debt_current | 228 | 228 | 200 | 24 | 4 | 0.877 | 0.877 | 0.893 | 0.000 |
| cost_of_revenue | 298 | 298 | 264 | 31 | 3 | 0.886 | 0.886 | 0.895 | 0.000 |
| accounts_payable | 357 | 357 | 318 | 33 | 6 | 0.891 | 0.891 | 0.906 | 0.000 |
| eps_basic | 105 | 105 | 94 | 9 | 2 | 0.895 | 0.895 | 0.913 | 0.000 |
| rd_expense | 194 | 194 | 175 | 16 | 3 | 0.902 | 0.902 | 0.916 | 0.000 |
| operating_income | 366 | 366 | 332 | 31 | 3 | 0.907 | 0.907 | 0.915 | 0.000 |
| financing_cash_flow | 446 | 446 | 406 | 36 | 4 | 0.910 | 0.910 | 0.919 | 0.000 |
| revenue | 422 | 422 | 388 | 31 | 3 | 0.919 | 0.919 | 0.926 | 0.000 |
| total_liabilities | 326 | 326 | 301 | 22 | 3 | 0.923 | 0.923 | 0.932 | 0.000 |
| eps_diluted | 118 | 118 | 109 | 8 | 1 | 0.924 | 0.924 | 0.932 | 0.000 |
| operating_cash_flow | 463 | 463 | 428 | 31 | 4 | 0.924 | 0.924 | 0.932 | 0.000 |
| inventory | 340 | 340 | 317 | 19 | 4 | 0.932 | 0.932 | 0.943 | 0.000 |
| gross_profit | 259 | 259 | 242 | 14 | 3 | 0.934 | 0.934 | 0.945 | 0.000 |
| total_assets | 467 | 467 | 444 | 21 | 2 | 0.951 | 0.951 | 0.955 | 0.000 |

## Per fiscal year

| year | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2017 | 1477 | 1477 | 1199 | 259 | 19 | 0.812 | 0.812 | 0.822 | 0.000 |
| 2019 | 1610 | 1610 | 1356 | 217 | 37 | 0.842 | 0.842 | 0.862 | 0.000 |
| 2018 | 1433 | 1432 | 1220 | 192 | 20 | 0.852 | 0.852 | 0.864 | 0.000 |
| 2020 | 1755 | 1754 | 1505 | 226 | 23 | 0.858 | 0.858 | 0.869 | 0.000 |
| 2021 | 1841 | 1841 | 1590 | 213 | 38 | 0.864 | 0.864 | 0.882 | 0.000 |
| 2022 | 1884 | 1882 | 1655 | 192 | 35 | 0.879 | 0.879 | 0.896 | 0.000 |

## Per ground-truth source

| source | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 17 | 17 | 11 | 6 | 0 | 0.647 | 0.647 | 0.647 | 0.000 |
| edgar + alphavantage | 4398 | 4396 | 3620 | 663 | 113 | 0.823 | 0.823 | 0.845 | 0.000 |
| edgar | 5585 | 5583 | 4894 | 630 | 59 | 0.877 | 0.877 | 0.886 | 0.000 |

## Per unit class

| unit_class | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shares | 344 | 343 | 218 | 121 | 4 | 0.636 | 0.636 | 0.643 | 0.000 |
| monetary | 9433 | 9430 | 8104 | 1161 | 165 | 0.859 | 0.859 | 0.875 | 0.000 |
| per_share | 223 | 223 | 203 | 17 | 3 | 0.910 | 0.910 | 0.923 | 0.000 |
