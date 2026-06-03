# Needle-in-a-haystack KPI benchmark — nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8

- Responses: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/output/nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-FP8/responses.jsonl`
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Match tolerance: ±1.00% (strict: ±0.050%)
- Queries scored: 10000

## Headline

| metric | value |
| --- | --- |
| queries_scored | 10000 |
| eval_n (matched+wrong+not_found) | 9981 |
| matched | 1577 |
| wrong | 8219 |
| not_found | 185 |
| no_response | 19 |
| skipped | 0 |
| accuracy | 0.1580 |
| accuracy_strict | 0.1504 |
| attempt_rate | 0.9815 |
| precision_when_found | 0.1610 |
| median_abs_rel_error | 0.9990 |

## Wrong-answer diagnostics

How the `wrong` answers break down (systematic failure modes):

| bucket | count |
| --- | --- |
| scale_error(x1e-3) | 5208 |
| scale_error(x1e-6) | 2126 |
| other | 756 |
| scope_factor | 92 |
| sign_error | 12 |
| year_shift(-1) | 7 |
| scale_error(x1e-9) | 5 |
| year_shift(+1) | 5 |
| year_shift(+2) | 4 |
| scale_error(x1e+3) | 3 |
| year_shift(-2) | 1 |

## Per KPI

| kpi | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_term_debt_total | 260 | 258 | 13 | 244 | 1 | 0.050 | 0.047 | 0.051 | 0.999 |
| inventory | 340 | 338 | 18 | 320 | 0 | 0.053 | 0.053 | 0.053 | 0.999 |
| cost_of_revenue | 298 | 298 | 16 | 282 | 0 | 0.054 | 0.047 | 0.054 | 0.999 |
| accounts_receivable | 354 | 353 | 19 | 334 | 0 | 0.054 | 0.054 | 0.054 | 0.999 |
| long_term_debt_current | 228 | 228 | 13 | 215 | 0 | 0.057 | 0.057 | 0.057 | 0.999 |
| income_tax_expense | 396 | 396 | 28 | 368 | 0 | 0.071 | 0.066 | 0.071 | 0.999 |
| stockholders_equity_incl_nci | 227 | 225 | 18 | 187 | 20 | 0.080 | 0.076 | 0.088 | 0.999 |
| long_term_debt_noncurrent | 276 | 276 | 24 | 251 | 1 | 0.087 | 0.087 | 0.087 | 0.999 |
| total_liabilities | 326 | 326 | 29 | 274 | 23 | 0.089 | 0.089 | 0.096 | 0.999 |
| sga_expense | 358 | 356 | 32 | 307 | 17 | 0.090 | 0.087 | 0.094 | 0.999 |
| accounts_payable | 357 | 357 | 33 | 324 | 0 | 0.092 | 0.092 | 0.092 | 0.999 |
| operating_income | 366 | 364 | 35 | 322 | 7 | 0.096 | 0.093 | 0.098 | 0.999 |
| investing_cash_flow | 444 | 444 | 43 | 401 | 0 | 0.097 | 0.097 | 0.097 | 0.999 |
| short_term_borrowings | 123 | 122 | 12 | 109 | 1 | 0.098 | 0.098 | 0.099 | 1.000 |
| financing_cash_flow | 446 | 446 | 45 | 399 | 2 | 0.101 | 0.096 | 0.101 | 0.999 |
| interest_expense | 301 | 300 | 33 | 267 | 0 | 0.110 | 0.103 | 0.110 | 0.999 |
| stockholders_equity | 436 | 435 | 49 | 351 | 35 | 0.113 | 0.106 | 0.122 | 0.999 |
| cash_incl_restricted | 174 | 174 | 20 | 154 | 0 | 0.115 | 0.115 | 0.115 | 0.999 |
| total_assets | 467 | 467 | 56 | 381 | 30 | 0.120 | 0.116 | 0.128 | 0.999 |
| operating_cash_flow | 463 | 462 | 56 | 399 | 7 | 0.121 | 0.119 | 0.123 | 0.999 |
| depreciation_amortization | 427 | 427 | 52 | 375 | 0 | 0.122 | 0.119 | 0.122 | 0.999 |
| net_income | 459 | 457 | 63 | 391 | 3 | 0.138 | 0.136 | 0.139 | 0.999 |
| capex | 369 | 369 | 51 | 318 | 0 | 0.138 | 0.122 | 0.138 | 0.999 |
| gross_profit | 259 | 258 | 49 | 209 | 0 | 0.190 | 0.186 | 0.190 | 0.999 |
| cash_and_equivalents | 456 | 456 | 93 | 362 | 1 | 0.204 | 0.204 | 0.204 | 0.999 |
| revenue | 422 | 421 | 99 | 310 | 12 | 0.235 | 0.223 | 0.242 | 0.999 |
| rd_expense | 194 | 194 | 46 | 147 | 1 | 0.237 | 0.227 | 0.238 | 0.999 |
| dividends_paid | 207 | 207 | 74 | 133 | 0 | 0.357 | 0.329 | 0.357 | 0.999 |
| shares_outstanding | 344 | 344 | 258 | 63 | 23 | 0.750 | 0.660 | 0.804 | 0.000 |
| eps_diluted | 118 | 118 | 103 | 15 | 0 | 0.873 | 0.856 | 0.873 | 0.000 |
| eps_basic | 105 | 105 | 97 | 7 | 1 | 0.924 | 0.895 | 0.933 | 0.000 |

## Per fiscal year

| year | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | 1884 | 1876 | 241 | 1592 | 43 | 0.128 | 0.124 | 0.131 | 0.999 |
| 2021 | 1841 | 1841 | 280 | 1539 | 22 | 0.152 | 0.143 | 0.154 | 0.999 |
| 2018 | 1433 | 1433 | 232 | 1165 | 36 | 0.162 | 0.154 | 0.166 | 0.999 |
| 2020 | 1755 | 1755 | 291 | 1435 | 29 | 0.166 | 0.158 | 0.169 | 0.999 |
| 2019 | 1610 | 1610 | 274 | 1314 | 22 | 0.170 | 0.163 | 0.173 | 0.999 |
| 2017 | 1477 | 1466 | 259 | 1174 | 33 | 0.177 | 0.166 | 0.181 | 0.999 |

## Per ground-truth source

| source | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edgar | 6549 | 6549 | 680 | 5758 | 111 | 0.104 | 0.098 | 0.106 | 0.999 |
| yfinance | 17 | 17 | 4 | 13 | 0 | 0.235 | 0.235 | 0.235 | 0.999 |
| edgar + alphavantage | 3434 | 3415 | 893 | 2448 | 74 | 0.261 | 0.250 | 0.267 | 0.999 |

## Per unit class

| unit_class | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| monetary | 9433 | 9414 | 1119 | 8134 | 161 | 0.119 | 0.115 | 0.121 | 0.999 |
| shares | 344 | 344 | 258 | 63 | 23 | 0.750 | 0.660 | 0.804 | 0.000 |
| per_share | 223 | 223 | 200 | 22 | 1 | 0.897 | 0.874 | 0.901 | 0.000 |
