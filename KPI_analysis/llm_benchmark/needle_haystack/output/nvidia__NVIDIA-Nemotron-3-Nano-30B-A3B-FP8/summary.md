# Needle-in-a-haystack KPI benchmark — nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8

- Responses: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/output/nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-FP8/responses.jsonl`
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Match tolerance: ±0.05% (strict: ±0.050%)
- Queries scored: 10000

## Headline

| metric | value |
| --- | --- |
| queries_scored | 10000 |
| eval_n (matched+wrong+not_found) | 9981 |
| matched | 1501 |
| wrong | 8295 |
| not_found | 185 |
| no_response | 19 |
| skipped | 0 |
| accuracy | 0.1504 |
| accuracy_strict | 0.1504 |
| attempt_rate | 0.9815 |
| precision_when_found | 0.1532 |
| median_abs_rel_error | 0.9990 |

## Wrong-answer diagnostics

How the `wrong` answers break down (systematic failure modes):

| bucket | count |
| --- | --- |
| scale_error(x1e-3) | 5074 |
| scale_error(x1e-6) | 2077 |
| other | 941 |
| scope_factor | 174 |
| sign_error | 10 |
| year_shift(-1) | 6 |
| scale_error(x1e-9) | 5 |
| scale_error(x1e+3) | 3 |
| year_shift(+1) | 3 |
| year_shift(+2) | 2 |

## Per KPI

| kpi | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_term_debt_total | 260 | 258 | 12 | 245 | 1 | 0.047 | 0.047 | 0.047 | 0.999 |
| cost_of_revenue | 298 | 298 | 14 | 284 | 0 | 0.047 | 0.047 | 0.047 | 0.999 |
| inventory | 340 | 338 | 18 | 320 | 0 | 0.053 | 0.053 | 0.053 | 0.999 |
| accounts_receivable | 354 | 353 | 19 | 334 | 0 | 0.054 | 0.054 | 0.054 | 0.999 |
| long_term_debt_current | 228 | 228 | 13 | 215 | 0 | 0.057 | 0.057 | 0.057 | 0.999 |
| income_tax_expense | 396 | 396 | 26 | 370 | 0 | 0.066 | 0.066 | 0.066 | 0.999 |
| stockholders_equity_incl_nci | 227 | 225 | 17 | 188 | 20 | 0.076 | 0.076 | 0.083 | 0.999 |
| long_term_debt_noncurrent | 276 | 276 | 24 | 251 | 1 | 0.087 | 0.087 | 0.087 | 0.999 |
| sga_expense | 358 | 356 | 31 | 308 | 17 | 0.087 | 0.087 | 0.091 | 0.999 |
| total_liabilities | 326 | 326 | 29 | 274 | 23 | 0.089 | 0.089 | 0.096 | 0.999 |
| accounts_payable | 357 | 357 | 33 | 324 | 0 | 0.092 | 0.092 | 0.092 | 0.999 |
| operating_income | 366 | 364 | 34 | 323 | 7 | 0.093 | 0.093 | 0.095 | 0.999 |
| financing_cash_flow | 446 | 446 | 43 | 401 | 2 | 0.096 | 0.096 | 0.097 | 0.999 |
| investing_cash_flow | 444 | 444 | 43 | 401 | 0 | 0.097 | 0.097 | 0.097 | 0.999 |
| short_term_borrowings | 123 | 122 | 12 | 109 | 1 | 0.098 | 0.098 | 0.099 | 1.000 |
| interest_expense | 301 | 300 | 31 | 269 | 0 | 0.103 | 0.103 | 0.103 | 0.999 |
| stockholders_equity | 436 | 435 | 46 | 354 | 35 | 0.106 | 0.106 | 0.115 | 0.999 |
| cash_incl_restricted | 174 | 174 | 20 | 154 | 0 | 0.115 | 0.115 | 0.115 | 0.999 |
| total_assets | 467 | 467 | 54 | 383 | 30 | 0.116 | 0.116 | 0.124 | 0.999 |
| operating_cash_flow | 463 | 462 | 55 | 400 | 7 | 0.119 | 0.119 | 0.121 | 0.999 |
| depreciation_amortization | 427 | 427 | 51 | 376 | 0 | 0.119 | 0.119 | 0.119 | 0.999 |
| capex | 369 | 369 | 45 | 324 | 0 | 0.122 | 0.122 | 0.122 | 0.999 |
| net_income | 459 | 457 | 62 | 392 | 3 | 0.136 | 0.136 | 0.137 | 0.999 |
| gross_profit | 259 | 258 | 48 | 210 | 0 | 0.186 | 0.186 | 0.186 | 0.999 |
| cash_and_equivalents | 456 | 456 | 93 | 362 | 1 | 0.204 | 0.204 | 0.204 | 0.999 |
| revenue | 422 | 421 | 94 | 315 | 12 | 0.223 | 0.223 | 0.230 | 0.999 |
| rd_expense | 194 | 194 | 44 | 149 | 1 | 0.227 | 0.227 | 0.228 | 0.999 |
| dividends_paid | 207 | 207 | 68 | 139 | 0 | 0.329 | 0.329 | 0.329 | 0.999 |
| shares_outstanding | 344 | 344 | 227 | 94 | 23 | 0.660 | 0.660 | 0.707 | 0.000 |
| eps_diluted | 118 | 118 | 101 | 17 | 0 | 0.856 | 0.856 | 0.856 | 0.000 |
| eps_basic | 105 | 105 | 94 | 10 | 1 | 0.895 | 0.895 | 0.904 | 0.000 |

## Per fiscal year

| year | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | 1884 | 1876 | 233 | 1600 | 43 | 0.124 | 0.124 | 0.127 | 0.999 |
| 2021 | 1841 | 1841 | 264 | 1555 | 22 | 0.143 | 0.143 | 0.145 | 0.999 |
| 2018 | 1433 | 1433 | 220 | 1177 | 36 | 0.154 | 0.154 | 0.157 | 0.999 |
| 2020 | 1755 | 1755 | 278 | 1448 | 29 | 0.158 | 0.158 | 0.161 | 0.999 |
| 2019 | 1610 | 1610 | 263 | 1325 | 22 | 0.163 | 0.163 | 0.166 | 0.999 |
| 2017 | 1477 | 1466 | 243 | 1190 | 33 | 0.166 | 0.166 | 0.170 | 0.999 |

## Per ground-truth source

| source | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edgar | 5585 | 5585 | 470 | 5028 | 87 | 0.084 | 0.084 | 0.085 | 0.999 |
| edgar + alphavantage | 4398 | 4379 | 1027 | 3254 | 98 | 0.235 | 0.235 | 0.240 | 0.999 |
| yfinance | 17 | 17 | 4 | 13 | 0 | 0.235 | 0.235 | 0.235 | 0.999 |

## Per unit class

| unit_class | n | eval_n | matched | wrong | not_found | accuracy | accuracy_strict | precision_when_found | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| monetary | 9433 | 9414 | 1079 | 8174 | 161 | 0.115 | 0.115 | 0.117 | 0.999 |
| shares | 344 | 344 | 227 | 94 | 23 | 0.660 | 0.660 | 0.707 | 0.000 |
| per_share | 223 | 223 | 195 | 27 | 1 | 0.874 | 0.874 | 0.878 | 0.000 |
