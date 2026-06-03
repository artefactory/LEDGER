# LLM KPI extraction benchmark — summary

- Tolerance: ±1.0%
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Test-set scope: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/test_set_reports.txt` (494 reports, 13455 ground-truth KPIs)
- Predictions root: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/multi_kpi_extraction/output/mistralai__Ministral-3-14B-Instruct-2512/raw`
- Reports loaded: 977 (ok=944, failed=33, error=0)

## Overall

| metric | value |
| --- | --- |
| n_predictions | 14178 |
| n_ground_truth | 13455 |
| matched | 5732 |
| wrong | 7065 |
| missing | 658 |
| extra | 1381 |
| recall (matched/gt) | 0.4260 |
| precision (matched/(matched+wrong)) | 0.4479 |

## Per KPI

| kpi | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_term_borrowings | 301 | 29 | 216 | 56 | 126 | 0.096 | 0.118 | 1.000 |
| long_term_debt_total | 351 | 83 | 265 | 3 | 133 | 0.236 | 0.239 | 0.999 |
| gross_profit | 446 | 136 | 290 | 20 | 47 | 0.305 | 0.319 | 0.999 |
| cash_incl_restricted | 349 | 107 | 172 | 70 | 105 | 0.307 | 0.384 | 0.999 |
| stockholders_equity_incl_nci | 258 | 80 | 149 | 29 | 145 | 0.310 | 0.349 | 0.999 |
| cost_of_revenue | 487 | 158 | 305 | 24 | 7 | 0.324 | 0.341 | 0.999 |
| rd_expense | 315 | 106 | 144 | 65 | 28 | 0.337 | 0.424 | 0.999 |
| sga_expense | 491 | 166 | 294 | 31 | 3 | 0.338 | 0.361 | 0.999 |
| inventory | 468 | 166 | 233 | 69 | 12 | 0.355 | 0.416 | 0.999 |
| long_term_debt_noncurrent | 300 | 107 | 181 | 12 | 161 | 0.357 | 0.372 | 0.999 |
| dividends_paid | 346 | 128 | 193 | 25 | 114 | 0.370 | 0.399 | 0.999 |
| depreciation_amortization | 492 | 184 | 296 | 12 | 1 | 0.374 | 0.383 | 0.978 |
| income_tax_expense | 486 | 185 | 277 | 24 | 8 | 0.381 | 0.400 | 0.999 |
| long_term_debt_current | 340 | 132 | 169 | 39 | 131 | 0.388 | 0.439 | 0.987 |
| interest_expense | 426 | 167 | 240 | 19 | 68 | 0.392 | 0.410 | 0.999 |
| operating_income | 483 | 197 | 277 | 9 | 9 | 0.408 | 0.416 | 0.999 |
| net_income | 493 | 202 | 290 | 1 | 1 | 0.410 | 0.411 | 0.999 |
| capex | 490 | 201 | 278 | 11 | 4 | 0.410 | 0.420 | 0.880 |
| accounts_receivable | 459 | 193 | 248 | 18 | 29 | 0.420 | 0.438 | 0.999 |
| accounts_payable | 460 | 196 | 240 | 24 | 34 | 0.426 | 0.450 | 0.998 |
| operating_cash_flow | 494 | 217 | 270 | 7 | 0 | 0.439 | 0.446 | 0.999 |
| revenue | 494 | 217 | 276 | 1 | 0 | 0.439 | 0.440 | 0.999 |
| total_liabilities | 494 | 217 | 261 | 16 | 0 | 0.439 | 0.454 | 0.999 |
| stockholders_equity | 486 | 215 | 267 | 4 | 8 | 0.442 | 0.446 | 0.999 |
| investing_cash_flow | 491 | 219 | 254 | 18 | 3 | 0.446 | 0.463 | 0.999 |
| financing_cash_flow | 494 | 227 | 249 | 18 | 0 | 0.460 | 0.477 | 0.999 |
| total_assets | 494 | 229 | 263 | 2 | 0 | 0.464 | 0.465 | 0.999 |
| cash_and_equivalents | 488 | 227 | 243 | 18 | 6 | 0.465 | 0.483 | 0.999 |
| shares_outstanding | 419 | 305 | 107 | 7 | 75 | 0.728 | 0.740 | 0.000 |
| eps_basic | 435 | 370 | 60 | 5 | 58 | 0.851 | 0.860 | 0.000 |
| eps_diluted | 425 | 366 | 58 | 1 | 65 | 0.861 | 0.863 | 0.000 |

## Per year

| year | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | 2289 | 913 | 1270 | 106 | 232 | 0.399 | 0.418 | 0.999 |
| 2017 | 2087 | 842 | 1145 | 100 | 211 | 0.403 | 0.424 | 0.954 |
| 2022 | 2537 | 1036 | 1356 | 145 | 245 | 0.408 | 0.433 | 0.998 |
| 2018 | 2003 | 886 | 1018 | 99 | 218 | 0.442 | 0.465 | 0.081 |
| 2019 | 2115 | 944 | 1077 | 94 | 223 | 0.446 | 0.467 | 0.717 |
| 2021 | 2424 | 1111 | 1199 | 114 | 252 | 0.458 | 0.481 | 0.093 |

## Per ground-truth source

| source | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 50 | 6 | 44 | 0 | 0 | 0.120 | 0.120 | 1.000 |
| edgar + alphavantage | 6586 | 2671 | 3504 | 411 | 0 | 0.406 | 0.433 | 0.751 |
| edgar | 6819 | 3055 | 3517 | 247 | 0 | 0.448 | 0.465 | 0.667 |
