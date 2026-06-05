# LLM KPI extraction benchmark — summary

- Tolerance: ±0.1%
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Test-set scope: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/test_set_reports.txt` (494 reports, 13455 ground-truth KPIs)
- Predictions root: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/multi_kpi_extraction/output/mistralai__Ministral-3-14B-Instruct-2512/raw`
- Reports loaded: 977 (ok=944, failed=33, error=0)

## Overall

| metric | value |
| --- | --- |
| n_predictions | 14178 |
| n_ground_truth | 13455 |
| matched | 5576 |
| wrong | 7221 |
| missing | 658 |
| extra | 1381 |
| recall (matched/gt) | 0.4144 |
| precision (matched/(matched+wrong)) | 0.4357 |

## Per KPI

| kpi | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_term_borrowings | 301 | 28 | 217 | 56 | 126 | 0.093 | 0.114 | 1.000 |
| long_term_debt_total | 351 | 70 | 278 | 3 | 133 | 0.199 | 0.201 | 0.999 |
| stockholders_equity_incl_nci | 258 | 74 | 155 | 29 | 145 | 0.287 | 0.323 | 0.999 |
| gross_profit | 446 | 132 | 294 | 20 | 47 | 0.296 | 0.310 | 0.999 |
| cash_incl_restricted | 349 | 107 | 172 | 70 | 105 | 0.307 | 0.384 | 0.999 |
| cost_of_revenue | 487 | 150 | 313 | 24 | 7 | 0.308 | 0.324 | 0.999 |
| sga_expense | 491 | 157 | 303 | 31 | 3 | 0.320 | 0.341 | 0.999 |
| rd_expense | 315 | 104 | 146 | 65 | 28 | 0.330 | 0.416 | 0.999 |
| long_term_debt_noncurrent | 300 | 101 | 187 | 12 | 161 | 0.337 | 0.351 | 0.999 |
| inventory | 468 | 166 | 233 | 69 | 12 | 0.355 | 0.416 | 0.999 |
| depreciation_amortization | 492 | 179 | 301 | 12 | 1 | 0.364 | 0.373 | 0.978 |
| dividends_paid | 346 | 128 | 193 | 25 | 114 | 0.370 | 0.399 | 0.999 |
| income_tax_expense | 486 | 182 | 280 | 24 | 8 | 0.374 | 0.394 | 0.999 |
| long_term_debt_current | 340 | 131 | 170 | 39 | 131 | 0.385 | 0.435 | 0.987 |
| interest_expense | 426 | 165 | 242 | 19 | 68 | 0.387 | 0.405 | 0.999 |
| operating_income | 483 | 189 | 285 | 9 | 9 | 0.391 | 0.399 | 0.999 |
| capex | 490 | 194 | 285 | 11 | 4 | 0.396 | 0.405 | 0.880 |
| net_income | 493 | 197 | 295 | 1 | 1 | 0.400 | 0.400 | 0.999 |
| accounts_receivable | 459 | 189 | 252 | 18 | 29 | 0.412 | 0.429 | 0.999 |
| total_liabilities | 494 | 207 | 271 | 16 | 0 | 0.419 | 0.433 | 0.999 |
| stockholders_equity | 486 | 204 | 278 | 4 | 8 | 0.420 | 0.423 | 0.999 |
| accounts_payable | 460 | 195 | 241 | 24 | 34 | 0.424 | 0.447 | 0.998 |
| revenue | 494 | 210 | 283 | 1 | 0 | 0.425 | 0.426 | 0.999 |
| operating_cash_flow | 494 | 216 | 271 | 7 | 0 | 0.437 | 0.444 | 0.999 |
| investing_cash_flow | 491 | 217 | 256 | 18 | 3 | 0.442 | 0.459 | 0.999 |
| financing_cash_flow | 494 | 225 | 251 | 18 | 0 | 0.455 | 0.473 | 0.999 |
| total_assets | 494 | 226 | 266 | 2 | 0 | 0.457 | 0.459 | 0.999 |
| cash_and_equivalents | 488 | 227 | 243 | 18 | 6 | 0.465 | 0.483 | 0.999 |
| shares_outstanding | 419 | 277 | 135 | 7 | 75 | 0.661 | 0.672 | 0.000 |
| eps_basic | 435 | 366 | 64 | 5 | 58 | 0.841 | 0.851 | 0.000 |
| eps_diluted | 425 | 363 | 61 | 1 | 65 | 0.854 | 0.856 | 0.000 |

## Per year

| year | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2017 | 2087 | 808 | 1179 | 100 | 211 | 0.387 | 0.407 | 0.954 |
| 2020 | 2289 | 893 | 1290 | 106 | 232 | 0.390 | 0.409 | 0.999 |
| 2022 | 2537 | 1016 | 1376 | 145 | 245 | 0.400 | 0.425 | 0.998 |
| 2018 | 2003 | 856 | 1048 | 99 | 218 | 0.427 | 0.450 | 0.081 |
| 2019 | 2115 | 916 | 1105 | 94 | 223 | 0.433 | 0.453 | 0.717 |
| 2021 | 2424 | 1087 | 1223 | 114 | 252 | 0.448 | 0.471 | 0.093 |

## Per ground-truth source

| source | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 50 | 6 | 44 | 0 | 0 | 0.120 | 0.120 | 1.000 |
| edgar + alphavantage | 6586 | 2581 | 3594 | 411 | 0 | 0.392 | 0.418 | 0.751 |
| edgar | 6819 | 2989 | 3583 | 247 | 0 | 0.438 | 0.455 | 0.667 |
