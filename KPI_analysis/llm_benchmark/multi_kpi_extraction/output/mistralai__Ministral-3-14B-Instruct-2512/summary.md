# LLM KPI extraction benchmark — summary

- Tolerance: ±1.0%
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Test-set scope: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/needle_haystack/test_set_reports.txt` (494 reports, 13265 ground-truth KPIs)
- Predictions root: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/output/mistralai__Ministral-3-14B-Instruct-2512/raw`
- Reports loaded: 977 (ok=944, failed=33, error=0)

## Overall

| metric | value |
| --- | --- |
| n_predictions | 14178 |
| n_ground_truth | 13265 |
| matched | 5696 |
| wrong | 6929 |
| missing | 640 |
| extra | 1553 |
| recall (matched/gt) | 0.4294 |
| precision (matched/(matched+wrong)) | 0.4512 |

## Per KPI

| kpi | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_term_borrowings | 261 | 28 | 187 | 46 | 156 | 0.107 | 0.130 | 1.000 |
| long_term_debt_total | 331 | 82 | 246 | 3 | 153 | 0.248 | 0.250 | 0.999 |
| cash_incl_restricted | 349 | 107 | 172 | 70 | 105 | 0.307 | 0.384 | 0.999 |
| stockholders_equity_incl_nci | 258 | 80 | 149 | 29 | 145 | 0.310 | 0.349 | 0.999 |
| gross_profit | 434 | 135 | 279 | 20 | 59 | 0.311 | 0.326 | 0.998 |
| cost_of_revenue | 477 | 156 | 297 | 24 | 17 | 0.327 | 0.344 | 0.999 |
| sga_expense | 489 | 166 | 292 | 31 | 5 | 0.339 | 0.362 | 0.999 |
| rd_expense | 308 | 106 | 142 | 60 | 30 | 0.344 | 0.427 | 0.999 |
| inventory | 467 | 166 | 232 | 69 | 13 | 0.355 | 0.417 | 0.999 |
| long_term_debt_noncurrent | 300 | 107 | 181 | 12 | 161 | 0.357 | 0.372 | 0.999 |
| depreciation_amortization | 492 | 184 | 296 | 12 | 1 | 0.374 | 0.383 | 0.978 |
| income_tax_expense | 484 | 185 | 275 | 24 | 10 | 0.382 | 0.402 | 0.999 |
| dividends_paid | 327 | 125 | 180 | 22 | 130 | 0.382 | 0.410 | 0.999 |
| interest_expense | 410 | 157 | 234 | 19 | 84 | 0.383 | 0.402 | 0.999 |
| long_term_debt_current | 322 | 129 | 154 | 39 | 149 | 0.401 | 0.456 | 0.894 |
| net_income | 487 | 199 | 287 | 1 | 7 | 0.409 | 0.409 | 0.999 |
| capex | 490 | 201 | 278 | 11 | 4 | 0.410 | 0.420 | 0.880 |
| operating_income | 480 | 197 | 274 | 9 | 12 | 0.410 | 0.418 | 0.999 |
| accounts_receivable | 451 | 191 | 242 | 18 | 37 | 0.424 | 0.441 | 0.908 |
| accounts_payable | 459 | 196 | 239 | 24 | 35 | 0.427 | 0.451 | 0.997 |
| operating_cash_flow | 494 | 217 | 270 | 7 | 0 | 0.439 | 0.446 | 0.999 |
| revenue | 494 | 217 | 276 | 1 | 0 | 0.439 | 0.440 | 0.999 |
| total_liabilities | 494 | 217 | 261 | 16 | 0 | 0.439 | 0.454 | 0.999 |
| stockholders_equity | 485 | 215 | 266 | 4 | 9 | 0.443 | 0.447 | 0.999 |
| investing_cash_flow | 491 | 219 | 254 | 18 | 3 | 0.446 | 0.463 | 0.999 |
| financing_cash_flow | 494 | 227 | 249 | 18 | 0 | 0.460 | 0.477 | 0.999 |
| total_assets | 494 | 229 | 263 | 2 | 0 | 0.464 | 0.465 | 0.999 |
| cash_and_equivalents | 486 | 227 | 241 | 18 | 8 | 0.467 | 0.485 | 0.999 |
| shares_outstanding | 397 | 295 | 95 | 7 | 97 | 0.743 | 0.756 | 0.000 |
| eps_basic | 435 | 370 | 60 | 5 | 58 | 0.851 | 0.860 | 0.000 |
| eps_diluted | 425 | 366 | 58 | 1 | 65 | 0.861 | 0.863 | 0.000 |

## Per year

| year | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | 2257 | 907 | 1245 | 105 | 263 | 0.402 | 0.421 | 0.999 |
| 2017 | 2054 | 833 | 1126 | 95 | 239 | 0.406 | 0.425 | 0.954 |
| 2022 | 2492 | 1031 | 1319 | 142 | 287 | 0.414 | 0.439 | 0.998 |
| 2018 | 1984 | 879 | 1009 | 96 | 234 | 0.443 | 0.466 | 0.083 |
| 2019 | 2091 | 943 | 1056 | 92 | 245 | 0.451 | 0.472 | 0.495 |
| 2021 | 2387 | 1103 | 1174 | 110 | 285 | 0.462 | 0.484 | 0.062 |

## Per ground-truth source

| source | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 50 | 6 | 44 | 0 | 0 | 0.120 | 0.120 | 1.000 |
| edgar + alphavantage | 5264 | 2130 | 2759 | 375 | 0 | 0.405 | 0.436 | 0.637 |
| edgar | 7951 | 3560 | 4126 | 265 | 0 | 0.448 | 0.463 | 0.585 |
