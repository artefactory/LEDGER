# LLM KPI extraction benchmark — summary

- Tolerance: ±1.0%
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Predictions root: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/output/raw`
- Reports loaded: 10 (ok=10, failed=0, error=0)

## Overall

| metric | value |
| --- | --- |
| n_predictions | 399 |
| n_ground_truth | 234 |
| matched | 178 |
| wrong | 44 |
| missing | 12 |
| extra | 177 |
| recall (matched/gt) | 0.7607 |
| precision (matched/(matched+wrong)) | 0.8018 |

## Per KPI

| kpi | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_term_debt_total | 0 | 0 | 0 | 0 | 10 | — | — | — |
| cash_incl_restricted | 8 | 1 | 0 | 7 | 1 | 0.125 | 1.000 | 0.000 |
| rd_expense | 5 | 1 | 1 | 3 | 3 | 0.200 | 0.500 | 0.306 |
| interest_expense | 4 | 1 | 3 | 0 | 12 | 0.250 | 0.250 | 0.023 |
| depreciation_amortization | 9 | 5 | 4 | 0 | 7 | 0.556 | 0.556 | 0.000 |
| sga_expense | 9 | 5 | 3 | 1 | 5 | 0.556 | 0.625 | 0.000 |
| operating_income | 6 | 4 | 2 | 0 | 10 | 0.667 | 0.667 | 0.000 |
| accounts_receivable | 6 | 4 | 2 | 0 | 7 | 0.667 | 0.667 | 0.000 |
| capex | 9 | 7 | 2 | 0 | 7 | 0.778 | 0.778 | 0.000 |
| cost_of_revenue | 9 | 7 | 2 | 0 | 7 | 0.778 | 0.778 | 0.000 |
| dividends_paid | 9 | 7 | 2 | 0 | 5 | 0.778 | 0.778 | 0.000 |
| gross_profit | 9 | 7 | 2 | 0 | 7 | 0.778 | 0.778 | 0.000 |
| income_tax_expense | 9 | 7 | 2 | 0 | 7 | 0.778 | 0.778 | 0.000 |
| net_income | 9 | 7 | 2 | 0 | 7 | 0.778 | 0.778 | 0.000 |
| revenue | 9 | 7 | 2 | 0 | 7 | 0.778 | 0.778 | 0.000 |
| accounts_payable | 9 | 7 | 2 | 0 | 4 | 0.778 | 0.778 | 0.000 |
| shares_outstanding | 9 | 7 | 2 | 0 | 2 | 0.778 | 0.778 | 0.000 |
| long_term_debt_current | 5 | 4 | 0 | 1 | 6 | 0.800 | 1.000 | 0.000 |
| long_term_debt_noncurrent | 5 | 4 | 1 | 0 | 5 | 0.800 | 0.800 | 0.000 |
| eps_basic | 9 | 8 | 1 | 0 | 7 | 0.889 | 0.889 | 0.000 |
| eps_diluted | 9 | 8 | 1 | 0 | 7 | 0.889 | 0.889 | 0.000 |
| financing_cash_flow | 9 | 8 | 1 | 0 | 7 | 0.889 | 0.889 | 0.000 |
| investing_cash_flow | 9 | 8 | 1 | 0 | 7 | 0.889 | 0.889 | 0.000 |
| operating_cash_flow | 9 | 8 | 1 | 0 | 7 | 0.889 | 0.889 | 0.000 |
| cash_and_equivalents | 9 | 8 | 1 | 0 | 4 | 0.889 | 0.889 | 0.000 |
| inventory | 9 | 8 | 1 | 0 | 4 | 0.889 | 0.889 | 0.000 |
| stockholders_equity | 9 | 8 | 1 | 0 | 4 | 0.889 | 0.889 | 0.000 |
| total_assets | 9 | 8 | 1 | 0 | 4 | 0.889 | 0.889 | 0.000 |
| total_liabilities | 9 | 8 | 1 | 0 | 4 | 0.889 | 0.889 | 0.000 |
| short_term_borrowings | 3 | 3 | 0 | 0 | 2 | 1.000 | 1.000 | 0.000 |
| stockholders_equity_incl_nci | 3 | 3 | 0 | 0 | 1 | 1.000 | 1.000 | 0.000 |

## Per year

| year | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2015 | 0 | 0 | 0 | 0 | 32 | — | — | — |
| 2016 | 0 | 0 | 0 | 0 | 56 | — | — | — |
| 2020 | 0 | 0 | 0 | 0 | 40 | — | — | — |
| 2022 | 51 | 12 | 36 | 3 | 2 | 0.235 | 0.250 | 0.103 |
| 2021 | 27 | 24 | 1 | 2 | 26 | 0.889 | 0.960 | 0.000 |
| 2018 | 52 | 47 | 2 | 3 | 7 | 0.904 | 0.959 | 0.000 |
| 2019 | 52 | 47 | 2 | 3 | 7 | 0.904 | 0.959 | 0.000 |
| 2017 | 52 | 48 | 3 | 1 | 7 | 0.923 | 0.941 | 0.000 |

## Per ground-truth source

| source | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 27 | 11 | 13 | 3 | 0 | 0.407 | 0.458 | 0.011 |
| edgar | 207 | 167 | 31 | 9 | 0 | 0.807 | 0.843 | 0.000 |
