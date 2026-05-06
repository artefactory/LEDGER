# LLM KPI extraction benchmark — summary

- Tolerance: ±1.0%
- Ground truth: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/output/kpis_long.csv`
- Predictions root: `/home/cmoslonka/ardian_dataset_bench/KPI_analysis/llm_benchmark/output/raw`
- Reports loaded: 152 (ok=152, failed=0, error=0)

## Overall

| metric | value |
| --- | --- |
| n_predictions | 5349 |
| n_ground_truth | 3858 |
| matched | 3037 |
| wrong | 671 |
| missing | 150 |
| extra | 1641 |
| recall (matched/gt) | 0.7872 |
| precision (matched/(matched+wrong)) | 0.8190 |

## Per KPI

| kpi | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cash_incl_restricted | 109 | 21 | 6 | 82 | 18 | 0.193 | 0.778 | 0.000 |
| short_term_borrowings | 43 | 22 | 13 | 8 | 28 | 0.512 | 0.629 | 0.000 |
| stockholders_equity_incl_nci | 92 | 54 | 6 | 32 | 23 | 0.587 | 0.900 | 0.000 |
| depreciation_amortization | 151 | 96 | 55 | 0 | 49 | 0.636 | 0.636 | 0.000 |
| sga_expense | 136 | 90 | 40 | 6 | 58 | 0.662 | 0.692 | 0.000 |
| interest_expense | 110 | 77 | 31 | 2 | 89 | 0.700 | 0.713 | 0.000 |
| long_term_debt_total | 48 | 34 | 13 | 1 | 124 | 0.708 | 0.723 | 0.000 |
| rd_expense | 95 | 70 | 15 | 10 | 43 | 0.737 | 0.824 | 0.000 |
| operating_income | 140 | 108 | 32 | 0 | 60 | 0.771 | 0.771 | 0.000 |
| gross_profit | 132 | 102 | 30 | 0 | 69 | 0.773 | 0.773 | 0.000 |
| accounts_payable | 150 | 117 | 33 | 0 | 28 | 0.780 | 0.780 | 0.000 |
| accounts_receivable | 119 | 93 | 26 | 0 | 59 | 0.782 | 0.782 | 0.000 |
| long_term_debt_current | 60 | 47 | 11 | 2 | 90 | 0.783 | 0.810 | 0.000 |
| capex | 151 | 119 | 32 | 0 | 50 | 0.788 | 0.788 | 0.000 |
| income_tax_expense | 151 | 122 | 27 | 2 | 50 | 0.808 | 0.819 | 0.000 |
| long_term_debt_noncurrent | 81 | 66 | 15 | 0 | 90 | 0.815 | 0.815 | 0.000 |
| net_income | 150 | 125 | 25 | 0 | 51 | 0.833 | 0.833 | 0.000 |
| total_liabilities | 151 | 126 | 24 | 1 | 27 | 0.834 | 0.840 | 0.000 |
| inventory | 139 | 116 | 23 | 0 | 39 | 0.835 | 0.835 | 0.000 |
| revenue | 151 | 128 | 23 | 0 | 50 | 0.848 | 0.848 | 0.000 |
| cash_and_equivalents | 128 | 109 | 19 | 0 | 50 | 0.852 | 0.852 | 0.000 |
| dividends_paid | 81 | 69 | 10 | 2 | 51 | 0.852 | 0.873 | 0.000 |
| eps_basic | 143 | 122 | 21 | 0 | 58 | 0.853 | 0.853 | 0.000 |
| total_assets | 151 | 129 | 22 | 0 | 27 | 0.854 | 0.854 | 0.000 |
| eps_diluted | 143 | 123 | 20 | 0 | 58 | 0.860 | 0.860 | 0.000 |
| cost_of_revenue | 145 | 125 | 18 | 2 | 56 | 0.862 | 0.874 | 0.000 |
| investing_cash_flow | 151 | 131 | 20 | 0 | 50 | 0.868 | 0.868 | 0.000 |
| operating_cash_flow | 151 | 131 | 20 | 0 | 50 | 0.868 | 0.868 | 0.000 |
| financing_cash_flow | 151 | 132 | 19 | 0 | 50 | 0.874 | 0.874 | 0.000 |
| stockholders_equity | 140 | 123 | 17 | 0 | 38 | 0.879 | 0.879 | 0.000 |
| shares_outstanding | 115 | 110 | 5 | 0 | 58 | 0.957 | 0.957 | 0.000 |

## Per year

| year | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2015 | 0 | 0 | 0 | 0 | 367 | — | — | — |
| 2016 | 0 | 0 | 0 | 0 | 695 | — | — | — |
| 2022 | 660 | 493 | 137 | 30 | 85 | 0.747 | 0.783 | 0.000 |
| 2021 | 635 | 488 | 120 | 27 | 114 | 0.769 | 0.803 | 0.000 |
| 2017 | 641 | 501 | 122 | 18 | 86 | 0.782 | 0.804 | 0.000 |
| 2018 | 642 | 508 | 108 | 26 | 93 | 0.791 | 0.825 | 0.000 |
| 2020 | 635 | 518 | 93 | 24 | 115 | 0.816 | 0.848 | 0.000 |
| 2019 | 645 | 529 | 91 | 25 | 86 | 0.820 | 0.853 | 0.000 |

## Per ground-truth source

| source | n_gt | matched | wrong | missing | extra | recall | precision | median_abs_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | 27 | 11 | 13 | 3 | 0 | 0.407 | 0.458 | 0.011 |
| edgar + alphavantage | 155 | 122 | 30 | 3 | 0 | 0.787 | 0.803 | 0.000 |
| edgar | 3676 | 2904 | 628 | 144 | 0 | 0.790 | 0.822 | 0.000 |
