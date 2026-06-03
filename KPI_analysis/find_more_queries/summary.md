# Extended KPI Coverage Summary

Generated from `scan_and_fetch.py` output.  Source OCR tree: `/data/workspace/charles/pdf_ocr_deepseek/DeepSeekOCR_Ardian_raw_3kdocs`

## Overall

| Metric | Value |
|--------|-------|
| Companies in OCR corpus | 292 |
| Successfully fetched (no error) | 282 |
| Companies with at least one KPI | 268 |
| Total potential (ticker, year, KPI) triples | 33,982 |
| Overall KPI×year fill rate (all 31 KPIs) | 55.2% (33,982/61,535) |

## By Industry

| Industry | Companies | No Error | Potential Queries | Core Coverage % |
|----------|-----------|----------|-------------------|-----------------|
| basic-materials-specialty-chemicals | 57 | 54 | 7,671 | 69.4% |
| consumer-cyclical-auto-parts | 46 | 44 | 6,046 | 76.6% |
| consumer-defensive-packaged-foods | 47 | 44 | 6,791 | 77.0% |
| energy-oil-gas-e-p | 65 | 63 | 5,946 | 50.3% |
| energy-oil-gas-equipment-services | 42 | 42 | 4,611 | 68.3% |
| real-estate-reit-mortgage | 35 | 35 | 2,917 | 54.6% |

## By Exchange (OCR label)

| Exchange | Companies | No Error | Potential Queries |
|----------|-----------|----------|-------------------|
| AMEX | 5 | 5 | 781 |
| LSE | 25 | 20 | 171 |
| NASDAQ | 84 | 83 | 10,690 |
| NYSE | 178 | 174 | 22,340 |

## Per-KPI Coverage (core set, years present in OCR)

| KPI | Companies with Data | Total (ticker,year) Pairs | Fill Rate |
|-----|---------------------|---------------------------|-----------|
| revenue | 233/292 | 1,319/1,985 | 66.4% |
| gross_profit | 135/292 | 741/1,985 | 37.3% |
| operating_income | 214/292 | 1,191/1,985 | 60.0% |
| net_income | 255/292 | 1,462/1,985 | 73.7% |
| total_assets | 257/292 | 1,461/1,985 | 73.6% |
| total_liabilities | 257/292 | 1,456/1,985 | 73.4% |
| cash_and_equivalents | 249/292 | 1,494/1,985 | 75.3% |
| operating_cash_flow | 253/292 | 1,482/1,985 | 74.7% |
| capex | 200/292 | 1,124/1,985 | 56.6% |

## Per-KPI Coverage (all 31 KPIs, years present in OCR)

| KPI | Companies with Data | Fill Rate |
|-----|---------------------|-----------|
| revenue | 233/292 | 66.4% |
| cost_of_revenue | 169/292 | 48.1% |
| gross_profit | 135/292 | 37.3% |
| rd_expense | 114/292 | 32.4% |
| sga_expense | 246/292 | 69.8% |
| operating_income | 214/292 | 60.0% |
| interest_expense | 219/292 | 60.9% |
| income_tax_expense | 244/292 | 70.7% |
| net_income | 255/292 | 73.7% |
| eps_basic | 249/292 | 68.4% |
| eps_diluted | 247/292 | 67.9% |
| total_assets | 257/292 | 73.6% |
| total_liabilities | 257/292 | 73.4% |
| stockholders_equity | 248/292 | 72.0% |
| stockholders_equity_incl_nci | 153/292 | 41.4% |
| cash_and_equivalents | 249/292 | 75.3% |
| cash_incl_restricted | 198/292 | 28.1% |
| long_term_debt_total | 170/292 | 34.5% |
| long_term_debt_noncurrent | 157/292 | 36.9% |
| long_term_debt_current | 138/292 | 29.1% |
| short_term_borrowings | 54/292 | 11.9% |
| inventory | 184/292 | 50.6% |
| accounts_receivable | 168/292 | 47.2% |
| accounts_payable | 194/292 | 52.8% |
| shares_outstanding | 207/292 | 55.0% |
| operating_cash_flow | 253/292 | 74.7% |
| investing_cash_flow | 253/292 | 66.6% |
| financing_cash_flow | 252/292 | 66.3% |
| capex | 200/292 | 56.6% |
| depreciation_amortization | 235/292 | 67.9% |
| dividends_paid | 167/292 | 42.6% |

## Metadata Matching (OCR vs Cleaned CSVs)

| Status | Count |
|--------|-------|
| Matched in cleaned/ CSVs | 245 |
| Not found in cleaned/ CSVs | 47 |

### Companies not in cleaned/ CSVs

These companies exist in the OCR corpus but were not found in `tickers_lists/cleaned/` (likely filtered out during cleaning or added later).  KPIs were still fetched directly using the OCR ticker.

| Ticker (OCR) | Exchange | Industry Dir |
|-------------|---------|--------------|
| NGS | AMEX | energy-oil-gas-equipment-services |
| TWO | AMEX | real-estate-reit-mortgage |
| BAKK.L | LSE | consumer-defensive-packaged-foods |
| CRC | LSE | energy-oil-gas-e-p |
| CYAN | LSE | consumer-defensive-packaged-foods |
| DX | LSE | real-estate-reit-mortgage |
| EOG | LSE | energy-oil-gas-e-p |
| FLO | LSE | consumer-defensive-packaged-foods |
| FUL | LSE | basic-materials-specialty-chemicals |
| LOOP | LSE | basic-materials-specialty-chemicals |
| MTR | LSE | energy-oil-gas-e-p |
| NOG | LSE | energy-oil-gas-e-p |
| OPTI | LSE | consumer-cyclical-auto-parts |
| PPG | LSE | basic-materials-specialty-chemicals |
| PRM | LSE | basic-materials-specialty-chemicals |
| SMP | LSE | consumer-cyclical-auto-parts |
| SSTY | LSE | basic-materials-specialty-chemicals |
| THS | LSE | consumer-defensive-packaged-foods |
| VAL | LSE | energy-oil-gas-equipment-services |
| WBI | LSE | energy-oil-gas-equipment-services |
| CRKN | NASDAQ | basic-materials-specialty-chemicals |
| CYAN | NASDAQ | consumer-defensive-packaged-foods |
| GPOR | NASDAQ | energy-oil-gas-e-p |
| RIBT | NASDAQ | consumer-defensive-packaged-foods |
| TORM | NASDAQ | basic-materials-specialty-chemicals |
| USNA | NASDAQ | consumer-defensive-packaged-foods |
| APA | NYSE | energy-oil-gas-e-p |
| AXL | NYSE | consumer-cyclical-auto-parts |
| BATL | NYSE | energy-oil-gas-e-p |
| BKR | NYSE | energy-oil-gas-equipment-services |
| CIVI | NYSE | energy-oil-gas-e-p |
| CMT | NYSE | basic-materials-specialty-chemicals |
| CPB | NYSE | consumer-defensive-packaged-foods |
| ENSV | NYSE | energy-oil-gas-equipment-services |
| GT | NYSE | consumer-cyclical-auto-parts |
| GTX | NYSE | consumer-cyclical-auto-parts |
| HYLN | NYSE | consumer-cyclical-auto-parts |
| LIN | NYSE | basic-materials-specialty-chemicals |
| LSF | NYSE | consumer-defensive-packaged-foods |
| NINE | NYSE | energy-oil-gas-equipment-services |
| PED | NYSE | energy-oil-gas-e-p |
| QS | NYSE | consumer-cyclical-auto-parts |
| REI | NYSE | energy-oil-gas-e-p |
| REPX | NYSE | energy-oil-gas-e-p |
| SACH | NYSE | real-estate-reit-mortgage |
| THS | NYSE | consumer-defensive-packaged-foods |
| VC | NYSE | consumer-cyclical-auto-parts |

## Year Range in OCR Corpus

| Year | Reports in OCR |
|------|----------------|
| 2009 | 134 |
| 2010 | 130 |
| 2011 | 148 |
| 2012 | 150 |
| 2013 | 165 |
| 2014 | 176 |
| 2015 | 191 |
| 2016 | 198 |
| 2017 | 53 |
| 2018 | 64 |
| 2019 | 73 |
| 2020 | 86 |
| 2021 | 99 |
| 2022 | 104 |
| 2023 | 213 |
| 2024 | 1 |
