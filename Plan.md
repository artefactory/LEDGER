# Notes for benchmark and task framing.

We want a dataset that utilizes the OCRed data from all of our financial reports.
We already have 6000 reports OCRed in our database.
But we need structuring from sector data + few years in order to have a comprehensive set of reports to work with.

So maybe the very first step is to gather the sector data I already have, with a BIG dictionnary to have a one-to-one correspondance.

First get all tickers for all of our stock exchanges. Then fuse it with the data I already have.

Cluster by sectors, check with LLM if there's a real competition between the companies.

Industries that offer a wide array of company choices, are highly sensitive to the same macroeconomic shocks, and where strict financial KPIs are the best tools for separating the winners from the losers:

**Subject to Physical Supply Chain & Inflation Shocks:**
* **Consumer Cyclical / Auto Parts** (Sensitive to global freight, raw materials, and factory shutdowns)
* **Basic Materials / Specialty Chemicals** (Sensitive to industrial demand, energy feedstock prices, and shipping logistics)
* **Consumer Defensive / Packaged Foods** (Sensitive to agricultural commodity prices and domestic freight costs)

**Subject to Global Commodity Prices & Capital Cycles:**
* **Energy / Oil & Gas E&P** (Pure price takers driven entirely by the global spot price of crude/natural gas)
* **Energy / Oil & Gas Equipment & Services** (Driven entirely by the capital expenditure budgets of the E&P sector)

**Subject to Interest Rates & Monetary Policy Shocks:**
* **Financial Services / Banks - Regional** (Driven by central bank rates, yield curves, and local economic health)
* **Real Estate / REIT - Mortgage** (Driven by borrowing costs and highly sensitive to interest rate volatility) 

