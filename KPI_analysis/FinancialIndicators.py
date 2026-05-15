from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from fetch_kpis import tickers_from_selected

HERE = Path(__file__).resolve().parent
INDUSTRY_CACHE = HERE / "cache" / "industry_indicators"


def find_ticker_industry(ticker: str) -> str | None:
	for entry in tickers_from_selected():
		if entry["ticker"] == ticker:
			return entry["industry"]


def getIndustryInfoFromTicker(ticker: str, bench_start: date, bench_end: date) -> list[pd.DataFrame] | None:
	from fetch_filing_returns import fetch_prices
	industry = find_ticker_industry(ticker)
	tickers_industry = tickers_from_selected(industry)
	tickers_industry = [e['ticker'] for e in tickers_industry]
	dfs = [fetch_prices(t, bench_start, bench_end) for t in tickers_industry]
	return dfs

def GetIndicatorsForPrices(prices:pd.DataFrame,) -> pd.DataFrame:
	prices = prices.copy()

	prices['Volume_ATS'] = prices['Volume'] / prices['Volume'].mean()
	prices['returns'] = prices['Close'].pct_change()
	prices['Volatility'] = prices['Close'].pct_change().rolling(window=20).std()

	# Cumulative returns from t-10 to t+10 relative to each day
	for lag in range(-10, 11):
		if lag == 0:
			prices[f'return_t{lag}'] = 0.0
		elif lag > 0:
			# return from day 0 to day +lag: Close[t+lag]/Close[t] - 1
			prices[f'return_t{lag}'] = prices['Close'].shift(-lag) / prices['Close'] - 1
		else:
			# return from day lag to day 0: Close[t]/Close[t+|lag|] - 1
			prices[f'return_t{lag}'] = prices['Close'] / prices['Close'].shift(-lag) - 1

	return prices

def GetIndustryIndicatorByDate(list_prices: list[pd.DataFrame | None]) -> pd.DataFrame:
	returns = []
	volumes = []
	volatility = []
	raw_volumes = []  # for computing weights
	cum_returns = {lag: [] for lag in range(-10, 11)}  # return_t{lag} per ticker
	for prices in list_prices:
		if prices is None or prices.empty or "Close" not in prices:
			continue

		prices = GetIndicatorsForPrices(prices)
		returns.append(prices["returns"])
		volumes.append(prices["Volume_ATS"])
		volatility.append(prices["Volatility"])
		raw_volumes.append(prices["Volume"])
		for lag in range(-10, 11):
			cum_returns[lag].append(prices[f"return_t{lag}"])

	# Equal-weighted averages
	mean_returns = pd.concat(returns, axis=1).mean(axis=1, skipna=True)
	mean_returns.name = "returns"

	mean_volumes = pd.concat(volumes, axis=1).mean(axis=1, skipna=True)
	mean_volumes.name = "volumes"

	mean_volatility = pd.concat(volatility, axis=1).mean(axis=1, skipna=True)
	mean_volatility.name = "volatility"

	# cum return
	mean_cum_returns = []
	for lag in range(-10, 11):
		cr_df = pd.concat(cum_returns[lag], axis=1)
		mean_cum_returns.append(cr_df.mean(axis=1, skipna=True))
		mean_cum_returns[-1].name = f"return_t{lag}"

	# Volume-weighted averages: w_i = mean(V_i) / sum(mean(V_j))
	avg_vols = [rv.mean(skipna=True) for rv in raw_volumes]
	total_avg_vol = sum(avg_vols)
	if total_avg_vol > 0:
		weights = [av / total_avg_vol for av in avg_vols]
	else:
		weights = [1.0 / len(raw_volumes)] * len(raw_volumes)

	# Weighted returns
	ret_df = pd.concat(returns, axis=1)
	weighted_returns = sum(w * ret_df.iloc[:, i].fillna(0) for i, w in enumerate(weights))
	weighted_returns.name = "returns_vw"

	# Weighted volumes (ATS)
	vol_df = pd.concat(volumes, axis=1)
	weighted_volumes = sum(w * vol_df.iloc[:, i].fillna(0) for i, w in enumerate(weights))
	weighted_volumes.name = "volumes_vw"

	# Weighted volatility
	volat_df = pd.concat(volatility, axis=1)
	weighted_volatility = sum(w * volat_df.iloc[:, i].fillna(0) for i, w in enumerate(weights))
	weighted_volatility.name = "volatility_vw"

	# Weighted cumulative returns for each horizon t-10..t+10
	weighted_cum_returns = []
	for lag in range(-10, 11):
		cr_df = pd.concat(cum_returns[lag], axis=1)
		wcr = sum(w * cr_df.iloc[:, i].fillna(0) for i, w in enumerate(weights))
		wcr.name = f"return_t{lag}_vw"
		weighted_cum_returns.append(wcr)

	return pd.concat([mean_returns, mean_volumes, mean_volatility,
	                  weighted_returns, weighted_volumes, weighted_volatility] + weighted_cum_returns + mean_cum_returns, axis=1)


def _industry_cache_path(industry: str) -> Path:
	safe = industry.replace("/", "_").replace(" ", "_").replace(":", "_")
	return INDUSTRY_CACHE / f"{safe}.csv"


def GetIndustryDataFrame(ticker: str, bench_start: date, bench_end: date, *, refresh: bool = False) -> pd.DataFrame:
	industry = find_ticker_industry(ticker)
	if industry is None:
		return pd.DataFrame(columns=["returns", "volumes", "volatility"])

	INDUSTRY_CACHE.mkdir(parents=True, exist_ok=True)
	path = _industry_cache_path(industry)

	if path.exists() and not refresh:
		try:
			df = pd.read_csv(path, index_col=0, parse_dates=True)
			if not df.empty:
				return df
		except Exception:
			pass

	dfs = getIndustryInfoFromTicker(ticker, bench_start, bench_end)
	industry_df = GetIndustryIndicatorByDate(dfs)
	industry_df.to_csv(path)
	return industry_df


	
