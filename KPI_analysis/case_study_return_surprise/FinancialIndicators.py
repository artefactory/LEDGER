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
	print(f"  [Industry] Fetching prices for {len(tickers_industry)} tickers in '{industry}'...")
	dfs = []
	for j, t in enumerate(tickers_industry):
		print(f"    [{j+1}/{len(tickers_industry)}] {t}", end=" ", flush=True)
		df = fetch_prices(t, bench_start, bench_end)
		print("ok" if df is not None and not df.empty else "empty")
		dfs.append(df)
	return dfs

def GetIndicatorsForPrices(prices:pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
	prices = prices.copy()

	# Drop any pre-existing computed columns to avoid duplicates on re-read from cache
	existing_return_cols = [c for c in prices.columns if c.startswith("return_t")]
	if existing_return_cols:
		prices = prices.drop(columns=existing_return_cols)
	for col in ("Volume_ATS", "returns", "Volatility"):
		if col in prices.columns:
			prices = prices.drop(columns=[col])

	prices['Volume_ATS'] = prices['Volume'] / prices['Volume'].mean()
	prices['returns'] = prices['Close'].pct_change()
	prices['Volatility'] = prices['Close'].pct_change().rolling(window=20).std()

	# Cumulative returns from t-max_lag to t+max_lag relative to each day
	# return_t{d} = Close[t+d] / Close[t] - 1  (anchored at day 0 = 0)
	new_cols = {}
	for lag in range(-max_lag, max_lag + 1):
		col_name = f'return_t{lag}'
		if lag == 0:
			new_cols[col_name] = pd.Series(0.0, index=prices.index)
		else:
			new_cols[col_name] = prices['Close'].shift(-lag) / prices['Close'] - 1
	prices = pd.concat([prices, pd.DataFrame(new_cols, index=prices.index)], axis=1)

	return prices

def GetIndustryIndicatorByDate(list_prices: list[pd.DataFrame | None], max_lag: int = 10) -> pd.DataFrame:
	returns = []
	volumes = []
	volatility = []
	raw_volumes = []  # for computing weights
	cum_returns = {lag: [] for lag in range(-max_lag, max_lag + 1)}  # return_t{lag} per ticker
	n_valid = 0
	for prices in list_prices:
		if prices is None or prices.empty or "Close" not in prices:
			continue
		n_valid += 1
		print(f"  [Industry indicators] Processing ticker {n_valid}...", flush=True)
		prices = GetIndicatorsForPrices(prices, max_lag=max_lag)
		returns.append(prices["returns"])
		volumes.append(prices["Volume_ATS"])
		volatility.append(prices["Volatility"])
		raw_volumes.append(prices["Volume"])
		for lag in range(-max_lag, max_lag + 1):
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
	for lag in range(-max_lag, max_lag + 1):
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

	# Weighted cumulative returns for each horizon
	weighted_cum_returns = []
	for lag in range(-max_lag, max_lag + 1):
		cr_df = pd.concat(cum_returns[lag], axis=1)
		wcr = sum(w * cr_df.iloc[:, i].fillna(0) for i, w in enumerate(weights))
		wcr.name = f"return_t{lag}_vw"
		weighted_cum_returns.append(wcr)

	return pd.concat([mean_returns, mean_volumes, mean_volatility,
	                  weighted_returns, weighted_volumes, weighted_volatility] + weighted_cum_returns + mean_cum_returns, axis=1)


def _industry_cache_path(industry: str) -> Path:
	safe = industry.replace("/", "_").replace(" ", "_").replace(":", "_")
	return INDUSTRY_CACHE / f"{safe}.csv"


def GetIndustryDataFrame(ticker: str, bench_start: date, bench_end: date, *, refresh: bool = False, max_lag: int = 10) -> pd.DataFrame:
	industry = find_ticker_industry(ticker)
	if industry is None:
		return pd.DataFrame(columns=["returns", "volumes", "volatility"])

	INDUSTRY_CACHE.mkdir(parents=True, exist_ok=True)
	path = _industry_cache_path(industry)

	if path.exists() and not refresh:
		try:
			df = pd.read_csv(path, index_col=0, parse_dates=True)
			if not df.empty:
				# Check if cached df has the required lag columns
				if f"return_t{max_lag}" in df.columns and f"return_t{-max_lag}" in df.columns:
					print(f"  [Industry] Using cache for '{industry}' (max_lag={max_lag})")
					return df
				else:
					print(f"  [Industry] Cache outdated for '{industry}' (missing lag {max_lag}), recomputing...")
		except Exception:
			pass
	else:
		print(f"  [Industry] No cache for '{industry}', computing (max_lag={max_lag})...")

	dfs = getIndustryInfoFromTicker(ticker, bench_start, bench_end)
	industry_df = GetIndustryIndicatorByDate(dfs, max_lag=max_lag)
	industry_df.to_csv(path)
	print(f"  [Industry] Saved cache for '{industry}'")
	return industry_df


	
