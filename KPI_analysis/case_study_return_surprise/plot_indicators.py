"""Plot financial indicators for a given ticker.

Usage:
    uv run python KPI_analysis/plot_indicators.py STRT
    uv run python KPI_analysis/plot_indicators.py STRT --start 2020-01-01 --end 2023-12-31
"""
import argparse
import os
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import edgar_filings as ef

import edgar
from fetch_filing_returns import fetch_prices
from FinancialIndicators import (
	GetIndicatorsForPrices,
	GetIndustryDataFrame,

)


def annual_publication_dates(
	ticker: str,
	*,
	originals_only: bool = False,
) -> pd.DataFrame:
	cik_map = edgar.load_ticker_cik_map()
	cik = edgar.ticker_to_cik(ticker, mapping=cik_map)
	if cik is None:
		return pd.DataFrame(
			columns=[
				"form",
				"report_date",
				"filing_date",
				"publication_date_et",
				"accession",
			]
		)

	filings = ef.all_annual_filings(cik)
	if originals_only:
		filings = [f for f in filings if f.form in ef.ORIGINAL_ANNUAL_FORMS]

	return pd.DataFrame(
		[
			{
				"form": f.form,
				"report_date": f.report_date,
				"filing_date": f.filing_date,
				"publication_date_et": ef.acceptance_in_et(f),
				"accession": f.accession,
			}
			for f in filings
		]
	)


def publication_points(series: pd.Series, publication_dates: pd.Series) -> tuple[pd.Index, pd.Series]:
	series = series.dropna()
	matched = series.loc[series.index.isin(publication_dates)]
	return matched.index, matched


HERE = Path(__file__).resolve().parent


def plot_ticker(ticker: str, bench_start: date, bench_end: date, *, refresh: bool = False) -> None:
	prices = fetch_prices(ticker, bench_start, bench_end)
	prices = GetIndicatorsForPrices(prices)
	industry_df = GetIndustryDataFrame(ticker, bench_start, bench_end, refresh=refresh)

	

	returns_without_bias = prices["returns"] - industry_df["returns"]
	stock_returns = prices["returns"]
	close_prices = prices["Close"]
	mean_returns = industry_df["returns"]
	volatility_without_bias = prices["Volatility"] - industry_df["volatility"]
	volume_without_bias = prices["Volume_ATS"] - industry_df["volumes"]

	# publication dates
	pub_dates = annual_publication_dates(ticker)["publication_date_et"]
	pub_dates = pd.to_datetime(pub_dates).dt.tz_localize(None).dt.normalize()

	returns_pub_x, returns_pub_y = publication_points(returns_without_bias, pub_dates)
	stock_pub_x, stock_pub_y = publication_points(stock_returns, pub_dates)
	price_pub_x, price_pub_y = publication_points(close_prices, pub_dates)
	vol_pub_x, vol_pub_y = publication_points(volume_without_bias, pub_dates)
	volatility_pub_x, volatility_pub_y = publication_points(volatility_without_bias, pub_dates)

	out_dir = HERE / "output" / "plots" / ticker
	os.makedirs(out_dir, exist_ok=True)

	# --- Unbiased Returns ---
	print("nb nan", returns_without_bias.isna().sum())
	print("nb nan base", stock_returns.isna().sum())
	print("nb nan ind", industry_df["returns"].isna().sum())
	print('start industry date', industry_df.index.min())
	print('end industry date', industry_df.index.max())
	print('start stock date', stock_returns.index.min()
	   )
	print('end stock date', stock_returns.index.max())
	
	plt.figure(figsize=(15, 6))
	plt.plot(returns_without_bias, label="Unbiased Returns", alpha=0.5)
	rolling_mean = returns_without_bias.ffill().rolling(20).mean()
	rolling_std = returns_without_bias.ffill().rolling(20).std()
	plt.plot(rolling_mean, label="Mean (20-day rolling mean)", linestyle="--", alpha=1, color="green")
	plt.plot(rolling_mean + rolling_std, label="Volatility (20-day rolling std)", linestyle="--", alpha=1, color="black")
	plt.plot(rolling_mean - rolling_std, label="Neg Volatility (20-day rolling std)", linestyle="--", alpha=1, color="black")
	plt.fill_between(
		returns_without_bias.index,
		rolling_mean - 1.96 * rolling_std / 20**0.5,
		rolling_mean + 1.96 * rolling_std / 20**0.5,
		color="orange",
		alpha=0.3,
	)
	plt.scatter(returns_pub_x, returns_pub_y, color="red", label="Publication Dates")
	plt.xlabel("Date")
	plt.ylabel("Returns")
	plt.title(f"Unbiased Returns for {ticker}")
	plt.legend()
	plt.grid()
	plt.savefig(out_dir / f"returns_without_bias_{ticker}.png")
	plt.show()

	# --- Stock Returns ---
	plt.figure(figsize=(15, 6))
	plt.plot(stock_returns, label="Stock Returns", alpha=0.5)
	plt.scatter(stock_pub_x, stock_pub_y, color="red", label="Publication Dates")
	plt.plot(mean_returns, label="Mean Industry Returns", linestyle="--", alpha=0.5)
	plt.xlabel("Date")
	plt.ylabel("Returns")
	plt.title(f"Stock Returns for {ticker}")
	plt.legend()
	plt.grid()
	plt.savefig(out_dir / f"stock_returns_{ticker}.png")
	plt.show()

	# --- Stock Prices ---
	plt.figure(figsize=(15, 6))
	plt.plot(close_prices, label="Prices", alpha=0.5)
	plt.scatter(price_pub_x, price_pub_y, color="red", label="Publication Dates")
	plt.xlabel("Date")
	plt.ylabel("Prices")
	plt.title(f"Stock Prices for {ticker}")
	plt.legend()
	plt.grid()
	plt.savefig(out_dir / f"stock_prices_{ticker}.png")
	plt.show()

	# --- Unbiased Volatility ---
	plt.figure(figsize=(15, 6))
	plt.plot(volatility_without_bias, label="Unbiased Volatility (20-day rolling std)", alpha=0.5)
	plt.scatter(volatility_pub_x, volatility_pub_y, color="red", label="Publication Dates")
	plt.xlabel("Date")
	plt.ylabel("Unbiased Volatility (20-day rolling std)")
	plt.title(f"Stock Unbiased Volatility for {ticker}")
	plt.legend()
	plt.grid()
	plt.savefig(out_dir / f"stock_volatility_{ticker}.png")
	plt.show()

	# --- Unbiased Volume ---
	plt.figure(figsize=(15, 6))
	plt.plot(volume_without_bias, label="Unbiased Volume (ATS)", alpha=0.5)
	plt.scatter(vol_pub_x, vol_pub_y, color="red", label="Publication Dates")
	plt.xlabel("Date")
	plt.ylabel("Unbiased Volume (ATS)")
	plt.title(f"Stock Unbiased Volume (ATS) for {ticker}")
	plt.legend()
	plt.grid()
	plt.savefig(out_dir / f"stock_volume_ats_{ticker}.png")
	plt.show()

	# top anomalous days
	top_10 = returns_without_bias.nlargest(10).index

	pub_df = annual_publication_dates(ticker)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Plot financial indicators for a ticker")
	parser.add_argument("ticker", help="Ticker symbol (e.g. STRT)")
	parser.add_argument("--start", default="2022-01-01", help="Bench start date (YYYY-MM-DD)")
	parser.add_argument("--end", default="2023-12-31", help="Bench end date (YYYY-MM-DD)")
	parser.add_argument("--refresh", action="store_true", help="Force re-fetch industry cache")
	args = parser.parse_args()

	plot_ticker(
		args.ticker,
		date.fromisoformat(args.start),
		date.fromisoformat(args.end),
		refresh=args.refresh,
	)
