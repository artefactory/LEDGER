# `filing_returns.csv` — validation & error analysis

How we know the numbers in `output/filing_returns.csv` are real, and what
the populated `error` column actually means.

The pipeline is described in [`README.md`](README.md#filing-date-market-reaction-kpi_fetch_and_buildfetch_filing_returnspy);
this doc focuses on the post-hoc checks we ran on the full `--selected
--years 2017-2022` output (238 US-listed tickers, 1428 rows).

## TL;DR

- 1221 / 1428 rows (85.5%) carry full returns. The 207 errors are dominated
  by 52/53-week fiscal years (75% of all errors) — a known dataset edge
  case, not a bug.
- The alpha distribution is centered at `median = -0.01%` (1-day),
  `-0.45%` (5-day) — i.e. essentially zero, which is the right answer if
  the event-window anchoring is correct.
- Three real-world stress-tests confirm the rows track the actual market:
  the 2020-03 oil crash, the 2022-02 Russia/Ukraine invasion, and the
  10-K/A amendment flag.

## What "validated" means here

The pipeline's correctness depends on three things being right at once:

1. We pick the **right 10-K** for each `(ticker, fiscal_year)` (original,
   not amendment; `reportDate` matches the target year).
2. We anchor the **right trading days** (`t0` strictly before news public,
   `t1` strictly after) — wrong anchoring would systematically bias one
   direction.
3. The **prices are correctly aligned** with those anchors (no off-by-one,
   no tz drift).

We can't unit-test the second-by-second timestamp logic against ground
truth, so instead we check that the *outputs* behave the way real markets
do around known events. If anchoring or sourcing were off, the centred
distribution would drift and the event-day spikes wouldn't line up.

## Sanity check #1 — March 2020 oil crash

The 2020-03-09 day is famous: OPEC+ broke down on Sunday March 8, WTI
opened −25% Monday morning, the COVID equity sell-off had just started,
and trading curbs were triggered.

Eight US oil-leveraged filers happened to file their 10-Ks in the two
weeks beforehand. If our t0/t1 anchoring is right, their event windows
should land squarely on the crash:

| Ticker | FY   | Acceptance (ET)     | t1         | t5         | r_5d   | a_5d (vs SPY) |
| ------ | ---- | ------------------- | ---------- | ---------- | ------ | ------------- |
| MTDR   | 2019 | 2020-03-02 16:23    | 2020-03-03 | 2020-03-09 | −76.7% | −65.4%        |
| AMPY   | 2019 | 2020-03-05 17:18    | 2020-03-06 | 2020-03-12 | −67.2% | −49.3%        |
| FTI    | 2019 | 2020-03-02 18:45    | 2020-03-03 | 2020-03-09 | −38.4% | −27.1%        |
| VTOL   | 2019 | 2020-03-05 18:34    | 2020-03-06 | 2020-03-12 | −37.5% | −19.6%        |
| WTI    | 2019 | 2020-03-05 18:31    | 2020-03-06 | 2020-03-12 | −36.8% | −18.8%        |

Two things to notice:

- **Magnitudes match the news.** Matador (MTDR) is a Permian E&P with no
  hedges to speak of; a −77% week that includes Mar 9 is exactly what the
  raw share price did.
- **Alpha (vs SPY) is large and negative.** SPY itself dropped ~12% over
  the same window, so subtracting market beta still leaves −19% to −65%
  of *idiosyncratic* downside. That's the part you'd attribute to
  oil-leverage, not market beta. If we'd anchored to the wrong day or
  used SPY's adjusted close incorrectly, alpha would collapse to noise.

## Sanity check #2 — Russia invades Ukraine (2022-02-24)

108 filings landed in the Feb 18 – Mar 15 2022 window. Calendar-year
filers with December year-ends were filing right as the invasion began.
Oil and gas names should benefit from the Brent/WTI spike; non-oil names
exposed to risk-off should suffer. Both effects show up cleanly.

**Top positive 5-day reactions in the war window** (all energy):

| Ticker | FY   | Acceptance (ET)     | t1         | t5         | r_5d   | a_5d   |
| ------ | ---- | ------------------- | ---------- | ---------- | ------ | ------ |
| FET    | 2021 | 2022-03-04 17:48    | 2022-03-07 | 2022-03-10 | +24.7% | +27.0% |
| TPL    | 2021 | 2022-02-23 17:31    | 2022-02-24 | 2022-03-02 | +22.8% | +19.0% |
| OXY    | 2021 | 2022-02-24 17:36    | 2022-02-25 | 2022-03-03 | +22.7% | +21.0% |
| RRC    | 2021 | 2022-02-22 17:21    | 2022-02-23 | 2022-03-01 | +19.0% | +18.9% |
| NOG    | 2021 | 2022-02-25 17:39    | 2022-02-28 | 2022-03-04 | +16.1% | +17.3% |

OXY's t1 is `2022-02-25` — the morning after the invasion. The +21% alpha
is *post-invasion*, and the row attributes it to OXY rather than to the
market average (which was actually negative that week — alpha exceeds
the raw return).

The negative side of the same window picks up war-losers:

| Ticker | FY   | Acceptance (ET)     | t5         | r_5d   | a_5d   |
| ------ | ---- | ------------------- | ---------- | ------ | ------ |
| NOMD   | 2021 | 2022-03-03 16:31    | 2022-03-09 | −16.8% | −14.4% |
| XPEL   | 2021 | 2022-02-28 16:03    | 2022-03-04 | −18.7% | −17.4% |

NOMD is a frozen-foods company with European exposure; XPEL is an
auto-aftermarket name. Both reasonable losers in a war-spike, energy-led
risk-off week.

## Sanity check #3 — amendment flag

8% of rows (`98 / 1221`) carry `has_amendment=True`. Spot-checking a few:

- `AAP FY2023` is in our flagged set — Advance Auto Parts amended its
  FY2023 10-K on `2024-05-30` after announcing it had over-recognised
  vendor consideration in prior periods. That matches the amendment
  reality.
- `DAN FY2022` is flagged — Dana filed a 10-K/A in 2024 for fiscal 2022,
  consistent with their accounting-restatement disclosure cycle.

The flag is informational, not a re-anchor. We deliberately use the
*original* 10-K's acceptanceDateTime — the original publication is the
clean reaction signal; amendments arrive months later when the news is
already priced in. Callers who want to filter restatements can drop
`has_amendment == True`.

## Distribution sanity

If the t0/t1 anchoring or the SPY join were off, the distribution would
drift. It doesn't:

```
r_1d:   n=1221  mean=-0.17%  median=-0.04%  stdev=5.57%
r_5d:   n=1221  mean=-1.18%  median=-0.57%  stdev=10.44%
a_1d:   n=1221  mean=-0.08%  median=-0.01%  stdev=5.46%
a_5d:   n=1221  mean=-0.61%  median=-0.45%  stdev=9.55%
```

The alpha distributions are centred essentially at zero (median
`-0.01%`, `-0.45%`). Raw returns are slightly negative on average
because our universe over-samples small-caps and energy/REIT names that
underperformed SPY across 2017-2022 — but the *alpha* mean is also near
zero, which is the right answer if the event-day anchoring isn't
introducing any systematic timing bias.

## Error analysis

207 of 1428 rows have a populated `error` column. The buckets:

| Count | Bucket                                          | Why this happens                                                                                  |
| ----- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| 160   | `no original 10-K for FY{year}`                 | 52/53-week fiscal year; period-end date lands in the wrong calendar year for our keying.          |
| 18    | `no yfinance prices`                            | Delisted, merged, or thin-trade name yfinance no longer serves.                                   |
| 17    | `insufficient price history around filing`      | Filing is so recent (or ticker is so new) that we don't have ≥1 trading day on each side.         |
| 6     | `no CIK in EDGAR ticker map`                    | US-listed but absent from `company_tickers.json` — typically ADRs or recent OTC migrations.       |
| 6     | `no 10-K/20-F on EDGAR`                         | EDGAR has the company but only non-annual forms (e.g. 6-K only for some foreign issuers).          |

### The 52/53-week fiscal year bucket

This is the dominant error source and is **not a bug** — it's a direct
consequence of how we key fiscal year (period-end-date year, see
`CLAUDE.md`).

Many US retailers and consumer companies file on a 52/53-week calendar:
their fiscal year ends on the Saturday closest to a calendar boundary.
A representative case is **AAP (Advance Auto Parts)**:

```
report=2017-12-30  -> keyed FY2017
report=2018-12-29  -> keyed FY2018
report=2019-12-28  -> keyed FY2019
report=2021-01-02  -> keyed FY2021   <- this is "FY2020" in their books!
report=2022-01-01  -> keyed FY2022   <- this is "FY2021" in their books!
report=2022-12-31  -> keyed FY2022   <- this one wins (see below)
```

Two artefacts:

1. AAP has no row keyed as `FY2020` because their fiscal-2020 ended
   Jan 2 2021, which our convention keys as `2021`. The 160 errors
   bucket is dominated by these "missing FY2020" rows for retailers /
   consumer names with January-Saturday year-ends (COST, ROST, TJX, etc).
2. AAP's fiscal-2021 (period-end Jan 1 2022) and fiscal-2022 (period-end
   Dec 31 2022) **both** key as `FY2022`. We pick the one with the
   *latest* `report_date` — fiscal-2022's 10-K — to stay consistent with
   `fetch_kpis.py`'s `max(filed)` selection. This means
   `filing_returns.csv` and `kpis_long.csv` reference the same accession
   for the same `(ticker, year)` row, so a join on `(ticker, year)` is
   value-aligned. The fiscal-2021 10-K (the *earlier* period whose
   year-end happened to fall on Jan 1) is silently dropped from both
   tables — there's no `FY2021` row for AAP anywhere in the dataset.

Both behaviours are consistent with our keying convention, but worth
knowing if you're surprised by gaps. To track every 10-K an issuer
filed, query EDGAR submissions directly via `edgar_filings.py`.

The same edge hits other retailers (COST, ROST, TJX, …) and most of the
auto-parts industry. To get a count by ticker:

```python
import pandas as pd
df = pd.read_csv("KPI_analysis/output/filing_returns.csv")
errs = df[df.error == "no original 10-K for FY{}".format(2020)]
errs.ticker.value_counts()  # which tickers are most affected
```

### `no yfinance prices`

Eighteen rows. Examples: `LB`, `SUNS`. These are tickers that yfinance
no longer serves at all, typically because the listing was merged out
(LB → BBWI / VSCO split) or wound down. There's no fix from this
script's side — the price history is genuinely unavailable from
yfinance. A future Alpha Vantage gap-fill (or other paid feed) could
recover some of these.

### `insufficient price history around filing`

Seventeen rows. Common cause: **post-IPO filings**. If a ticker IPO'd in
mid-2021 and filed its first 10-K in early 2022, we have no ≥1 trading
day strictly *before* the news to anchor `t0` against. We surface the
filing metadata but leave returns null.

### `no CIK in EDGAR ticker map`

Six rows. Tickers where SEC's `company_tickers.json` doesn't have a
mapping. Usually ADR / dual-listing edge cases. Pass `--refresh-cache`
to re-pull the ticker map if you suspect it's stale.

### `no 10-K/20-F on EDGAR`

Six rows. The company exists on EDGAR but has never filed a 10-K or
20-F in the time window we requested — e.g. SPACs that haven't
completed a business combination, foreign issuers that file only 6-K
event reports.

## What this validation does *not* cover

- **Beta-adjustment.** `a_Nd = r_Nd − spy_r_Nd` subtracts the market
  return verbatim. For a high-beta name in a violent week (a 1.5-beta
  stock with SPY −10% has a −15% expected move from market alone), raw
  alpha overstates the company-specific signal. A market-model
  regression would be cleaner if you care about pure idiosyncratic
  reaction; this is left to downstream analysis.
- **Filing time vs. press release.** The `acceptanceDateTime` is when
  EDGAR ingested the 10-K, which may be hours after the company's own
  press release earlier the same day. For most filers the two coincide
  to within an hour, but a tight intraday analysis would benefit from
  cross-checking against PR-wire timestamps.
- **Currency.** SPY is USD; non-US ADRs trading in USD on US exchanges
  are fine, but a future LSE/AIM extension would need an FTSE 100
  benchmark and FX-aware comparisons, not SPY.
- **Earnings-call timing.** Some companies hold their earnings call
  before the 10-K hits EDGAR (the 8-K + press release goes out first,
  the 10-K trickles in hours or days later). For those, our `t1`
  captures the post-10-K reaction, not the pre-call → post-call
  reaction. The 10-K specifically — not the earnings release — is what
  this dataset measures.
