# Intraday Momentum Strategy for SPY

Backtest of *Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)* (Zarattini, Aziz, Barbon, 2025) and my own variation with an RSI-scaled leverage, a lunch-time entry filter and a "breadth" filter with RSP (S&P 500 Equal Weight ETF), tested in two opposite directions.

**Short version**
- **The replication works.** In train (2020–2023) the paper strategy reaches a Sharpe ratio of 1.88 after costs, and its monthly returns match the paper's own table (FAQ Q24) with a correlation of 0.99.
- **Out of sample the edge is small.** In test (2024 – Oct 2026) the Sharpe ratio is 0.29 and 2025–2026 are negative, while SPY Buy & Hold returned 20.6% p.a. (Sharpe 1.28).
- **None of my add-ons beats the paper baseline in train**, so a train-only selection keeps the paper strategy. The breadth filter does not help in either direction: breadth carries little information about the quality of an intraday SPY breakout.

## Files

| File | Purpose |
|---|---|
| `config.py` | Data feed (SIP), sample period and train/test boundary. Used everywhere. |
| `data.py` | Downloads and caches Alpaca data, cleans 1-minute bars, builds the daily tables, loads the dividend-adjusted SPY benchmark. |
| `metrics.py` | Performance metrics, monthly table in the layout of paper FAQ Q24. |
| `paper_strategy.py` | The strategy as described in the paper. |
| `my_strategy.py` | Paper strategy plus add-ons, trade counting and the ablation. |
| `charts.ipynb` | Charts and tables for the paper strategy, comparison with the paper. |
| `my_strategy_charts.ipynb` | Charts and tables for my variation, both RSP directions and the ablation. |

## 1. Paper strategy

**Idea.** Intraday trends appear when there is a persistent demand/supply imbalance. The strategy only trades when the price leaves the range of normal intraday moves, the Noise Area.

**Noise Area.** For each time of day, `sigma` is the average of `|Close(t) / Open - 1|` over the previous 14 days. The bounds are `max(Open, PrevClose) * (1 + sigma)` and `min(Open, PrevClose) * (1 - sigma)`. The previous close accounts for overnight gaps.

**Trading rules.** Decisions every half hour from 10:00 to 15:30: long above the upper bound, short below the lower bound. The stop is the band or the VWAP, whichever is tighter (`band_vwap`). The alternative `opposite_band` reverses only at the opposite bound. Everything is closed at 16:00.

**Sizing.** `leverage = min(4, 2% / sigma_SPY)`, where `sigma_SPY` is the volatility of the last 14 daily returns. Less exposure in volatile markets, at most 4x. A fixed 100% variant is shown for comparison.

**Costs.** $0.0035 commission + $0.001 slippage per share and side, as in the paper. A reversal counts as two sides.

**Data and evaluation.**
- 1-minute bars from Alpaca (SIP feed, unadjusted), regular trading hours. Days with fewer than 370 bars (half days) are removed. 2020-01-02 to 2026-10-01, 1,687 SPY days. I used Alpaca because I already knew its API; the IQFeed history from 2007 that the paper uses was not available to me. The downside is the shorter sample.
- Train until 2023-12-31, test from 2024-01-01. The boundary was fixed in advance.
- Indicators use only past data (`shift(1)`), and position size uses the open price before the first decision at 10:00. Trades are filled at the close of the signal bar, the paper's convention.
- Annual returns are annualised over calendar years. Volatility and Sharpe ratio use √252 and no risk-free rate.
- Benchmark: SPY Buy & Hold, dividends reinvested, on every trading day.

## 2. My strategy add-ons

The Noise Area signal is unchanged. The add-ons only change *how much* is traded and *which* entries are allowed. Exits are never blocked.

1. **RSI scaling of the leverage** (`USE_RSI_SCALING`). Paper section 4.5 reports that a lower 5-day RSI of SPY is followed by higher strategy returns, which the authors link to dealers' hedging of their gamma exposure. The leverage is scaled smoothly by `clip(1 + 0.5 * (50 - RSI) / 50, 0.5, 1.5)`. The RSI only uses closes up to the previous day.
2. **Lunch filter** (`USE_LUNCH_FILTER`). Paper FAQ Q18: trends pause around lunch. No new entries at 12:30, 13:00 and 13:30.
3. **RSP breadth filter** (`RSP_MODE`). RSP is the Invesco equal-weight S&P 500 ETF. Its Noise Area is computed with the same rule as for SPY. Two opposite hypotheses:
   - `confirm`: only trade a SPY breakout if RSP breaks out in the same direction. Idea: a move carried by the whole market is a stronger imbalance.
   - `contrarian`: only trade a SPY breakout if RSP does **not** break out. Idea: in a market led by a few tech/AI mega caps/hyperscalers, the trends driven by them are the persistent ones.

   RSP confirms 56.5% of the upward and 61.8% of the downward SPY breakouts. RSP is less liquid, and on 125 days (105 of them in 2020) it does not have enough minute bars. **All strategy variants are therefore traded on the same 1,562 days with data for both ETFs.** The benchmark still covers every trading day.

## 3. Results

Same days, same costs ($0.0045 per share and side), same split. "Mine" = paper + RSI scaling + lunch filter + RSP filter.

| | Train return | Train vol | Train Sharpe | Train max DD | Test return | Test vol | Test Sharpe | Test max DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed 100% | 9.8% | 8.1% | 1.31 | 10.8% | 0.6% | 5.7% | 0.14 | 10.3% |
| **Paper (dynamic sizing)** | 27.4% | 14.8% | **1.88** | 10.1% | 3.1% | 14.5% | 0.29 | 23.0% |
| Mine, RSP confirm | 10.9% | 12.5% | 0.98 | 16.8% | 0.6% | 12.8% | 0.11 | 13.2% |
| Mine, RSP contrarian | 8.5% | 7.9% | 1.17 | 5.6% | 1.1% | 9.4% | 0.16 | 21.6% |
| SPY Buy & Hold | 12.5% | 22.4% | 0.64 | 28.5% | 20.6% | 15.5% | **1.28** | 18.8% |

**Ablation** (`run_ablation()`, all 12 combinations in the notebook). Effect of each add-on on its own, as the change in Sharpe ratio versus the paper baseline:

| Add-on alone | Δ Sharpe train | Δ Sharpe test | Trades train / test |
|---|---:|---:|---:|
| RSI scaling | −0.27 | +0.02 | 792 / 623 |
| Lunch filter | −0.07 | +0.01 | 738 / 570 |
| RSP confirm | −0.60 | −0.24 | 529 / 339 |
| RSP contrarian | −0.35 | +0.06 | 384 / 387 |

**RSP confirm vs. contrarian** (filter alone on top of the paper strategy). Average net return per trade, in basis points of the account:

| | Train | Test |
|---|---:|---:|
| Paper (all breakouts) | 12.2 bp | 1.8 bp |
| RSP confirm | 10.8 bp | 0.4 bp |
| RSP contrarian | 11.8 bp | 2.4 bp |

## 4. Interpretation

- **Train vs. test.** The weakness is concentrated in 2025–2026; 2024 was still +27%. The paper itself shows similar losing years (2016: −12.8%, 2017: −6.9%). Possible reasons: market makers are long gamma thus slowing down intraday trends to stay neutral, post-publication crowding.
- **Breadth.** RSP-confirmed breakouts earn less per trade, the contrarian filter about the same, so breadth carries little information about an intraday SPY breakout.
- **RSI scaling** lowered the average leverage in a rising market and did not help, **the lunch filter** is roughly neutral.
- **Evidence is weak overall:** about 7 years of data is not enough

## 5. Next steps

1. Longer history
2. Breadth as a sizing signal instead of a hard filter

## Usage

`pip install -r requirements.txt` (tested with Python 3.9.6) and run everything from this folder. Feed, period and train/test split are set in `config.py`, strategy settings at the top of `my_strategy.py`.

- `python paper_strategy.py` and `python my_strategy.py` print the metrics for the full, train and test period.

- `charts.ipynb` (paper replication) and `my_strategy_charts.ipynb` (own variation): run from the top after changing settings.

**Data.** The market data is not part of the repository. The first run downloads it from Alpaca and needs a free Alpaca account: set `ALPACA_KEY` and `ALPACA_SECRET` in the environment or in a `.env` file in this folder (loaded automatically). The download takes a few minutes; afterwards everything is read from `data_cache/`.