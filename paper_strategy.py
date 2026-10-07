from datetime import time

import numpy as np
import pandas as pd

from config import END, START, TEST_START, TRAIN_END
from data import clean_minute_bars, load_daily_total_returns, load_minute_bars, make_daily_tables
from metrics import compute_metrics

FIRST_TRADE_TIME = time(10, 0)
LAST_TRADE_TIME = time(15, 30)
COMMISSION_PER_SHARE = 0.0035
SLIPPAGE_PER_SHARE = 0.001


def compute_sigma(close, day_open, lookback=14):
    moves = close.div(day_open, axis=0).sub(1).abs()
    return moves.rolling(lookback).mean().shift(1)


def compute_noise_area(close, day_open, prev_close, lookback=14, vm=1.0):
    sigma = compute_sigma(close, day_open, lookback)
    upper_reference = np.maximum(day_open, prev_close)
    lower_reference = np.minimum(day_open, prev_close)
    upper = (1 + vm * sigma).mul(upper_reference, axis=0)
    lower = (1 - vm * sigma).mul(lower_reference, axis=0)
    return upper, lower


def compute_vwap(close, volume):
    cumulative_value = (close * volume).cumsum(axis=1)
    cumulative_volume = volume.cumsum(axis=1)
    return cumulative_value / cumulative_volume


def get_trading_times(columns):
    trading_times = []
    for column in columns:
        is_half_hour = column.minute in (0, 30)
        is_in_window = FIRST_TRADE_TIME <= column <= LAST_TRADE_TIME
        if is_half_hour and is_in_window:
            trading_times.append(column)
    return trading_times


def compute_daily_volatility(close, lookback=14):
    daily_close = close.iloc[:, -1]
    daily_returns = daily_close.pct_change()
    return daily_returns.rolling(lookback).std().shift(1)


def compute_shares(aum, day_open, daily_volatility, target_volatility=0.02, max_leverage=4):
    leverage = min(max_leverage, target_volatility / daily_volatility)
    return np.floor(aum * leverage / day_open)


def compute_trading_costs(shares, commission=COMMISSION_PER_SHARE, slippage=SLIPPAGE_PER_SHARE):
    return abs(shares) * (commission + slippage)


def get_target_position(position, price, upper, lower, vwap, stop_mode):
    if stop_mode == "band_vwap":
        if price > max(upper, vwap):
            return 1
        if price < min(lower, vwap):
            return -1
        return 0

    if position == 1:
        if price < lower:
            return -1
        return 1
    if position == -1:
        if price > upper:
            return 1
        return -1
    if price > upper:
        return 1
    if price < lower:
        return -1
    return 0


def simulate_day(prices, uppers, lowers, vwaps, final_price, shares, stop_mode):
    position = 0
    entry_price = 0.0
    pnl = 0.0
    traded_shares = 0

    for price, upper, lower, vwap in zip(prices, uppers, lowers, vwaps):
        target = get_target_position(position, price, upper, lower, vwap, stop_mode)
        if target != position:
            pnl += position * shares * (price - entry_price)
            traded_shares += abs(target - position) * shares
            position = target
            entry_price = price

    pnl += position * shares * (final_price - entry_price)
    traded_shares += abs(position) * shares
    return pnl, traded_shares


def run_backtest(
    tables,
    stop_mode="band_vwap",
    sizing="dynamic",
    initial_capital=100_000,
    lookback=14,
    vm=1.0,
    commission=COMMISSION_PER_SHARE,
    slippage=SLIPPAGE_PER_SHARE,
    benchmark_returns=None,
):
    close = tables["close"]
    day_open = tables["day_open"]
    upper, lower = compute_noise_area(close, day_open, tables["prev_close"], lookback, vm)
    vwap = compute_vwap(close, tables["volume"])
    daily_volatility = compute_daily_volatility(close)
    trading_times = get_trading_times(close.columns)

    prices = close[trading_times].to_numpy()
    uppers = upper[trading_times].to_numpy()
    lowers = lower[trading_times].to_numpy()
    vwaps = vwap[trading_times].to_numpy()
    final_prices = close.iloc[:, -1].to_numpy()
    open_prices = day_open.to_numpy()
    volatilities = daily_volatility.to_numpy()

    aum = initial_capital
    returns = {}
    for i, date in enumerate(close.index):
        if np.isnan(uppers[i, 0]) or np.isnan(volatilities[i]):
            continue

        if sizing == "dynamic":
            shares = compute_shares(aum, open_prices[i], volatilities[i])
        else:
            shares = np.floor(aum / open_prices[i])

        pnl, traded_shares = simulate_day(prices[i], uppers[i], lowers[i], vwaps[i], final_prices[i], shares, stop_mode)
        net_pnl = pnl - compute_trading_costs(traded_shares, commission, slippage)
        returns[date] = net_pnl / aum
        aum += net_pnl

    daily_returns = pd.Series(returns)
    return daily_returns, compute_metrics(daily_returns, benchmark_returns)


# for terminal summary
if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    raw = load_minute_bars("SPY", START, END)
    tables = make_daily_tables(clean_minute_bars(raw))
    benchmark_returns = load_daily_total_returns("SPY", START, END)

    variants = {
        "Opp.Band, 100%": ("opposite_band", "fixed"),
        "Band+VWAP, 100%": ("band_vwap", "fixed"),
        "Band+VWAP, Dyn.": ("band_vwap", "dynamic"),
    }
    strategy_returns = {}
    for name, (stop_mode, sizing) in variants.items():
        strategy_returns[name], _ = run_backtest(tables, stop_mode, sizing)

    spy_returns = benchmark_returns.loc[strategy_returns["Band+VWAP, Dyn."].index[0] :]
    periods = {
        "FULL": slice(None),
        "TRAIN": slice(None, TRAIN_END),
        "TEST": slice(TEST_START, None),
    }

    for period_name, period in periods.items():
        rows = {}
        for name, daily_returns in strategy_returns.items():
            rows[name] = compute_metrics(daily_returns.loc[period], benchmark_returns)
        rows["SPY Buy&Hold"] = compute_metrics(spy_returns.loc[period])
        print(period_name)
        print(pd.DataFrame(rows).T.round(3))
        print()
