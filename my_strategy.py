from datetime import time

import numpy as np
import pandas as pd

from config import END, START, TEST_START, TRAIN_END
from data import clean_minute_bars, load_daily_total_returns, load_minute_bars, make_daily_tables
from metrics import compute_metrics
from paper_strategy import (
    COMMISSION_PER_SHARE,
    SLIPPAGE_PER_SHARE,
    compute_daily_volatility,
    compute_noise_area,
    compute_shares,
    compute_trading_costs,
    compute_vwap,
    get_target_position,
    get_trading_times,
)

# ----------------------------------------------------------------------------
# Settings (the notebook reads everything from here)
# ----------------------------------------------------------------------------
STOP_MODE = "band_vwap"  # "band_vwap" or "opposite_band"

# Leverage (paper direction: high volatility -> low leverage)
# Paper values: TARGET_VOLATILITY = 0.02 | MAX_LEVERAGE = 4 | LEVERAGE_MULTIPLIER = 1.0
TARGET_VOLATILITY = 0.02
MAX_LEVERAGE = 4
LEVERAGE_MULTIPLIER = 1.0

# Add-on 1: RSI scaling of the leverage (dynamic sizing only)
USE_RSI_SCALING = True
RSI_PERIOD = 5
RSI_CENTER = 50
RSI_SENSITIVITY = 0.5  # 0 = no effect; scale = 1 + sensitivity * (center - RSI) / 50
RSI_MIN_SCALE = 0.5
RSI_MAX_SCALE = 1.5

# Add-on 2: no new entries during the lunch window
USE_LUNCH_FILTER = True
NO_ENTRY_START = time(12, 30)
NO_ENTRY_END = time(14, 0)

# Add-on 3: breadth filter with RSP (S&P 500 equal weight). The RSP Noise Area is computed exactly like
# the SPY one (same lookback and vm). Exits and stops are untouched.
#   "off"        : no filter
#   "confirm"    : a long (short) SPY entry needs RSP above its upper (below its lower) band as well,
#                  i.e. the move is carried by the broad market
#   "contrarian" : a long (short) SPY entry needs RSP NOT above its upper (NOT below its lower) band,
#                  i.e. only moves carried by the large caps are traded
# Only days with clean data for both etfs are traded.
RSP_MODE = "confirm"
RSP_MODES = ("off", "confirm", "contrarian")
RSP_SYMBOL = "RSP"

# (off/confirm/contrarian) = 2 x 2 x 3 = 12 variants
ADDONS = ("use_rsi_scaling", "use_lunch_filter")
ADDON_LABELS = {"use_rsi_scaling": "RSI", "use_lunch_filter": "Lunch"}
RSP_LABELS = {"off": None, "confirm": "RSP confirm", "contrarian": "RSP contrarian"}


def compute_scaled_shares(
    aum,
    day_open,
    daily_volatility,
    leverage_multiplier=LEVERAGE_MULTIPLIER,
    target_volatility=TARGET_VOLATILITY,
    max_leverage=MAX_LEVERAGE,
):
    """Number of shares with the paper's volatility-targeted leverage, scaled by `leverage_multiplier`."""
    # leverage = min(max_leverage, multiplier * target_vol / sigma)
    # high sigma -> low leverage, low sigma -> high leverage (as in the paper)
    return compute_shares(
        aum,
        day_open,
        daily_volatility,
        target_volatility=target_volatility * leverage_multiplier,
        max_leverage=max_leverage,
    )


def compute_rsi(close, period=RSI_PERIOD):
    """RSI (Wilder smoothing) of the daily closes, shifted by one day so it is known at the open."""
    daily_close = close.iloc[:, -1]
    delta = daily_close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    return rsi.shift(1)


def compute_rsi_scale(
    rsi, sensitivity=RSI_SENSITIVITY, center=RSI_CENTER, min_scale=RSI_MIN_SCALE, max_scale=RSI_MAX_SCALE
):
    """Leverage scale from the RSI: above 1 after declines (RSI < center), below 1 after rallies."""
    # Paper: lower RSI -> higher expected return -> more leverage.
    # linear scale around 1.0 instead of a hard filter.
    scale = 1 + sensitivity * (center - rsi) / 50
    return scale.clip(min_scale, max_scale).fillna(1.0)


def get_common_days(spy_tables, rsp_tables):
    """Days with clean minute data for both SPY and RSP, the only days an RSP filter can be evaluated on."""
    return spy_tables["close"].index.intersection(rsp_tables["close"].index)


def compute_rsp_filter(spy_tables, rsp_tables, trading_times, mode="confirm", lookback=14, vm=1.0):
    """Which SPY entries the RSP filter allows, as two boolean arrays (days x trading times).

    "confirm" allows a long (short) only if RSP is above its upper (below its lower) Noise Area band,
    "contrarian" only if it is not. Missing RSP data blocks the entry in both modes.
    """
    if mode not in ("confirm", "contrarian"):
        raise ValueError(f"unknown RSP mode: {mode!r}")
    upper, lower = compute_noise_area(
        rsp_tables["close"], rsp_tables["day_open"], rsp_tables["prev_close"], lookback, vm
    )
    index = spy_tables["close"].index
    price = rsp_tables["close"][trading_times].reindex(index)
    upper = upper[trading_times].reindex(index)
    lower = lower[trading_times].reindex(index)
    if mode == "confirm":
        allow_long, allow_short = price > upper, price < lower
    else:
        allow_long, allow_short = price <= upper, price >= lower
    return allow_long.to_numpy(), allow_short.to_numpy()


def compute_entry_filters(tables, rsp_tables, trading_times, rsp_mode, days=None, lookback=14, vm=1.0):
    """Allowed long/short entries per day and time slot, plus the days to trade.

    With an RSP filter, `days` is restricted to the days with data for both ETFs.
    """
    n_days = len(tables["close"])
    if rsp_mode == "off":
        allow = np.ones((n_days, len(trading_times)), dtype=bool)
        return allow, allow, days
    if rsp_tables is None:
        raise ValueError("an RSP filter needs rsp_tables (make_daily_tables of the RSP minute bars)")
    allow_long, allow_short = compute_rsp_filter(tables, rsp_tables, trading_times, rsp_mode, lookback, vm)
    common_days = get_common_days(tables, rsp_tables)
    days = common_days if days is None else pd.Index(days).intersection(common_days)
    return allow_long, allow_short, days


def compute_entry_allowed(trading_times, use_lunch_filter=USE_LUNCH_FILTER, start=NO_ENTRY_START, end=NO_ENTRY_END):
    """Boolean per trading time: False inside the lunch window [start, end) if the filter is on."""
    if not use_lunch_filter:
        return np.ones(len(trading_times), dtype=bool)
    return np.array([not (start <= t < end) for t in trading_times])


def simulate_day_filtered(
    prices, uppers, lowers, vwaps, final_price, shares, stop_mode, entry_allowed, allow_long=None, allow_short=None
):
    """Like paper_strategy.simulate_day, but new entries can be blocked per time slot.

    Blocked entries (lunch filter, RSP filter) keep the account flat. Exits are always executed, so a
    blocked flip closes the old position. Returns the day's PnL and the traded shares.
    """
    if allow_long is None:
        allow_long = np.ones(len(prices), dtype=bool)
    if allow_short is None:
        allow_short = np.ones(len(prices), dtype=bool)

    position = 0
    entry_price = 0.0
    pnl = 0.0
    traded_shares = 0

    for price, upper, lower, vwap, allowed, long_ok, short_ok in zip(
        prices, uppers, lowers, vwaps, entry_allowed, allow_long, allow_short
    ):
        target = get_target_position(position, price, upper, lower, vwap, stop_mode)
        if target != position and target != 0:
            if not allowed or (target == 1 and not long_ok) or (target == -1 and not short_ok):
                target = 0
        if target != position:
            pnl += position * shares * (price - entry_price)
            traded_shares += abs(target - position) * shares
            position = target
            entry_price = price

    pnl += position * shares * (final_price - entry_price)
    traded_shares += abs(position) * shares
    return pnl, traded_shares


def run_my_backtest(
    tables,
    stop_mode=STOP_MODE,
    sizing="dynamic",
    initial_capital=100_000,
    lookback=14,
    vm=1.0,
    leverage_multiplier=LEVERAGE_MULTIPLIER,
    target_volatility=TARGET_VOLATILITY,
    max_leverage=MAX_LEVERAGE,
    use_rsi_scaling=USE_RSI_SCALING,
    use_lunch_filter=USE_LUNCH_FILTER,
    commission=COMMISSION_PER_SHARE,
    slippage=SLIPPAGE_PER_SHARE,
    rsp_mode=RSP_MODE,
    rsp_tables=None,
    days=None,
    benchmark_returns=None,
):
    """Backtest of the paper strategy with the add-ons. Returns the daily net returns and their metrics.

    `days` restricts the dates that are traded (others are skipped, not counted as flat). Indicators are
    always computed on the full SPY history.
    """
    close = tables["close"]
    day_open = tables["day_open"]
    upper, lower = compute_noise_area(close, day_open, tables["prev_close"], lookback, vm)
    vwap = compute_vwap(close, tables["volume"])
    daily_volatility = compute_daily_volatility(close)
    trading_times = get_trading_times(close.columns)
    entry_allowed = compute_entry_allowed(trading_times, use_lunch_filter)
    allow_long, allow_short, days = compute_entry_filters(
        tables, rsp_tables, trading_times, rsp_mode, days, lookback, vm
    )

    if use_rsi_scaling:
        rsi_scale = compute_rsi_scale(compute_rsi(close)).to_numpy()
    else:
        rsi_scale = np.ones(len(close))

    prices = close[trading_times].to_numpy()
    uppers = upper[trading_times].to_numpy()
    lowers = lower[trading_times].to_numpy()
    vwaps = vwap[trading_times].to_numpy()
    final_prices = close.iloc[:, -1].to_numpy()
    open_prices = day_open.to_numpy()
    volatilities = daily_volatility.to_numpy()

    is_trading_day = np.ones(len(close), dtype=bool) if days is None else close.index.isin(days)

    aum = initial_capital
    returns = {}
    for i, date in enumerate(close.index):
        if not is_trading_day[i] or np.isnan(uppers[i, 0]) or np.isnan(volatilities[i]):
            continue

        if sizing == "dynamic":
            shares = compute_scaled_shares(
                aum,
                open_prices[i],
                volatilities[i],
                leverage_multiplier * rsi_scale[i],
                target_volatility,
                max_leverage,
            )
        else:
            shares = np.floor(aum / open_prices[i])

        pnl, traded_shares = simulate_day_filtered(
            prices[i],
            uppers[i],
            lowers[i],
            vwaps[i],
            final_prices[i],
            shares,
            stop_mode,
            entry_allowed,
            allow_long[i],
            allow_short[i],
        )
        net_pnl = pnl - compute_trading_costs(traded_shares, commission, slippage)
        returns[date] = net_pnl / aum
        aum += net_pnl

    daily_returns = pd.Series(returns)
    return daily_returns, compute_metrics(daily_returns, benchmark_returns)


def count_daily_trades(
    tables,
    stop_mode=STOP_MODE,
    use_lunch_filter=USE_LUNCH_FILTER,
    rsp_mode=RSP_MODE,
    rsp_tables=None,
    days=None,
    lookback=14,
    vm=1.0,
    **_,
):
    """Number of entries per day, with the same signals and days as run_my_backtest.

    Simulated with one share, so trades = traded shares / 2. A flip long -> short counts as a new trade.
    """
    close = tables["close"]
    upper, lower = compute_noise_area(close, tables["day_open"], tables["prev_close"], lookback, vm)
    vwap = compute_vwap(close, tables["volume"])
    volatilities = compute_daily_volatility(close).to_numpy()
    trading_times = get_trading_times(close.columns)
    entry_allowed = compute_entry_allowed(trading_times, use_lunch_filter)
    allow_long, allow_short, days = compute_entry_filters(
        tables, rsp_tables, trading_times, rsp_mode, days, lookback, vm
    )
    is_trading_day = np.ones(len(close), dtype=bool) if days is None else close.index.isin(days)

    prices, uppers, lowers, vwaps = (frame[trading_times].to_numpy() for frame in (close, upper, lower, vwap))
    final_prices = close.iloc[:, -1].to_numpy()
    counts = {}
    for i, date in enumerate(close.index):
        if not is_trading_day[i] or np.isnan(uppers[i, 0]) or np.isnan(volatilities[i]):
            continue
        _, traded = simulate_day_filtered(
            prices[i],
            uppers[i],
            lowers[i],
            vwaps[i],
            final_prices[i],
            1,
            stop_mode,
            entry_allowed,
            allow_long[i],
            allow_short[i],
        )
        counts[date] = traded // 2
    return pd.Series(counts, dtype=float)


def ablation_variants():
    """Settings of all 12 combinations of RSI scaling, lunch filter and RSP mode, paper baseline first."""
    variants = {}
    for rsp_mode in RSP_MODES:
        for mask in range(2 ** len(ADDONS)):
            flags = {addon: bool(mask >> k & 1) for k, addon in enumerate(ADDONS)}
            on = [ADDON_LABELS[a] for a in ADDONS if flags[a]] + [RSP_LABELS[rsp_mode]] * (rsp_mode != "off")
            name = "Paper" if not on else "Paper + " + " + ".join(on)
            variants[name] = dict(sizing="dynamic", leverage_multiplier=1.0, rsp_mode=rsp_mode, **flags)
    return variants


def run_ablation(tables, rsp_tables, days, periods, stop_mode=STOP_MODE):
    """Metrics and trades of every ablation variant per period, on the same days and with the same costs.

    Returns a DataFrame indexed by (period, variant), including the Sharpe change versus the paper baseline.
    """
    runs = {}
    for name, kwargs in ablation_variants().items():
        returns, _ = run_my_backtest(tables, stop_mode, rsp_tables=rsp_tables, days=days, **kwargs)
        trades = count_daily_trades(tables, stop_mode, rsp_tables=rsp_tables, days=days, **kwargs)
        runs[name] = (returns, trades)

    rows = {}
    for period_name, period in periods.items():
        for name, (returns, trades) in runs.items():
            metrics = compute_metrics(returns.loc[period])
            rows[(period_name, name)] = {
                "sharpe_ratio": metrics["sharpe_ratio"],
                "annual_return": metrics["annual_return"],
                "annual_volatility": metrics["annual_volatility"],
                "max_drawdown": metrics["max_drawdown"],
                "trades": trades.loc[period].sum(),
            }
    result = pd.DataFrame(rows).T
    result.index.names = ["period", "variant"]
    paper_sharpe = result.xs("Paper", level="variant")["sharpe_ratio"]
    result["sharpe_vs_paper"] = (
        result["sharpe_ratio"] - paper_sharpe.reindex(result.index.get_level_values("period")).to_numpy()
    )
    return result


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    raw = load_minute_bars("SPY", START, END)
    tables = make_daily_tables(clean_minute_bars(raw))
    benchmark_returns = load_daily_total_returns("SPY", START, END)
    rsp_tables = make_daily_tables(clean_minute_bars(load_minute_bars(RSP_SYMBOL, START, END)))
    common_days = get_common_days(tables, rsp_tables)
    print(f"{len(common_days)} of {len(tables['close'])} SPY days have RSP data, all variants use these days\n")

    off = dict(use_rsi_scaling=False, use_lunch_filter=False, rsp_mode="off")
    mine = dict(
        sizing="dynamic",
        leverage_multiplier=LEVERAGE_MULTIPLIER,
        use_rsi_scaling=USE_RSI_SCALING,
        use_lunch_filter=USE_LUNCH_FILTER,
    )
    variants = {
        "Fixed 100%": dict(sizing="fixed", **off),
        "Paper (x1.0)": dict(sizing="dynamic", leverage_multiplier=1.0, **off),
        "Mine, RSP confirm": dict(rsp_mode="confirm", **mine),
        "Mine, RSP contrarian": dict(rsp_mode="contrarian", **mine),
    }
    strategy_returns = {}
    for name, kwargs in variants.items():
        strategy_returns[name], _ = run_my_backtest(
            tables, STOP_MODE, rsp_tables=rsp_tables, days=common_days, **kwargs
        )

    # Buy & hold on every trading day from the first strategy day on (dividends included), not only on the
    # common days: skipping days is not something a buy & hold investor can do
    spy_returns = benchmark_returns.loc[strategy_returns["Fixed 100%"].index[0] :]
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

    print("ABLATION (all 12 add-on combinations, same days and costs)")
    ablation = run_ablation(tables, rsp_tables, common_days, {"TRAIN": periods["TRAIN"], "TEST": periods["TEST"]})
    print(ablation.round(3))
