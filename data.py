import os
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config import FEED

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

TIMEZONE = "America/New_York"
CACHE_DIR = Path("data_cache")


def get_client():
    if "ALPACA_KEY" not in os.environ or "ALPACA_SECRET" not in os.environ:
        raise RuntimeError(
            "The requested data is not in data_cache/ and no Alpaca keys were found. "
            "Set ALPACA_KEY and ALPACA_SECRET in the environment or in a .env file."
        )
    return StockHistoricalDataClient(os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"])


def download_minute_bars(symbol, start, end, feed=FEED):
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=pd.Timestamp(start, tz=TIMEZONE),
        end=pd.Timestamp(end, tz=TIMEZONE),
        adjustment=Adjustment.RAW,
        feed=DataFeed(feed),
    )
    bars = get_client().get_stock_bars(request).df
    return bars.droplevel("symbol")


def load_minute_bars(symbol, start, end, feed=FEED):
    path = CACHE_DIR / f"{symbol}_{start}_{end}_{feed}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    CACHE_DIR.mkdir(exist_ok=True)
    bars = download_minute_bars(symbol, start, end, feed)
    bars.to_parquet(path)
    return bars


def load_daily_total_returns(symbol, start, end, feed=FEED):
    path = CACHE_DIR / f"{symbol}_{start}_{end}_{feed}_daily_adjusted.parquet"
    if path.exists():
        bars = pd.read_parquet(path)
    else:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=pd.Timestamp(start, tz=TIMEZONE),
            end=pd.Timestamp(end, tz=TIMEZONE),
            adjustment=Adjustment.ALL,
            feed=DataFeed(feed),
        )
        bars = get_client().get_stock_bars(request).df.droplevel("symbol")
        CACHE_DIR.mkdir(exist_ok=True)
        bars.to_parquet(path)
    close = bars["close"].copy()
    close.index = close.index.tz_convert(TIMEZONE).normalize().tz_localize(None)
    return close.pct_change().dropna()


def clean_minute_bars(raw, min_bars=370):
    bars = raw[["open", "high", "low", "close", "volume"]].copy()
    bars.index = bars.index.tz_convert(TIMEZONE)
    bars = bars.between_time("09:30", "15:59")
    # Alpaca uses the start of the minute, I want the end to prevent look aheads
    bars.index = bars.index + pd.Timedelta(minutes=1)

    days = []
    for date, day in bars.groupby(bars.index.date):
        if len(day) < min_bars:
            continue
        full_index = pd.date_range(f"{date} 09:31", f"{date} 16:00", freq="min", tz=TIMEZONE)
        day = day.reindex(full_index)
        day["volume"] = day["volume"].fillna(0)
        # Add values to missing minutes
        days.append(day.ffill().bfill())
    return pd.concat(days)


def make_daily_tables(bars):
    bars = bars.copy()
    bars["date"] = bars.index.normalize().tz_localize(None)
    bars["time"] = bars.index.time

    # reshape: one row per day and one col per min
    close = bars.pivot(index="date", columns="time", values="close")
    volume = bars.pivot(index="date", columns="time", values="volume")
    day_open = bars.groupby("date")["open"].first()
    # get last min from prev trading day for overnight gap
    prev_close = close.iloc[:, -1].shift(1)

    return {"close": close, "volume": volume, "day_open": day_open, "prev_close": prev_close}
