import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_equity_curve(daily_returns, initial_capital=100_000):
    return initial_capital * (1 + daily_returns).cumprod()


def compute_metrics(daily_returns, benchmark_returns=None):
    equity = (1 + daily_returns).cumprod()
    # Calendar years, not len / 252: skipped days (e.g. no RSP data) would otherwise shorten the period
    years = (daily_returns.index[-1] - daily_returns.index[0]).days / 365.25
    drawdown = equity / equity.cummax() - 1

    metrics = {
        "total_return": equity.iloc[-1] - 1,
        "annual_return": equity.iloc[-1] ** (1 / years) - 1,
        "annual_volatility": daily_returns.std() * np.sqrt(TRADING_DAYS),
        "sharpe_ratio": daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS),
        "hit_ratio": (daily_returns > 0).sum() / (daily_returns != 0).sum(),
        "max_drawdown": -drawdown.min(),
        "alpha": np.nan,
        "beta": np.nan,
    }

    if benchmark_returns is not None:
        aligned = pd.concat([daily_returns, benchmark_returns], axis=1, join="inner").dropna()
        beta, daily_alpha = np.polyfit(aligned.iloc[:, 1], aligned.iloc[:, 0], 1)
        metrics["alpha"] = daily_alpha * TRADING_DAYS
        metrics["beta"] = beta

    return metrics


MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Paper monthly return table (Q24)
PAPER_Q24 = pd.DataFrame(
    {
        2020: [5.3, 1.9, -0.5, -1.8, -0.2, 8.0, 0.1, -1.8, 14.3, 2.8, -1.1, -1.9, 26.8],
        2021: [7.8, 3.1, 0.7, 6.1, 2.6, -1.0, 2.8, 1.7, 2.8, 3.3, -3.2, 4.1, 34.8],
        2022: [-5.5, 2.8, 0.2, 9.0, 2.1, 0.5, 0.2, 6.3, -1.0, 5.8, 1.6, 0.7, 24.4],
        2023: [2.9, -1.3, 7.8, 1.8, 2.9, 4.1, 2.2, 6.0, 2.9, -1.1, 0.5, 3.8, 37.2],
        2024: [8.8, -1.5, -0.4, 5.8, -4.3, 1.6, 8.2, -2.8, 4.1, 6.7, -2.6, 5.7, 32.2],
        2025: [-1.2] + [np.nan] * 11 + [-1.2],
    },
    index=MONTH_NAMES + ["Yearly"],
).T
PAPER_Q24.index.name = "Year"


def compute_q24_tables(daily_returns, test_start="2024-01-01"):
    daily_returns = daily_returns.copy()  # do not modify the caller's Series
    daily_returns.index = pd.to_datetime(daily_returns.index)
    growth = 1 + daily_returns

    monthly = growth.groupby([growth.index.year, growth.index.month]).prod() - 1
    monthly_table = monthly.unstack().reindex(columns=range(1, 13)) * 100
    monthly_table.columns = MONTH_NAMES
    monthly_table["Yearly"] = (growth.groupby(growth.index.year).prod() - 1) * 100
    monthly_table.index.name = "Year"

    periods = {
        "Train": daily_returns[daily_returns.index < pd.Timestamp(test_start)],
        "Test": daily_returns[daily_returns.index >= pd.Timestamp(test_start)],
        "Full": daily_returns,
    }
    columns = ["annual_return", "annual_volatility", "sharpe_ratio", "max_drawdown"]
    period_summary = pd.DataFrame({name: compute_metrics(r) for name, r in periods.items()}).T[columns]

    return monthly_table, period_summary


def compare_with_paper_q24(monthly_table):
    mine = monthly_table.reindex(index=PAPER_Q24.index, columns=PAPER_Q24.columns)
    difference = mine - PAPER_Q24
    a = mine[MONTH_NAMES].to_numpy().ravel()
    b = PAPER_Q24[MONTH_NAMES].to_numpy().ravel()
    valid = ~np.isnan(a) & ~np.isnan(b)
    correlation = np.corrcoef(a[valid], b[valid])[0, 1] if valid.sum() > 2 else np.nan
    # The yearly figure is only comparable if both tables have all 12 months for that year
    partial = mine[MONTH_NAMES].isna().any(axis=1) | PAPER_Q24[MONTH_NAMES].isna().any(axis=1)
    difference.loc[partial, "Yearly"] = np.nan
    return difference.dropna(how="all"), correlation
