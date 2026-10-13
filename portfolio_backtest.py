#!/usr/bin/env python3
"""
Portfolio Backtesting Model
===========================

Builds a portfolio of 10-20 stocks, simulates its historical performance with
the chosen weighting and rebalancing rules, and compares it against a benchmark
(QQQ by default).

Reported metrics
----------------
Initial investment, final portfolio value, benchmark final value, total return,
CAGR, Sharpe ratio, max drawdown, alpha, beta, outperformance, and an overall
investment grade (Excellent / Good / Average / Poor).

Usage
-----
    pip install -r requirements.txt
    python portfolio_backtest.py

Edit the "CUSTOM PORTFOLIO INPUTS" section below to change the portfolio.

Methodology notes
-----------------
* Prices are split- and dividend-adjusted closes from Yahoo Finance, so returns
  are total returns (dividends reinvested).
* The portfolio is simulated by holding share counts day to day. Weights drift
  with prices between rebalances and are reset to target on the first trading
  day of each new month / quarter / year.
* Market-cap weighting uses each company's *current* shares outstanding times
  the price on each rebalance date. Historical share counts are not available
  from Yahoo, so this is an approximation (buybacks/issuance are ignored).
* Sharpe ratio and beta use daily returns, annualized with 252 trading days.
* Alpha is Jensen's alpha: the annualized intercept of a regression of the
  portfolio's daily excess returns on the benchmark's daily excess returns.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - dependency guard
    sys.exit("Missing dependency: install with `pip install yfinance pandas numpy`.")


# =============================================================================
# CUSTOM PORTFOLIO INPUTS
# =============================================================================

# Portfolio tickers (10-20 stocks)
TICKERS = [
    "AAPL", "MSFT", "NVDA", "HOOD", "GOOG", "META", "AVGO",
    "TSLA", "SPCX", "MU", "ORCL", "PLTR", "AMZN", "SNDK",
]

# Benchmark ticker. QQQ (Nasdaq-100) suits a tech-heavy portfolio like the one
# above; use "SPY" for a broad-market comparison.
BENCHMARK = "QQQ"

# Backtest period (YYYY-MM-DD). Set END_DATE = None to run up to today.
START_DATE = "2020-01-01"
END_DATE = None

# Starting capital in dollars
INITIAL_INVESTMENT = 10_000.00

# Weighting method: "equal", "market_cap", or "custom"
WEIGHTING_METHOD = "equal"

# Only used when WEIGHTING_METHOD = "custom". Weights are normalized to sum
# to 1, so you may enter percentages (e.g. 15) or fractions (e.g. 0.15).
CUSTOM_WEIGHTS = {
    "AAPL": 10, "MSFT": 10, "NVDA": 10, "HOOD": 4, "GOOG": 9,
    "META": 9, "AVGO": 8, "TSLA": 6, "SPCX": 3, "MU": 5, "ORCL": 5,
    "PLTR": 6, "AMZN": 9, "SNDK": 6,
}

# Rebalancing frequency: "none", "monthly", "quarterly", or "yearly"
REBALANCE_FREQUENCY = "quarterly"

# Annual risk-free rate as a decimal (0.04 = 4%)
RISK_FREE_RATE = 0.04

# Transaction cost per rebalance trade, in basis points of traded value
# (0 = frictionless). Applied to the initial purchase and every rebalance.
TRANSACTION_COST_BPS = 0.0

# What to do when a ticker has no price data at START_DATE (e.g. a recent IPO):
#   "align" - move the start date forward until every ticker has data
#   "drop"  - remove those tickers and keep the original start date
MISSING_DATA_POLICY = "align"

# =============================================================================
# END OF INPUTS
# =============================================================================


MIN_STOCKS, MAX_STOCKS = 10, 20
TRADING_DAYS = 252
VALID_WEIGHTING = {"equal", "market_cap", "custom"}
REBALANCE_RULES = {"none": None, "monthly": "M", "quarterly": "Q", "yearly": "Y"}


@dataclass
class Config:
    tickers: list[str]
    benchmark: str
    start: date
    end: date
    initial_investment: float
    weighting: str
    custom_weights: dict[str, float]
    rebalance: str
    risk_free_rate: float
    cost_bps: float
    missing_policy: str


@dataclass
class Metrics:
    final_value: float
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float


# -----------------------------------------------------------------------------
# Configuration & validation
# -----------------------------------------------------------------------------

def _parse_date(value: str | date | None, default: date) -> date:
    if value is None:
        return default
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid date {value!r}; expected YYYY-MM-DD.") from exc


def build_config() -> Config:
    tickers = list(dict.fromkeys(t.strip().upper() for t in TICKERS if t.strip()))
    benchmark = BENCHMARK.strip().upper()
    weighting = WEIGHTING_METHOD.strip().lower()
    rebalance = REBALANCE_FREQUENCY.strip().lower()
    policy = MISSING_DATA_POLICY.strip().lower()
    start = _parse_date(START_DATE, date.today())
    end = _parse_date(END_DATE, date.today())

    if not MIN_STOCKS <= len(tickers) <= MAX_STOCKS:
        warn(f"Portfolio has {len(tickers)} tickers; {MIN_STOCKS}-{MAX_STOCKS} is recommended.")
    if not tickers:
        raise ValueError("TICKERS is empty.")
    if benchmark in tickers:
        warn(f"Benchmark {benchmark} is also a portfolio holding.")
    if start >= end:
        raise ValueError(f"START_DATE ({start}) must be before END_DATE ({end}).")
    if INITIAL_INVESTMENT <= 0:
        raise ValueError("INITIAL_INVESTMENT must be positive.")
    if weighting not in VALID_WEIGHTING:
        raise ValueError(f"WEIGHTING_METHOD must be one of {sorted(VALID_WEIGHTING)}.")
    if rebalance not in REBALANCE_RULES:
        raise ValueError(f"REBALANCE_FREQUENCY must be one of {sorted(REBALANCE_RULES)}.")
    if policy not in {"align", "drop"}:
        raise ValueError('MISSING_DATA_POLICY must be "align" or "drop".')
    if not -0.05 <= RISK_FREE_RATE <= 0.25:
        raise ValueError("RISK_FREE_RATE looks wrong; enter it as a decimal (0.04 = 4%).")
    if TRANSACTION_COST_BPS < 0:
        raise ValueError("TRANSACTION_COST_BPS cannot be negative.")

    custom = {k.strip().upper(): float(v) for k, v in CUSTOM_WEIGHTS.items()}
    if weighting == "custom":
        missing = [t for t in tickers if t not in custom]
        extra = [t for t in custom if t not in tickers]
        if missing:
            raise ValueError(f"CUSTOM_WEIGHTS is missing weights for: {', '.join(missing)}")
        if extra:
            warn(f"Ignoring CUSTOM_WEIGHTS for tickers not in TICKERS: {', '.join(extra)}")
        if any(custom[t] < 0 for t in tickers):
            raise ValueError("CUSTOM_WEIGHTS cannot be negative (no short positions).")
        if sum(custom[t] for t in tickers) <= 0:
            raise ValueError("CUSTOM_WEIGHTS must sum to a positive number.")

    return Config(
        tickers=tickers,
        benchmark=benchmark,
        start=start,
        end=end,
        initial_investment=float(INITIAL_INVESTMENT),
        weighting=weighting,
        custom_weights=custom,
        rebalance=rebalance,
        risk_free_rate=float(RISK_FREE_RATE),
        cost_bps=float(TRANSACTION_COST_BPS),
        missing_policy=policy,
    )


def warn(message: str) -> None:
    print(f"  [WARNING] {message}")


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def download_prices(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Return adjusted close prices (columns = tickers) from Yahoo Finance."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    if raw is None or raw.empty:
        raise RuntimeError("No price data returned. Check tickers, dates, and your connection.")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:  # single ticker
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.reindex(columns=tickers)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.sort_index()


def clean_prices(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Drop empty tickers and handle late-starting histories per the policy."""
    stock_cols = [t for t in cfg.tickers if t in prices.columns]

    no_data = [t for t in stock_cols if prices[t].dropna().empty]
    if no_data:
        warn(f"No price data for {', '.join(no_data)} - removed from the portfolio.")
        stock_cols = [t for t in stock_cols if t not in no_data]
    if prices[cfg.benchmark].dropna().empty:
        raise RuntimeError(f"No price data for benchmark {cfg.benchmark}.")
    if not stock_cols:
        raise RuntimeError("None of the portfolio tickers returned price data.")

    first_valid = {t: prices[t].first_valid_index() for t in stock_cols + [cfg.benchmark]}
    first_date = prices.index[0]
    late = {t: d for t, d in first_valid.items() if d > first_date}

    if late:
        late_str = ", ".join(f"{t} (from {d.date()})" for t, d in sorted(late.items(), key=lambda x: x[1]))
        if cfg.missing_policy == "drop":
            late_stocks = [t for t in late if t != cfg.benchmark]
            if late_stocks:
                warn(f"Data starts late for {late_str}. Dropping: {', '.join(late_stocks)}.")
            stock_cols = [t for t in stock_cols if t not in late]
            if not stock_cols:
                raise RuntimeError("Every ticker starts after START_DATE; try MISSING_DATA_POLICY = 'align'.")
        new_start = max(first_valid[t] for t in stock_cols + [cfg.benchmark])
        if new_start > first_date:
            warn(f"Data starts late for {late_str}. Backtest start moved to {new_start.date()}.")
        prices = prices.loc[new_start:]

    prices = prices[stock_cols + [cfg.benchmark]].ffill().dropna()
    if len(prices) < 2:
        raise RuntimeError("Not enough overlapping price history to run a backtest.")
    if len(prices) < TRADING_DAYS:
        warn(f"Only {len(prices)} trading days of data; annualized metrics will be noisy.")
    if len(stock_cols) < MIN_STOCKS:
        warn(f"Portfolio now holds {len(stock_cols)} stocks (below the {MIN_STOCKS} minimum).")
    return prices


def fetch_shares_outstanding(tickers: list[str]) -> pd.Series:
    """Current shares outstanding per ticker, used for market-cap weighting."""
    shares = {}
    for t in tickers:
        value = None
        try:
            value = yf.Ticker(t).fast_info.get("shares")
        except Exception:
            pass
        if not value:
            try:
                value = yf.Ticker(t).info.get("sharesOutstanding")
            except Exception:
                pass
        if not value:
            raise RuntimeError(
                f"Could not get shares outstanding for {t}; market-cap weighting unavailable. "
                "Use 'equal' or 'custom' weighting instead."
            )
        shares[t] = float(value)
    return pd.Series(shares)


# -----------------------------------------------------------------------------
# Portfolio simulation
# -----------------------------------------------------------------------------

def target_weights(cfg: Config, tickers: list[str], prices_today: pd.Series,
                   shares_out: pd.Series | None) -> pd.Series:
    if cfg.weighting == "equal":
        w = pd.Series(1.0, index=tickers)
    elif cfg.weighting == "custom":
        w = pd.Series({t: cfg.custom_weights[t] for t in tickers})
    else:  # market_cap
        w = shares_out[tickers] * prices_today[tickers]
    total = w.sum()
    if total <= 0:
        raise RuntimeError("Target weights sum to zero.")
    return w / total


def rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> set[pd.Timestamp]:
    """First trading day of each new period (excluding the very first day)."""
    rule = REBALANCE_RULES[frequency]
    if rule is None:
        return set()
    periods = index.to_period(rule)
    is_new = periods != np.roll(periods, 1)
    is_new[0] = False
    return set(index[is_new])


def simulate_portfolio(prices: pd.DataFrame, cfg: Config, tickers: list[str],
                       shares_out: pd.Series | None) -> tuple[pd.Series, int]:
    """Simulate daily portfolio value. Returns (value series, number of rebalances)."""
    cost_rate = cfg.cost_bps / 10_000
    px = prices[tickers]
    rebal_days = rebalance_dates(px.index, cfg.rebalance)

    first = px.iloc[0]
    weights = target_weights(cfg, tickers, first, shares_out)
    investable = cfg.initial_investment * (1 - cost_rate)
    holdings = investable * weights / first

    values = np.empty(len(px))
    rebalances = 0
    for i, (day, row) in enumerate(px.iterrows()):
        value = float((holdings * row).sum())
        if day in rebal_days:
            weights = target_weights(cfg, tickers, row, shares_out)
            target_value = value * weights
            turnover = float((target_value - holdings * row).abs().sum())
            value -= turnover * cost_rate
            holdings = value * weights / row
            rebalances += 1
        values[i] = value

    return pd.Series(values, index=px.index, name="Portfolio"), rebalances


def simulate_benchmark(prices: pd.DataFrame, cfg: Config) -> pd.Series:
    series = prices[cfg.benchmark]
    investable = cfg.initial_investment * (1 - cfg.cost_bps / 10_000)
    return (investable * series / series.iloc[0]).rename("Benchmark")


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def daily_rf(annual_rate: float) -> float:
    return (1 + annual_rate) ** (1 / TRADING_DAYS) - 1


def compute_metrics(values: pd.Series, initial: float, rf: float) -> Metrics:
    returns = values.pct_change().dropna()
    final = float(values.iloc[-1])
    total_return = final / initial - 1

    years = (values.index[-1] - values.index[0]).days / 365.25
    cagr = (final / initial) ** (1 / years) - 1 if years > 0 and final > 0 else float("nan")

    excess = returns - daily_rf(rf)
    std = excess.std(ddof=1)
    sharpe = float(excess.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else float("nan")
    volatility = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))

    running_peak = values.cummax()
    max_dd = float((values / running_peak - 1).min())

    return Metrics(final, total_return, cagr, volatility, sharpe, max_dd)


def alpha_beta(portfolio: pd.Series, benchmark: pd.Series, rf: float) -> tuple[float, float]:
    """Jensen's alpha (annualized) and beta from daily excess returns."""
    r_p = portfolio.pct_change().dropna() - daily_rf(rf)
    r_b = benchmark.pct_change().dropna() - daily_rf(rf)
    r_p, r_b = r_p.align(r_b, join="inner")
    var_b = r_b.var(ddof=1)
    if len(r_b) < 2 or var_b == 0:
        return float("nan"), float("nan")
    beta = float(r_p.cov(r_b) / var_b)
    daily_alpha = r_p.mean() - beta * r_b.mean()
    alpha = float((1 + daily_alpha) ** TRADING_DAYS - 1)
    return alpha, beta


# -----------------------------------------------------------------------------
# Investment grade
# -----------------------------------------------------------------------------

def _score(value: float, thresholds: tuple[float, float, float]) -> int:
    """0-3 points: one point for each threshold the value meets or beats."""
    if np.isnan(value):
        return 0
    return sum(value >= t for t in thresholds)


def investment_grade(port: Metrics, bench: Metrics, alpha: float) -> tuple[str, list[tuple[str, int]]]:
    """
    Score four dimensions 0-3 each (max 12):
      Risk-adjusted return  Sharpe ratio             >= 0.5 / 1.0 / 1.5
      Skill vs. benchmark   Alpha                    >= -2% / 0% / +5%
      Relative growth       CAGR minus bench CAGR    >= -2% / 0% / +3%
      Downside protection   Max drawdown             >= -50% / -35% / -20%
    Grade: 10-12 Excellent, 7-9 Good, 4-6 Average, 0-3 Poor.
    """
    breakdown = [
        ("Sharpe ratio", _score(port.sharpe, (0.5, 1.0, 1.5))),
        ("Alpha", _score(alpha, (-0.02, 0.0, 0.05))),
        ("CAGR vs. benchmark", _score(port.cagr - bench.cagr, (-0.02, 0.0, 0.03))),
        ("Max drawdown", _score(port.max_drawdown, (-0.50, -0.35, -0.20))),
    ]
    total = sum(points for _, points in breakdown)
    if total >= 10:
        grade = "Excellent"
    elif total >= 7:
        grade = "Good"
    elif total >= 4:
        grade = "Average"
    else:
        grade = "Poor"
    return grade, breakdown


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def money(x: float) -> str:
    return f"${x:,.2f}"


def pct(x: float, signed: bool = False) -> str:
    if np.isnan(x):
        return "n/a"
    return f"{x:+.2%}" if signed else f"{x:.2%}"


def num(x: float) -> str:
    return "n/a" if np.isnan(x) else f"{x:.2f}"


def print_report(cfg: Config, tickers: list[str], weights: pd.Series, prices: pd.DataFrame,
                 port: Metrics, bench: Metrics, alpha: float, beta: float,
                 rebalances: int, grade: str, breakdown: list[tuple[str, int]]) -> None:
    line = "=" * 64
    sub = "-" * 64
    start, end = prices.index[0].date(), prices.index[-1].date()
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    outperformance = port.total_return - bench.total_return

    print(f"\n{line}\n{'PORTFOLIO BACKTEST REPORT':^64}\n{line}")
    print(f"  Period            : {start} to {end} ({years:.1f} years, {len(prices)} trading days)")
    print(f"  Benchmark         : {cfg.benchmark}")
    print(f"  Weighting         : {cfg.weighting.replace('_', ' ').title()}")
    print(f"  Rebalancing       : {cfg.rebalance.title()} ({rebalances} rebalances)")
    print(f"  Risk-free rate    : {cfg.risk_free_rate:.2%}")
    if cfg.cost_bps:
        print(f"  Transaction cost  : {cfg.cost_bps:g} bps")

    print(f"\n  Starting allocation ({len(tickers)} stocks):")
    for t, w in weights.sort_values(ascending=False).items():
        print(f"    {t:<8}{w:>8.2%}")

    print(f"\n{sub}\n  {'RESULTS':<30}\n{sub}")
    rows = [
        ("Initial Investment", money(cfg.initial_investment)),
        ("Final Portfolio Value", money(port.final_value)),
        ("Benchmark Final Value", money(bench.final_value)),
        ("Total Return %", pct(port.total_return)),
        ("CAGR", pct(port.cagr)),
        ("Sharpe Ratio", num(port.sharpe)),
        ("Max Drawdown", pct(port.max_drawdown)),
        ("Alpha (annualized)", pct(alpha, signed=True)),
        ("Beta", num(beta)),
        ("Outperformance %", pct(outperformance, signed=True)),
        ("Investment Grade", grade.upper()),
    ]
    for label, value in rows:
        print(f"  {label:<30}{value:>20}")

    print(f"\n{sub}\n  {'PORTFOLIO vs. BENCHMARK':<30}{'Portfolio':>15}{cfg.benchmark:>15}\n{sub}")
    comparison = [
        ("Final Value", money(port.final_value), money(bench.final_value)),
        ("Total Return", pct(port.total_return), pct(bench.total_return)),
        ("CAGR", pct(port.cagr), pct(bench.cagr)),
        ("Volatility (annualized)", pct(port.volatility), pct(bench.volatility)),
        ("Sharpe Ratio", num(port.sharpe), num(bench.sharpe)),
        ("Max Drawdown", pct(port.max_drawdown), pct(bench.max_drawdown)),
    ]
    for label, p, b in comparison:
        print(f"  {label:<30}{p:>15}{b:>15}")

    print(f"\n{sub}\n  INVESTMENT GRADE BREAKDOWN (0-3 points each)\n{sub}")
    for label, points in breakdown:
        print(f"  {label:<30}{'#' * points:<4}{points}/3")
    total = sum(p for _, p in breakdown)
    print(f"  {'Total':<30}{'':<4}{total}/12  ->  {grade}")
    print("  Scale: 10-12 Excellent | 7-9 Good | 4-6 Average | 0-3 Poor")
    print(f"{line}\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run() -> None:
    cfg = build_config()
    print(f"Downloading prices for {len(cfg.tickers)} stocks + {cfg.benchmark} "
          f"({cfg.start} to {cfg.end})...")

    raw = download_prices(cfg.tickers + [cfg.benchmark], cfg.start, cfg.end)
    prices = clean_prices(raw, cfg)
    tickers = [c for c in prices.columns if c != cfg.benchmark]

    shares_out = None
    if cfg.weighting == "market_cap":
        print("Fetching shares outstanding for market-cap weights...")
        shares_out = fetch_shares_outstanding(tickers)

    start_weights = target_weights(cfg, tickers, prices[tickers].iloc[0], shares_out)
    port_values, rebalances = simulate_portfolio(prices, cfg, tickers, shares_out)
    bench_values = simulate_benchmark(prices, cfg)

    port = compute_metrics(port_values, cfg.initial_investment, cfg.risk_free_rate)
    bench = compute_metrics(bench_values, cfg.initial_investment, cfg.risk_free_rate)
    alpha, beta = alpha_beta(port_values, bench_values, cfg.risk_free_rate)
    grade, breakdown = investment_grade(port, bench, alpha)

    print_report(cfg, tickers, start_weights, prices, port, bench,
                 alpha, beta, rebalances, grade, breakdown)


if __name__ == "__main__":
    try:
        run()
    except (ValueError, RuntimeError) as exc:
        sys.exit(f"\nError: {exc}")
