# Portfolio Analysis & Backtesting Model

A single Python script (`portfolio_backtest.py`) that builds a 10-20 stock portfolio,
backtests it against a benchmark (QQQ by default), and reports performance and risk metrics.

## Quick start

```bash
pip install -r requirements.txt
python portfolio_backtest.py
```

Edit the **CUSTOM PORTFOLIO INPUTS** section at the top of the script:

| Input | Options |
|---|---|
| `TICKERS` | 10-20 stock symbols |
| `BENCHMARK` | Default `QQQ` (use `SPY` for broad market) |
| `START_DATE` / `END_DATE` | `YYYY-MM-DD` (`END_DATE = None` → today) |
| `INITIAL_INVESTMENT` | Starting capital in dollars |
| `WEIGHTING_METHOD` | `equal`, `market_cap`, `custom` (+ `CUSTOM_WEIGHTS`) |
| `REBALANCE_FREQUENCY` | `none`, `monthly`, `quarterly`, `yearly` |
| `RISK_FREE_RATE` | Annual rate as a decimal (`0.04` = 4%) |
| `TRANSACTION_COST_BPS` | Optional trading cost per rebalance |
| `MISSING_DATA_POLICY` | `align` (shift start date) or `drop` (remove late-listed tickers) |

## Output

Initial investment, final portfolio value, benchmark final value, total return %, CAGR,
Sharpe ratio, max drawdown, alpha, beta, outperformance %, and an investment grade
(Excellent / Good / Average / Poor), plus a side-by-side comparison with the benchmark
and a breakdown of how the grade was scored.

## Methodology

- Split- and dividend-adjusted prices from Yahoo Finance (total return).
- Share-count simulation: weights drift between rebalances and reset on the first
  trading day of each new month / quarter / year.
- Sharpe and beta from daily returns, annualized with 252 trading days.
- Alpha is Jensen's alpha (annualized regression intercept on excess returns).
- Outperformance = portfolio total return − benchmark total return (percentage points).
- Market-cap weights use *current* shares outstanding × price on each rebalance date
  (historical share counts aren't available from Yahoo).
- Investment grade scores Sharpe, alpha, CAGR vs. benchmark and max drawdown 0-3 points
  each: 10-12 Excellent, 7-9 Good, 4-6 Average, 0-3 Poor.
