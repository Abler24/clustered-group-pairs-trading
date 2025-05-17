# plot_equity_vs_benchmarks.py
#
# Run *after* backtest.py so portfolioEquity is already in memory,
# OR import the equity curve from a saved CSV / pickle if you prefer.

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

###############################################################################
# 1. Grab (or load) your strategy equity curve
###############################################################################
try:
    # Option A – import the equity Series that backtest.py wrote to disk
    # (Uncomment if you saved it, otherwise comment this block out.)
    # portfolioEquity = pd.read_csv("portfolio_equity.csv", index_col=0, parse_dates=True)["equity"]

    # Option B – if you ran backtest.py in the *same* session and still have
    #           the variable in memory, you can import it directly:
    from backtest import portfolioEquity
except Exception:
    raise RuntimeError(
        "Couldn’t find portfolioEquity. Run backtest.py first or save / load the "
        "equity curve (see Option A)."
    )

###############################################################################
# 2. Download benchmark prices
###############################################################################
bench_tickers = {
    "SPY":  "S&P‑500",
    "IEF":  "Treasury Bond (7‑10 yr)",
    "GLD":  "Gold",
    "QAI": "Hedge Fund Multi-Strategy",
}

start = portfolioEquity.index.min().strftime("%Y-%m-%d")
end   = portfolioEquity.index.max().strftime("%Y-%m-%d")

#
# yfinance switched its default to auto_adjust=True which removes the
# multi‑index "Adj Close" level.  Force auto_adjust=False so we always
# get the familiar multi‑index format.
bench_prices = (
    yf.download(
        list(bench_tickers.keys()),
        start=start,
        end=end,
        auto_adjust=False,   # keep "Adj Close" column
        progress=False,
    )["Adj Close"]
      .dropna(how="all")
      .ffill()
)

###############################################################################
# 3. Normalise everything to 100 on the first date
###############################################################################
equity_norm = 100 * portfolioEquity / portfolioEquity.iloc[0]
bench_norm  = 100 * bench_prices / bench_prices.iloc[0]

###############################################################################
# 4. Plot
###############################################################################
plt.figure(figsize=(12, 6))
plt.plot(equity_norm, label="Strategy", linewidth=2)

for ticker, label in bench_tickers.items():
    if ticker in bench_norm.columns:
        plt.plot(bench_norm[ticker], label=label, alpha=0.7)

plt.title("Equity Curve vs. Benchmarks")
plt.ylabel("Growth of $100")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save &/or show
plt.savefig("equity_vs_benchmarks.png", dpi=300)
plt.show()