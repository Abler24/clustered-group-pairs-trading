import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.dates as mdates

# Load cointegrated pairs from `find.py`
# from find import cointegrated_pairs as COINTEGRATED_PAIRS
COINTEGRATED_PAIRS = [('CFG', 'KEY'), ('CFG', 'PNC'), ('CMA', 'KEY'), ('CMA', 'PNC'), ('CMA', 'RF'), ('ED', 'D'), ('COO', 'NVR'), ('CPRT', 'TJX'), ('CMI', 'PNR'), ('DHI', 'LIN'), ('DHI', 'PHM'), ('FANG', 'XOM')]
DATA_PATH = "stock_data.parquet"
df = pd.read_parquet(DATA_PATH)
df = df.ffill().bfill()  # Fill missing values to preserve full data range

# Define test period (2024) using data that was unseen during cointegration learning in find.py
TEST_START_DATE = "2024-01-01"
df_test = df.loc[TEST_START_DATE:]
print(f"Test data range: {df_test.index.min()} to {df_test.index.max()}")

# Function to compute spread and Z-score
def compute_spread_zscore(series1, series2, window=20):
    model = sm.OLS(series1, sm.add_constant(series2)).fit()
    beta = model.params[1]  # Optimal hedge ratio
    spread = series1 - beta * series2
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    z_score = (spread - mean) / std
    return spread, z_score, beta

# Backtest parameters
ENTRY_Z = 2.0  # Enter trade when Z-score > +1 or < -1
EXIT_Z = 0.0   # Exit trade when Z-score reverts to 0
INITIAL_CAPITAL = 100000  # Starting capital
TRADE_ALLOCATION = 0.5  # Allocate 5% of capital per trade

# Backtest function
def backtest_pair(stock1, stock2):
    s1 = df_test[stock1].squeeze()
    s2 = df_test[stock2].squeeze()
    spread, z_score, beta = compute_spread_zscore(s1, s2)

    # Calculate signals for annotation
    long_signal = (z_score < -ENTRY_Z)
    short_signal = (z_score > ENTRY_Z)
    exit_signal = (abs(z_score) < EXIT_Z)

    # Track capital and dynamic positions
    capital = INITIAL_CAPITAL
    position = np.zeros(len(df_test))
    capital_per_trade = capital * TRADE_ALLOCATION

    for i in range(1, len(df_test)):
        if long_signal.iloc[i]:
            position[i] = 1  # Long stock1, short stock2
        elif short_signal.iloc[i]:
            position[i] = -1  # Short stock1, long stock2
        elif exit_signal.iloc[i]:
            position[i] = 0   # Exit trade

    # Dynamic position sizing based on available capital
    s1_prices = s1.values
    s2_prices = s2.values
    shares_s1 = capital_per_trade / s1_prices[:-1]
    shares_s2 = (capital_per_trade / s2_prices[:-1]) * beta

    # Compute PnL as percentage of capital allocated per trade
    s1_returns = np.diff(s1_prices) * shares_s1
    s2_returns = np.diff(s2_prices) * shares_s2
    returns = position[:-1] * (s1_returns - s2_returns)

    # Update portfolio value dynamically
    portfolio_value = INITIAL_CAPITAL + returns.cumsum()

    # Compute percentage gain relative to invested capital
    final_pnl = portfolio_value[-1] - INITIAL_CAPITAL
    percentage_gain = (final_pnl / INITIAL_CAPITAL) * 100

    # Print Stats
    print(f"\n{stock1} - {stock2} Backtest Results:")
    print(f"Final PnL: ${final_pnl:.2f}")
    print(f"Percentage Gain: {percentage_gain:.2f}%")

    # **Plot Portfolio Performance**
    plt.figure(figsize=(12, 6))
    plt.plot(df_test.index[1:], portfolio_value, label="Portfolio Value", color='b')
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.title(f"Portfolio Performance: {stock1} - {stock2}")
    plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', label="Initial Capital")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.show()

    return final_pnl, percentage_gain

# Run backtest on first 5 pairs
pnl_results = {}
for stock1, stock2 in COINTEGRATED_PAIRS[:5]:  # Test first 5 pairs
    pnl, pct_gain = backtest_pair(stock1, stock2)
    pnl_results[(stock1, stock2)] = {"PnL": pnl, "Percentage Gain": pct_gain}

# Print final results
print("\nFinal Backtest Results:")
for pair, stats in pnl_results.items():
    print(f"{pair}: PnL: ${stats['PnL']:.2f}, Gain: {stats['Percentage Gain']:.2f}%")