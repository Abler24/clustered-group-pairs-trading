import pandas as pd
import numpy as np
import statsmodels.api as sm

# Global Configuration

dataPath            = "stock_data.parquet"
clusterMapPath      = "selected_clusters.csv"    # produced by validate.py

startDate = "2023-01-20"
endDate   = "2025-04-20"

entryThres   = 2
exitThres    = 0
rollingWindow = 60
targetVolPct  = 0.01        # target daily return volatility per cluster
capitalPerGroup = 100000
totalCapital   = 1000000  #capital available for all clusters
tradingDays    = 252   # annual trading days for Sharpe calculation
latencyDays    = 1     # simulate order execution delay (trading days)
impactBps      = 0.001 # 10 bps per trade for market impact
stopLossPct    = 0.05    # 5% trailing drawdown stop-out per cluster
volSlippageWindow = 20      # days to estimate realized volatility for slippage
volSlippageMult   = 0.9     # multiplier for vol-based slippage adjustment

# 0b. Realistic Trading Costs

commissionBps   = 0.0005   # 5 bps per trade (round‑trip)
slippageBps     = 0.0005   # 5 bps per side

# 1. Data Loading

try:
    prices = pd.read_parquet(dataPath)
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices.ffill().ffill()
    priceData  = prices.loc[startDate:endDate]
    print(f"Loaded data: {priceData.index[0]} to {priceData.index[-1]}")
except Exception as e:
    raise RuntimeError(f"Error loading price data: {e}")

try:
    clusterMap = pd.read_csv(clusterMapPath)
    clusterMap["cluster"] = clusterMap["cluster"].astype(int)
    print("Selected cluster mapping loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Error loading cluster mapping: {e}")

clusterGroups = clusterMap.groupby("cluster")["ticker"].apply(list).to_dict()
print(f"Clusters: {list(clusterGroups.keys())}")

# 2. Basket Construction (identical to validate)
def construct_basket_spread(tickers, priceData, winLen=rollingWindow):
    """Compute spread and z-score for tickers."""
    sub_df = priceData[tickers].dropna()
    if len(sub_df) < winLen:
        return None, None

    logPrices = np.log(sub_df)
    spread     = pd.Series(index=logPrices.index, dtype=float)

    for i in range(winLen - 1, len(logPrices)):
        window_data = logPrices.iloc[i - winLen + 1 : i + 1]
        y = window_data.iloc[:, 0]
        X = sm.add_constant(window_data.iloc[:, 1:])
        hedge = sm.OLS(y, X).fit().params

        row   = logPrices.iloc[i]
        X_now = np.insert(row[1:].values.reshape(1, -1), 0, 1, axis=1)
        spread.iloc[i] = row.iloc[0] - np.dot(X_now, hedge.values).item()

    spread = spread.dropna()
    zScore  = (spread - spread.expanding(min_periods=2).mean()) / spread.expanding(min_periods=2).std()
    return spread, zScore

# 3. Signal Generation (identical to validate)
def generate_signals(zScore, entryThres=entryThres, exitThres=exitThres):
    sig, position = pd.Series(0, zScore.index), 0
    for t in range(1, len(zScore)):
        z = zScore.iloc[t]
        if position == 0:
            if z < -entryThres:
                position = 1
            elif z > entryThres:
                position = -1
        else:
            if (position == 1 and z >= exitThres) or (position == -1 and z <= exitThres):
                position = 0
        sig.iloc[t] = position
    return sig

# 4. Backtest Basket (identical metric logic, with trading costs)
def backtest_basket(spread, sig):
    print(f"[backtest] Starting cluster with {len(spread)} points")
    spreadDelta   = spread.diff()
    # raw pnl
    realizedVol = spreadDelta.rolling(volSlippageWindow).std().shift(latencyDays).fillna(method='bfill')
    leverage = (targetVolPct * capitalPerGroup) / realizedVol.replace(0, np.nan)
    rawPnl = sig.shift(latencyDays).fillna(0) * spreadDelta * leverage
    # costs
    volSeries = spread.diff().rolling(volSlippageWindow).std()
    slipBps = slippageBps + volSeries.shift(latencyDays).fillna(0) * volSlippageMult
    trades = sig.diff().abs().shift(latencyDays).fillna(0)
    slipCost   = trades * slipBps * leverage
    commCost = trades * commissionBps * leverage
    impactCost     = trades * impactBps     * leverage
    # calc metrics
    pnl = rawPnl - slipCost - commCost - impactCost
    pnl = pnl.fillna(0)
    retSeries = pnl / capitalPerGroup

    cumPnl = retSeries.cumsum()
    if (cumPnl < -stopLossPct).any():
        stop_date = cumPnl[cumPnl < -stopLossPct].index[0]
        pnl.loc[stop_date:] = 0
        retSeries.loc[stop_date:] = 0
    cumPnl = pnl.cumsum()

    sharpRatio = pnl.mean() / pnl.std() * np.sqrt(tradingDays) if pnl.std() else 0
    drawdown = cumPnl - cumPnl.cummax()

    print(f"[backtestBasket] total PnL: {cumPnl.iloc[-1]:.2f}, Sharpe: {sharpRatio:.2f}")

    return {
        "pnl_series"     : pnl,
        "cum_pnl_series" : cumPnl,
        "sharpRatio"   : sharpRatio,
        "max_drawdown"   : drawdown.min(),
        "total_pnl"      : cumPnl.iloc[-1],
        "num_trades"     : sig.diff().abs().sum() / 2,
        "return_series"   : retSeries,
    }

# 5. Out‑of‑Sample Backtest Loop
resList = []
returnsByCluster = {}   # daily return series for each cluster
for cluster, tickers in clusterGroups.items():
    if len(tickers) < 2:
        print(f"Skipping cluster {cluster}: not enough tickers.")
        continue

    print(f"Backtesting cluster {cluster}: {tickers}")
    spread, zScore = construct_basket_spread(tickers, priceData)

    if spread is None:
        print("  → Insufficient data.")
        continue

    signals = generate_signals(zScore)
    metrics = backtest_basket(spread, signals)

    metrics.update({
        "cluster"        : cluster,
        "tickers"        : ",".join(tickers),
        "percentReturn" : metrics["total_pnl"] / capitalPerGroup * 100
    })
    cluster_returns = metrics.pop("return_series")
    returnsByCluster[cluster] = cluster_returns
    resList.append(metrics)

if not resList:
    print("No clusters were backtested.")
    exit()

print("Results summary:")
for r in resList:
    print(f" - {r['cluster']}: Sharpe={r['sharpRatio']:.2f}, #Trades={r['num_trades']:.2f}, Return={r['percentReturn']:.2f}%")

# 6. Portfolio Construction – dynamic inverse‑volatility weighting
if returnsByCluster:
    retsDf = pd.concat(returnsByCluster, axis=1).sort_index().fillna(0)   # columns = cluster IDs

    rebalanceDates = retsDf.resample("BQS-JAN").first().index

    lookback_days  = 60
    minWeight, maxWeight   = 0.05, 0.35   # tighter concentration cap (5%–35%)

    alloc = pd.DataFrame(index=retsDf.index,
                           columns=retsDf.columns,
                           dtype=float)

    currentWeights = pd.Series(1 / len(retsDf.columns), index=retsDf.columns)

    print(f"[portfolio] Rebalancing on {len(rebalanceDates)} dates")

    for current_date in retsDf.index:
        if current_date in rebalanceDates:
            window_loc = retsDf.index.get_loc(current_date)
            start_loc  = max(0, window_loc - lookback_days + 1)
            vol            = retsDf.iloc[start_loc:window_loc + 1].std()
            inv_vol        = 1 / vol.replace(0, np.nan)
            raw_w          = inv_vol / inv_vol.sum()
            clipped        = raw_w.clip(lower=minWeight, upper=maxWeight)
            currentWeights   = clipped / clipped.sum()
        alloc.loc[current_date] = currentWeights

    alloc = alloc.ffill()

    portfolio_returns = (alloc * retsDf).sum(axis=1)
    portfolioEquity  = totalCapital * (1 + portfolio_returns).cumprod()
    portfolioPnl     = portfolioEquity - totalCapital

    portSharp   = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(tradingDays)
    portDrawdown = (portfolioEquity - portfolioEquity.cummax()).min()

    print(f"Portfolio final equity: {portfolioEquity[-1]:.2f}, Sharpe {portSharp:.2f}, DrawDown {portDrawdown:.2f}")
portfolioEquity.to_csv("portfolio_equity.csv")