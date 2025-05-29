import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import matplotlib.pyplot as plt

# Global Configuration

dataPath            = "stock_data.parquet"
clusterMapPath      = "selected_clusters.csv"    # produced by validate.py

startDate = "2023-01-20"
endDate   = "2025-04-20"

entryThres   = 2
exitThres    = 0
rollingWindow = 60
targetVolPct  = 0.02        # target daily return volatility per cluster
capitalPerGroup = 100000
totalCapital   = 1000000  #capital available for all clusters
tradingDays    = 252   # annual trading days for Sharpe calculation
latencyDays    = 1     # simulate order execution delay (trading days)
impactBps      = 0.002 # 10 bps per trade for market impact
stopLossPct    = 0.05    # 5% trailing drawdown stop-out per cluster
volSlippageWindow = 30      # days to estimate realized volatility for slippage
volSlippageMult   = 1     # multiplier for vol-based slippage adjustment

# 0b. Realistic Trading Costs

commissionBps   = 0.0005   # 5 bps per trade (round‑trip)
slippageBps     = 0.001   # 5 bps per side

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
    """Compute spread and z-score using best-y-by-R² selection"""
    sub_df = priceData[tickers].dropna()
    if len(sub_df) < winLen or len(tickers) < 2:
        return None, None

    logPrices = np.log(sub_df)

    # --- 1. Choose best y by R²
    best_y_idx = 0
    best_r2 = -np.inf
    for i in range(len(tickers)):
        y = logPrices.iloc[:, i]
        X = sm.add_constant(logPrices.drop(columns=logPrices.columns[i]))
        try:
            model = sm.OLS(y, X).fit()
            r2 = model.rsquared
            if r2 > best_r2:
                best_r2 = r2
                best_y_idx = i
        except Exception:
            continue

    # --- 2. Build spread
    spread = pd.Series(index=logPrices.index, dtype=float)
    for i in range(winLen - 1, len(logPrices)):
        window_data = logPrices.iloc[i - winLen + 1 : i + 1]
        y = window_data.iloc[:, best_y_idx]
        X = window_data.drop(columns=window_data.columns[best_y_idx])
        X = sm.add_constant(X)
        hedge = sm.OLS(y, X).fit().params

        row = logPrices.iloc[i]
        X_now = row.drop(labels=row.index[best_y_idx])
        X_now = np.insert(X_now.values.reshape(1, -1), 0, 1, axis=1)
        spread.iloc[i] = row.iloc[best_y_idx] - np.dot(X_now, hedge.values).item()

    spread = spread.dropna()
    zScore = (spread - spread.expanding(min_periods=2).mean()) / spread.expanding(min_periods=2).std()
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

def plot_cluster_signals(cluster, tickers, priceData, zScore, signals,
                          entryThres=2, exitThres=0):
    """
    Plot log price of one anchor stock versus the equal weighted average of the
    rest of the cluster, together with the z score and long/short signals.

    * The first ticker in `tickers` is treated as the anchor (asset i).
    * The equal weighted average of all tickers is treated as the basket (asset j).
    * Long signals are highlighted in green, short signals in red  on BOTH series.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    anchor = tickers[0]                           # “Asset i”
    log_px_anchor = np.log(priceData[anchor].loc[zScore.index]).dropna()
    log_px_group  = np.log(priceData[tickers].mean(axis=1).loc[zScore.index]).dropna()

    # Align the three series to the same dates
    common_idx = log_px_anchor.index.intersection(log_px_group.index).intersection(zScore.index)
    log_px_anchor = log_px_anchor.loc[common_idx]
    log_px_group  = log_px_group.loc[common_idx]
    zScore        = zScore.loc[common_idx]
    signals       = signals.loc[common_idx]

    # Long/short masks
    long_mask  = signals == 1          # long anchor, short group
    short_mask = signals == -1         # short anchor, long group

    fig, ax_price = plt.subplots(figsize=(14, 8))

    # Price series
    ax_price.plot(common_idx, log_px_anchor, label=f'Log Price: {anchor}', linewidth=1.5)
    ax_price.plot(common_idx, log_px_group,  label='Log Price: Group Avg', linewidth=1.5)

    # Signal markers (green = long, red = short)
    ax_price.scatter(common_idx[long_mask],  log_px_anchor[long_mask],
                     color='green', marker='o', s=45, label=f'Long {anchor}')
    ax_price.scatter(common_idx[short_mask], log_px_anchor[short_mask],
                     color='red',   marker='o', s=45, label=f'Short {anchor}')

    ax_price.scatter(common_idx[short_mask], log_px_group[short_mask],
                     color='green', marker='o', s=45, label='Long Group')
    ax_price.scatter(common_idx[long_mask],  log_px_group[long_mask],
                     color='red',   marker='o', s=45, label='Short Group')

    # Z‑score on secondary axis
    ax_z = ax_price.twinx()
    ax_z.plot(common_idx, zScore, color='grey', linestyle='--', linewidth=1.2, label='Z‑Score')
    ax_z.axhline(entryThres,  linestyle='--', linewidth=1, color='grey')
    ax_z.axhline(-entryThres, linestyle='--', linewidth=1, color='grey')
    ax_z.axhline(exitThres,   linestyle='-.', linewidth=1, color='grey')

    # Formatting
    ax_price.set_title(f'Cluster {cluster}: Log Prices & Z‑Score with Trade Signals', fontsize=14)
    ax_price.set_xlabel('Date', fontsize=12)
    ax_price.set_ylabel('Log Price', fontsize=12)
    ax_z.set_ylabel('Z‑Score', fontsize=12)
    ax_price.grid(True, linestyle='--', alpha=0.5)

    # Combine legends from both axes
    lines1, labels1 = ax_price.get_legend_handles_labels()
    lines2, labels2 = ax_z.get_legend_handles_labels()
    ax_price.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

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
signalsByCluster = {}   # store daily position series for each cluster
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

    # Visualize price series, z‑score, and trade signals
    #plot_cluster_signals(cluster, tickers, priceData, zScore, signals,
    #                    entryThres=entryThres, exitThres=exitThres)

    signalsByCluster[cluster] = signals
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

# 6. Portfolio Construction w inverse volatility weighting
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

    # --- Final Metrics ---
    annReturn = portfolio_returns.mean() * tradingDays
    annVol    = portfolio_returns.std() * np.sqrt(tradingDays)
    hitRate   = (portfolio_returns > 0).sum() / len(portfolio_returns)
    maxDuration = (portfolioEquity != portfolioEquity.cummax()).astype(int).groupby((portfolioEquity == portfolioEquity.cummax()).cumsum()).cumsum().max()

    # Average holding period per trade
    total_days_in_position = sum((sig != 0).sum() for sig in signalsByCluster.values())
    total_trades = sum((sig.diff().abs() == 1).sum() for sig in signalsByCluster.values())
    avgHoldingPeriod = total_days_in_position / total_trades if total_trades else 0

    # Load SPY as benchmark
    try:
        if 'SPY' in prices.columns:
            spy = prices['SPY'].loc[portfolio_returns.index]
        else:
            spy = yf.download('SPY', start=startDate, end=endDate)['Adj Close'].reindex(portfolio_returns.index).fillna(method='ffill')
        spy_ret = np.log(spy / spy.shift(1)).fillna(0)
        benchmark_returns = spy_ret.reindex_like(portfolio_returns).fillna(0)
        active_ret = portfolio_returns - benchmark_returns
        infoRatio = active_ret.mean() / active_ret.std() * np.sqrt(tradingDays)
    except Exception as e:
        print(f"Failed to compute Information Ratio: {e}")
        infoRatio = np.nan

    print()
    print(f"Annualized Return:       {annReturn * 100:.2f}%")
    print(f"Annualized Volatility:   {annVol * 100:.2f}%")
    print(f"Sharpe Ratio:            {portSharp:.2f}")
    print(f"Hit Rate:                {hitRate * 100:.2f}%")
    print(f"Max Drawdown:            {portDrawdown * .0001:.2f}%")
    print(f"Max Drawdown Duration:   {maxDuration} days")
    print(f"Average Holding Period:  {avgHoldingPeriod:.2f} days")
    print()

portfolioEquity.to_csv("portfolio_equity.csv")