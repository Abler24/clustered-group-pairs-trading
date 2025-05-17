import pandas as pd
import numpy as np
import statsmodels.api as sm

# 0. Global Configuration
dataPath = "stock_data.parquet"
clusterMapPath = "cluster_mapping.csv"   # raw candidate clusters from Phase 1

# Date window for validation
valStart = "2022-01-01"
valEnd   = "2022-12-31"

# Trading parameters
entryThres   = 1.8
exitThres    = 0.0
rollingWindow    = 20          # look‑back window for rolling OLS & z‑score
riskMultiplier   = 100_00    
capitalPerGroup = 100_000
tradingDays      = 252

try:
    fullPrices = pd.read_parquet(dataPath)
    fullPrices.index = pd.to_datetime(fullPrices.index, errors="coerce")
    fullPrices = fullPrices.ffill().ffill()
    priceData  = fullPrices.loc[valStart:valEnd]
    print(f"Loaded price data {priceData.index.min()} to {priceData.index.max()}")
except Exception as e:
    raise RuntimeError(f"Error loading price data: {e}")

try:
    clusterMap = pd.read_csv(clusterMapPath)
    clusterMap["cluster"] = clusterMap["cluster"].astype(int)
    print("Loaded cluster mapping")
except Exception as e:
    raise RuntimeError(f"Error loading cluster mapping: {e}")

clusterGroups = clusterMap.groupby("cluster")["ticker"].apply(list).to_dict()

def construct_basket_spread(tickers, priceFrame, windowLen=rollingWindow):
    """Return spread series and z-score series """
    sub_df = priceFrame[tickers].dropna()
    if len(sub_df) < windowLen:
        return None, None

    logPrices = np.log(sub_df)
    spreadSeries     = pd.Series(index=logPrices.index, dtype=float)

    for i in range(windowLen - 1, len(logPrices)):
        window_data = logPrices.iloc[i - windowLen + 1 : i + 1]
        y = window_data.iloc[:, 0]
        X = sm.add_constant(window_data.iloc[:, 1:])
        hedge = sm.OLS(y, X).fit().params

        row   = logPrices.iloc[i]
        X_now = np.insert(row[1:].values.reshape(1, -1), 0, 1, axis=1)
        spreadSeries.iloc[i] = row.iloc[0] - np.dot(X_now, hedge.values).item()

    spreadSeries = spreadSeries.dropna()
    zScores      = (spreadSeries - spreadSeries.expanding(min_periods=2).mean()) / spreadSeries.expanding(min_periods=2).std()
    return spreadSeries, zScores

def generate_signals(zscore, entryThres=entryThres, exitThres=exitThres):
    sigSeries, position = pd.Series(0, zscore.index), 0
    for t in range(1, len(zscore)):
        z = zscore.iloc[t]
        if position == 0:
            if z < -entryThres:
                position = 1
            elif z > entryThres:
                position = -1
        else:
            if (position == 1 and z >= exitThres) or (position == -1 and z <= exitThres):
                position = 0
        sigSeries.iloc[t] = position
    return sigSeries

def backtest_basket(spreadSeries, sigSeries):
    priceDelta   = spreadSeries.diff()
    pnlSeries     = sigSeries.shift(1).fillna(0) * priceDelta * riskMultiplier
    pnlSeries     = pnlSeries.fillna(0)
    cumPnl = pnlSeries.cumsum()

    sharpRatio = pnlSeries.mean() / pnlSeries.std() * np.sqrt(tradingDays) if pnlSeries.std() else 0
    drawdown = cumPnl - cumPnl.cummax()

    print(f"[validate] cluster {clusterId}: PnL {cumPnl.iloc[-1]:.2f}, Sharpe {sharpRatio:.2f}")

    return {
        "pnlSeries"     : pnlSeries,
        "cumPnlSeries" : cumPnl,
        "sharpeRatio"   : sharpRatio,
        "maxDrawdown"   : drawdown.min(),
        "totalPnl"      : cumPnl.iloc[-1],
        "numTrades"     : sigSeries.diff().abs().sum() / 2
    }

results = []
for clusterId, tickers in clusterGroups.items():
    if len(tickers) < 2:
        print(f"Skipping cluster {clusterId}: not enough tickers.")
        continue

    print(f"Checking cluster {clusterId}: {tickers}")
    spreadSeries, zscore = construct_basket_spread(tickers, priceData)

    if spreadSeries is None:
        print("  → Insufficient data.")
        continue

    signals = generate_signals(zscore)
    stats = backtest_basket(spreadSeries, signals)

    stats.update({
        "cluster"        : clusterId,
        "tickers"        : ",".join(tickers),
        "percentReturn" : stats["totalPnl"] / capitalPerGroup * 100
    })
    results.append(stats)
    print(f"... done, return {stats['totalPnl']:.2f}%")

if not results:
    print("No clusters validated.")
    exit()

print("Summary:")
for r in sorted(results, key=lambda x: x['sharpeRatio'], reverse=True):
    print(f"{r['cluster']}: Sharpe {r['sharpeRatio']:.2f}, Return " +
          f"{r['percentReturn']:.2f}%, PnL {r['totalPnl']:.2f}, Trades " +
          f"{r['numTrades']:.1f}, Drawdown {r['maxDrawdown']:.2f}")
    
for recStats in results:
    recStats['score'] = recStats['sharpeRatio'] + recStats['percentReturn'] + recStats['numTrades'] * 0.5

topClusters = sorted(results, key=lambda x: x['sharpeRatio'], reverse=True)[:10]


if topClusters:
    selectedClusters = [rec['cluster'] for rec in topClusters]
    selectedDf = clusterMap[clusterMap['cluster'].isin(selectedClusters)]
    selectedDf.to_csv("selected_clusters.csv", index=False)
    print(f"Saved {len(topClusters)} clusters → selected_clusters.csv")
else:
    print("No clusters met the selection criteria.")