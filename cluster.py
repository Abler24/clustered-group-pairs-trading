import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
from collections import Counter


# Define start and end dates for clustering period
clusterStart = "2018-01-01"
clusterEnd = "2022-12-31"

dataPath = "stock_data.parquet"

try:
    # Load cleaned price data (from fetch.py)
    df = pd.read_parquet(dataPath)
    # Some data sources store columns as multi‑index tuples, e.g. ('MMM','MMM').
    # Flatten any tuple columns to simple ticker strings.
    if isinstance(df.columns[0], tuple):
        df.columns = [col[0] for col in df.columns]
    df.index = pd.to_datetime(df.index, errors="coerce")
    # Forward and backward fill remaining missing values
    df = df.ffill().bfill()
    print(f"Data loaded successfully! Date range: {df.index.min()} to {df.index.max()}")
    
    trainPrices = df.loc[clusterStart:clusterEnd]
    testPrices = df.loc[clusterEnd:]
    print(f"Clustering data: {trainPrices.index.min()} to {trainPrices.index.max()}")
    
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

log_returns = np.log(trainPrices / trainPrices.shift(1)).dropna()

stocks_raw = log_returns.columns.tolist()
stocks = [col[0] if isinstance(col, tuple) else col for col in stocks_raw]
returnsMatrix = log_returns[stocks].T.values  # shape: (num_stocks, num_days)

n_components = 5
pca = PCA(n_components=n_components)
pcaTransformed = pca.fit_transform(returnsMatrix)

n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusterLabels = kmeans.fit_predict(pcaTransformed)

clusterMap = {}
for ticker, label in zip(stocks, clusterLabels):
    clusterMap.setdefault(label, []).append(ticker)

print("\nClusters found:")
for label, tickers in clusterMap.items():
    print(f"Cluster {label}: {tickers}")

top_labels = [label for label, _ in Counter(clusterLabels).most_common(6)]
subset_idx = [i for i, lbl in enumerate(clusterLabels) if lbl in top_labels]

plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    pcaTransformed[subset_idx, 0],
    pcaTransformed[subset_idx, 1],
    c=[clusterLabels[i] for i in subset_idx],
    cmap='tab10',  # 10 distinguishable colors
    alpha=0.7
)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Top 5 Clusters via PCA + KMeans')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.tight_layout()
plt.savefig("top_clusters_pca.png", dpi=300)
plt.show()

mean_returns = log_returns.mean()
volatilities = log_returns.std()

# Plot
cluster_sizes = Counter(clusterLabels)
labels = list(cluster_sizes.keys())
sizes = list(cluster_sizes.values())

plt.figure(figsize=(10, 6))
plt.bar(labels, sizes, color='skyblue')
plt.xlabel("Cluster Label")
plt.ylabel("Number of Stocks")
plt.title("Size of Each Cluster")
plt.tight_layout()
plt.savefig("cluster_size_barplot.png", dpi=300)
plt.show()

maxSubsetSize = 10
minSubsetSize = 2

size_thresholds = {
    2: 0.95,
    3: 0.92, 4: 0.92,
    5: 0.88, 6: 0.88,
    7: 0.85, 8: 0.85,
    9: 0.85, 10: 0.85
}

def getCointegrationScores(tickers, price_data):
    """Return DataFrame of 1 - p‑value for all ticker pairs."""
    scores = pd.DataFrame(index=tickers, columns=tickers, dtype=float)
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if j <= i:
                continue
            try:
                _, pval, _ = coint(price_data[t1], price_data[t2])
                score = 1 - pval
            except Exception:
                score = np.nan
            scores.loc[t1, t2] = score
            scores.loc[t2, t1] = score
        scores.loc[t1, t1] = 1.0
    return scores

def spread_volatility(tickers, price_data):
    """
    Compute volatility of an equal-weight log-price basket spread.
    Returns the standard deviation of daily differences of the basket.
    """
    log_prices = np.log(price_data[tickers])
    basket = log_prices.mean(axis=1)          
    spread = basket - basket.mean()           
    return spread.diff().std()

selected_cluster_dict = {}
clusterStats = []

for label, tickers in clusterMap.items():
    print(f"\nEvaluating cluster {label}...")

    if len(tickers) > maxSubsetSize:
        corr_matrix = trainPrices[tickers].corr().abs()
        avg_corr = corr_matrix.mean().sort_values(ascending=False)
        tickers = avg_corr.index.tolist()[:maxSubsetSize]
        print(f"  Reduced to top 10 by correlation: {tickers}")
    else:
        print(f"  All {len(tickers)} tickers used.")

    scoreMatrix = getCointegrationScores(tickers, trainPrices)
    workingSet = tickers.copy()

    while len(workingSet) > minSubsetSize:
        size = len(workingSet)
        mask = np.triu(np.ones((size, size)), 1).astype(bool)
        current_avg = scoreMatrix.loc[workingSet, workingSet].where(mask).stack().mean()
        current_vol = spread_volatility(workingSet, trainPrices)

        threshold = size_thresholds.get(size, size_thresholds[maxSubsetSize])
        if current_avg >= threshold:
            print(f"  Size {size} meets threshold {threshold:.2f} (avg {current_avg:.4f}).")
            final_avg = current_avg
            break

        per_ticker_avg = scoreMatrix.loc[workingSet, workingSet].mean()
        worst = per_ticker_avg.idxmin()
        print(f"    Size {size} avg coint = {current_avg:.4f} < {threshold:.2f}, vol = {current_vol:.5f}. Dropping {worst}.")
        workingSet.remove(worst)

    finalTickers = workingSet.copy()
    
    vol = spread_volatility(finalTickers, trainPrices)

    comboScore = final_avg * vol

    final_mask = np.triu(np.ones((len(finalTickers), len(finalTickers))), 1).astype(bool)
    final_avg = scoreMatrix.loc[finalTickers, finalTickers].where(final_mask).stack().mean()
    selected_cluster_dict[label] = finalTickers
    clusterStats.append({
        "cluster": label,
        "initial_size": len(clusterMap[label]),
        "chosen_size": len(finalTickers),
        "avg_coint_score": final_avg,
        "volatility": vol,
        "combined_score": comboScore
    })
    print(f"  Final subset for Cluster {label}: {finalTickers} | Avg coint {final_avg:.4f} | Vol {vol:.5f}")

mappingDf = pd.DataFrame(
    [(label, ticker) for label, tickers in selected_cluster_dict.items() for ticker in tickers],
    columns=["cluster", "ticker"]
)
mappingDf.to_csv("cluster_mapping.csv", index=False)
print("\n Mapping saved to file.")

statsDf = pd.DataFrame(clusterStats)
statsDf.to_csv("cluster_stats.csv", index=False)
print("\n Stats saved to file.")