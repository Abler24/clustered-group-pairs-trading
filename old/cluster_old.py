import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
import itertools
from statsmodels.tsa.stattools import coint

# Define start and end dates for clustering period
CLUSTER_START = "2018-01-01"
CLUSTER_END = "2022-12-31"

## Number of stocks to select per cluster based on correlation
GROUP_SIZE = 5

# --------------------------
# 1. Load and Clean Data
# --------------------------
DATA_PATH = "stock_data.parquet"

try:
    # Load cleaned price data (from fetch.py)
    df = pd.read_parquet(DATA_PATH)
    # Some data sources store columns as multi‑index tuples, e.g. ('MMM','MMM').
    # Flatten any tuple columns to simple ticker strings.
    if isinstance(df.columns[0], tuple):
        df.columns = [col[0] for col in df.columns]
    df.index = pd.to_datetime(df.index, errors="coerce")
    # Forward and backward fill remaining missing values
    df = df.ffill().bfill()
    print(f"Data loaded successfully! Date range: {df.index.min()} to {df.index.max()}")
    
    # --------------------------
    # Split Data into Training and Testing Sets
    # --------------------------
    train_df = df.loc[CLUSTER_START:CLUSTER_END]
    test_df = df.loc[CLUSTER_END:]
    print(f"Clustering data: {train_df.index.min()} to {train_df.index.max()}")
    
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --------------------------
# 2. Feature Engineering for Clustering
# --------------------------
# Compute daily log returns (each column represents a stock)
log_returns = np.log(train_df / train_df.shift(1)).dropna()

# Transpose so each stock is a data point (features: daily returns)
stocks_raw = log_returns.columns.tolist()
stocks = [col[0] if isinstance(col, tuple) else col for col in stocks_raw]
X = log_returns[stocks].T.values  # shape: (num_stocks, num_days)

# PCA for Dimensionality Reduction
n_components = 5
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# --------------------------
# 3. Clustering Stocks into Groups with optimal k
# --------------------------
n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Build a dictionary mapping each cluster label to its list of tickers
cluster_dict = {}
for ticker, label in zip(stocks, clusters):
    cluster_dict.setdefault(label, []).append(ticker)

print("\nClusters found:")
for label, tickers in cluster_dict.items():
    print(f"Cluster {label}: {tickers}")

# --------------------------
# 3a. Enforce size-based cointegration thresholds
# --------------------------
MAX_SUBSET_SIZE = 10
MIN_SUBSET_SIZE = 2

# Minimum avg cointegration thresholds by subset size
size_thresholds = {
    2: 0.95,
    3: 0.92, 4: 0.92,
    5: 0.88, 6: 0.88,
    7: 0.85, 8: 0.85,
    9: 0.85, 10: 0.85
}

def cointegration_score_matrix(tickers, price_data):
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
    basket = log_prices.mean(axis=1)           # equal-weight basket
    spread = basket - basket.mean()            # demeaned spread
    return spread.diff().std()

selected_cluster_dict = {}
cluster_stats = []

for label, tickers in cluster_dict.items():
    print(f"\n--- Applying thresholds for Cluster {label} (initial {len(tickers)} stocks) ---")

    # Pre-filter large clusters by correlation
    if len(tickers) > MAX_SUBSET_SIZE:
        corr_matrix = train_df[tickers].corr().abs()
        avg_corr = corr_matrix.mean().sort_values(ascending=False)
        tickers = avg_corr.index.tolist()[:MAX_SUBSET_SIZE]
        print(f"  Reduced to top {MAX_SUBSET_SIZE} by correlation: {tickers}")
    else:
        print(f"  Using all tickers: {tickers}")

    scores_df = cointegration_score_matrix(tickers, train_df)
    S_current = tickers.copy()

    # Iteratively drop worst until subset meets its size threshold
    while len(S_current) > MIN_SUBSET_SIZE:
        size = len(S_current)
        # Calculate current avg cointegration
        mask = np.triu(np.ones((size, size)), 1).astype(bool)
        current_avg = scores_df.loc[S_current, S_current].where(mask).stack().mean()

        # Check if current group meets its threshold
        threshold = size_thresholds.get(size, size_thresholds[MAX_SUBSET_SIZE])
        if current_avg >= threshold:
            print(f"  Size {size} meets threshold {threshold:.2f} (avg {current_avg:.4f}). Keeping this set.")
            final_avg = current_avg
            break

        # Else drop the worst ticker and continue
        per_ticker_avg = scores_df.loc[S_current, S_current].mean()
        worst = per_ticker_avg.idxmin()
        print(f"    Size {size} avg {current_avg:.4f} < {threshold:.2f}. Dropping {worst}.")
        S_current.remove(worst)

    chosen_set = S_current.copy()
    
    # Calculate basket spread volatility
    vol = spread_volatility(chosen_set, train_df)

    # Combined score (cointegration * volatility) for ranking
    combined_score = final_avg * vol

    # Record final stats
    final_mask = np.triu(np.ones((len(chosen_set), len(chosen_set))), 1).astype(bool)
    final_avg = scores_df.loc[chosen_set, chosen_set].where(final_mask).stack().mean()
    cluster_stats.append({
        "cluster": label,
        "initial_size": len(cluster_dict[label]),
        "chosen_size": len(chosen_set),
        "avg_coint_score": final_avg,
        "volatility": vol,
        "combined_score": combined_score
    })
    print(f"  Final subset for Cluster {label}: {chosen_set} | Avg coint {final_avg:.4f} | Vol {vol:.5f}")

# --------------------------
# 4. Validate Cluster Internal Consistency
# --------------------------

def average_pairwise_correlation(tickers, price_data):
    """Compute the average pairwise Pearson correlation for a group of stocks."""
    sub_df = price_data[tickers]
    corr_matrix = sub_df.corr()
    # Extract lower triangle (excluding diagonal)
    mask = np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool)
    correlations = corr_matrix.where(mask).stack().values
    return np.mean(correlations) if correlations.size > 0 else np.nan

print("\nCluster Internal Consistency (Average Pairwise Correlation):")
for label, tickers in cluster_dict.items():
    avg_corr = average_pairwise_correlation(tickers, train_df)
    print(f"Cluster {label} (n={len(tickers)}): Average Correlation = {avg_corr:.3f}")


# --------------------------
# Save Cluster Mapping for Future Use
# --------------------------
cluster_mapping_df = pd.DataFrame(
    [(label, ticker) for label, tickers in selected_cluster_dict.items() for ticker in tickers],
    columns=["cluster", "ticker"]
)
cluster_mapping_df.to_csv("cluster_mapping.csv", index=False)
print("\nCluster mapping saved to cluster_mapping.csv")

# Save cluster statistics for inspection
stats_df = pd.DataFrame(cluster_stats)
stats_df.to_csv("cluster_stats.csv", index=False)
print("\nCluster statistics saved to cluster_stats.csv")