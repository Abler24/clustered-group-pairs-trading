# vCluster.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- 1. Load price data & compute features ---
price_data = pd.read_parquet('stock_data.parquet')
if isinstance(price_data.columns[0], tuple):
    price_data.columns = [c[0] for c in price_data.columns]
price_data = price_data.ffill().bfill()

returns = np.log(price_data / price_data.shift(1)).dropna()

mean_returns = returns.mean()
volatility   = returns.std()
sharpe_ratio = mean_returns / volatility
skewness     = returns.skew()

X = pd.DataFrame({
    'mean_return': mean_returns,
    'volatility':  volatility,
    'sharpe':      sharpe_ratio,
    'skewness':    skewness
})
X.index.name = 'ticker'
X.reset_index(inplace=True)

# --- 2. Load clusters from cluster.py output ---
mapping = pd.read_csv('cluster_mapping.csv')
X = X.merge(mapping, on='ticker', how='inner')
clusters = sorted(X['cluster'].unique())
nclust = len(clusters)

# --- 3. Bar chart of cluster sizes ---
cluster_counts = X['cluster'].value_counts().sort_index()
plt.figure(figsize=(10,6))
plt.bar([str(c) for c in cluster_counts.index], cluster_counts.values, alpha=0.8)
plt.xlabel('Cluster Label')
plt.ylabel('Number of Stocks')
plt.title(f'Cluster Sizes (K={nclust})')
plt.tight_layout()
plt.show()

# --- 4. Scatter plots using different axis combinations ---
def plot_scatter(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(12,8))
    scatter = plt.scatter(
        X[x],
        X[y],
        c=X['cluster'],
        cmap='tab20',
        s=50,
        alpha=0.7
    )
    centroids = X.groupby('cluster')[[x, y]].mean().loc[clusters].values
    plt.scatter(
        centroids[:,0],
        centroids[:,1],
        marker='X',
        s=200,
        c=clusters,
        cmap='tab20',
        edgecolor='k',
        linewidth=1.5,
        label='Centroids'
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.tight_layout()
    plt.show()

# A: Mean Return vs Volatility
plot_scatter('mean_return', 'volatility', 'Mean Return', 'Volatility', 'Mean Return vs Volatility')

# B: Sharpe Ratio vs Volatility
plot_scatter('sharpe', 'volatility', 'Sharpe Ratio', 'Volatility', 'Sharpe Ratio vs Volatility')

# C: PCA1 vs PCA2
pca = PCA(n_components=2)
pca_data = pca.fit_transform(returns.T)
X['PCA1'], X['PCA2'] = pca_data[:,0], pca_data[:,1]
plot_scatter('PCA1', 'PCA2', 'PCA Component 1', 'PCA Component 2', 'PCA1 vs PCA2')

# D: Skewness vs Mean Return
plot_scatter('skewness', 'mean_return', 'Skewness', 'Mean Return', 'Skewness vs Mean Return')