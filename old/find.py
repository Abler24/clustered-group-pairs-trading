import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import itertools
import matplotlib.pyplot as plt

# Load data (from fetch.py)
DATA_PATH = "stock_data.parquet"

try:
    df = pd.read_parquet(DATA_PATH)
    df.index = pd.to_datetime(df.index, errors="coerce")  # Ensure index is datetime
    df = df.droplevel(0, axis=1) if isinstance(df.columns, pd.MultiIndex) else df  # Flatten MultiIndex if present
    df = df.dropna(axis=1, how="all")  # Drop columns with all NaNs
    
    # Remove stocks that are entirely zero
    df = df.loc[:, (df != 0).any(axis=0)]
    
    print(f"Parquet file loaded successfully! Data range: {df.index.min()} to {df.index.max()}")
except Exception as e:
    print(f"Error loading Parquet file: {e}")
    exit()

print(f"Data range before cleaning: {df.index.min()} to {df.index.max()}")
df = df.ffill().bfill()  # Fill missing values forward and backward
print(f"Data range after cleaning: {df.index.min()} to {df.index.max()}")

# Debug: Print Data Stats
print(f"DataFrame Shape: {df.shape}")
print(f"NaN Count Per Column:\n{df.isna().sum()}")

# Compute log returns (useful for correlation)
log_returns = np.log(df / df.shift(1)).dropna()

# Debug: Check Log Returns
print(f"Log Returns Data Shape: {log_returns.shape}")
print(log_returns.head())

# Function to compute correlation matrix with a smaller window
def get_correlation_matrix(df, window=50):  # Reduced from 200 to 50
    rolling_corr = df.rolling(window).corr(pairwise=True)
    last_correlation = rolling_corr.iloc[-len(df.columns):]  # Extract last valid correlation matrix
    return last_correlation.droplevel(0) if isinstance(last_correlation.index, pd.MultiIndex) else last_correlation

# Drop columns with any NaNs before computing correlation
df = df.dropna(axis=1, how="any")

# Compute correlation matrix
cor_matrix = get_correlation_matrix(df)

# Print correlation matrix stats
print(f"Correlation Matrix Stats - Min: {cor_matrix.min().min()}, Max: {cor_matrix.max().max()}")

# Fix: Ensure correct extraction of stock names from DataFrame
valid_tickers = df.columns.tolist()  # Ensure tickers are strings

high_corr_pairs = [
    (s1, s2) for s1, s2 in itertools.combinations(valid_tickers, 2)
    if abs(cor_matrix.at[s1, s2]) > 0.95 and s1 != s2  
]

# Filter high correlation pairs to only include same-sector stocks
filtered_pairs = high_corr_pairs

if not filtered_pairs:
    print("No highly correlated pairs found.")
else:
    print("Highly Correlated Pairs:", len(filtered_pairs))

# Use training data range (2020-2023) for cointegration learning
train_data = df.loc["2020-01-01":"2023-12-31"]

# Function to test for cointegration with rolling windows
def test_cointegration(series1, series2, window=252):  # Use ~1 year of daily data
    min_p_value = 1  # Start with the highest possible p-value
    for start in range(0, len(series1) - window, window // 2):  # Overlapping windows
        s1_window = series1.iloc[start:start + window].dropna()
        s2_window = series2.iloc[start:start + window].dropna()
        
        if len(s1_window) < 2 or len(s2_window) < 2:
            continue  # Skip if insufficient data
        score, p_value, _ = coint(s1_window, s2_window)
        min_p_value = min(min_p_value, p_value)  # Keep the best p-value
    
    return min_p_value  # Return the lowest p-value found

# Identify cointegrated pairs from filtered_pairs
cointegrated_pairs = []
for stock1, stock2 in filtered_pairs:
    p_value = test_cointegration(train_data[stock1], train_data[stock2])
    if p_value < 0.1:  # More lenient p-value threshold
        cointegrated_pairs.append((stock1, stock2))

if not cointegrated_pairs:
    print("No cointegrated pairs found.")
else:
    print("Cointegrated Pairs:", cointegrated_pairs)