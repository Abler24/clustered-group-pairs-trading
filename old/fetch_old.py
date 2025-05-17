import yfinance as yf
import pandas as pd
import numpy as np
import os

# Load tickers from CSV file
TICKERS_FILE = "sp500_tickers.csv"
DATA_PATH = "stock_data.parquet"

def load_tickers(file_path):
    try:
        tickers_df = pd.read_csv(file_path)
        return tickers_df["symbol"].tolist()
    except Exception as e:
        print(f"Error loading tickers file: {e}")
        return []

TICKERS = load_tickers(TICKERS_FILE)

# Function to fetch historical stock data
def fetch_stock_data(tickers, start="2015-01-01", end="2025-01-01", interval="1d"):
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
            if df.empty:
                print(f"Warning: No data fetched for {ticker}")
                continue
            df = df[['Close']]
            df.rename(columns={"Close": ticker}, inplace=True)
            data[ticker] = df
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    return pd.concat(data.values(), axis=1) if data else None

# Function to preprocess data (log returns, rolling stats)
def preprocess_data(df, window=20):
    if df is None or df.empty:
        print("Error: No data available for preprocessing.")
        return None, None, None
    df = df.dropna()
    log_returns = np.log(df / df.shift(1)).dropna()
    rolling_mean = df.rolling(window=window).mean().dropna()
    rolling_std = df.rolling(window=window).std().dropna()
    return log_returns, rolling_mean, rolling_std

# Function to save data efficiently
def save_data(df, filename=DATA_PATH):
    min_date = df.index.min()
    max_date = df.index.max()
    print(f"Available data date range: {min_date} to {max_date}")
    if df is None or df.empty:
        print("Error: No data to save.")
        return
    df.index = pd.to_datetime(df.index)
    df.to_parquet(filename, engine="pyarrow")
    print(f"Data saved successfully to {filename}")
    print(f"DataFrame Shape: {df.shape}")


# Run Data Pipeline
raw_data = fetch_stock_data(TICKERS)
#log_returns, rolling_mean, rolling_std = preprocess_data(raw_data)
save_data(raw_data)

# Display first few rows if data is available
if raw_data is not None:
    print(raw_data.head())
    #print("\nLog Returns:\n", log_returns.head() if log_returns is not None else "No log returns available.")