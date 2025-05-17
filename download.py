import yfinance as yf
import pandas as pd

tickersFile = "sp500_tickers.csv"
dataPath = "stock_data.parquet"

def loadTickers(filePath):
    try:
        tickersDf = pd.read_csv(filePath)
        return tickersDf["symbol"].tolist()
    except Exception as e:
        print(f"Couldn’t load ticker list: {e}")
        return []

tickers = loadTickers(tickersFile)

def fetchStockData(tickers, start="2015-01-01", end="2025-04-20", interval="1d"):
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
    if data:
        dfAll = pd.concat(data.values(), axis=1)
        dfAll.index = pd.to_datetime(dfAll.index)
        return dfAll
    else:
        return None

def cleanData(df, liquidityThreshold=0.9):
    if df is None or df.empty:
        print("Error: No data available for cleaning.")
        return None
    cleaned = df.ffill()
    
    # filter stocks with too many missing days
    minValid = liquidityThreshold * len(cleaned)
    illiquidStocks = cleaned.columns[cleaned.notna().sum() < minValid].tolist()
    
    if illiquidStocks:
        print(f"Filtered out illiquid stocks: {illiquidStocks}")
        cleaned = cleaned.drop(columns=illiquidStocks)
    
    return cleaned

def saveData(df, filename=dataPath):
    if df is None or df.empty:
        print("Error: No data to save.")
        return
    minDate = df.index.min()
    maxDate = df.index.max()
    print(f"Available data date range: {minDate} to {maxDate}")
    df.to_parquet(filename, engine="pyarrow")
    print(f"✅ Saved cleaned data to {filename}")
    print(f"DataFrame Shape: {df.shape}")

rawData = fetchStockData(tickers)
if rawData is not None:
    cleanedData = cleanData(rawData)
    saveData(cleanedData)
    print("Data Saved")
else:
    print("No data fetched.")