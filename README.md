# Statistical Arbitrage Using a Clustering Based Group Pairs Trading Strategy

This project explores a group based statistical arbitrage strategy. The pipeline from data collection to backtesting aims to identify and trade on mean reverting relationships between small baskets of cointegrated stocks.

The standard pairs trading framework (long one asset, short another) is well studied. But in modern equity markets, relationships often extend beyond simple pairs. Think of airlines, payments, or homebuilders: groups of related stocks that drift together, diverge temporarily, and frequently revert.

This repository builds a generalization of the pairs trading idea. Instead of handpicking ticker pairs, I use unsupervised learning to cluster similar stocks. From there, I filter clusters using cointegration, and trade these spreads with a rules based approach.

Results:

<img width="500" alt="Screenshot 2025-05-21 at 12 04 20 AM" src="https://github.com/user-attachments/assets/f992e864-f594-4da3-beeb-af9f80cb2dd8" />

Return compared to benchmarks: 

<img width="900" alt="Screenshot 2025-05-21 at 12 19 18 AM" src="https://github.com/user-attachments/assets/7e7d4bf2-a27d-4ee3-a4e3-8afd5b889e8e" />


The project is composed of four Python scripts:

1. `download.py`: Pulls and cleans daily price data.
2. `cluster.py`: Builds PCA-based return vectors, clusters stocks, and evaluates cointegration within each group.
3. `validate.py`: Provides sanity checks on cluster quality.
4. `backtest.py`: Constructs long-short baskets and runs a market realistic backtest.

---

## High-Level Overview

The project consists of three phases:

- **Discovery:** Build daily return vectors and cluster stocks using PCA and KMeans.
- **Filtering:** Score each cluster based on internal cointegration and only keep the best scores.
- **Execution:** Trade each cluster as a spread basket, targeting reversion to a statistical mean.

---

## Step 1: Data Download (download.py)

The first step is collecting clean, daily adjusted close prices. I use the S&P 500 stocks, supplemented with a few historically relevant tickers to avoid survivorship bias.

The data is downloaded using Yahoo Finance and filtered for:

- Adequate coverage (95%+ data presence)
- No extreme outliers or stale prices
- Reasonable return distributions

Missing values are handled with forward and back filling to a reasonable extent. The result is a Parquet file with a datetime index and one column per ticker.

---

## Step 2: Clustering Stocks (cluster.py)

I clustered correlated stocks to find unique groups to pairs trade.
The process:

1. Compute daily log returns for each stock.
2. Represent each stock by its return time series.
3. Reduce dimensionality via PCA (keeping the top 5 components).
4. Cluster the transformed data using KMeans into 20 groups.

I chose 20 clusters as a balance between:

- Having enough diversity across groups
- Avoiding dilution of statistical structure
- Ensuring trade volume per cluster

The clustering output is saved as a stock to cluster mapping.

<img width="900" alt="Screenshot 2025-05-19 at 7 58 21 AM" src="https://github.com/user-attachments/assets/2a30a2d7-0954-48d9-a495-8c09a1b22277" />

A bar chart showing the number of tickers per cluster.

<img width="900" alt="Screenshot 2025-05-19 at 7 55 56 AM" src="https://github.com/user-attachments/assets/f058f0a2-4ea4-4512-b4f5-4a666b1c4171" />

A PCA scatterplot (colored by cluster for the top 5 groups)
  
These visuals help check the unsupervised learning step before I move on.

---

## Step 3: Filtering with Cointegration (cluster.py)

Not every cluster is tradable. Even if returns are similar, they may not be statistically connected in a way that supports a mean reversion trade.

So I filter.

For each cluster:

- If the size > 10, I keep only the 10 tickers with highest average correlation to peers.
- I compute all pairwise cointegration scores using the Engle Granger method.
- The score is 1 - p-value. A higher score means stronger statistical linkage.
- I iteratively drop the worst performing ticker until the cluster’s average score exceeds a size specific threshold.

Thresholds:

- 2 tickers: ≥ 0.95
- 3-4 tickers: ≥ 0.92
- 5-6 tickers: ≥ 0.88
- 7-10 tickers: ≥ 0.85

This results in a final set of cointegrated clusters each with 2–10 tickers:

<img width="900" alt="Screenshot 2025-05-21 at 11 12 16 AM" src="https://github.com/user-attachments/assets/b9170f91-d19d-4b6c-9e85-8ff72f763703" />


**What Kinds of Stocks Are Being Traded?**

Even though I didn’t tell the model anything about what each company does, a lot of the clusters that ended up working well had something in common.

Here’s a breakdown of a few clusters that stood out:
- Cluster 2: [NOW, ADSK, ETSY, EPAM, PAYC] Mostly software and tech names. These are high growth companies that move a lot on earnings, interest rates, and big market news. This cluster had one of the best returns.
- Cluster 9: [DG, PEP, MDLZ, PGR, JNJ, GIS, AMGN, LLY] Big, steady companies like snacks, drinks, healthcare, and insurance. These stocks don’t move as wildly as tech but tend to bounce back when they dip. This group traded really well.
- Cluster 17: [KIM, HIG, DRI, SYY] A random mix at first glance, but they are all stable and tied to physical goods or services like restaurants, food supply, real estate, insurance. It worked well.
- Cluster 15: [SEDG, TSLA] This one didn’t work great. Two super high-volatility companies that are hard to predict. The strategy didn’t handle their wild price swings well.
- Cluster 16: [LNT, AEE, SO, DUK, WM, HSY] Mostly utility companies and Hershey for some reason. Very slow-moving stocks. The trades didn’t really have room to make much profit, so this one wasn’t very effective.

---

## Step 4: Building the Strategy

The strategy logic is simple and consistent across all clusters:

1.	Take log prices of the selected stocks in a cluster.
2.	Select the stock with the highest R² when OLS regressed on the rest.
3.	Use a 60 day rolling window to run an OLS regression of y against the other tickers in the cluster to estimate hedge ratios.
4.	Compute the spread as the residual between y and the predicted value from the hedge basket.
5.	Convert the spread into a z score using a 60 day rolling mean and standard deviation.
6.	Use the z score to trigger trades: long the spread when it’s too low, short when it’s too high.

Entry occurs when:

- z-score > +2: Short the asset basket, and long the dependent stock
- z-score < -2: Long the asset basket, and short the dependent stock
  
Exit occurs when:

- z-score crosses back through 0 (full mean reversion)

<img width="1394" alt="Screenshot 2025-05-21 at 2 12 23 PM" src="https://github.com/user-attachments/assets/e7051e3e-0a3f-4bc2-bd7c-ce534bed8065" />

This figure shows the log price of the anchor stock and group average, with green and red dots marking when to long or short each side. The grey line is the z-score that triggers trades.

---

## Step 5: Validation (validate.py)

Before putting capital at risk (even simulated), I validate the statistical integrity of the selected clusters.

- Load price data and cluster mappings, then filter for the year 2022.
- Backtest the strategy defined above on each cluster, then compute metrics like Sharpe ratio, drawdown, and number of trades.
- Rank clusters by Sharpe ratio and save the top performers to selected_clusters.csv for future use.

The top ten clusters are used for the official backtest. 

---

## Step 6: Execution and Risk Controls (backtest.py)

Each cluster is allocated a fixed capitalPerGroup of $100,000, with a total strategy budget of $1,000,000, to simulate a real portfolio.

- Trades are executed with a 1-day latency (latencyDays=1), meaning the model makes decisions today but executes them using tomorrow’s prices, simulating signal decay and order queuing.
- The backtest models transaction costs using three components: Commission (commissionBps=0.0005), Slippage: (slippageBps=0.001 + realized vol × volSlippageMult), and Market impact: (impactBps=0.002)
- Position sizes are scaled to achieve a target volatility of 1% daily (targetVolPct=0.01) by inversely weighting trades based on recent 30-day spread volatility, simulating a risk-managed exposure per group.
- A 5% trailing drawdown stop-loss (stopLossPct=0.05) per cluster zeroes out all future trades once cumulative returns fall below that threshold, mimicking capital risk controls used by real-world funds.
  
---

## Step 7: Results (backtest.py)

<img width="500" alt="Screenshot 2025-05-21 at 12 04 20 AM" src="https://github.com/user-attachments/assets/f992e864-f594-4da3-beeb-af9f80cb2dd8" />

- Annualized Return: Average daily return × 252 trading days
- Annualized Volatility: Daily standard deviation × √252
- Sharpe Ratio: (Mean daily PnL / SD) × √252
- Hit Rate: % of days with positive returns
- Maximum Drawdown: Largest peak to trough equity drop
- Drawdown Duration: Longest stretch below a previous high
- Average Holding Period: Total days in position / number of trades

Even after costs and slippage, the strategy has a lot of success.

---

## Final Thoughts

This project started as a way for me to learn more about quant trading by building something end to end. I wanted to learn how data is cleaned, how signals are built, how risk is managed. Along the way, I realized how much structure matters. The strategy isn’t perfect, but it’s consistent and realistic. It gave me a better feel for how quant firms think and what it takes to turn an idea into something tradable.

Thanks for reading.
