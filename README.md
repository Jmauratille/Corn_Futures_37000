# Corn_Futures_37000

We will use the DataBento data to pull historical corn trade price and volume data to analyze whether there are tradable trading patterns in the corn market. The focus is on CBOT corn futures (ZC), traded on CME Globex.

Rough outline:
- We will need to pull historical data, prices and sizes, having the aggressor side would also be useful.  We will likely want to pull about 6 months of a single contract, for example the Mar2027 contract (ZCH7). Trade-level data (with the aggressor side) lets us see whether buyers or sellers are driving price moves, and we can aggregate it into bars (e.g. 1-minute or 5-minute) for analysis.

- We will want to create some utilities for analyzing trading features.  Examples of features would be moving averages, bollinger bands, etc. These will be stratified as the project progresses. Other candidates include returns over different horizons, realized volatility, volume, and order-flow imbalance (buy volume minus sell volume).

- We will use these features to see if we can predict future returns, or if we can design a trading strategy with a clear profitable pattern. Any strategy should be tested on data it was not built on, and should account for transaction costs (tick size and fees) so that results are realistic.

- Lastly, we can use some non-trading-based data to influence our predictions.  Namely, the USDA NASS data regarding corn production and stocks can be used to see what is the influence under prediction and over prediction of the market estimate is on the actual number.  We can look up historical estimates, and compare with the actual number, and create a simple linear model to estimate the price impact of the degree of over/under estimation. The key reports here are the monthly Crop Production report and the quarterly Grain Stocks report, where the surprise (actual minus market expectation) can be regressed against the price move after the release.

## Repository layout

- `src/finm37000/` – course helper package (DataBento access, futures utilities, aggregation, plotting).
- `src/*.ipynb` – course example notebooks.
- `Group_Roles.md` – team roles.

## Team

See [Group_Roles.md](Group_Roles.md) for who is responsible for what.
