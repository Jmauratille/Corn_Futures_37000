# Corn_Futures_37000

We will use the DataBento data to pull historical corn trade price and volume data to analyze whether there are tradable trading patterns in the corn market. 

Rough outline:
- we will need to pull historical data, prices and sizes, having the aggressor side would also be useful.  We will probably want to pull about 6 months of a single contract, say the Mar2027 contract.

- we will want to create some utilities for analyzing trading features.  Some features would be moving averages, bollinger bands, etc.

- we will use these features to see if we can predict future returns, or if we can design a trading strategy with a clear profitable pattern.

- Lastly, we can use some non-trading data to influence our predictions.  Namely, the USDA NASS data regarding corn production and stocks can be used to see what is the influence under prediction and over prediction of the market estimate is on the actual number.  We can look up historical estimates, and compare with the actual number, and create a simple linear model to estimate the price impact of the degree of over/under estimation



