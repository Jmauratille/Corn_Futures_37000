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


## Setup and first run

Use Python 3.12 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install pytest
python examples/prediction_dataset.py
python -m pytest
```

The example uses synthetic bars and does not require a Databento account or API
key. It prints dataset sizes, the purged rows, and the train/test timestamp
boundary; it does not report investment performance.

For the existing historical-data notebook, install the notebook dependencies:

```bash
python -m pip install jupyter ipykernel matplotlib
python -m jupyter notebook src/project_data.ipynb
```

Read the notebook configuration and cost estimate before running its download
cells. It uses `get_databento_api_key()` to read a key from
`~/.databento_api_key`; keep that file private and never commit the key or raw
market data. The notebook also supports an existing `DATABENTO_API_KEY` environment variable.
It caches data under `.cache/`, which is ignored
by Git. The reusable `pull_data` command and `data/raw/` storage are proposed in
[PR #9](https://github.com/Jmauratille/Corn_Futures_37000/pull/9) and are not yet
available on main at the time of this update.

## Prediction dataset

`build_prediction_dataset` accepts one contract's bars with `close` and `volume`
columns and a unique, increasing `DatetimeIndex` labeled at **bar end time**.
The notebook's plotting helper uses start-labeled bars; do not pass those
chart bars directly as if their timestamps were closing times. Build separate
analysis bars with the existing helper, for example:

```python
from finm37000 import make_ohlcv

# trades_ct is the event-time-indexed, single-contract frame from the notebook.
bars = make_ohlcv(
    trades_ct, "5min", index_name="event_time", label="right",
)
```

All features use information available through that close. The target is the
simple return over the next `horizon` observed bars; intervals with no trades
are not forward-filled, so a horizon is a count of observations rather than a
fixed elapsed duration. Missing and invalid input values are rejected.

```python
from finm37000 import (
    FEATURE_COLUMNS, build_prediction_dataset, chronological_split,
)

dataset = build_prediction_dataset(bars, window=20, horizon=3)
train, test = chronological_split(dataset, test_fraction=0.2)
X_train = train[list(FEATURE_COLUMNS)]
y_train = train["target_return"]
X_test = test[list(FEATURE_COLUMNS)]
y_test = test["target_return"]
```

The features are one-bar simple return, close relative to its rolling average,
Bollinger bandwidth, unannualized rolling log-return volatility, and bar volume.
Warm-up rows and rows without known future targets are removed. `target_end`
records the future close's timestamp. The split reserves the latest rows for
testing and removes training rows whose targets end at or after the first test
observation; `gap` can exclude additional dataset rows before testing.

Fit preprocessing and models using training rows only. Neither `target_return`
nor `target_end` belongs in model inputs. This prepares research data; it does
not implement a trading strategy or assume an executable fill at the signal's
closing price. Execution timing and transaction costs belong in the evaluation
stage. Order-flow imbalance can be added after validating the side convention
and aligning its bar boundaries with the price bars.

## Baseline prediction

`run_baseline` builds the prediction dataset, makes the purged chronological
split, fits a standardized linear regression (optionally ridge) on the training
rows only, and scores the holdout once against a benchmark that always predicts
a zero return. Feature means, scales and coefficients come from training rows;
changing holdout prices cannot change the fitted model.

```bash
# Real data, after pulling ZCH7 trades with finm37000.pull_data:
python examples/baseline_predictor.py --symbol ZCH7 --bar 5min --horizon 3 \
    --output baseline_report.json
# No Databento key needed (tick-rounded random walk, should show no edge):
python examples/baseline_predictor.py --synthetic
```

The JSON report records every setting (symbol, bar, window, horizon, test
fraction, gap, ridge, features), the train and test date ranges and row counts,
the purged rows, the coefficients, and holdout metrics:

- `mae`, `rmse` for the model and for the zero-return benchmark.
- `oos_r2_vs_zero` = `1 - SSE_model / SSE_zero`; above zero beats the benchmark.
- `hit_rate`: share of scored rows whose predicted and realized returns have
  the same sign. Corn moves in 0.25-cent ticks, so many short-horizon returns
  are exactly zero. Rows with a zero realized return are not scored (no
  direction), and a zero prediction is an abstention. `coverage` is the share
  of nonzero-return rows that were scored; `zero_actual_share` reports how
  often the price did not move. The zero benchmark has coverage 0 and no hit
  rate (`null`).

Fix settings before looking at holdout results. To choose between settings,
split the training rows chronologically again and compare there; rerunning on
the holdout until the numbers look good is tuning on the test set. The stored
trade bars are built on `ts_event`, labeled at bar end.

## Roadmap and completion criteria

| Stage | Existing work or proposed task | Completion criteria |
| --- | --- | --- |
| Environment | [Issue #4](https://github.com/Jmauratille/Corn_Futures_37000/issues/4) | A teammate can install the package, run the synthetic example and tests, then open the data notebook. |
| Historical trades | [Issues #1](https://github.com/Jmauratille/Corn_Futures_37000/issues/1) and [#2](https://github.com/Jmauratille/Corn_Futures_37000/issues/2); [PR #9](https://github.com/Jmauratille/Corn_Futures_37000/pull/9) | One agreed ZCH7 date range, reusable local storage, validation, and raw data excluded from Git. |
| Features | [Issue #7](https://github.com/Jmauratille/Corn_Futures_37000/issues/7), completed by [PR #8](https://github.com/Jmauratille/Corn_Futures_37000/pull/8) | Reusable tested indicators; validate aggressor-side semantics before interpreting order flow. |
| Prediction dataset | [Issue #10](https://github.com/Jmauratille/Corn_Futures_37000/issues/10), this contribution | Causal features, explicit future-return targets, chronological split with overlapping training targets removed, and synthetic tests. |
| Baseline prediction | [Issue #11](https://github.com/Jmauratille/Corn_Futures_37000/issues/11); `finm37000.baseline` and `examples/baseline_predictor.py` | Training-only preprocessing, a simple baseline model and a zero-return benchmark, evaluated on an untouched chronological holdout. |
| Strategy evaluation | [Issue #12](https://github.com/Jmauratille/Corn_Futures_37000/issues/12) | Explicit entry timing, position changes, configurable fees/slippage, and both gross and net results on holdout data. |
| USDA analysis | [Issue #13](https://github.com/Jmauratille/Corn_Futures_37000/issues/13), optional feasibility task | Identify a historical expectations source, actual releases and timestamps; analyze surprises without using data published later. Defer if expectations data is unavailable. |

The team should agree on the contract, date range, bar duration, prediction
horizon, and cost assumptions before comparing results. README changes and
issues are places to record that agreement; each member should review the
other members' PRs with specific questions or suggestions.
