# Corn strategy starter

This copy contains the uploaded src tree, plus:

- src/finm37000/backtest/bollinger.py: concrete Strategy implementation and synthetic main demo.
- examples/corn_signal.py: load saved raw trades, build completed bars, print the latest target.
- tests/test_bollinger_strategy.py: eight behavioral unit tests using unittest.
- src/finm37000/backtest/engine.py: missing imports added; run_backtest now explicitly raises NotImplementedError instead of silently returning None.

The Strategy protocol and PortfolioState stay unchanged. The backtest engine is still a skeleton. This starter verifies signals, not strategy profitability.

## Rule

The latest closed bar is included in a trailing window of 20 observed bars.
Close below the Bollinger middle line -> target +1.
Close above the middle line -> target -1.
Equal to the middle line, insufficient history, or missing/nonfinite prices in the window -> hold the current position.
There are no extra entries when already at the target. Reversal is immediate at the target level: -1 to +1 requires buying two contracts. A flat portfolio remains flat until it has a signal.

This is a moving-average mean-reversion rule. It uses the middle line of your existing Bollinger helper; the upper/lower bands and num_std have no effect on this rule. A strategy entering at the outer bands and exiting at the middle would be a separate rule.

## Quick check

Copy src/, examples/, and tests/ into your existing project root, preserving other files. Use the environment with your existing project dependencies installed. The source zip has no pyproject.toml or environment specification, so this is not a standalone environment installer.

From the project root, if finm37000 is already installed in editable mode:

    python -m finm37000.backtest.bollinger
    python -m unittest discover -s tests -p test_bollinger_strategy.py -v

Otherwise add src to the module path. Windows Anaconda Prompt (cmd.exe):

    set "PYTHONPATH=%CD%\src"
    python -m finm37000.backtest.bollinger

PowerShell:

    $env:PYTHONPATH = "$PWD\src"
    python -m finm37000.backtest.bollinger

Bash:

    PYTHONPATH=src python -m finm37000.backtest.bollinger

The main demo prints scenario, current position, target, and target-minus-position. It does not send orders or simulate execution.

## Historical data

Your existing pull_data module downloads and caches raw history (uses Databento credits):

    python -m finm37000.pull_data --symbol ZCH7 --months 6

Then run the example against the stored file, supplying the ACTUAL exclusive endpoint of the download, with timezone:

    python examples/corn_signal.py --symbol ZCH7 --end YOUR_ACTUAL_COVERAGE_ENDPOINT --window 20 --position 0

Replace YOUR_ACTUAL_COVERAGE_ENDPOINT with the exact timestamp used by the successful pull, for example an ISO timestamp ending in Z or -05:00. The example does not download data. The existing pull CLI logs its requested endpoint but does not store it in the parquet metadata: retain that log. Re-running it against a cached file does not establish a new coverage endpoint. Do not infer complete coverage from the last trade timestamp or use today's date for older cached data.

build_analysis_bars requires exchange event time: a ts_event column or a timezone-aware event index named ts_event/event_time. If your parquet has only receive time, retrieve/preserve exchange event timestamps; renaming receive time would change the meaning of the data. It also requires symbol, price, size, and side columns for a single contract. Nonempty five-minute intervals become bars, missing intervals are omitted, and incomplete trailing intervals are excluded. Twenty observations therefore need not represent 100 uninterrupted minutes.

The --position parameter is the current actual portfolio position. The example prints one decision for the latest completed bar. It neither tracks historical fills nor measures P&L. All history passed to on_bar must be closed and available at the decision time; the function cannot detect future rows supplied by the caller.

## Complete the engine next

For each bar i, in chronological order:

1. Execute the previous pending order at this bar's open with NextOpenExecution. With bar-end labels, the actual open/execution time must be represented separately; bars.index[i] labels the close, not the open. For omitted empty intervals use the next observed bar's first available trade, not an invented price.
2. Apply the fill: update position, average entry price, realized P&L and cumulative fees. A reversal closes the old side and opens the new side; charge costs on the full two-contract order.
3. Mark unrealized P&L at this bar's close and record equity as initial_capital + realized_pnl + unrealized_pnl - cumulative_fees (when realized_pnl is gross of fees).
4. Pass bars.iloc[:i+1] and the updated immutable PortfolioState to strategy.on_bar.
5. Validate the target against max_position and queue target - position for the next observed bar's open. Zero quantity means no order.

Do not fill the signal using the same bar's open: it preceded the close that generated the signal. The next open is a simplified execution model; it does not model queueing, latency or spread.

Do not execute the last signal without a later bar. Choose and document whether to leave final exposure marked or liquidate at an explicitly modeled executable price. Use dataclasses.replace or create a new PortfolioState because the dataclass is frozen.

Before interpreting dollar results, confirm that dollars_per_price_unit and tick_size match the price units in the loaded file. Include per-contract fees and slippage, then assess the fixed rule on an untouched later period. This starter does not supply verified market contract specifications or empirical profitability results.

## Validation in this session

Eight unit tests and the synthetic module demo passed using the actual strategy, features, models, engine imports, and backtest package initializer. The test harness used a temporary minimal top-level finm37000 initializer to avoid unrelated optional imports. Databento and Plotly are unavailable in this runtime, so a full installed-package import, market-data download, and real-history example could not be run here. The temporary harness is not part of this starter. Use your configured project environment for the commands above.
