# Backtest engine implementation

Copy these files into your existing project, preserving their paths. Only engine.py replaces an existing source file. This uses your existing models.py, execution.py, strategy.py (Strategy protocol), and bollinger.py (BollingerMeanReversion). No changes to their public interfaces are required. Keep PortfolioState imported as finm37000.backtest.models.PortfolioState or .models.PortfolioState in your strategy, rather than as a top-level models module.

## Run a synthetic backtest

From the project root in your existing configured virtual environment:

    python examples/run_corn_backtest.py
    python -m unittest discover -s tests -p test_backtest_engine.py -v

If the project is not installed editable, set PYTHONPATH to src first. In Windows cmd.exe:

    set "PYTHONPATH=%CD%\src"

The demo uses synthetic prices and illustrative contract settings. It is a functionality check, not evidence of trading performance. Its final marked P&L is -$17.50 for the supplied demo configuration.

## Use real cached history

    import pandas as pd
    from finm37000 import build_analysis_bars
    from finm37000.data_store import load_history
    from finm37000.backtest.bollinger import BollingerMeanReversion
    from finm37000.backtest.engine import run_backtest
    from finm37000.backtest.models import BacktestConfig

    trades = load_history("ZCH7")
    # Set this to the actual exclusive endpoint of your successful download.
    # coverage_end must be a timezone-aware pandas Timestamp.
    bars = build_analysis_bars(trades, rule="5min", end=coverage_end)

    config = BacktestConfig(
        symbol="ZCH7",
        initial_capital=100_000.0,
        dollars_per_price_unit=dollars_per_price_unit,
        tick_size=tick_size,
        fee_per_contract=fee_per_contract,
        slippage_ticks=1,
        max_position=1,
    )
    result = run_backtest(
        bars, BollingerMeanReversion(window=20), config,
        bar_duration="5min",
    )
    print(result.trades)
    print(result.history.tail())
    result.history["equity"].plot(title="Corn strategy equity")

coverage_end and the three cost/unit variables are inputs you must supply. Match dollars_per_price_unit and tick_size to the actual units of the stored price column, and use your chosen per-contract, per-fill fees. For unit arithmetic: a contract representing Q physical units has a multiplier Q when prices are in dollars per physical unit, or Q/100 when prices are in cents. Check the stored data rather than assuming its price units. No raw data or verified market contract settings were supplied or downloaded in this implementation.

## Processing order

Each bar first executes the previous pending order at its open. It then marks the portfolio at its close, calls the strategy with a copy of bars.iloc[:i+1], and queues target minus actual position for the next observed bar. The strategy receives no later rows. Same-target signals create no order. A -1 to +1 reversal buys two contracts and pays fees and slippage on both.

Average entry prices support additions, reductions, full closes, and reversals. Realized P&L is gross of fees; net_pnl = realized_pnl + unrealized_pnl - cumulative_fees. Equity = initial_capital + net_pnl. This is futures exposure accounting: opening a position does not subtract the full contract notional.

result.trades contains one row per executed order, including signal_time, execution_time, signed quantity, slipped price, and fees. A reversal appears as one quantity +/-2 fill. These rows are fills, not matched round-trip trades.

result.history contains bar-close position, entry price, realized/unrealized P&L, fees, close, equity, target_position, and order_quantity. order_quantity is the newly queued order, not a fill on that row. The last queued order is never filled without another bar. Final positions remain open and marked; liquidation costs are not invented.

## Timing and limitations

bar_duration is required and must match the aggregation frequency. Bar labels are interval ends. Modeled execution_time is the next observed bar end minus bar_duration. For contiguous bars this equals the previous signal timestamp; that represents an idealized zero-latency boundary fill. Actual first-trade times are unavailable from OHLC alone, and this model cannot guarantee execution at the recorded opening price. Missing intervals remain omitted; an order waits for the next observed bar.

The engine validates OHLC consistency, ordered timezone-aware timestamps, symbol agreement when present, target limits, and nonoverlapping modeled intervals. It does not model partial fills, liquidity, bid/ask spread, latency, margin calls, funding, daily cash variation margin, or contract rolls. Max_position is a position limit, not a capital or margin constraint. It accepts one contract; assess any roll strategy separately.

The direct history copies prioritize a clear causal implementation and may become costly for very long histories. Optimize only after profiling, while preserving the no-future-data contract. Passing a prefix cannot prevent an arbitrary strategy from independently accessing external future data.

## Validation

Ten new engine tests passed, alongside eight prior strategy tests, and the synthetic demo ran successfully. Tests cover known dollar outcomes, reversals, long/short P&L, adverse slippage, fees, additions/reductions, causal history prefixes, unchanged inputs, gaps, final unfilled signals, invalid data/targets, and Bollinger integration.

Validation ran against the actual backtest modules with a temporary minimal top-level package initializer to bypass unavailable Databento/Plotly dependencies. That harness is not included. A full import in the user's configured environment and a real market-data backtest have not been run here.
