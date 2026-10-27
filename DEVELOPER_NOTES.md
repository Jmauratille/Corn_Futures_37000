# Developer notes

Background and implementation detail that sits behind the report in
[README.md](README.md). This file is for contributors, not the headline
audience.

## Original project plan

The project began from this outline (kept for reference; see the README for what
was actually built):

- Pull ~6 months of trade-level ZCH7 data from Databento, with aggressor side,
  and aggregate into intraday bars.
- Build reusable trading features: moving averages, Bollinger bands, returns over
  horizons, realized volatility, volume, and order-flow imbalance.
- Use the features to predict future returns or design a trading rule, tested
  out-of-sample and net of transaction costs (tick size and fees).
- Optionally overlay USDA Crop Production / Grain Stocks "surprises" (actual
  minus pre-report market expectation) against post-release price moves.

All four were implemented except the USDA regression, which was scoped to a
feasibility assessment (see `docs/usda_feasibility.md`).

## Backtest engine internals

`finm37000.backtest.run_backtest(bars, strategy, config, bar_duration=...)` is a
causal, event-driven simulator. For each bar `i`, in chronological order:

1. Execute the previous pending order at this bar's open using
   `NextOpenExecution` (next-open fill with per-tick slippage).
2. Apply the fill: update position, average entry price, realized P&L, and
   cumulative fees. A reversal (e.g. −1 to +1) closes the old side and opens the
   new one, paying costs on the full two-contract order.
3. Mark unrealized P&L at this bar's close.
4. Call `strategy.on_bar(bars.iloc[:i+1], portfolio)` — the strategy sees only
   closed bars up to and including `i`, never later rows.
5. Validate the returned target against `max_position` and queue
   `target - position` for the **next** observed bar's open.

Accounting is futures-style: `net_pnl = realized_pnl + unrealized_pnl -
cumulative_fees`, and `equity = initial_capital + net_pnl`. `result.trades` holds
one row per executed fill; `result.history` holds per-bar position, P&L, fees,
close, equity, and the newly queued order.

### Execution model and its limits

Modeled execution time is the next observed bar end minus `bar_duration`; for
contiguous bars this is the prior signal timestamp (an idealized zero-latency
boundary fill). The last queued order is never filled without a following bar, so
final exposure stays open and marked (no invented liquidation price). The model
does **not** capture partial fills, bid/ask spread, queueing, latency, margin, or
contract rolls; `max_position` is a position limit, not a capital constraint.

### Contract units (important for dollar P&L)

`dollars_per_price_unit` and `tick_size` must match the price units of the loaded
data. CBOT corn (ZC) is 5,000 bushels, quoted in cents/bushel, tick = ¼ cent =
$12.50. So for cents-quoted prices use `dollars_per_price_unit=50`,
`tick_size=0.25`; for dollars-quoted prices use `5000` and `0.0025`. The
historical backtest script sets these automatically from `--price-unit`.

## Strategy rule

`BollingerMeanReversion(window=N)` targets +1 when the latest close is below the
N-bar Bollinger middle (moving average) and −1 when above; it holds on an exact
tie, insufficient history, or nonfinite prices in the window. Only the middle
line is used — the bands and `num_std` do not affect this rule. An entry-at-band /
exit-at-middle variant would be a separate strategy.

## Testing and conventions

The suite runs on synthetic data (no Databento key) and covers each stage's value
behavior, look-ahead rejection, and input validation. Code follows the repo's
`ruff` + `mypy` configuration (Google docstrings, full type annotations, strict
typing); run `uv run ruff check src/finm37000 tests` and `uv run mypy` before
pushing.
