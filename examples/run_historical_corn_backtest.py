"""Backtest six calendar months of ZCH7 trades and save a net-P&L PNG.

Cached history (use its actual exclusive coverage endpoint):
    python examples/run_historical_corn_backtest.py --end 2026-10-26T00:00:00-05:00

Download and cache six months through the latest available complete day:
    python examples/run_historical_corn_backtest.py --download

Downloads consume Databento credits. No fills are printed and no orders are sent.
"""

import argparse
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from finm37000 import build_analysis_bars
from finm37000.backtest.bollinger import BollingerMeanReversion
from finm37000.backtest.engine import run_backtest
from finm37000.backtest.models import BacktestConfig

SYMBOL = "ZCH7"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the historical backtest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end",
                        help="Exclusive coverage endpoint, including timezone.")
    parser.add_argument("--download", action="store_true",
                        help="Download into a range-specific cache if absent; "
                             "costs credits.")
    parser.add_argument("--data-dir", type=Path,
                        help="Directory with ZCH7_trades.parquet; "
                             "default: project data/raw.")
    parser.add_argument("--bar-rule", default="5min")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--fee-per-contract", type=float, default=0.0,
                        help="Dollars per contract per fill, not round trip. "
                             "Default: 0.")
    parser.add_argument("--slippage-ticks", type=int, default=1)
    parser.add_argument("--price-unit", choices=("cents", "dollars"),
                        default="cents",
                        help="Units already used by the price column. "
                             "No price conversion is done.")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/ZCH7_net_pnl.png"))
    args = parser.parse_args(argv)
    if not args.download and not args.end:
        parser.error("Cached data requires --end: supply the actual exclusive "
                     "coverage endpoint.")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png.")
    return args


def timezone_timestamp(value: str) -> pd.Timestamp:
    """Parse a timezone-aware timestamp, rejecting naive or missing values."""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tz is None:
        msg = "--end must be a nonmissing timestamp with a timezone."
        raise ValueError(msg)
    return timestamp


def load_trades(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Load local history or download a range into a separate reusable cache."""
    from finm37000.data_store import (  # noqa: PLC0415 (defer databento import)
        DEFAULT_DATASET,
        default_range,
        get_data_dir,
        history_path,
        load_history,
        pull_history,
    )

    data_dir = args.data_dir if args.data_dir is not None else get_data_dir()
    if not args.download:
        end = timezone_timestamp(args.end)
        start = end - pd.DateOffset(months=6)
        # --end is the caller's assertion of this file's actual coverage.
        return load_history(SYMBOL, data_dir=data_dir), start, end

    import databento as db  # noqa: PLC0415 (only needed when downloading)

    from finm37000.db_env_util import get_databento_api_key  # noqa: PLC0415

    client = db.Historical(get_databento_api_key())
    if args.end:
        end = timezone_timestamp(args.end)
        start = end - pd.DateOffset(months=6)
    else:
        start, end = default_range(client, months=6)
    # Include the exact requested UTC boundaries, avoiding accidental reuse of
    # the existing symbol-only parquet for a different six-month period.
    cache_name = f"{start.tz_convert('UTC').value}_{end.tz_convert('UTC').value}"
    directory = data_dir / "backtests" / SYMBOL / cache_name
    directory.mkdir(parents=True, exist_ok=True)
    path = history_path(SYMBOL, data_dir=directory)
    if not path.exists():
        cost = client.metadata.get_cost(
            dataset=DEFAULT_DATASET, symbols=[SYMBOL], stype_in="raw_symbol",
            schema="trades", start=start, end=end,
        )
        print(f"Downloading {SYMBOL} [{start}, {end}); "
              f"estimated cost ${float(cost):.4f}", flush=True)
        pull_history(client, SYMBOL, start, end, data_dir=directory)
    else:
        print(f"Using cached six-month download: {path}")
    return load_history(SYMBOL, data_dir=directory), start, end


def select_trades(
    trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
) -> pd.DataFrame:
    """Filter by exchange event time; never substitute receive timestamps."""
    if "ts_event" in trades:
        events = pd.DatetimeIndex(pd.to_datetime(trades["ts_event"]))
    elif isinstance(trades.index, pd.DatetimeIndex) and trades.index.name in {
        "ts_event", "event_time",
    }:
        events = trades.index
    else:
        msg = "Need ts_event or an event-time index; receive time is not enough."
        raise ValueError(msg)
    if events.tz is None or events.hasnans:
        msg = "Event times must be timezone-aware and nonmissing."
        raise ValueError(msg)
    selected = trades.loc[(events >= start) & (events < end)].copy()
    if selected.empty:
        msg = "No trades in the requested six-month range."
        raise ValueError(msg)
    if "ts_event" in selected:
        selected["ts_event"] = pd.to_datetime(selected["ts_event"])
    if "symbol" not in selected:
        # load_history(SYMBOL) selects a single-contract cache. Some stored
        # Databento frames omit symbol when symbol mapping was not retained.
        if ("instrument_id" in selected
                and len(selected["instrument_id"].unique()) != 1):
            msg = "Expected a single instrument in the contract cache."
            raise ValueError(msg)
        selected["symbol"] = SYMBOL
    elif selected["symbol"].isna().any() or not selected["symbol"].eq(SYMBOL).all():
        msg = f"Expected only {SYMBOL} in the cached data."
        raise ValueError(msg)
    return selected


def save_pnl_plot(  # noqa: PLR0913 (plot annotations describe the run)
    history: pd.DataFrame,
    output: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    window: int,
    rule: str,
    fee: float,
    slippage: int,
) -> Path:
    """Save actual bar-close net P&L, including fees and unrealized exposure."""
    import matplotlib as mpl  # noqa: PLC0415 (optional plotting dependency)

    mpl.use("Agg")
    import matplotlib.dates as mdates  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.ticker import StrMethodFormatter  # noqa: PLC0415

    if history.empty or "net_pnl" not in history:
        msg = "Backtest produced no net_pnl history."
        raise ValueError(msg)
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    try:
        ax.plot(history.index, history["net_pnl"], color="#176b87", linewidth=1.4)
        ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
        ax.set_title(f"{SYMBOL} | Bollinger middle-line mean reversion\n"
                     f"{start.date()} to {end.date()} (end exclusive) | "
                     f"{rule} bars | window={window}")
        ax.set_xlabel("Date (America/Chicago)")
        ax.set_ylabel("Cumulative net P&L (USD)")
        ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
            locator, tz=ZoneInfo("America/Chicago")))
        ax.grid(alpha=0.2)
        final = history["net_pnl"].iloc[-1]
        ax.text(0.01, 0.97, f"Final marked P&L: ${final:,.2f}\n"
                f"Fee: ${fee:g}/contract/fill | slippage: {slippage} tick(s)\n"
                "Includes unrealized P&L; no final liquidation",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        fig.savefig(output, dpi=200)
    finally:
        plt.close(fig)
    return output


def main(argv: list[str] | None = None) -> Path:
    """Run the six-month ZCH7 Bollinger backtest and save the net-P&L chart."""
    args = parse_args(argv)
    trades, start, end = load_trades(args)
    trades = select_trades(trades, start, end)
    bars = build_analysis_bars(trades, rule=args.bar_rule, end=end)
    # Do not include an initial partial interval if the requested start falls
    # inside a bar. The engine checks fixed durations and overlapping bars.
    duration = pd.Timedelta(pd.tseries.frequencies.to_offset(args.bar_rule))
    bars = bars.loc[bars.index - duration >= start]
    if len(bars) <= args.window:
        msg = "Not enough completed bars for warm-up plus execution."
        raise ValueError(msg)
    multiplier, tick_size = ((50.0, 0.25) if args.price_unit == "cents"
                             else (5000.0, 0.0025))
    config = BacktestConfig(
        symbol=SYMBOL, initial_capital=args.initial_capital,
        dollars_per_price_unit=multiplier, tick_size=tick_size,
        fee_per_contract=args.fee_per_contract,
        slippage_ticks=args.slippage_ticks, max_position=1,
    )
    print(f"Backtesting {len(bars):,} completed bars from "
          f"{bars.index[0]} to {bars.index[-1]}")
    print(f"Price units: {args.price_unit}/bushel; "
          f"first close: {bars['close'].iloc[0]:g}")
    result = run_backtest(bars, BollingerMeanReversion(window=args.window),
                          config, bar_duration=args.bar_rule)
    output = save_pnl_plot(result.history, args.output, start=start, end=end,
                           window=args.window, rule=args.bar_rule,
                           fee=args.fee_per_contract, slippage=args.slippage_ticks)
    print(f"Saved net P&L chart: {output}")
    return output


if __name__ == "__main__":
    main()
