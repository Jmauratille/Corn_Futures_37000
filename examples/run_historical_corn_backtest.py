"""Backtest six calendar months of ZCH7 trades; report gross vs net P&L.

The rule and its settings are fixed before the run. The final
``--holdout-fraction`` of bars is reported separately as held-out data;
do not tune --window, --bar-rule or costs on its results.

Cached history (use its actual exclusive coverage endpoint):
    python examples/run_historical_corn_backtest.py --end 2026-10-26T00:00:00-05:00

Download and cache six months through the latest available complete day:
    python examples/run_historical_corn_backtest.py --download

Downloads consume Databento credits. No fills are printed and no orders are sent.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from finm37000 import build_analysis_bars
from finm37000.backtest.bollinger import BollingerMeanReversion
from finm37000.backtest.engine import run_backtest
from finm37000.backtest.metrics import PeriodSummary, summarize_period
from finm37000.backtest.models import BacktestConfig, BacktestResult

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
    parser.add_argument("--holdout-fraction", type=float, default=0.3,
                        help="Final fraction of bars reported as held-out data.")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/ZCH7_net_pnl.png"),
                        help="PNG chart; a .json report is written beside it.")
    args = parser.parse_args(argv)
    if not 0 < args.holdout_fraction < 1:
        parser.error("--holdout-fraction must be between 0 and 1.")
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


def gross_pnl_path(result: BacktestResult) -> pd.Series:
    """Cumulative P&L before slippage and fees, at each bar close."""
    history, config = result.history, result.config
    traded = history["position"].diff().fillna(history["position"]).abs()
    slippage = (traded.cumsum() * config.slippage_ticks * config.tick_size
                * config.dollars_per_price_unit)
    return history["net_pnl"] + history["cumulative_fees"] + slippage


def save_pnl_plot(  # noqa: PLR0913 (plot annotations describe the run)
    result: BacktestResult,
    output: Path,
    *,
    holdout_start: pd.Timestamp,
    start: pd.Timestamp,
    end: pd.Timestamp,
    window: int,
    rule: str,
    fee: float,
    slippage: int,
) -> Path:
    """Save bar-close gross and net P&L, marking the held-out period."""
    import matplotlib as mpl  # noqa: PLC0415 (optional plotting dependency)

    mpl.use("Agg")
    import matplotlib.dates as mdates  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.ticker import StrMethodFormatter  # noqa: PLC0415

    history = result.history
    if history.empty or "net_pnl" not in history:
        msg = "Backtest produced no net_pnl history."
        raise ValueError(msg)
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    try:
        ax.plot(history.index, gross_pnl_path(result), color="#9a9a9a",
                linewidth=1.1, label="Gross (before slippage and fees)")
        ax.plot(history.index, history["net_pnl"], color="#176b87",
                linewidth=1.4, label="Net")
        ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
        ax.axvspan(holdout_start, history.index[-1], color="#e8a33d",
                   alpha=0.12, label="Held-out period")
        ax.legend(loc="center left", fontsize=9)
        ax.set_title(f"{SYMBOL} | Bollinger middle-line mean reversion\n"
                     f"{start.date()} to {end.date()} (end exclusive) | "
                     f"{rule} bars | window={window}")
        ax.set_xlabel("Date (America/Chicago)")
        ax.set_ylabel("Cumulative P&L (USD)")
        ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
        chicago = ZoneInfo("America/Chicago")
        ax.xaxis.set_major_locator(mdates.MonthLocator(tz=chicago))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y", tz=chicago))
        ax.grid(alpha=0.2)
        final = history["net_pnl"].iloc[-1]
        ax.text(0.01, 0.03, f"Final marked net P&L: ${final:,.2f}\n"
                f"Fee: ${fee:g}/contract/fill | slippage: {slippage} tick(s)\n"
                "Includes unrealized P&L; no final liquidation",
                transform=ax.transAxes, va="bottom", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        fig.savefig(output, dpi=200)
    finally:
        plt.close(fig)
    return output


def format_summary(name: str, summary: PeriodSummary) -> str:
    """One readable block of gross/net/turnover figures for a period."""
    return (
        f"{name}: {summary.start} to {summary.end} ({summary.bars:,} bars)\n"
        f"  gross P&L      ${summary.gross_pnl:>12,.2f}"
        f"  (${summary.gross_pnl_per_contract:,.2f}/contract traded)\n"
        f"  slippage       ${summary.slippage_cost:>12,.2f}\n"
        f"  fees           ${summary.fees:>12,.2f}\n"
        f"  net P&L        ${summary.net_pnl:>12,.2f}\n"
        f"  turnover       {summary.contracts_traded:>13,} contracts in "
        f"{summary.fills:,} fills\n"
        f"  net max DD     ${summary.net_max_drawdown:>12,.2f}"
        f" | daily net Sharpe {summary.daily_net_sharpe:.2f}"
        f" | in market {summary.time_in_market:.0%}"
    )


def write_report(  # noqa: PLR0913 (records every assumption of the run)
    path: Path,
    args: argparse.Namespace,
    config: BacktestConfig,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    periods: dict[str, PeriodSummary],
) -> Path:
    """Write the run's assumptions and per-period results as JSON."""
    report = {
        "contract": SYMBOL,
        "dataset": "GLBX.MDP3 trades",
        "sample": {"start": str(start), "end_exclusive": str(end)},
        "bar_duration": args.bar_rule,
        "rule": f"BollingerMeanReversion(window={args.window}): +1 below the "
                "rolling mean of closes, -1 above, hold on ties",
        "horizon": "position held until the next signal change; signal "
                   "evaluated at every bar close",
        "execution": "next observed bar open (one-bar delay), market order",
        "costs": {
            "slippage_ticks_per_contract": config.slippage_ticks,
            "tick_size": config.tick_size,
            "fee_per_contract_per_fill": config.fee_per_contract,
            "dollars_per_price_unit": config.dollars_per_price_unit,
        },
        "max_position": config.max_position,
        "holdout_fraction": args.holdout_fraction,
        "settings_chosen_before_holdout": True,
        "periods": {
            name: {key: str(value) if isinstance(value, pd.Timestamp) else value
                   for key, value in asdict(summary).items()}
            | {"gross_pnl_per_contract": summary.gross_pnl_per_contract,
               "cost_per_contract": summary.cost_per_contract}
            for name, summary in periods.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    return path


def main(argv: list[str] | None = None) -> Path:
    """Run the six-month ZCH7 Bollinger backtest; save chart and report."""
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
    holdout_start = bars.index[int(len(bars) * (1 - args.holdout_fraction))]
    periods = {
        "full": summarize_period(result),
        "development": summarize_period(result, end=holdout_start),
        "holdout": summarize_period(result, start=holdout_start),
    }
    for name, summary in periods.items():
        print(format_summary(name, summary))
    output = save_pnl_plot(result, args.output, holdout_start=holdout_start,
                           start=start, end=end, window=args.window,
                           rule=args.bar_rule, fee=args.fee_per_contract,
                           slippage=args.slippage_ticks)
    report = write_report(args.output.with_suffix(".json"), args, config,
                          start=start, end=end, periods=periods)
    print(f"Saved gross/net P&L chart: {output}")
    print(f"Saved report: {report}")
    return output


if __name__ == "__main__":
    main()
