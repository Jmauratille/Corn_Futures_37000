"""Immutable configuration, state, order, fill and result records."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """Contract, cost and risk settings for a backtest run."""

    symbol: str
    initial_capital: float
    dollars_per_price_unit: float
    tick_size: float
    fee_per_contract: float = 0.0
    slippage_ticks: int = 0
    max_position: int = 1


@dataclass(frozen=True)
class PortfolioState:
    """Position, average entry price and running P&L/fees."""

    position: int = 0
    average_entry_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    cumulative_fees: float = 0.0


@dataclass(frozen=True)
class Order:
    """A queued order; positive quantity buys, negative sells."""

    signal_time: pd.Timestamp
    quantity: int


@dataclass(frozen=True)
class Fill:
    """An executed order with its slipped price and fees."""

    signal_time: pd.Timestamp
    execution_time: pd.Timestamp
    quantity: int
    price: float
    fees: float


@dataclass
class BacktestResult:
    """Backtest output: the config plus per-bar history and per-fill trades."""

    config: BacktestConfig
    history: pd.DataFrame
    trades: pd.DataFrame
