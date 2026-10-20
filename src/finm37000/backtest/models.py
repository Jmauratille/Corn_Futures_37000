from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str
    initial_capital: float
    dollars_per_price_unit: float
    tick_size: float
    fee_per_contract: float = 0.0
    slippage_ticks: int = 0
    max_position: int = 1


@dataclass(frozen=True)
class PortfolioState:
    position: int = 0
    average_entry_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    cumulative_fees: float = 0.0


@dataclass(frozen=True)
class Order:
    signal_time: pd.Timestamp
    quantity: int  # Positive = buy; negative = sell.


@dataclass(frozen=True)
class Fill:
    signal_time: pd.Timestamp
    execution_time: pd.Timestamp
    quantity: int
    price: float
    fees: float


@dataclass
class BacktestResult:
    config: BacktestConfig
    history: pd.DataFrame
    trades: pd.DataFrame