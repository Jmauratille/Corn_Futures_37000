"""Bar-based backtesting: config, state, execution, engine and strategies."""

from .engine import run_backtest
from .metrics import PeriodSummary, summarize_period
from .models import BacktestConfig, BacktestResult, PortfolioState
from .strategy import Strategy

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "PeriodSummary",
    "PortfolioState",
    "Strategy",
    "run_backtest",
    "summarize_period",
]
