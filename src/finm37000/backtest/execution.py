"""Next-open fill model with fixed per-tick slippage and per-contract fees."""

import pandas as pd

from .models import BacktestConfig, Fill, Order


class NextOpenExecution:
    """Fill an order at the next bar's open, adjusted for slippage and fees."""

    def __init__(self, config: BacktestConfig) -> None:
        """Store the backtest config supplying tick size, slippage and fees."""
        self.config = config

    def fill(
        self, order: Order, execution_time: pd.Timestamp, open_price: float,
    ) -> Fill:
        """Return the fill: open price slipped against the order side, plus fees."""
        direction = 1 if order.quantity > 0 else -1

        price = (
            open_price
            + direction
            * self.config.slippage_ticks
            * self.config.tick_size
        )

        return Fill(
            signal_time=order.signal_time,
            execution_time=execution_time,
            quantity=order.quantity,
            price=price,
            fees=abs(order.quantity) * self.config.fee_per_contract,
        )
