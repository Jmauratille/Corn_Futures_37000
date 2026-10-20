from .models import BacktestConfig, Fill, Order


class NextOpenExecution:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def fill(self, order: Order, execution_time, open_price: float) -> Fill:
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