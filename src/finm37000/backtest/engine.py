def run_backtest(
    bars: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig,
) -> BacktestResult:
    ...

def validate_bars(bars):
    ...

def apply_fill(portfolio, fill, config):
    """Handle additions, reductions, closes, and reversals."""
    ...

def mark_to_market(portfolio, close_price, config):
    ...

def make_history_record(timestamp, portfolio, config):
    ...