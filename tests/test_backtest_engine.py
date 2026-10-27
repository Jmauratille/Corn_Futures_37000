import unittest
from typing import Any

import pandas as pd

from finm37000.backtest.bollinger import BollingerMeanReversion
from finm37000.backtest.engine import apply_fill, run_backtest
from finm37000.backtest.models import BacktestConfig, Fill, PortfolioState


def config(**kwargs: Any) -> BacktestConfig:  # noqa: ANN401
    values: dict[str, Any] = {
        "symbol": "TEST", "initial_capital": 1000., "dollars_per_price_unit": 10.,
        "tick_size": 0.25, "fee_per_contract": 1., "max_position": 1,
    }
    return BacktestConfig(**(values | kwargs))


def bars() -> pd.DataFrame:
    return pd.DataFrame({
        "open": [100., 102., 104., 101.], "close": [101., 103., 102., 100.],
        "high": [101., 103., 104., 101.], "low": [100., 102., 102., 100.],
        "symbol": ["TEST"] * 4,
    }, index=pd.date_range("2026-10-01 14:05", periods=4, freq="5min", tz="UTC"))


class ScriptedStrategy:
    # Holds scripted targets, including deliberately invalid ones for the
    # rejection test, so the element type is intentionally Any.
    def __init__(self, targets: list[Any]) -> None:  # noqa: ANN401
        self.targets = targets
        self.observed: list[tuple[pd.DataFrame, PortfolioState]] = []

    def on_bar(self, history: pd.DataFrame, portfolio: PortfolioState) -> int:
        self.observed.append((history.copy(), portfolio))
        target: int = self.targets[len(history) - 1]
        return target


class TestEngine(unittest.TestCase):
    def test_known_long_reversal_close_and_fees(self) -> None:
        result = run_backtest(bars(), ScriptedStrategy([1, -1, 0, 0]),
                              config(), bar_duration="5min")
        self.assertEqual(result.trades.quantity.tolist(), [1, -2, 1])
        self.assertEqual(result.trades.price.tolist(), [102., 104., 101.])
        self.assertEqual(result.history.position.tolist(), [0, 1, -1, 0])
        self.assertEqual(result.history.realized_pnl.tolist(), [0., 0., 20., 50.])
        self.assertEqual(result.history.unrealized_pnl.tolist(), [0., 10., 20., 0.])
        self.assertEqual(result.history.cumulative_fees.tolist(), [0., 1., 3., 4.])
        self.assertEqual(result.history.equity.tolist(),
                         [1000., 1009., 1037., 1046.])
        self.assertEqual(result.trades.execution_time.iloc[0], bars().index[0])

    def test_adverse_slippage_both_directions(self) -> None:
        result = run_backtest(bars(), ScriptedStrategy([1, -1, 0, 0]),
                              config(slippage_ticks=1), bar_duration="5min")
        self.assertEqual(result.trades.price.tolist(), [102.25, 103.75, 101.25])
        self.assertEqual(result.history.equity.iloc[-1], 1036.)

    def test_history_is_causal_and_input_unchanged(self) -> None:
        source = bars()
        original = source.copy(deep=True)
        strategy = ScriptedStrategy([1, 1, 1, 1])
        result = run_backtest(source, strategy, config(), bar_duration="5min")
        self.assertEqual([len(h) for h, _ in strategy.observed], [1, 2, 3, 4])
        self.assertEqual([p.position for _, p in strategy.observed], [0, 1, 1, 1])
        self.assertEqual(len(result.trades), 1)
        pd.testing.assert_frame_equal(source, original)

    def test_last_signal_unfilled_and_final_position_marked(self) -> None:
        result = run_backtest(bars(), ScriptedStrategy([0, 0, 1, -1]),
                              config(), bar_duration="5min")
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.history.position.iloc[-1], 1)
        self.assertEqual(result.history.target_position.iloc[-1], -1)
        self.assertEqual(result.history.order_quantity.iloc[-1], -2)
        self.assertEqual(result.history.net_pnl.iloc[-1], -11.)

    def test_no_trades_has_stable_columns(self) -> None:
        result = run_backtest(bars(), ScriptedStrategy([0] * 4), config(),
                              bar_duration="5min")
        self.assertTrue(result.trades.empty)
        self.assertIn("fees", result.trades.columns)
        self.assertEqual(result.history.equity.tolist(), [1000.] * 4)

    def test_gap_fills_at_next_observed_open(self) -> None:
        source = bars().iloc[[0, 3]]
        result = run_backtest(source, ScriptedStrategy([1, 1]), config(),
                              bar_duration="5min")
        self.assertEqual(result.trades.execution_time.iloc[0],
                         source.index[1] - pd.Timedelta("5min"))
        self.assertEqual(result.trades.price.iloc[0], 101.)

    def test_add_reduce_reverse_and_close(self) -> None:
        cfg = config(max_position=4)
        p = PortfolioState()
        timestamp = bars().index[0]

        def fill(q: int, price: float) -> Fill:
            return Fill(timestamp, timestamp, q, price, abs(q))

        p = apply_fill(p, fill(2, 100.), cfg)
        p = apply_fill(p, fill(2, 102.), cfg)
        self.assertEqual(p.average_entry_price, 101.)
        p = apply_fill(p, fill(-1, 104.), cfg)
        self.assertEqual((p.position, p.average_entry_price, p.realized_pnl),
                         (3, 101., 30.))
        p = apply_fill(p, fill(-4, 99.), cfg)
        self.assertEqual((p.position, p.average_entry_price, p.realized_pnl),
                         (-1, 99., -30.))
        p = apply_fill(p, fill(1, 98.), cfg)
        self.assertEqual((p.position, p.average_entry_price, p.realized_pnl),
                         (0, 0., -20.))
        self.assertEqual(p.cumulative_fees, 10.)

    def test_reject_invalid_targets(self) -> None:
        for target in (2, 0.5, True):
            with self.subTest(target=target), self.assertRaises(ValueError):
                run_backtest(bars(), ScriptedStrategy([target] * 4), config(),
                             bar_duration="5min")

    def test_reject_bad_bars(self) -> None:
        invalid = [bars().iloc[::-1], bars().iloc[:0], bars().assign(high=1.),
                   bars().assign(close=float("nan")), bars().assign(symbol="OTHER")]
        for source in invalid:
            with self.assertRaises(ValueError):
                run_backtest(source, ScriptedStrategy([0] * 4), config(),
                             bar_duration="5min")
        with self.assertRaises(ValueError):
            run_backtest(bars(), ScriptedStrategy([0] * 4), config(),
                         bar_duration="10min")

    def test_bollinger_integration(self) -> None:
        result = run_backtest(bars(), BollingerMeanReversion(2), config(),
                              bar_duration="5min")
        self.assertEqual(result.history.target_position.tolist(), [0, -1, 1, 1])
        self.assertEqual(result.trades.quantity.tolist(), [-1, 2])
        self.assertEqual(result.history.net_pnl.iloc[-1], 17.)


if __name__ == "__main__":
    unittest.main()
