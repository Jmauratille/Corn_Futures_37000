import math
import unittest

from finm37000.backtest import run_backtest, summarize_period

from .test_backtest_engine import ScriptedStrategy, bars, config


def run(targets: list[int], **kwargs: float) -> object:
    return run_backtest(bars(), ScriptedStrategy(targets), config(**kwargs),
                        bar_duration="5min")


class TestSummarizePeriod(unittest.TestCase):
    def test_gross_net_and_turnover_full_sample(self) -> None:
        summary = summarize_period(run([1, -1, 0, 0], slippage_ticks=1))
        # Fills of +1, -2 (reversal) and +1: four contracts in three fills.
        self.assertEqual(summary.contracts_traded, 4)
        self.assertEqual(summary.fills, 3)
        self.assertEqual(summary.fees, 4.)
        self.assertEqual(summary.slippage_cost, 4 * 0.25 * 10.)
        self.assertEqual(summary.net_pnl, 36.)
        self.assertEqual(summary.gross_pnl, 50.)
        self.assertEqual(summary.cost_per_contract, 3.5)
        self.assertEqual(summary.bars, 4)

    def test_gross_does_not_depend_on_costs(self) -> None:
        free = summarize_period(run([1, -1, 0, 0], fee_per_contract=0.))
        costly = summarize_period(run([1, -1, 0, 0], slippage_ticks=2,
                                      fee_per_contract=5.))
        self.assertEqual(free.gross_pnl, free.net_pnl)
        self.assertEqual(costly.gross_pnl, free.gross_pnl)
        self.assertEqual(costly.net_pnl,
                         costly.gross_pnl - costly.slippage_cost - costly.fees)

    def test_period_uses_only_in_period_moves_and_fills(self) -> None:
        result = run([1, -1, 0, 0], slippage_ticks=1)
        split = bars().index[2]
        dev = summarize_period(result, end=split)
        hold = summarize_period(result, start=split)
        self.assertEqual((dev.bars, hold.bars), (2, 2))
        self.assertEqual((dev.contracts_traded, hold.contracts_traded), (1, 3))
        self.assertEqual(hold.start, split)
        for field in ("gross_pnl", "net_pnl", "fees", "slippage_cost"):
            self.assertAlmostEqual(getattr(dev, field) + getattr(hold, field),
                                   getattr(summarize_period(result), field))

    def test_flat_strategy_has_no_turnover_or_pnl(self) -> None:
        summary = summarize_period(run([0, 0, 0, 0], slippage_ticks=1))
        self.assertEqual(summary.contracts_traded, 0)
        self.assertEqual(summary.net_pnl, 0.)
        self.assertEqual(summary.gross_pnl, 0.)
        self.assertEqual(summary.time_in_market, 0.)
        self.assertTrue(math.isnan(summary.cost_per_contract))

    def test_drawdown_and_time_in_market(self) -> None:
        summary = summarize_period(run([1, -1, 0, 0], fee_per_contract=0.))
        # Net P&L path is 0, 10, 40, 50: never below its running peak.
        self.assertEqual(summary.net_max_drawdown, 0.)
        self.assertEqual(summary.time_in_market, 0.5)

    def test_empty_period_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize_period(run([0, 0, 0, 0]), start=bars().index[-1]
                             + bars().index.freq)
