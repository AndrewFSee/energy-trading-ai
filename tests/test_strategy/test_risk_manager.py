"""Tests for the RiskManager module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.risk_manager import RiskManager, RiskReport


class TestRiskManager:
    """Unit tests for RiskManager."""

    def test_init_defaults(self) -> None:
        """RiskManager should initialise with correct defaults."""
        rm = RiskManager()
        assert rm.max_position_size == 0.20
        assert rm.max_drawdown_limit == 0.15
        assert rm.var_confidence == 0.99

    def test_check_position_size_within_limit(self) -> None:
        """Position within limit should not be modified."""
        rm = RiskManager(max_position_size=0.20)
        result = rm.check_position_size(0.15)
        assert abs(result - 0.15) < 1e-9

    def test_check_position_size_capped(self) -> None:
        """Position exceeding limit should be capped."""
        rm = RiskManager(max_position_size=0.20)
        result = rm.check_position_size(0.35)
        assert abs(result - 0.20) < 1e-9

    def test_check_position_size_short_capped(self) -> None:
        """Short position exceeding limit should be capped to -max."""
        rm = RiskManager(max_position_size=0.20)
        result = rm.check_position_size(-0.30)
        assert abs(result - (-0.20)) < 1e-9

    def test_update_drawdown_no_halt(self) -> None:
        """Drawdown within limit should not halt trading."""
        rm = RiskManager(max_drawdown_limit=0.15)
        # Portfolio drops 10% — below limit
        drawdown = rm.update_drawdown(0.90)
        assert abs(drawdown - (-0.10)) < 1e-9
        assert not rm.is_trading_halted()

    def test_update_drawdown_triggers_halt(self) -> None:
        """Drawdown exceeding limit should halt trading."""
        rm = RiskManager(max_drawdown_limit=0.10)
        rm.update_drawdown(1.0)  # Set peak
        rm.update_drawdown(0.85)  # 15% drawdown > 10% limit
        assert rm.is_trading_halted()

    def test_reset_halt(self) -> None:
        """reset_halt() should clear the trading halt flag."""
        rm = RiskManager(max_drawdown_limit=0.05)
        rm.update_drawdown(1.0)
        rm.update_drawdown(0.90)  # 10% drawdown > 5% limit
        assert rm.is_trading_halted()
        rm.reset_halt()
        assert not rm.is_trading_halted()

    def test_compute_var(self) -> None:
        """VaR should be a positive float."""
        rm = RiskManager()
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 500))
        var = rm.compute_var(returns, confidence=0.99)
        assert var > 0
        assert isinstance(var, float)

    def test_compute_var_95_less_than_99(self) -> None:
        """95% VaR should be less than 99% VaR (smaller loss at 95%)."""
        rm = RiskManager()
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 500))
        var_95 = rm.compute_var(returns, confidence=0.95)
        var_99 = rm.compute_var(returns, confidence=0.99)
        assert var_95 <= var_99

    def test_compute_stop_loss_long(self) -> None:
        """Stop loss for long should be below entry price."""
        rm = RiskManager(stop_loss_atr_multiple=2.0)
        entry = 80.0
        atr = 2.0
        stop = rm.compute_stop_loss(entry, atr, direction=1)
        assert stop < entry
        assert abs(stop - (entry - 2.0 * atr)) < 1e-9

    def test_compute_stop_loss_short(self) -> None:
        """Stop loss for short should be above entry price."""
        rm = RiskManager(stop_loss_atr_multiple=2.0)
        entry = 80.0
        atr = 2.0
        stop = rm.compute_stop_loss(entry, atr, direction=-1)
        assert stop > entry

    def test_compute_take_profit_long(self) -> None:
        """Take profit for long should be above entry price."""
        rm = RiskManager(take_profit_atr_multiple=4.0)
        entry = 80.0
        atr = 2.0
        tp = rm.compute_take_profit(entry, atr, direction=1)
        assert tp > entry
        assert abs(tp - (entry + 4.0 * atr)) < 1e-9

    def test_generate_report_returns_risk_report(self) -> None:
        """generate_report() should return a RiskReport instance."""
        rm = RiskManager()
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 252))
        report = rm.generate_report(
            portfolio_value=1_000_000,
            returns=returns,
            positions={"wti": 0.15, "natgas": 0.10},
        )
        assert isinstance(report, RiskReport)
        assert report.current_leverage > 0
        assert report.var_95 > 0
        assert report.var_99 > 0

    def test_position_size_zero_unchanged(self) -> None:
        """Zero position should remain zero after risk check."""
        rm = RiskManager()
        result = rm.check_position_size(0.0)
        assert result == 0.0
