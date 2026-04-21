"""
Tests para backtest helpers: sharpe, sortino, dd, cagr, apply_signal_returns.
"""
import numpy as np
import pandas as pd
import pytest

from vix_controller.quant.backtest import (
    apply_signal_returns, sharpe, sortino, max_drawdown, cagr,
)


class TestApplySignalReturns:
    def test_no_lookahead(self):
        sig = pd.Series([0, 0, 1, 1, 0, 0])
        ret = pd.Series([0.01, -0.01, 0.02, -0.03, 0.01, 0.02])
        out = apply_signal_returns(sig, ret, cost_bps=0)
        # sig.shift(1) = [NaN, 0, 0, 1, 1, 0]
        expected_gross = [0.0, 0.0, 0.0, -0.03, 0.01, 0.0]
        np.testing.assert_allclose(out.values, expected_gross, atol=1e-8)

    def test_costs_reduce_returns(self):
        sig = pd.Series([0, 1, 1, 0])
        ret = pd.Series([0.0, 0.01, 0.01, 0.01])
        out_no_cost = apply_signal_returns(sig, ret, cost_bps=0)
        out_cost = apply_signal_returns(sig, ret, cost_bps=10)
        # Hay turnover en t=1 y t=3 → menos retorno con costos
        assert out_cost.sum() < out_no_cost.sum()


class TestSharpeSortino:
    def test_sharpe_constant_positive(self):
        # Retornos constantes positivos → std=0 → NaN
        r = pd.Series([0.001] * 100)
        assert np.isnan(sharpe(r))

    def test_sharpe_random(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.0005, 0.01, 500))
        s = sharpe(r)
        assert np.isfinite(s)
        # Sharpe ≈ 0.0005/0.01 · √252 ≈ 0.79
        assert 0.3 < s < 1.3

    def test_sortino_greater_than_sharpe_for_positive_skew(self):
        # Positive skew → downside std < total std → sortino > sharpe
        r = pd.Series([-0.005] * 20 + [0.01] * 80)
        sh = sharpe(r)
        so = sortino(r)
        assert np.isfinite(sh) and np.isfinite(so)
        assert so > sh


class TestMaxDrawdown:
    def test_monotonic_up_zero_dd(self):
        eq = pd.Series(np.arange(1, 101, dtype=float))
        assert max_drawdown(eq) == 0.0

    def test_known_dd(self):
        eq = pd.Series([100, 110, 120, 60, 80])  # peak=120, trough=60
        dd = max_drawdown(eq)
        assert dd == pytest.approx(-0.5, abs=1e-8)


class TestCAGR:
    def test_exact_double_in_one_year(self):
        # 252 días → 1 año
        eq = pd.Series(np.linspace(100, 200, 252))
        c = cagr(eq)
        assert c == pytest.approx(1.0, abs=0.02)
