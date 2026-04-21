"""
Tests para rates: fallback cuando yfinance no disponible (offline).
"""
import pytest

from vix_controller import config as cfg
from vix_controller.rates import get_risk_free_rate, get_dividend_yield, _cache


@pytest.fixture(autouse=True)
def clear_cache():
    _cache.clear()
    yield
    _cache.clear()


class TestRatesFallback:
    def test_rf_returns_decimal_in_range(self):
        r = get_risk_free_rate()
        # Debe devolver decimal (no % bruto) en rango razonable
        assert 0.0 <= r <= 0.20

    def test_div_yield_returns_decimal_in_range(self):
        q = get_dividend_yield("SPY")
        assert 0.0 <= q <= 0.15

    def test_rf_cached(self):
        # Segunda llamada debe devolver exactamente el mismo valor
        r1 = get_risk_free_rate()
        r2 = get_risk_free_rate()
        assert r1 == r2

    def test_unknown_ticker_falls_back(self):
        q = get_dividend_yield("ZZZZ_NONEXISTENT")
        # Debe retornar el fallback (sin crashear)
        assert q == cfg.FALLBACK_DIV_YIELD
