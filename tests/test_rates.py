"""
Tests para rates — mockea yfinance para NO depender de red.

Cubre: parse OK, cache TTL, fallback si yfinance falla.
"""
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from vix_controller import config as cfg
from vix_controller.rates import get_risk_free_rate, get_dividend_yield, _cache


@pytest.fixture(autouse=True)
def clear_cache():
    _cache.clear()
    yield
    _cache.clear()


def _mk_history(close_value: float) -> pd.DataFrame:
    idx = pd.date_range("2026-04-14", periods=5, freq="D")
    return pd.DataFrame({"Close": [close_value] * 5}, index=idx)


class TestGetRiskFreeRate:
    def test_parses_irx_close_as_percent(self):
        ticker = MagicMock()
        ticker.history.return_value = _mk_history(4.25)  # 4.25% anual
        with patch("vix_controller.rates.yf.Ticker", return_value=ticker):
            r = get_risk_free_rate()
        assert r == pytest.approx(0.0425, abs=1e-6)

    def test_fallback_when_yf_raises(self):
        with patch("vix_controller.rates.yf.Ticker", side_effect=ConnectionError("no net")):
            r = get_risk_free_rate()
        assert r == cfg.FALLBACK_RF_ANNUAL

    def test_fallback_when_out_of_range(self):
        ticker = MagicMock()
        ticker.history.return_value = _mk_history(50.0)  # 50% → rechaza
        with patch("vix_controller.rates.yf.Ticker", return_value=ticker):
            r = get_risk_free_rate()
        assert r == cfg.FALLBACK_RF_ANNUAL

    def test_cached_second_call_does_not_hit_yf(self):
        ticker = MagicMock()
        ticker.history.return_value = _mk_history(4.0)
        with patch("vix_controller.rates.yf.Ticker", return_value=ticker) as mocked:
            r1 = get_risk_free_rate()
            r2 = get_risk_free_rate()
        assert r1 == r2
        assert mocked.call_count == 1  # segunda llamada vino del cache


class TestGetDividendYield:
    def test_uses_info_trailing_yield(self):
        ticker = MagicMock()
        ticker.info = {"trailingAnnualDividendYield": 0.0145}
        with patch("vix_controller.rates.yf.Ticker", return_value=ticker):
            q = get_dividend_yield("SPY")
        assert q == pytest.approx(0.0145, abs=1e-6)

    def test_fallback_when_ticker_raises(self):
        with patch("vix_controller.rates.yf.Ticker", side_effect=ConnectionError("no net")):
            q = get_dividend_yield("SPY")
        assert q == cfg.FALLBACK_DIV_YIELD

    def test_fallback_when_info_missing_and_no_divs(self):
        ticker = MagicMock()
        ticker.info = {}
        ticker.dividends = pd.Series(dtype=float)   # vacío
        ticker.history.return_value = _mk_history(450.0)
        with patch("vix_controller.rates.yf.Ticker", return_value=ticker):
            q = get_dividend_yield("SPY")
        assert q == cfg.FALLBACK_DIV_YIELD

    def test_computes_ttm_from_dividends_when_info_empty(self):
        ticker = MagicMock()
        ticker.info = {}
        idx = pd.date_range("2025-06-01", periods=4, freq="91D")
        ticker.dividends = pd.Series([1.5, 1.5, 1.5, 1.5], index=idx)
        ticker.history.return_value = _mk_history(500.0)  # ttm=6, price=500 → 1.2%
        with patch("vix_controller.rates.yf.Ticker", return_value=ticker):
            q = get_dividend_yield("SPY")
        assert q == pytest.approx(6.0 / 500.0, abs=1e-6)
