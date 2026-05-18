"""
rates.py — Risk-free rate y dividend yield dinámicos.

Reemplaza los `r=0.043, q=0.013` hardcoded en app.py. Usa yfinance
para leer ^IRX (13-week T-Bill) y SPY TTM dividend yield.

Fallback a config.FALLBACK_* si la red falla.
"""
from __future__ import annotations
import logging
from functools import lru_cache
from time import time

import yfinance as yf

from .config import FALLBACK_RF_ANNUAL, FALLBACK_DIV_YIELD, CACHE_TTL

log = logging.getLogger(__name__)

# TTL-based cache manual (no dependemos de st.cache_data aquí para
# poder testear sin Streamlit).
_cache: dict[str, tuple[float, float]] = {}


def _ttl_get(key: str, ttl: int):
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, val = entry
    if time() - ts > ttl:
        return None
    return val


def _ttl_set(key: str, val: float) -> None:
    _cache[key] = (time(), val)


def get_risk_free_rate(ttl: int = None) -> float:
    """
    Lee ^IRX (13-week T-Bill yield, cotizado en % anualizado).
    Retorna decimal (0.045 = 4.5%). Fallback: FALLBACK_RF_ANNUAL.

    Captura cualquier excepción de yfinance (incluye YFRateLimitError) y
    cae al fallback — nunca propaga.
    """
    ttl = ttl or CACHE_TTL["rates"]
    cached = _ttl_get("rf", ttl)
    if cached is not None:
        return cached
    try:
        h = yf.Ticker("^IRX").history(period="5d")
        if not h.empty and "Close" in h.columns:
            val = float(h["Close"].iloc[-1]) / 100.0
            if 0.0 <= val <= 0.20:
                _ttl_set("rf", val)
                return val
    except Exception as e:   # incluye YFRateLimitError, OSError, etc.
        log.warning("get_risk_free_rate: fallback por %s", e)
    _ttl_set("rf", FALLBACK_RF_ANNUAL)
    return FALLBACK_RF_ANNUAL


def get_dividend_yield(ticker: str = "SPY", ttl: int = None) -> float:
    """
    Dividend yield TTM para el ticker dado (decimal).
    Usa yfinance .info.trailingAnnualDividendYield cuando está disponible;
    si no, calcula TTM como sum(dividends últimos 365d) / price.

    Captura cualquier excepción de yfinance (incluye YFRateLimitError) y
    cae al fallback — nunca propaga.
    """
    ttl = ttl or CACHE_TTL["rates"]
    key = f"div_{ticker}"
    cached = _ttl_get(key, ttl)
    if cached is not None:
        return cached
    try:
        t = yf.Ticker(ticker)
        info = getattr(t, "info", {}) or {}
        y = info.get("trailingAnnualDividendYield") or info.get("dividendYield")
        if y is not None and 0.0 <= y <= 0.15:
            _ttl_set(key, float(y))
            return float(y)
        divs = t.dividends
        h = t.history(period="5d")
        if divs is not None and not divs.empty and not h.empty:
            import pandas as pd
            cutoff = divs.index.max() - pd.Timedelta(days=365)
            ttm = float(divs[divs.index > cutoff].sum())
            price = float(h["Close"].iloc[-1])
            if price > 0:
                y = ttm / price
                if 0.0 <= y <= 0.15:
                    _ttl_set(key, y)
                    return y
    except Exception as e:   # incluye YFRateLimitError
        log.warning("get_dividend_yield(%s): fallback por %s", ticker, e)
    _ttl_set(key, FALLBACK_DIV_YIELD)
    return FALLBACK_DIV_YIELD
