"""
bs.py — Black-Scholes pricing y IV inversion.

Funciones puras, sin dependencias de Streamlit. Testeable.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from ..config import IV_LOWER, IV_UPPER, IV_TOL


def _d1_d2(S: float, X: float, r: float, T: float, v: float, q: float):
    """Devuelve (d1, d2) de Black-Scholes con dividendo continuo."""
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / X) + (r - q + 0.5 * v * v) * T) / (v * sqrtT)
    d2 = d1 - v * sqrtT
    return d1, d2


def bs_call(S: float, X: float, r: float, T: float, v: float, q: float) -> float:
    """Black-Scholes call price con dividendo continuo."""
    if S <= 0 or X <= 0 or T <= 0:
        return max(S - X, 0.0)
    if v <= 0:
        return max(S * np.exp(-q * T) - X * np.exp(-r * T), 0.0)
    d1, d2 = _d1_d2(S, X, r, T, v, q)
    return S * np.exp(-q * T) * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S: float, X: float, r: float, T: float, v: float, q: float) -> float:
    """Black-Scholes put price con dividendo continuo."""
    if S <= 0 or X <= 0 or T <= 0:
        return max(X - S, 0.0)
    if v <= 0:
        return max(X * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    d1, d2 = _d1_d2(S, X, r, T, v, q)
    return X * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_gamma(S: float, X: float, r: float, T: float, v: float, q: float) -> float:
    """Gamma (idéntica para calls y puts)."""
    if S <= 0 or X <= 0 or T <= 0 or v <= 0:
        return 0.0
    d1, _ = _d1_d2(S, X, r, T, v, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * v * np.sqrt(T))


def bs_gamma_vec(S: float, X: np.ndarray, r: float, T: float,
                 v: np.ndarray, q: float) -> np.ndarray:
    """
    Gamma vectorizada sobre arrays de strikes/IVs (misma fórmula que bs_gamma).
    Entradas inválidas (X<=0, v<=0, NaN) producen gamma 0 en esa posición.
    """
    X = np.asarray(X, dtype=float)
    v = np.asarray(v, dtype=float)
    out = np.zeros_like(X)
    ok = (X > 0) & (v > 0) & np.isfinite(X) & np.isfinite(v)
    if S <= 0 or T <= 0 or not ok.any():
        return out
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / X[ok]) + (r - q + 0.5 * v[ok] ** 2) * T) / (v[ok] * sqrtT)
    out[ok] = np.exp(-q * T) * norm.pdf(d1) / (S * v[ok] * sqrtT)
    return out


def bs_iv(S: float, X: float, r: float, T: float, price: float,
          option_type: str, q: float, tol: float = IV_TOL,
          lo: float = IV_LOWER, hi: float = IV_UPPER) -> float:
    """
    IV via Brent's method. Retorna NaN si no converge, está fuera de bounds
    de no-arbitraje, o la función no cambia de signo en [lo, hi].

    Validación adicional (anti-ruido):
      - Bracket de precio teórico
      - IV clamped a [lo, hi] = [IV_LOWER, IV_UPPER] (default 1%-300%)
      - Verifica cambio de signo antes de llamar brentq (evita ValueError)
    """
    if T <= 0 or S <= 0 or X <= 0 or not np.isfinite(price) or price <= 0:
        return np.nan

    fn = bs_call if option_type == "C" else bs_put

    price_lo = (max(S * np.exp(-q * T) - X * np.exp(-r * T), 0.0)
                if option_type == "C"
                else max(X * np.exp(-r * T) - S * np.exp(-q * T), 0.0))
    price_hi = (S * np.exp(-q * T) if option_type == "C"
                else X * np.exp(-r * T))
    if not (price_lo <= price <= price_hi):
        return np.nan

    f = lambda v: price - fn(S, X, r, T, v, q)
    f_lo, f_hi = f(lo), f(hi)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi):
        return np.nan
    if f_lo * f_hi > 0:
        return np.nan

    try:
        iv = brentq(f, lo, hi, xtol=tol, maxiter=100)
    except (ValueError, RuntimeError):
        return np.nan

    if iv <= tol or iv >= hi - tol:
        return np.nan
    return float(iv)
