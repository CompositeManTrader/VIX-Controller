"""
config.py — Single source of truth para constantes, TTLs y defaults.

Elimina magic numbers dispersos por app.py. Todos los parámetros que
podrían querer ajustarse (ventanas, thresholds, tasas fallback, pesos)
viven aquí.
"""
from __future__ import annotations
from zoneinfo import ZoneInfo

# ──────────────────────────────────────────────────────────────
# TIMEZONES
# ──────────────────────────────────────────────────────────────
CDMX_TZ = ZoneInfo("America/Mexico_City")
NYSE_TZ = ZoneInfo("America/New_York")

# ──────────────────────────────────────────────────────────────
# RATES (fallback, solo si fetch dinámico falla)
# ──────────────────────────────────────────────────────────────
# Risk-free: 13-week T-Bill (^IRX) anualizado, como decimal.
# Dividend yield: SPY TTM dividend / price, como decimal.
FALLBACK_RF_ANNUAL = 0.043       # 4.3% — actualizar periódicamente
FALLBACK_DIV_YIELD = 0.013       # 1.3% — SPY típico

# ──────────────────────────────────────────────────────────────
# IMPLIED VOLATILITY (Black-Scholes + Brent)
# ──────────────────────────────────────────────────────────────
IV_LOWER = 0.01                  # 1% — IVs menores son ruido o pricing error
IV_UPPER = 3.00                  # 300% — acima es fuera de rango razonable SPX
IV_TOL   = 1e-6

# ──────────────────────────────────────────────────────────────
# TIME CONVENTIONS
# ──────────────────────────────────────────────────────────────
CAL_DAYS_YEAR = 365.0            # para descuento (r, q) y BS pricing
TRADING_DAYS_YEAR = 252.0        # para vol annualization de returns

# ──────────────────────────────────────────────────────────────
# STRATEGY — BB × CONTANGO
# ──────────────────────────────────────────────────────────────
BB_WINDOW = 20
BB_STD_MULT = 2.0
CONTANGO_MIN_DAYS = 60           # mínimo de historia para percentiles

# ──────────────────────────────────────────────────────────────
# BAROMETER VTS (ventanas y pesos)
# ──────────────────────────────────────────────────────────────
VTS_ROLLING_WINDOW = 252         # 1 año trading ≈ 252 días
VTS_MIN_OBS_RATIO = 1 / 3        # min_obs = max(60, window // 3)
VTS_MIN_OBS_FLOOR = 60

# Pesos ponderados del score (normalizados al final). Todos > 0.
VTS_WEIGHTS = {
    "m1_m2":           1.2,
    "m4_m7":           1.0,
    "vx30_vix_roll":   1.3,
    "vix_vvix_voli":   1.0,
    "cvo":             1.0,
    "vix_voli_resid":  0.8,
    "traders_vrp":     1.0,
    "skew":            0.8,
    "vix_vix3m":       1.2,
    "extreme_pc":      0.8,
    "vix9d_vix":       1.1,
    "volpocalypse":    1.5,
    "vix_level":       0.7,
}

# ──────────────────────────────────────────────────────────────
# CACHE TTLs (segundos). Un solo lugar.
# ──────────────────────────────────────────────────────────────
CACHE_TTL = {
    "cboe_scrape":   55,
    "yahoo_spot":    55,
    "yahoo_options": 3600,
    "edge_extra":    300,
    "barometer":     300,
    "backtest":      3600,
    "rates":         3600,
}

# ──────────────────────────────────────────────────────────────
# BACKTEST
# ──────────────────────────────────────────────────────────────
WF_WINDOW_MONTHS = 6
MIN_BACKTEST_LEN = 60
TRANSACTION_COST_BPS = 5         # 0.05% por lado — conservador (SVXY típico)

# ──────────────────────────────────────────────────────────────
# GEX
# ──────────────────────────────────────────────────────────────
GEX_DEALER_SIGN_CALL = -1        # dealers short calls
GEX_DEALER_SIGN_PUT  = +1        # dealers long puts
GEX_CONTRACT_MULT    = 100       # 1 contrato = 100 shares
GEX_POINT_MULT       = 0.01      # GEX por 1% movimiento

# ──────────────────────────────────────────────────────────────
# HAR-RV
# ──────────────────────────────────────────────────────────────
HAR_HORIZONS = (1, 5, 22)        # daily, weekly, monthly

# ──────────────────────────────────────────────────────────────
# SVI (Gatheral 2004)
# ──────────────────────────────────────────────────────────────
SVI_MAX_ITER = 500
SVI_MIN_POINTS = 5
SVI_RHO_MAX = 0.99               # rechaza fits con |rho| >= 0.99
