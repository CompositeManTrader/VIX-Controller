"""
backtest.py — Helpers puros para métricas de performance.

Extraídos de run_walkforward_backtest para ser testeables sin Streamlit.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..config import TRADING_DAYS_YEAR, TRANSACTION_COST_BPS


def apply_signal_returns(sig: pd.Series, vehicle_ret: pd.Series,
                         cost_bps: float = TRANSACTION_COST_BPS) -> pd.Series:
    """
    Aplica señal al retorno del vehículo SIN data leakage:
        ret_t = sig_{t-1} · vehicle_ret_t - cost·|Δsig|

    cost_bps: costo total (ida+vuelta) en basis points del AUM.
    """
    sig_shift = sig.shift(1).fillna(0).astype(float)
    gross = sig_shift * vehicle_ret
    turnover = sig.diff().abs().fillna(0)
    cost = turnover * (cost_bps / 10_000.0)
    return gross - cost


def sharpe(returns: pd.Series, rf_daily: float = 0.0) -> float:
    r = returns.dropna()
    if len(r) < 20:
        return float("nan")
    excess = r - rf_daily
    std = excess.std(ddof=0)
    if std <= 0:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_YEAR))


def sortino(returns: pd.Series, rf_daily: float = 0.0) -> float:
    r = returns.dropna()
    if len(r) < 20:
        return float("nan")
    excess = r - rf_daily
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("nan")
    dstd = downside.std(ddof=0)
    if dstd <= 0:
        return float("nan")
    return float(excess.mean() / dstd * np.sqrt(TRADING_DAYS_YEAR))


def max_drawdown(equity: pd.Series) -> float:
    e = equity.dropna()
    if e.empty:
        return float("nan")
    peak = e.cummax()
    dd = (e / peak - 1).min()
    return float(dd)


def cagr(equity: pd.Series) -> float:
    e = equity.dropna()
    if len(e) < 2:
        return float("nan")
    total_years = len(e) / TRADING_DAYS_YEAR
    return float((e.iloc[-1] / e.iloc[0]) ** (1 / total_years) - 1)
