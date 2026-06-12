"""
cross_asset.py — Early warning de stress desde otros mercados.

El mercado de bonos y el de divisas suelen oler el stress ANTES de que
el VIX lo registre:

  1. MOVE (^MOVE)    — vol implícita de Treasuries. MOVE alto con VIX
                       dormido = divergencia clásica pre-spike.
  2. Crédito         — ratio HYG/LQD (high yield vs investment grade).
                       Cuando cae, el crédito está pagando el riesgo que
                       el equity todavía ignora. Fallback: HYG/IEF.
  3. DXY             — momentum 20d del dólar. Breakout alcista del USD
                       = tightening global de liquidez = risk-off.

Cada señal se convierte a un "stress percentile" 0-100 (100 = máximo
stress histórico en la ventana) y un semáforo:

    verde    < 60   sin señal
    amarillo 60-85  monitorear
    rojo     > 85   warning activo

Funciones puras, sin Streamlit. Testeable.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .percentile import rolling_percentile

LIGHT_GREEN, LIGHT_YELLOW, LIGHT_RED = "green", "yellow", "red"
_YELLOW_AT, _RED_AT = 60.0, 85.0


def _light(stress_pct: float) -> str:
    if stress_pct >= _RED_AT:
        return LIGHT_RED
    if stress_pct >= _YELLOW_AT:
        return LIGHT_YELLOW
    return LIGHT_GREEN


def _last_valid(s: pd.Series) -> float | None:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else None


def compute_stress_signals(move: pd.Series | None = None,
                           hyg: pd.Series | None = None,
                           lqd: pd.Series | None = None,
                           ief: pd.Series | None = None,
                           dxy: pd.Series | None = None,
                           window: int = 252) -> dict:
    """
    Calcula las 3 señales de stress cross-asset. Cada serie es Close diario.
    Series faltantes se omiten con gracia (la señal no aparece).

    Returns dict:
      signals   : lista de dicts {key, name, value, stress_pct, light, desc}
      composite : máximo stress de las señales disponibles (el eslabón
                  más débil manda — un solo rojo ya es warning)
      light     : semáforo del composite
    """
    signals = []

    # ── 1. MOVE: percentil del nivel ─────────────────────────────
    if move is not None and move.dropna().shape[0] > 120:
        m = move.dropna().astype(float)
        pct_series = rolling_percentile(m, window=window)
        pct = _last_valid(pct_series)
        if pct is not None:
            chg20 = (m.iloc[-1] / m.iloc[-21] - 1) * 100 if len(m) > 21 else np.nan
            signals.append({
                "key": "move",
                "name": "MOVE (bond vol)",
                "value": f"{m.iloc[-1]:.0f}" + (f" ({chg20:+.0f}% 20d)" if np.isfinite(chg20) else ""),
                "stress_pct": pct,
                "light": _light(pct),
                "desc": "Vol implícita de Treasuries — alta con VIX dormido = divergencia pre-spike",
            })

    # ── 2. Crédito: HYG/LQD (fallback HYG/IEF) momentum 20d ──────
    denom, denom_name = None, None
    if lqd is not None and lqd.dropna().shape[0] > 120:
        denom, denom_name = lqd, "LQD"
    elif ief is not None and ief.dropna().shape[0] > 120:
        denom, denom_name = ief, "IEF"
    if hyg is not None and denom is not None and hyg.dropna().shape[0] > 120:
        ratio = (hyg.astype(float) / denom.astype(float)).dropna()
        if len(ratio) > 140:
            mom20 = ratio.pct_change(20) * 100
            # Caída del ratio = stress → invertimos el percentil
            pct_series = 100.0 - rolling_percentile(mom20, window=window)
            pct = _last_valid(pct_series)
            if pct is not None:
                signals.append({
                    "key": "credit",
                    "name": f"Crédito HYG/{denom_name}",
                    "value": f"{mom20.dropna().iloc[-1]:+.2f}% 20d",
                    "stress_pct": pct,
                    "light": _light(pct),
                    "desc": "High yield vs grado de inversión — cae cuando el crédito paga el riesgo que equity ignora",
                })

    # ── 3. DXY: momentum alcista 20d ─────────────────────────────
    if dxy is not None and dxy.dropna().shape[0] > 140:
        d = dxy.dropna().astype(float)
        mom20 = d.pct_change(20) * 100
        pct_series = rolling_percentile(mom20, window=window)  # USD ↑ = stress
        pct = _last_valid(pct_series)
        if pct is not None:
            signals.append({
                "key": "dxy",
                "name": "DXY momentum",
                "value": f"{mom20.dropna().iloc[-1]:+.2f}% 20d",
                "stress_pct": pct,
                "light": _light(pct),
                "desc": "Breakout del dólar = tightening de liquidez global = risk-off",
            })

    if not signals:
        return {}

    composite = max(s["stress_pct"] for s in signals)
    return {
        "signals": signals,
        "composite": composite,
        "light": _light(composite),
    }
