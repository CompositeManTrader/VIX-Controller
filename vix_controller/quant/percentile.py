"""
percentile.py — Rolling percentile con soporte mínimo configurable.

Vectorizado con `rolling().rank(method="max", pct=True)`: para cada día,
fracción de valores válidos de la ventana trailing que son <= al actual.
Es exactamente lo que hacía el loop original en Python (que era O(n·window)
y dominaba el tiempo del Barómetro: 13 indicadores × ~8.000 filas).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..config import VTS_ROLLING_WINDOW, VTS_MIN_OBS_FLOOR, VTS_MIN_OBS_RATIO


def rolling_percentile(s: pd.Series, window: int = VTS_ROLLING_WINDOW,
                       min_obs: int | None = None) -> pd.Series:
    """
    Percentil del último valor vs la ventana histórica trailing. Devuelve 0-100.

    min_obs: observaciones válidas mínimas en la ventana.
    Por default max(VTS_MIN_OBS_FLOOR, window * VTS_MIN_OBS_RATIO).

    Robusto a NaN: un NaN en la entrada produce NaN en la salida y no cuenta
    como observación válida de la ventana.
    """
    if s is None or s.empty:
        return pd.Series(dtype=float)

    if min_obs is None:
        min_obs = max(VTS_MIN_OBS_FLOOR, int(window * VTS_MIN_OBS_RATIO))

    x = s.astype(float)
    # Con ventana menor que el soporte mínimo nunca hay obs suficientes:
    # el loop original devolvía todo NaN (pandas lanzaría ValueError).
    if window < min_obs:
        return pd.Series(np.nan, index=s.index, dtype=float)
    # method="max": los empates reciben el rango máximo → (valid <= cur).mean()
    out = x.rolling(window, min_periods=min_obs).rank(method="max", pct=True) * 100.0
    out.name = None
    return out
