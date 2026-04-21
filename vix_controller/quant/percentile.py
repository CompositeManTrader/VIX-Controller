"""
percentile.py — Rolling percentile con soporte mínimo configurable.
Extraído de app.py::_rolling_percentile para que sea testeable sin Streamlit.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..config import VTS_ROLLING_WINDOW, VTS_MIN_OBS_FLOOR, VTS_MIN_OBS_RATIO


def rolling_percentile(s: pd.Series, window: int = VTS_ROLLING_WINDOW,
                       min_obs: int | None = None) -> pd.Series:
    """
    Percentil del último valor vs la ventana histórica. Devuelve 0-100.

    min_obs: observaciones válidas mínimas en la ventana.
    Por default max(VTS_MIN_OBS_FLOOR, window * VTS_MIN_OBS_RATIO).
    """
    if s is None or s.empty:
        return pd.Series(dtype=float)

    if min_obs is None:
        min_obs = max(VTS_MIN_OBS_FLOOR, int(window * VTS_MIN_OBS_RATIO))

    arr = s.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)

    for i in range(min_obs - 1, n):
        start = max(0, i - window + 1)
        window_vals = arr[start:i + 1]
        cur = arr[i]
        if np.isnan(cur):
            continue
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) < min_obs:
            continue
        out[i] = (valid <= cur).mean() * 100.0

    return pd.Series(out, index=s.index)
