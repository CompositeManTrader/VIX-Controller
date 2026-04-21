"""
signals.py — Señales de trading puras (BB × Contango).

Separa la lógica de construcción de señal de la UI/cache de Streamlit.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..config import BB_WINDOW, BB_STD_MULT


def bb_contango_signal(vxx_close: pd.Series,
                       in_contango: pd.Series,
                       bb_window: int = BB_WINDOW,
                       bb_std_mult: float = BB_STD_MULT) -> pd.DataFrame:
    """
    Construye la señal LONG/CASH de la estrategia BB × Contango.

    Estados:
      - Entrada LONG: VXX < SMA(bb_window) (compresión de vol)
      - Salida CASH: VXX > BB_upper = SMA + bb_std_mult·σ (expansión)
      - Filtro: SOLO operar si in_contango == True

    Importante: la señal en t se usa para el retorno en t+1 (shift(1) en
    el backtest). Esto evita look-ahead.

    Returns DataFrame con columnas:
      sma, bb_upper, sig_raw, sig_bb, ct_filter, sig_final
    """
    vxx = vxx_close.astype(float)
    sma = vxx.rolling(bb_window, min_periods=bb_window).mean()
    std = vxx.rolling(bb_window, min_periods=bb_window).std(ddof=0)
    bb_upper = sma + bb_std_mult * std

    n = len(vxx)
    sig = np.zeros(n, dtype=int)
    pos = 0
    for i in range(n):
        p = vxx.iloc[i]
        s = sma.iloc[i]
        u = bb_upper.iloc[i]
        if np.isnan(p) or np.isnan(s) or np.isnan(u):
            sig[i] = pos
            continue
        if pos == 0 and p < s:
            pos = 1
        elif pos == 1 and p > u:
            pos = 0
        sig[i] = pos

    sig_series = pd.Series(sig, index=vxx.index, name="sig_raw")
    sig_bb = sig_series.shift(1).fillna(0).astype(int).rename("sig_bb")

    ct = in_contango.astype(bool).reindex(vxx.index).fillna(False).astype(int)
    ct = ct.rename("ct_filter")
    sig_final = (sig_bb * ct).rename("sig_final")

    return pd.DataFrame({
        "sma": sma, "bb_upper": bb_upper,
        "sig_raw": sig_series, "sig_bb": sig_bb,
        "ct_filter": ct, "sig_final": sig_final,
    })
