"""
vrp.py — Volatility Risk Premium tracker.

VRP = prima que cobra el vendedor de volatilidad por asumir el riesgo
de varianza. Dos convenciones:

  - Vol points:      VRP_vol = VIX − RV20            (la intuitiva)
  - Variance units:  VRP_var = (VIX² − RV20²) / 100  (Carr-Wu 2009; castiga
                     más los episodios de alta vol, que es donde el short
                     vol muere — es la métrica económicamente correcta)

El percentil rolling del VRP_var dice si la prima que cosechas hoy está
cara o barata vs su propia historia:

  - percentil < 20 : prima comprimida → mala compensación para short vol
  - 20 – 80        : régimen normal
  - percentil > 80 : prima gorda → entorno generoso para cosechar
  - VRP negativo   : la vol realizada SUPERA la implícita → short vol
                     está pagando por perder (señal de stress agudo)

Funciones puras, sin Streamlit. Testeable.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .percentile import rolling_percentile


def compute_vrp_tracker(vix: pd.Series, rv20: pd.Series,
                        window: int = 1260) -> dict:
    """
    Construye el tracker de VRP a partir de series alineadas de VIX y RV20
    (ambas en puntos de vol anualizados, p.ej. 18.5 = 18.5%).

    Returns dict con:
      df           : DataFrame con vrp_vol, vrp_var, vrp_pct (percentil rolling)
      vrp_vol      : último VRP en vol points
      vrp_var      : último VRP en variance units
      percentile   : percentil rolling del VRP_var (0-100)
      regime       : 'COMPRIMIDA' | 'NORMAL' | 'RICA' | 'NEGATIVA'
      mean_1y      : VRP_vol promedio último año (la cosecha típica)
      pct_negative_1y : % de días del último año con VRP negativo
    o dict vacío si no hay datos suficientes.
    """
    if vix is None or rv20 is None:
        return {}
    df = pd.DataFrame({"vix": vix.astype(float), "rv20": rv20.astype(float)}).dropna()
    if len(df) < 120:
        return {}

    df["vrp_vol"] = df["vix"] - df["rv20"]
    df["vrp_var"] = (df["vix"] ** 2 - df["rv20"] ** 2) / 100.0
    df["vrp_pct"] = rolling_percentile(df["vrp_var"], window=window)

    last = df.iloc[-1]
    pct = float(last["vrp_pct"]) if pd.notna(last["vrp_pct"]) else 50.0

    if last["vrp_vol"] < 0:
        regime = "NEGATIVA"
    elif pct < 20:
        regime = "COMPRIMIDA"
    elif pct > 80:
        regime = "RICA"
    else:
        regime = "NORMAL"

    tail_1y = df["vrp_vol"].tail(252)
    return {
        "df": df,
        "vrp_vol": float(last["vrp_vol"]),
        "vrp_var": float(last["vrp_var"]),
        "percentile": pct,
        "regime": regime,
        "mean_1y": float(tail_1y.mean()),
        "pct_negative_1y": float((tail_1y < 0).mean() * 100.0),
    }
