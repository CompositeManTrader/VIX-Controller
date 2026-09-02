"""
term_structure.py — Lectura operable de la curva de futuros del VIX.

Traduce la forma de la curva a una señal LONG VOL / SHORT VOL / NEUTRAL con
reglas transparentes. Es la MISMA lógica que muestra el tab Term Structure en
vivo, factorizada aquí para poder reproducirla sobre el histórico diario
(data/vx_curve_history.parquet) y ver cómo ha cambiado la señal en el tiempo.

Reglas (idénticas al panel en vivo):
  LONG VOL   si la curva de futuros se invierte: contango M1→M2 < 0
  SHORT VOL  si pasa el checklist completo (≥3 checks disponibles y todos ✓):
               contango > 0 · basis > 0 · percentil del contango > 25 · VIX < 22
  NEUTRAL    en el resto: contango comprimido (percentil < 15), spot por encima
             del M1 con la curva aún en contango (spike de contado: en 2024-26
             el VIX cayó −13,8 % de mediana en los 5 días siguientes — revierte,
             no es inversión), o señales mixtas.

Base VIX→M1 negativa NO es LONG VOL por sí sola (se midió: es el setup
clásico de reversión tras un spike). Sí bloquea el SHORT (falla el checklist).

Funciones puras, sin Streamlit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .percentile import rolling_percentile

SIGNAL_LONG = "LONG VOL"
SIGNAL_SHORT = "SHORT VOL"
SIGNAL_NEUTRAL = "NEUTRAL"

# Codificación numérica para graficar: +1 short vol favorable, −1 long vol, 0 neutral
SIGNAL_CODE = {SIGNAL_SHORT: 1, SIGNAL_NEUTRAL: 0, SIGNAL_LONG: -1}

VIX_MAX_SHORT = 22.0          # VIX "bajo control"
# En los últimos días antes del vencimiento M1 converge al spot por
# construcción: una base VIX→M1 negativa ahí es ruido de convergencia, no
# inversión. Con DTE <= este umbral la base no se usa como señal.
BASIS_MIN_DTE = 5.0
PCTILE_MIN_SHORT = 25.0       # contango no comprimido
PCTILE_COMPRESSED = 15.0      # contango comprimido → neutral aunque sea positivo
PCTILE_WINDOW = 1260          # 5 años de sesiones
PCTILE_MIN_OBS = 200

DESC = {
    SIGNAL_LONG: ("Curva invertida/estresada — el mercado paga por vol inmediata. "
                  "El viento de cola del roll favorece VXX/UVXY; el short vol "
                  "PIERDE el carry."),
    SIGNAL_SHORT: ("Contango sano con prima sobre spot — la convergencia del futuro "
                   "hacia el VIX paga el carry al short vol (SVXY/SVIX)."),
    "compressed": ("Contango comprimido (p<15) — el carry no compensa el riesgo de "
                   "spike. Esperar mejor entrada o reducir tamaño."),
    "spot_spike": ("Spot por encima del M1 con la curva aún en contango — spike de "
                   "contado que históricamente revierte (VIX −13,8 % de mediana a 5 "
                   "días). No es inversión: no abrir short vol hasta que el spot "
                   "vuelva bajo el M1, ni perseguir long vol."),
    "mixed": "Señales mixtas — sin edge claro de curva. Revisa Barómetro y VRP.",
}


def pct_change(a: float | None, b: float | None) -> float | None:
    """(b/a − 1)·100. None si falta algo o a ≤ 0."""
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return (b / a - 1.0) * 100.0


def curve_checks(front_ct: float | None, spot_m1: float | None,
                 ct_pctile: float | None, vix: float | None) -> list[tuple]:
    """Checklist (nombre, ok, valor_formateado). Solo incluye lo disponible."""
    checks = []
    if front_ct is not None:
        checks.append(("Contango M1→M2 positivo", front_ct > 0, f"{front_ct:+.2f}%"))
    if spot_m1 is not None:
        checks.append(("Prima M1 sobre VIX (basis > 0)", spot_m1 > 0, f"{spot_m1:+.2f}%"))
    if ct_pctile is not None and np.isfinite(ct_pctile):
        checks.append((f"Contango NO comprimido (percentil > {PCTILE_MIN_SHORT:.0f})",
                       ct_pctile > PCTILE_MIN_SHORT, f"p{ct_pctile:.0f} de 5 años"))
    if vix is not None and np.isfinite(vix):
        checks.append((f"VIX bajo control (< {VIX_MAX_SHORT:.0f})",
                       vix < VIX_MAX_SHORT, f"{vix:.1f}"))
    return checks


def curve_signal(front_ct: float | None, spot_m1: float | None,
                 ct_pctile: float | None, vix: float | None,
                 dte1: float | None = None) -> dict:
    """
    Señal compuesta de la curva. Devuelve dict:
      signal, code, desc, checks, n_ok, inverted

    dte1: días a vencimiento del M1. Si <= BASIS_MIN_DTE la base VIX→M1 se
    ignora (convergencia mecánica, no señal).
    """
    if dte1 is not None and np.isfinite(dte1) and dte1 <= BASIS_MIN_DTE:
        spot_m1 = None
    checks = curve_checks(front_ct, spot_m1, ct_pctile, vix)
    n_ok = sum(1 for _, ok, _ in checks if ok)
    inverted = front_ct is not None and front_ct < 0
    spot_spike = (not inverted) and spot_m1 is not None and spot_m1 < 0

    if inverted:
        sig, desc = SIGNAL_LONG, DESC[SIGNAL_LONG]
    elif len(checks) >= 3 and n_ok == len(checks):
        sig, desc = SIGNAL_SHORT, DESC[SIGNAL_SHORT]
    elif spot_spike:
        sig, desc = SIGNAL_NEUTRAL, DESC["spot_spike"]
    elif ct_pctile is not None and np.isfinite(ct_pctile) and ct_pctile < PCTILE_COMPRESSED:
        sig, desc = SIGNAL_NEUTRAL, DESC["compressed"]
    else:
        sig, desc = SIGNAL_NEUTRAL, DESC["mixed"]

    return {"signal": sig, "code": SIGNAL_CODE[sig], "desc": desc,
            "checks": checks, "n_ok": n_ok, "inverted": bool(inverted)}


def signal_history(curve: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce la señal día a día sobre el histórico de la curva.

    `curve` necesita columnas m1, m2, VIX (y opcionalmente dias_m1, dias_m2
    para el roll yield del ETP). Índice = fecha.

    El percentil del contango es ROLLING (solo pasado): la señal de un día
    no ve días posteriores. Devuelve DataFrame con:
      contango_pct, basis_pct, ct_pctile, vix, roll_ann, signal, code
    """
    need = {"m1", "m2", "VIX"}
    missing = need - set(curve.columns)
    if missing:
        raise ValueError(f"signal_history: faltan columnas {sorted(missing)}")

    d = curve[["m1", "m2", "VIX"]].astype(float).copy()
    d["contango_pct"] = (d["m2"] / d["m1"] - 1.0) * 100.0
    d["basis_pct"] = (d["m1"] / d["VIX"] - 1.0) * 100.0
    d.loc[(d["m1"] <= 0) | d["m1"].isna(), ["contango_pct", "basis_pct"]] = np.nan
    d["ct_pctile"] = rolling_percentile(d["contango_pct"], window=PCTILE_WINDOW,
                                        min_obs=PCTILE_MIN_OBS)
    d["vix"] = d["VIX"]

    if {"dias_m1", "dias_m2"} <= set(curve.columns):
        gap = (curve["dias_m2"] - curve["dias_m1"]).astype(float)
        gap = gap.where(gap > 0)
        d["roll_ann"] = d["contango_pct"] * (365.0 / gap)
    else:
        d["roll_ann"] = np.nan

    ct = d["contango_pct"].to_numpy()
    bs = d["basis_pct"].to_numpy()
    pc = d["ct_pctile"].to_numpy()
    vx = d["vix"].to_numpy()
    dt = (curve["dias_m1"].astype(float).to_numpy() if "dias_m1" in curve.columns
          else np.full(len(d), np.nan))
    sig = np.empty(len(d), dtype=object)
    code = np.zeros(len(d), dtype=int)
    for i in range(len(d)):
        r = curve_signal(
            None if np.isnan(ct[i]) else float(ct[i]),
            None if np.isnan(bs[i]) else float(bs[i]),
            None if np.isnan(pc[i]) else float(pc[i]),
            None if np.isnan(vx[i]) else float(vx[i]),
            dte1=None if np.isnan(dt[i]) else float(dt[i]),
        )
        sig[i], code[i] = r["signal"], r["code"]
    d["signal"] = sig
    d["code"] = code
    return d[["contango_pct", "basis_pct", "ct_pctile", "vix", "roll_ann",
              "signal", "code"]]
