"""
vol_desk.py — Medidas de la página «Volatilidad». Funciones puras, sin Streamlit.

Convenciones
  · Volatilidad realizada: raíz de la media de los rendimientos logarítmicos
    diarios del SPY al cuadrado × √252 × 100 (sin restar la media, como en
    los swaps de varianza).
  · El VIX es la volatilidad esperada a 30 días naturales ≈ 21 sesiones.
  · Prima de volatilidad (ex post) en t = VIX_t − realizada de t+1 … t+21.
    Se conoce 21 sesiones después: las 21 últimas filas son NaN. Es la que de
    verdad cobra o paga el vendedor de volatilidad.
  · Régimen en t: tramo de VIX y estado de la curva con el CIERRE de t. Lo que
    se mide después usa solo t+1 … t+21: no hay anticipación.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SESIONES_MES = 21
ANUAL = 252

CORTES_VIX = [0.0, 12.0, 15.0, 20.0, 25.0, 30.0, np.inf]
TRAMOS_VIX = ["< 12", "12–15", "15–20", "20–25", "25–30", "≥ 30"]
CORTES_REGIMEN = [0.0, 15.0, 20.0, 30.0, np.inf]
TRAMOS_REGIMEN = ["VIX < 15", "VIX 15–20", "VIX 20–30", "VIX ≥ 30"]
MUESTRA_MINIMA = 40           # por debajo, la fila se marca como poco fiable
VENTANA_PERCENTIL = 5 * ANUAL  # termómetros: el SKEW y el VVIX cambian de nivel con los años


# ──────────────────────────────────────────────────────────────────────
# Volatilidad realizada y prima
# ──────────────────────────────────────────────────────────────────────
def realizada(close: pd.Series, n: int = SESIONES_MES) -> pd.Series:
    """Realizada de las n sesiones que terminan en t (incluye el rendimiento de t)."""
    r = np.log(close.astype(float)).diff()
    return np.sqrt((r ** 2).rolling(n, min_periods=n).mean() * ANUAL) * 100.0


def realizada_futura(close: pd.Series, n: int = SESIONES_MES) -> pd.Series:
    """Realizada de t+1 … t+n colocada en t (ex post)."""
    return realizada(close, n).shift(-n)


def tabla_prima(vix: pd.Series, spy: pd.Series, n: int = SESIONES_MES) -> pd.DataFrame:
    """Serie diaria: vix, realizada pasada, realizada futura y prima."""
    d = pd.DataFrame({"vix": vix, "spy": spy}).dropna()
    d["rv"] = realizada(d["spy"], n)
    d["rv_fut"] = realizada_futura(d["spy"], n)
    d["prima"] = d["vix"] - d["rv_fut"]
    d["prima_hoy"] = d["vix"] - d["rv"]
    return d.drop(columns="spy")


def tramo_vix(vix: pd.Series, cortes=CORTES_VIX, etiquetas=TRAMOS_VIX) -> pd.Series:
    return pd.cut(vix, cortes, labels=etiquetas, right=False)


def prima_por_nivel(d: pd.DataFrame) -> pd.DataFrame:
    """
    Por tramo de VIX: cuánto se cobró de verdad. Solo filas con la realizada
    futura ya conocida. Columnas: dias, peso, vix, rv_fut, prima (mediana),
    gana (% de días con VIX > realizada futura), p10 (peor 10 % de la prima).
    """
    x = d.dropna(subset=["vix", "rv_fut"]).copy()
    x["tramo"] = tramo_vix(x["vix"])
    g = x.groupby("tramo", observed=False)
    out = pd.DataFrame({
        "dias": g.size(),
        "vix": g["vix"].mean(),
        "rv_fut": g["rv_fut"].mean(),
        "prima": g["prima"].median(),
        "gana": g["prima"].apply(lambda s: (s > 0).mean() * 100 if len(s) else np.nan),
        "p10": g["prima"].quantile(0.10),
    })
    out["peso"] = out["dias"] / max(int(out["dias"].sum()), 1) * 100
    return out


# ──────────────────────────────────────────────────────────────────────
# VXX reconstruido: índice de futuros del VIX a corto plazo
# ──────────────────────────────────────────────────────────────────────
def indice_corto_plazo(curva: pd.DataFrame) -> pd.Series:
    """
    Rendimiento diario del índice de futuros del VIX a ~1 mes (metodología del
    S&P 500 VIX Short-Term Futures, el que replica el VXX), reconstruido con la
    liquidación diaria de CBOE. Sin comisión del ETN ni coste de préstamo.

    Cada cierre se reparte entre el front (F) y el siguiente (S): peso en F =
    sesiones que faltan hasta la víspera del vencimiento de F / sesiones del
    periodo de roll. El día antes del vencimiento ya está todo en S. El
    rendimiento de t usa los pesos fijados al cierre de t−1 y los precios de
    los MISMOS contratos en t. Si falta un precio con peso > 0: NaN (no se rellena).
    """
    partes = []
    for k in range(1, 9):
        if f"m{k}" in curva.columns and f"exp_m{k}" in curva.columns:
            partes.append(pd.DataFrame({"fecha": curva.index,
                                        "exp": pd.to_datetime(curva[f"exp_m{k}"]),
                                        "px": curva[f"m{k}"].astype(float)}))
    if not partes:
        return pd.Series(dtype=float)
    largo = pd.concat(partes, ignore_index=True).dropna()
    largo = largo[largo["px"] > 0]
    px = largo.pivot_table(index="fecha", columns="exp", values="px", aggfunc="last").sort_index()
    exps = np.array(sorted(px.columns), dtype="datetime64[D]")
    fechas = px.index.values.astype("datetime64[D]")

    pos = np.searchsorted(exps, fechas, side="right")      # índice del front (exp > t)
    pesos = []
    for t, p in zip(fechas, pos):
        if p == 0 or p + 1 >= len(exps):
            pesos.append(None)
            continue
        prev_e, f, s = exps[p - 1], exps[p], exps[p + 1]
        dt = np.busday_count(prev_e, f)
        dr = max(np.busday_count(t, f) - 1, 0)
        pesos.append((f, s, dr / dt if dt > 0 else 0.0))

    col = {e: i for i, e in enumerate(px.columns.values.astype("datetime64[D]"))}
    m = px.to_numpy()
    out = np.full(len(fechas), np.nan)
    for i in range(1, len(fechas)):
        w = pesos[i - 1]
        if w is None:
            continue
        f, s, w1 = w
        if f not in col or s not in col:
            continue
        a0, b0 = m[i - 1, col[f]], m[i - 1, col[s]]
        a1, b1 = m[i, col[f]], m[i, col[s]]
        if w1 > 0 and (np.isnan(a0) or np.isnan(a1)):
            continue
        if np.isnan(b0) or np.isnan(b1):
            continue
        a0, a1 = (a0, a1) if w1 > 0 else (0.0, 0.0)
        den = w1 * a0 + (1 - w1) * b0
        if den > 0:
            out[i] = (w1 * a1 + (1 - w1) * b1) / den - 1.0
    return pd.Series(out, index=px.index, name="vxx_ret")


# ──────────────────────────────────────────────────────────────────────
# Régimen → lo que hizo el corto en VXX después
# ──────────────────────────────────────────────────────────────────────
def rendimiento_siguiente(r: pd.Series, n: int = SESIONES_MES) -> pd.Series:
    """Rendimiento compuesto de t+1 … t+n colocado en t. NaN si falta algún día."""
    lc = np.log1p(r.fillna(0.0)).cumsum()
    out = np.expm1(lc.shift(-n) - lc)
    faltan = r.isna().astype(int)[::-1].rolling(n, min_periods=1).sum()[::-1].shift(-1)
    return out.where(faltan == 0)


def regimen(vix: pd.Series, contango: pd.Series) -> pd.Series:
    """Etiqueta 'VIX 15–20 · contango' con el cierre de t."""
    t = tramo_vix(vix, CORTES_REGIMEN, TRAMOS_REGIMEN).astype(object)
    c = np.where(contango.astype(bool), "contango", "backwardation")
    return pd.Series([f"{a} · {b}" if isinstance(a, str) else np.nan for a, b in zip(t, c)],
                     index=vix.index, dtype=object)


def tabla_regimen(vix: pd.Series, contango: pd.Series, vxx_ret: pd.Series,
                  n: int = SESIONES_MES) -> pd.DataFrame:
    """
    Por régimen al cierre de t (tramo de VIX × curva según la regla de la
    estrategia): qué hizo un corto en VXX en las n sesiones siguientes.
      siempre  corto todo el periodo
      regla    corto solo los días en que la regla estaba dentro (decidido con
               el cierre anterior)
    Corto a exposición constante (Π(1 − r) − 1), sin costes: es una medida del
    riesgo de cada entorno, no una réplica del backtest.
    """
    d = pd.DataFrame({"vix": vix, "contango": contango, "r": vxx_ret})
    d = d.dropna(subset=["vix", "contango"])
    d["contango"] = d["contango"].astype(bool)
    corto = -d["r"]
    dentro = d["contango"].shift(1, fill_value=False)
    d["reg"] = regimen(d["vix"], d["contango"])
    d["siempre"] = rendimiento_siguiente(corto, n)
    d["regla"] = rendimiento_siguiente(corto.where(dentro, 0.0).where(d["r"].notna()), n)
    d = d.dropna(subset=["reg", "siempre", "regla"])
    filas = []
    for tramo in TRAMOS_REGIMEN:
        for curva in ("contango", "backwardation"):
            etiqueta = f"{tramo} · {curva}"
            x = d[d["reg"] == etiqueta]
            if x.empty:
                continue
            filas.append({
                "regimen": etiqueta, "tramo": tramo, "curva": curva, "dias": len(x),
                "siempre_media": x["siempre"].mean() * 100,
                "siempre_gana": (x["siempre"] > 0).mean() * 100,
                "siempre_p5": x["siempre"].quantile(0.05) * 100,
                "regla_media": x["regla"].mean() * 100,
                "regla_p5": x["regla"].quantile(0.05) * 100,
            })
    out = pd.DataFrame(filas)
    if not out.empty:
        out["peso"] = out["dias"] / out["dias"].sum() * 100
        out["fiable"] = out["dias"] >= MUESTRA_MINIMA
        out.attrs["desde"] = d.index[0]
        out.attrs["hasta"] = d.index[-1]
    return out


# ──────────────────────────────────────────────────────────────────────
# Termómetros de riesgo
# ──────────────────────────────────────────────────────────────────────
def percentil(s: pd.Series, valor: float | None = None) -> float:
    """% de la historia en o por debajo de `valor` (por defecto, el último)."""
    s = s.dropna()
    if s.empty:
        return np.nan
    v = float(s.iloc[-1]) if valor is None else float(valor)
    return float((s <= v).mean() * 100)


def caida_credito(hyg: pd.Series, ief: pd.Series, ventana: int = ANUAL) -> pd.Series:
    """HYG/IEF frente a su máximo de 1 año, en %. Negativo = el crédito se deteriora."""
    ratio = (hyg / ief).dropna()
    return (ratio / ratio.rolling(ventana, min_periods=ventana // 2).max() - 1.0) * 100.0


def termometro(s: pd.Series, n_mes: int = SESIONES_MES) -> dict:
    """Valor, fecha, percentil de 5 años y de 1 año, cambio a 1 mes."""
    s = s.dropna()
    if s.empty:
        return {}
    v = float(s.iloc[-1])
    hace = float(s.iloc[-n_mes - 1]) if len(s) > n_mes else np.nan
    return {"valor": v, "fecha": s.index[-1], "pct": percentil(s.tail(VENTANA_PERCENTIL)),
            "pct_1a": percentil(s.tail(ANUAL)), "cambio_mes": v - hace,
            "desde": s.index[0]}


def curva_indices(baro: pd.DataFrame, fechas: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """VIX9D, VIX, VIX3M y VIX6M en cada fecha pedida (último cierre ≤ fecha)."""
    cols = [c for c in ("VIX9D", "VIX", "VIX3M", "VIX6M") if c in baro.columns]
    b = baro[cols].dropna()
    filas = {}
    for nombre, f in fechas.items():
        sub = b[b.index <= f]
        if not sub.empty:
            filas[nombre] = sub.iloc[-1]
    return pd.DataFrame(filas).T
