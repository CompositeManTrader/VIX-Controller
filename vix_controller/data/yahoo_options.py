"""
yahoo_options.py — Cadenas de opciones de Yahoo Finance (vía yfinance). Sin Streamlit.

Lo usan la página Opciones (con caché) y scripts/update_options_snapshot.py
(foto diaria). Solo se aceptan opciones con mercado real: bid > 0 y ask > 0,
horquilla < 50 % del medio, interés abierto ≥ 10. Un último precio sin
cotización no es un precio: no se usa.

Vencimientos: los `cercanos` primeros con al menos un día de vida (donde se
concentra la gamma), los más próximos a 30/60/90 días (estructura temporal) y
el primero posterior a cada fecha de `eventos` (movimiento descontado).
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime

import pandas as pd

log = logging.getLogger("vix_controller")

OBJETIVOS_DTE = (30, 60, 90)
CERCANOS = 3


class YahooError(RuntimeError):
    """Yahoo no devolvió datos (límite de peticiones, fuera de servicio o vacío)."""


def es_limite(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(k in s for k in ("rate limit", "too many", "429", "throttl"))


def elegir_vencimientos(fechas: list[str], hoy: date, cercanos: int = CERCANOS,
                        objetivos=OBJETIVOS_DTE, eventos=(), min_dte: int = 1) -> list[tuple[str, int]]:
    """[(vencimiento, días)] sin duplicados y ordenados."""
    vivos = []
    for f in fechas:
        d = (datetime.strptime(f, "%Y-%m-%d").date() - hoy).days
        if d >= min_dte:
            vivos.append((f, d))
    vivos.sort(key=lambda x: x[1])
    if not vivos:
        return []
    elegidos = dict(vivos[:cercanos])
    for obj in objetivos:
        f, d = min(vivos, key=lambda x: abs(x[1] - obj))
        elegidos[f] = d
    for ev in eventos:
        ev = pd.Timestamp(ev).date()
        tras = [(f, d) for f, d in vivos if datetime.strptime(f, "%Y-%m-%d").date() >= ev]
        if tras:
            elegidos[tras[0][0]] = tras[0][1]
    return sorted(elegidos.items(), key=lambda x: x[1])


def limpiar(df: pd.DataFrame, spot: float, rango=(0.70, 1.30)) -> pd.DataFrame:
    """Filtros de calidad. `rango` = strike / spot admitido."""
    d = df.copy()
    for c in ("bid", "ask", "lastPrice", "openInterest", "volume", "strike"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0) if c in d.columns else 0.0
    d = d[(d["bid"] > 0) & (d["ask"] > 0) & (d["strike"] > 0)].copy()
    d["mid"] = 0.5 * (d["bid"] + d["ask"])
    d["moneyness"] = d["strike"] / spot
    d = d[d["moneyness"].between(*rango)]
    d = d[(d["ask"] - d["bid"]) / d["mid"] < 0.50]
    d = d[(d["openInterest"] >= 10) & (d["mid"] >= 0.05)]
    return d.sort_values("strike").reset_index(drop=True)


def _reintentar(fn, etiqueta: str, n: int = 4):
    for i in range(n):
        try:
            return fn()
        except Exception as e:                  # noqa: BLE001 — Yahoo falla de mil formas
            if es_limite(e) and i < n - 1:
                espera = 2 ** (i + 1)
                log.warning("%s: límite de Yahoo, espero %ss", etiqueta, espera)
                time.sleep(espera)
                continue
            raise
    return None


def descargar(ticker: str = "SPY", hoy: date | None = None, cercanos: int = CERCANOS,
              objetivos=OBJETIVOS_DTE, eventos=(), rango=(0.70, 1.30),
              pausa: float = 0.8) -> tuple[dict, float, str]:
    """
    ({vencimiento: {"calls", "puts", "dte"}}, spot, estado del mercado).
    Lanza YahooError si no consigue ni una cadena utilizable.
    """
    import yfinance as yf

    hoy = hoy or date.today()
    tk = yf.Ticker(ticker)
    try:
        fechas = _reintentar(lambda: tk.options, f"{ticker}.options")
    except Exception as e:                      # noqa: BLE001
        raise YahooError(f"{ticker}: sin lista de vencimientos ({e})") from e
    if not fechas:
        raise YahooError(f"{ticker}: Yahoo no devolvió vencimientos")
    sel = elegir_vencimientos(list(fechas), hoy, cercanos, objetivos, eventos)

    cadenas, spot, estado = {}, None, ""
    for f, dte in sel:
        try:
            ch = _reintentar(lambda f=f: tk.option_chain(f), f"{ticker} {f}")
        except Exception as e:                  # noqa: BLE001
            log.warning("%s %s: %s", ticker, f, e)
            continue
        und = getattr(ch, "underlying", None) or {}
        if spot is None:
            spot = float(und.get("regularMarketPrice") or 0) or None
            estado = str(und.get("marketState", ""))
        if not spot:
            continue
        calls, puts = limpiar(ch.calls, spot, rango), limpiar(ch.puts, spot, rango)
        if len(calls) >= 3 and len(puts) >= 3:
            cadenas[f] = {"calls": calls, "puts": puts, "dte": dte}
        time.sleep(pausa)
    if not cadenas or not spot:
        raise YahooError(f"{ticker}: ninguna cadena con cotizaciones válidas")
    log.info("Yahoo %s: %d cadenas · spot %.2f · %s", ticker, len(cadenas), spot, estado)
    return cadenas, float(spot), estado


def resumen_cadenas(cadenas: dict) -> pd.DataFrame:
    """Vencimiento, días y nº de opciones válidas por lado (diagnóstico)."""
    return pd.DataFrame([{"vencimiento": f, "dte": c["dte"], "calls": len(c["calls"]),
                          "puts": len(c["puts"])} for f, c in cadenas.items()]).sort_values("dte") \
        if cadenas else pd.DataFrame()

