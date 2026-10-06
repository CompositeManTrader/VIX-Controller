"""
cboe.py — Datos públicos de CBOE: curva de futuros VX, índices y settlement.

Sustituye al scraping con Playwright + Chromium (≈2 min de arranque en frío y
timeouts de 45 s) por los endpoints JSON/CSV que usa la propia web de CBOE.

Tres fuentes, cada una para un uso:

  · get_quotes_combined (JSON)   curva en vivo (retraso ~15 min): último,
                                 settlement, volumen, OI de cada contrato.
  · delayed_quotes/_<IDX>.json   cotización en vivo de VIX, VIX3M, VIX9D…
  · settlement/csv?dt=AAAA-MM-DD liquidación OFICIAL del día, publicada la
                                 misma tarde. Es la fuente de la señal de la
                                 estrategia (a = liquidación M2 / liquidación
                                 M1): los CSV por contrato llegan un día tarde.
  · daily_prices/<IDX>_History   cierres históricos de los índices.

Las funciones `parse_*` son puras (texto/JSON → DataFrame) y se testean sin
red. Las `fetch_*` solo descargan y delegan en ellas.
"""
from __future__ import annotations

import io
import json
import re
import time
import urllib.error
import urllib.request
from datetime import date

import numpy as np
import pandas as pd

FUTURES_URL = "https://www.cboe.com/us/futures/api/get_quotes_combined/?symbol=VX&rootsymbol=null"
INDEX_QUOTE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/quotes/_{sym}.json"
SETTLEMENT_URL = "https://www.cboe.com/us/futures/market_statistics/settlement/csv?dt={dt}"
INDEX_HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/csv, */*",
}

# Contrato MENSUAL: VX/V6. Los semanales llevan número (VX40/V6) y se excluyen.
MONTHLY_RE = re.compile(r"^VX/[A-Z]\d+$")
N_MONTHS = 8


class CboeError(RuntimeError):
    """Fallo al obtener o interpretar un dato de CBOE."""


# ──────────────────────────────────────────────────────────────────────
# Red
# ──────────────────────────────────────────────────────────────────────
def _get(url: str, timeout: float = 20.0, tries: int = 3) -> bytes:
    last: Exception | None = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                raise CboeError(f"{url}: HTTP {e.code}") from e
            last = e
        except Exception as e:                       # noqa: BLE001 — red inestable
            last = e
        time.sleep(1.0 * (k + 1))
    raise CboeError(f"{url}: {last}")


# ──────────────────────────────────────────────────────────────────────
# Curva en vivo
# ──────────────────────────────────────────────────────────────────────
def parse_futures_quotes(payload: dict | str, today: pd.Timestamp | None = None,
                         n_months: int = N_MONTHS) -> pd.DataFrame:
    """
    JSON de get_quotes_combined → DataFrame de contratos MENSUALES ordenados
    por vencimiento, con las columnas que consume la app:

        Symbol, Expiration, Last, Change, High, Low, Settlement,
        PrevSettlement, Volume, OpenInterest, Price, DTE

    Price = último si hay negociación hoy (>0); si no, el settlement.
    Solo contratos aún vivos (vencimiento >= hoy).
    """
    if isinstance(payload, (str, bytes)):
        payload = json.loads(payload)
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not rows:
        raise CboeError("Respuesta de futuros sin campo 'data'")

    df = pd.DataFrame(rows)
    if "symbol" not in df.columns or "expiration" not in df.columns:
        raise CboeError(f"Columnas inesperadas en futuros: {list(df.columns)[:8]}")

    df = df[df["symbol"].astype(str).str.match(MONTHLY_RE)].copy()
    if df.empty:
        raise CboeError("No hay contratos mensuales VX en la respuesta")

    out = pd.DataFrame({
        "Symbol": df["symbol"].astype(str),
        "Expiration": pd.to_datetime(df["expiration"], format="%m/%d/%Y", errors="coerce"),
        "Last": pd.to_numeric(df.get("last_price"), errors="coerce"),
        "Change": pd.to_numeric(df.get("change"), errors="coerce"),
        "High": pd.to_numeric(df.get("high"), errors="coerce"),
        "Low": pd.to_numeric(df.get("low"), errors="coerce"),
        "Settlement": pd.to_numeric(df.get("settlement"), errors="coerce"),
        "PrevSettlement": pd.to_numeric(df.get("prev_settlement"), errors="coerce"),
        "Volume": pd.to_numeric(df.get("volume"), errors="coerce"),
        "OpenInterest": pd.to_numeric(df.get("prev_open_int"), errors="coerce"),
    })
    today = (pd.Timestamp(today) if today is not None else pd.Timestamp.now()).normalize()
    out = out[out["Expiration"] >= today].sort_values("Expiration").reset_index(drop=True)
    out["Price"] = np.where(out["Last"].fillna(0) > 0, out["Last"], out["Settlement"])
    out["DTE"] = (out["Expiration"] - today).dt.days
    return out.head(n_months).reset_index(drop=True)


def fetch_futures_quotes(today: pd.Timestamp | None = None) -> pd.DataFrame:
    return parse_futures_quotes(json.loads(_get(FUTURES_URL)), today=today)


# ──────────────────────────────────────────────────────────────────────
# Índices en vivo
# ──────────────────────────────────────────────────────────────────────
def parse_index_quote(payload: dict | str) -> dict:
    """JSON delayed_quotes → dict(price, prev_close, open, high, low, change, ts)."""
    if isinstance(payload, (str, bytes)):
        payload = json.loads(payload)
    d = payload.get("data") or {}
    price = d.get("current_price")
    if price is None or not np.isfinite(float(price)) or float(price) <= 0:
        raise CboeError("Cotización de índice sin precio")
    prev = d.get("prev_day_close")
    return {
        "symbol": d.get("symbol"),
        "price": float(price),
        "prev_close": float(prev) if prev else None,
        "change": float(d["price_change"]) if d.get("price_change") is not None else None,
        "open": d.get("open"), "high": d.get("high"), "low": d.get("low"),
        "last_trade_time": d.get("last_trade_time"),
        "timestamp": payload.get("timestamp"),
    }


def fetch_index_quote(symbol: str) -> dict:
    """symbol sin caret: 'VIX', 'VIX3M', 'VIX9D', 'VIX6M', 'VVIX', 'SKEW'."""
    return parse_index_quote(json.loads(_get(INDEX_QUOTE_URL.format(sym=symbol.upper()))))


# ──────────────────────────────────────────────────────────────────────
# Settlement oficial del día
# ──────────────────────────────────────────────────────────────────────
def parse_settlement_csv(text: str | bytes, trade_date: date | str) -> pd.DataFrame:
    """
    CSV de settlement de un día → contratos MENSUALES VX vivos ese día,
    ordenados: Symbol, Expiration, Settle, DTE (días naturales).

    Un contrato deja de contar como M1 el día de su liquidación (misma regla
    que la curva de la Plataforma: liquidación > fecha).
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip() for c in df.columns]
    need = {"Product", "Symbol", "Expiration Date", "Price"}
    if not need <= set(df.columns):
        raise CboeError(f"CSV de settlement con columnas inesperadas: {list(df.columns)}")
    df = df[(df["Product"].astype(str).str.strip() == "VX")
            & df["Symbol"].astype(str).str.strip().str.match(MONTHLY_RE)].copy()
    td = pd.Timestamp(trade_date).normalize()
    df["Expiration"] = pd.to_datetime(df["Expiration Date"], errors="coerce")
    df["Settle"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df[(df["Expiration"] > td) & (df["Settle"] > 0)]
    df = df.drop_duplicates("Symbol").sort_values("Expiration").reset_index(drop=True)
    df["DTE"] = (df["Expiration"] - td).dt.days
    return df[["Symbol", "Expiration", "Settle", "DTE"]].rename(
        columns={"Symbol": "Symbol"})


def fetch_settlement(trade_date: date | str) -> pd.DataFrame:
    dt = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    df = parse_settlement_csv(_get(SETTLEMENT_URL.format(dt=dt)), dt)
    if df.empty:
        raise CboeError(f"Sin settlement VX para {dt} (¿festivo o aún no publicado?)")
    return df


def settlement_to_curve_row(settle: pd.DataFrame, n_months: int = N_MONTHS) -> dict:
    """Fila m1..mN / dias_mK / sym_mK / exp_mK a partir del settlement del día."""
    row: dict = {}
    for k, (_, r) in enumerate(settle.head(n_months).iterrows(), start=1):
        row[f"m{k}"] = float(r["Settle"])
        row[f"dias_m{k}"] = int(r["DTE"])
        row[f"sym_m{k}"] = str(r["Symbol"])
        row[f"exp_m{k}"] = pd.Timestamp(r["Expiration"])
    return row


# ──────────────────────────────────────────────────────────────────────
# Históricos de índices
# ──────────────────────────────────────────────────────────────────────
def parse_index_history(text: str | bytes, name: str) -> pd.Series:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().upper() for c in df.columns]
    if "DATE" not in df.columns:
        raise CboeError(f"{name}: CSV sin columna DATE")
    close_col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    s = pd.to_numeric(df.set_index("DATE")[close_col], errors="coerce").dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.index = pd.DatetimeIndex(s.index).normalize()
    return s.rename(name)


def fetch_index_history(symbol: str) -> pd.Series:
    return parse_index_history(_get(INDEX_HISTORY_URL.format(sym=symbol.upper()), timeout=45),
                               symbol.upper())
