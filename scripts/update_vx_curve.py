"""
update_vx_curve.py — Histórico DIARIO de la curva de futuros del VIX.

Fuente: CDN público y gratuito de CBOE (un CSV por contrato, más el índice
VIX). Construye un panel por fecha con M1..M8 (settle), días a vencimiento,
open interest, volumen y símbolo de cada mes, y las medidas de la curva que
usa el tab Term Structure (contango M1→M2, basis VIX→M1, roll yield del ETP).

Salida: data/vx_curve_history.parquet  (índice = fecha)

Modo incremental: si el parquet existe, solo baja los contratos vivos en las
últimas semanas y hace upsert por fecha. Sin parquet: backfill completo desde
2013 (~170 ficheros, un par de minutos).

Uso:
    python scripts/update_vx_curve.py            # incremental (o backfill si no hay parquet)
    python scripts/update_vx_curve.py --full     # fuerza backfill completo
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("update_vx_curve")

CDN = "https://cdn.cboe.com"
FUT_URL = f"{CDN}/data/us/futures/market_statistics/historical_data/VX/VX_%s.csv"
VIX_URL = f"{CDN}/api/global/us_indices/daily_prices/VIX_History.csv"
HEADERS = {"User-Agent": "vix-controller/1.0 (uso propio)"}
PAUSE = 0.25
N_MONTHS = 8
DEFAULT_OUTPUT = Path("data/vx_curve_history.parquet")
FIRST_YEAR = 2013


class TransientError(RuntimeError):
    """Fallo de red tras agotar reintentos: NO es un 404, el fichero existe."""


def fetch(url: str, tries: int = 5) -> bytes | None:
    """None solo si el recurso no existe (403/404). Un fallo de red persistente
    lanza TransientError: construir la curva con un contrato ausente desplaza
    todo el ranking M1..M8 en silencio, y eso es peor que no escribir."""
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            last = e
        except Exception as e:                    # noqa: BLE001 — red inestable
            last = e
        wait = 2.0 * (k + 1)
        log.warning("fetch %s: %s — reintento %d/%d en %.0fs", url, last, k + 1, tries, wait)
        time.sleep(wait)
    raise TransientError(f"{url}: {last}")


def settlement(year: int, month: int) -> pd.Timestamp:
    """Miércoles 30 días antes del tercer viernes del mes SIGUIENTE."""
    nxt = pd.Timestamp(year=year + (month == 12), month=1 if month == 12 else month + 1, day=1)
    first_friday = nxt + pd.Timedelta(days=(4 - nxt.weekday()) % 7)
    return first_friday + pd.Timedelta(days=14) - pd.Timedelta(days=30)


def all_settlements(first_year: int, today: pd.Timestamp) -> list[pd.Timestamp]:
    out = []
    for y in range(first_year, today.year + 2):
        for m in range(1, 13):
            s = settlement(y, m)
            if s <= today + pd.Timedelta(days=400):
                out.append(s)
    return out


def fetch_contract(settle_date: pd.Timestamp) -> pd.DataFrame | None:
    raw = fetch(FUT_URL % settle_date.strftime("%Y-%m-%d"))
    if raw is None:
        return None
    d = pd.read_csv(io.BytesIO(raw))
    d.columns = [c.strip() for c in d.columns]
    if "Trade Date" not in d.columns or "Settle" not in d.columns:
        return None
    d["fecha"] = pd.to_datetime(d["Trade Date"], errors="coerce")
    d["liquidacion"] = settle_date
    d["sym"] = d["Futures"].astype(str).str.strip() if "Futures" in d.columns else ""
    for c in ("Settle", "Total Volume", "Open Interest"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        else:
            d[c] = np.nan
    return d[["fecha", "liquidacion", "sym", "Settle", "Total Volume", "Open Interest"]] \
        .rename(columns={"Settle": "settle", "Total Volume": "vol", "Open Interest": "oi"})


def fetch_vix() -> pd.Series:
    raw = fetch(VIX_URL)
    if raw is None:
        raise RuntimeError("No se pudo bajar VIX_History.csv de CBOE")
    d = pd.read_csv(io.BytesIO(raw))
    d.columns = [c.strip().upper() for c in d.columns]
    d["DATE"] = pd.to_datetime(d["DATE"], errors="coerce")
    s = pd.to_numeric(d.set_index("DATE")["CLOSE"], errors="coerce").dropna()
    return s[~s.index.duplicated(keep="last")].sort_index().rename("VIX")


def build_curve(contracts: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """Panel diario M1..M8. El día de liquidación el contrato ya no es front."""
    d = contracts.dropna(subset=["fecha", "settle"])
    d = d[(d["settle"] > 0) & (d["liquidacion"] > d["fecha"])].copy()
    d["rank"] = d.groupby("fecha")["liquidacion"].rank(method="first").astype(int)
    d["dias"] = (d["liquidacion"] - d["fecha"]).dt.days
    d = d[d["rank"] <= N_MONTHS]

    out = pd.DataFrame(index=sorted(d["fecha"].unique()))
    out.index.name = "fecha"
    for k in range(1, N_MONTHS + 1):
        sub = d[d["rank"] == k].set_index("fecha")
        out[f"m{k}"] = sub["settle"]
        out[f"dias_m{k}"] = sub["dias"]
        out[f"oi_m{k}"] = sub["oi"]
        out[f"vol_m{k}"] = sub["vol"]
        out[f"sym_m{k}"] = sub["sym"]
        out[f"exp_m{k}"] = sub["liquidacion"]

    out["VIX"] = vix.reindex(out.index)
    out["contango_pct"] = (out["m2"] / out["m1"] - 1.0) * 100.0
    out["basis_pct"] = (out["m1"] / out["VIX"] - 1.0) * 100.0
    gap = (out["dias_m2"] - out["dias_m1"]).astype(float).where(lambda g: g > 0)
    out["roll_ann"] = out["contango_pct"] * (365.0 / gap)
    out["ratio_m2m1"] = out["m2"] / out["m1"]
    return out.sort_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()

    today = pd.Timestamp.now().normalize()
    old = pd.DataFrame()
    if a.output.exists() and not a.full:
        try:
            old = pd.read_parquet(a.output)
            old.index = pd.DatetimeIndex(old.index).normalize()
        except Exception as e:                   # noqa: BLE001
            log.error("Parquet ilegible (%s) — backfill completo", e)
            old = pd.DataFrame()

    settles = all_settlements(FIRST_YEAR, today)
    if not old.empty:
        # Solo contratos vivos desde 10 días antes del último dato
        since = old.index.max() - pd.Timedelta(days=10)
        settles = [s for s in settles if s > since]
        log.info("INCREMENTAL desde %s · %d contratos vivos", since.date(), len(settles))
    else:
        log.info("BACKFILL completo · %d contratos", len(settles))

    frames, ok = [], 0
    try:
        for i, s in enumerate(settles, 1):
            f = fetch_contract(s)
            if f is not None and not f.empty:
                frames.append(f)
                ok += 1
            if i % 25 == 0:
                log.info("  %d/%d · %d con datos", i, len(settles), ok)
            time.sleep(PAUSE)
    except TransientError as e:
        log.error("Fallo de red persistente — abortando SIN escribir el parquet: %s", e)
        sys.exit(2)
    if not frames:
        log.error("Ningún contrato descargado — abortando sin tocar el parquet")
        sys.exit(1)

    vix = fetch_vix()
    new = build_curve(pd.concat(frames, ignore_index=True), vix)
    if new.empty:
        log.error("Curva vacía — abortando")
        sys.exit(1)

    if not old.empty:
        # En incremental solo confiamos en fechas con M1..M4 completos: en los
        # primeros días de la ventana faltan contratos que ya vencieron.
        new = new.dropna(subset=["m1", "m2", "m3", "m4"])
        combined = pd.concat([old, new])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new

    # Informe de integridad: entre M1 y M2 debe haber ~1 mes (20-45 días). Un
    # hueco mayor = a esa fecha el CDN no tiene el contrato intermedio y el
    # "M2" es en realidad el de dos meses. Ocurre en la propia fuente (la
    # curva de la Plataforma, misma fuente, presenta los mismos huecos), así
    # que se INFORMA y se marca (columna gap_ok), no se aborta. Los fallos de
    # red sí abortan (TransientError, más arriba).
    gap = (combined["dias_m2"] - combined["dias_m1"])
    combined["gap_ok"] = (gap >= 20) & (gap <= 45)
    bad = combined.index[~combined["gap_ok"]]
    if len(bad):
        log.warning("INTEGRIDAD: %d fechas con hueco M1→M2 fuera de 20-45 días "
                    "(contrato intermedio ausente en el CDN): %s … %s",
                    len(bad), bad.min().date(), bad.max().date())

    a.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(a.output, compression="snappy")
    last = combined.index.max()
    log.info("✅ %s · %s filas · %s → %s · última M1=%s contango=%+.2f%%",
             a.output, f"{len(combined):,}", combined.index.min().date(), last.date(),
             combined.loc[last, "m1"], combined.loc[last, "contango_pct"])


if __name__ == "__main__":
    main()
