"""
update_baro_parquet.py — Actualizador del parquet histórico del Barómetro VTS

Descarga y mantiene actualizado `data/baro_history.parquet` con todos los
tickers que necesita el barómetro:

  Familia VIX: ^VIX, ^VIX9D, ^VIX3M, ^VIX6M, ^VIX1Y, ^VVIX, ^SKEW
  Equity/Credit: SPY, HYG, IEF
  ETPs de vol: VXX, SVXY

Uso:
    python scripts/update_baro_parquet.py                    # update incremental
    python scripts/update_baro_parquet.py --full             # redescarga completa
    python scripts/update_baro_parquet.py --output PATH      # ruta custom

GitHub Action recomendada (.github/workflows/update_baro.yml):

    name: Update Barómetro Parquet
    on:
      schedule:
        - cron: '0 22 * * 1-5'   # lunes-viernes 22:00 UTC (post-close NYSE)
      workflow_dispatch:          # permite trigger manual
    jobs:
      update:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - uses: actions/setup-python@v5
            with:
              python-version: '3.11'
          - run: pip install yfinance pandas pyarrow
          - run: python scripts/update_baro_parquet.py
          - name: Commit changes
            run: |
              git config user.name 'github-actions'
              git config user.email 'github-actions@github.com'
              git add data/baro_history.parquet
              git diff --staged --quiet || git commit -m "Auto: update baro_history parquet"
              git push
"""
from __future__ import annotations
import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("update_baro")

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

# Tickers que necesita el Barómetro VTS
# Clave = nombre interno · Valor = ticker Yahoo
TICKERS = {
    "VIX":   "^VIX",
    "VIX9D": "^VIX9D",
    "VIX3M": "^VIX3M",
    "VIX6M": "^VIX6M",
    "VIX1Y": "^VIX1Y",
    "VVIX":  "^VVIX",
    "SKEW":  "^SKEW",
    "SPY":   "SPY",
    "HYG":   "HYG",
    "IEF":   "IEF",
    "VXX":   "VXX",
    "SVXY":  "SVXY",
    # Cross-asset early warning (MOVE = bond vol, LQD = crédito IG, DXY = dólar)
    "MOVE":  "^MOVE",
    "LQD":   "LQD",
    "DXY":   "DX-Y.NYB",
}

DEFAULT_OUTPUT = Path("data/baro_history.parquet")


# ══════════════════════════════════════════════════════════════
# FETCH
# ══════════════════════════════════════════════════════════════

def fetch_one(name: str, sym: str, start: str | None = None,
              period: str = "max") -> pd.Series:
    """
    Descarga una serie de yfinance (solo columna Close).
    Si `start` está dado, descarga desde esa fecha.
    Si no, descarga `period` completo.
    """
    try:
        if start:
            h = yf.download(sym, start=start, progress=False,
                            auto_adjust=True, threads=False)
        else:
            h = yf.download(sym, period=period, progress=False,
                            auto_adjust=True, threads=False)

        if h is None or h.empty:
            log.warning(f"{name} ({sym}): sin datos")
            return pd.Series(dtype=float, name=name)

        # Manejar MultiIndex (pasa cuando yfinance retorna múltiples tickers)
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.get_level_values(0)

        if "Close" not in h.columns:
            log.warning(f"{name} ({sym}): sin columna Close")
            return pd.Series(dtype=float, name=name)

        s = h["Close"].copy()

        # Normalizar índice
        if hasattr(s.index, "tz") and s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s.index = pd.DatetimeIndex(s.index).normalize()
        s.name = name

        log.info(f"  {name} ({sym}): {len(s):,} filas · "
                 f"{s.index.min().date()} → {s.index.max().date()}")
        return s

    except Exception as e:
        log.error(f"{name} ({sym}): {e}")
        return pd.Series(dtype=float, name=name)


def fetch_all_tickers(start: str | None = None) -> pd.DataFrame:
    """Descarga todos los tickers y los combina en un DataFrame."""
    log.info(f"Descargando {len(TICKERS)} tickers"
             f"{f' desde {start}' if start else ' (histórico completo)'}…")

    series_list = []
    for name, sym in TICKERS.items():
        s = fetch_one(name, sym, start=start)
        if not s.empty:
            series_list.append(s)

    if not series_list:
        log.error("No se descargó ningún ticker")
        return pd.DataFrame()

    df = pd.concat(series_list, axis=1)
    df = df.sort_index()
    return df


# ══════════════════════════════════════════════════════════════
# UPDATE LOGIC
# ══════════════════════════════════════════════════════════════

def update_parquet(output_path: Path, full: bool = False) -> None:
    """
    Actualiza el parquet del barómetro.

    Si `full` o el parquet no existe: descarga historia completa.
    Si el parquet existe: descarga solo desde la última fecha - 5 días
    (buffer por si hay revisiones) y hace merge.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Caso 1: parquet no existe o force full → download completo
    if full or not output_path.exists():
        reason = "--full solicitado" if full else "parquet no existe"
        log.info(f"MODO COMPLETO ({reason}) — descargando historia completa")
        df_new = fetch_all_tickers(start=None)
        if df_new.empty:
            log.error("Descarga vacía — abortando")
            sys.exit(1)
        df_new.to_parquet(output_path, compression="snappy")
        log.info(f"✅ Escrito {output_path} · {len(df_new):,} filas · "
                 f"{len(df_new.columns)} tickers")
        return

    # Caso 2: parquet existe → update incremental
    log.info(f"MODO INCREMENTAL — cargando {output_path}")
    try:
        df_old = pd.read_parquet(output_path)
    except Exception as e:
        log.error(f"Parquet corrupto o ilegible ({e}) — cayendo a descarga full")
        df_new = fetch_all_tickers(start=None)
        if df_new.empty:
            log.error("Descarga full vacía — abortando sin sobrescribir parquet")
            sys.exit(1)
        df_new.to_parquet(output_path, compression="snappy")
        log.info(f"✅ Reconstruido {output_path} · {len(df_new):,} filas")
        return

    if df_old.empty:
        log.error("Parquet válido pero vacío — cayendo a descarga full")
        df_new = fetch_all_tickers(start=None)
        if df_new.empty:
            log.error("Descarga full vacía — abortando sin sobrescribir parquet")
            sys.exit(1)
        df_new.to_parquet(output_path, compression="snappy")
        log.info(f"✅ Reconstruido {output_path} · {len(df_new):,} filas")
        return

    df_old.index = pd.DatetimeIndex(df_old.index).normalize()
    df_old = df_old.sort_index()
    last_date = df_old.index.max()
    log.info(f"  Parquet actual: {len(df_old):,} filas · última fecha = {last_date.date()}")

    # Descargar desde last_date - 10 días (buffer por revisiones tardías, weekends,
    # holidays). 5d era marginal: si el job falla viernes, el lunes el gap es >5d.
    fetch_start = (last_date - timedelta(days=10)).strftime("%Y-%m-%d")
    df_new = fetch_all_tickers(start=fetch_start)

    if df_new.empty:
        log.warning("Descarga incremental vacía — parquet sin cambios")
        return

    # Merge: las filas nuevas sobrescriben las viejas (con mismo índice)
    df_combined = pd.concat([df_old, df_new])
    df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
    df_combined = df_combined.sort_index()

    # Asegurar que todas las columnas estén (por si se agregó un ticker nuevo)
    missing_cols = [c for c in TICKERS.keys() if c not in df_combined.columns]
    if missing_cols:
        log.info(f"Columnas nuevas detectadas: {missing_cols} — forzando download completo")
        df_combined = fetch_all_tickers(start=None)

    n_added = len(df_combined) - len(df_old)
    log.info(f"  Filas nuevas agregadas: {n_added}")

    # Diagnóstico: qué tan actualizado está cada ticker
    log.info("Última fecha por ticker:")
    for col in df_combined.columns:
        last = df_combined[col].dropna().index.max()
        if pd.notna(last):
            age = (datetime.now() - last.to_pydatetime()).days
            status = "✅" if age <= 3 else ("⚠️" if age <= 7 else "❌")
            log.info(f"  {status} {col}: {last.date()} (hace {age}d)")
        else:
            log.warning(f"  ❌ {col}: sin datos")

    # Guardar
    df_combined.to_parquet(output_path, compression="snappy")
    log.info(f"✅ Escrito {output_path} · {len(df_combined):,} filas · "
             f"{len(df_combined.columns)} tickers")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Actualiza data/baro_history.parquet con datos de yfinance"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Fuerza redescarga completa (no incremental)"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Ruta del parquet (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()

    log.info(f"VTS Barómetro · parquet updater · {datetime.now().isoformat(timespec='seconds')}")
    log.info(f"Output: {args.output.resolve()}")

    try:
        update_parquet(args.output, full=args.full)
        log.info("🎉 Listo")
    except KeyboardInterrupt:
        log.warning("Interrumpido por usuario")
        sys.exit(130)
    except Exception as e:
        log.exception(f"Error fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
