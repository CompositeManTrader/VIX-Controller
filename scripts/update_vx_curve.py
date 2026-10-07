"""
update_vx_curve.py — Histórico DIARIO de la curva de futuros del VIX.

Fuentes (CBOE, públicas y gratuitas):
  · CSV por contrato (CDN)          histórico de settles, OI y volumen
  · CSV de settlement por fecha     liquidación OFICIAL del día, publicada la
                                    misma tarde (los CSV por contrato llegan
                                    con un día de retraso)
  · CSV de índices (CDN)            cierres de VIX y VIX3M
  · Archivo CFE (CDN)               contratos 2004-2014 (antes de mayo de 2013
                                    el CDN nuevo no trae liquidaciones)

Salida: data/vx_curve_history.parquet  (índice = fecha)
  m1..m8, dias_mK, oi_mK, vol_mK, sym_mK, exp_mK, VIX, VIX3M,
  contango_pct, basis_pct, roll_ann, ratio_m2m1, ratio_vix3m,
  gap_ok, front_ok

Integridad (lo que fallaba antes y ya no puede pasar en silencio):
  1. VENCIMIENTOS FESTIVOS. La fórmula estándar (miércoles 30 días antes del
     tercer viernes del mes siguiente) no conoce festivos: con Viernes Santo
     o Juneteenth CBOE adelanta el vencimiento y el fichero con la fecha
     calculada no existe. Se prueba el día hábil anterior. Sin esto el
     contrato front desaparecía un mes entero (2014, 2019, 2022, 2024, 2025).
  2. FALLOS DE RED abortan sin escribir: un contrato ausente corre todo el
     ranking M1..M8.
  3. front_ok: el M1 de cada fecha debe ser el vencimiento mensual más
     cercano conocido. Un M1 equivocado no se ve mirando solo el hueco M1→M2.
  4. El settlement por fecha solo se acepta si trae el front esperado (para
     fechas antiguas CBOE devuelve listas truncadas sin el front).

Uso:
    python scripts/update_vx_curve.py            # incremental
    python scripts/update_vx_curve.py --full     # backfill completo (2004 →)
    python scripts/update_vx_curve.py --archive  # solo añade 2004-2013 al parquet actual
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vix_controller.data import cboe  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("update_vx_curve")

CDN = "https://cdn.cboe.com"
FUT_URL = f"{CDN}/data/us/futures/market_statistics/historical_data/VX/VX_%s.csv"
# Archivo histórico de CFE (2004-2014): un CSV por contrato, CFE_K04_VX.csv.
# La última fila de cada fichero es el día de vencimiento (liquidación final).
ARCHIVE_URL = f"{CDN}/resources/futures/archive/volume-and-price/CFE_%s%02d_VX.csv"
MONTH_CODES = "FGHJKMNQUVXZ"
ARCHIVE_YEARS = (2004, 2014)
# Hasta el 26/03/2007 los futuros cotizaban a 10× el VIX (escala VBI).
RESCALE_UNTIL = pd.Timestamp("2007-03-26")
HEADERS = {"User-Agent": "vix-controller/1.0 (uso propio)"}
PAUSE = 0.2
N_MONTHS = 8
DEFAULT_OUTPUT = Path("data/vx_curve_history.parquet")
FIRST_YEAR = 2013
CATCHUP_DAYS = 12          # días naturales que se re-verifican con el settlement oficial


class TransientError(RuntimeError):
    """Fallo de red tras agotar reintentos: NO es un 404, el fichero existe."""


def fetch(url: str, tries: int = 5) -> bytes | None:
    """None solo si el recurso no existe (403/404). Fallo de red → TransientError."""
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


def standard_settlement(year: int, month: int) -> pd.Timestamp:
    """Miércoles 30 días antes del tercer viernes del mes SIGUIENTE (sin festivos)."""
    nxt = pd.Timestamp(year=year + (month == 12), month=1 if month == 12 else month + 1, day=1)
    first_friday = nxt + pd.Timedelta(days=(4 - nxt.weekday()) % 7)
    return first_friday + pd.Timedelta(days=14) - pd.Timedelta(days=30)


# Compatibilidad con código que importaba `settlement`
settlement = standard_settlement


def candidate_dates(std: pd.Timestamp) -> list[pd.Timestamp]:
    """El vencimiento estándar y, si cae en festivo, los días hábiles previos."""
    return [std, std - pd.offsets.BDay(1), std - pd.offsets.BDay(2)]


def parse_contract_csv(raw: bytes, expiration: pd.Timestamp) -> pd.DataFrame:
    d = pd.read_csv(io.BytesIO(raw))
    d.columns = [c.strip() for c in d.columns]
    if "Trade Date" not in d.columns or "Settle" not in d.columns:
        return pd.DataFrame()
    d["fecha"] = pd.to_datetime(d["Trade Date"], errors="coerce")
    d["liquidacion"] = pd.Timestamp(expiration)
    d["sym"] = d["Futures"].astype(str).str.strip() if "Futures" in d.columns else ""
    for c in ("Settle", "Total Volume", "Open Interest"):
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan
    return d[["fecha", "liquidacion", "sym", "Settle", "Total Volume", "Open Interest"]] \
        .rename(columns={"Settle": "settle", "Total Volume": "vol", "Open Interest": "oi"})


def fetch_contract(year: int, month: int) -> tuple[pd.DataFrame | None, pd.Timestamp | None]:
    """Descarga el contrato probando el vencimiento estándar y los desplazados."""
    for cand in candidate_dates(standard_settlement(year, month)):
        raw = fetch(FUT_URL % cand.strftime("%Y-%m-%d"))
        if raw is not None:
            df = parse_contract_csv(raw, cand)
            return (df if not df.empty else None), cand
        time.sleep(PAUSE)
    return None, None


def parse_archive_csv(raw: bytes) -> pd.DataFrame:
    """CSV del archivo CFE. Vencimiento = última fecha del fichero; precios
    anteriores al 26/03/2007 divididos entre 10 (cambio de escala de CBOE).
    Algunos ficheros de 2004-2005 terminan ciertas líneas con una coma de más."""
    lineas = raw.decode("utf-8", "replace").splitlines()
    texto = "\n".join(linea.rstrip().rstrip(",") for linea in lineas)
    d = pd.read_csv(io.StringIO(texto))
    d.columns = [c.strip() for c in d.columns]
    if "Trade Date" not in d.columns or "Settle" not in d.columns:
        return pd.DataFrame()
    d["fecha"] = pd.to_datetime(d["Trade Date"].astype(str).str.strip(), format="%m/%d/%Y",
                                errors="coerce")
    d = d.dropna(subset=["fecha"]).drop_duplicates(subset=["fecha"], keep="last")
    if d.empty:
        return pd.DataFrame()
    d["liquidacion"] = d["fecha"].max()
    d["sym"] = d["Futures"].astype(str).str.strip() if "Futures" in d.columns else ""
    for c in ("Settle", "Total Volume", "Open Interest"):
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan
    d.loc[d["fecha"] < RESCALE_UNTIL, "Settle"] /= 10.0
    return d[["fecha", "liquidacion", "sym", "Settle", "Total Volume", "Open Interest"]] \
        .rename(columns={"Settle": "settle", "Total Volume": "vol", "Open Interest": "oi"})


def fetch_archive() -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Todos los contratos del archivo CFE. Los meses sin fichero no se listaron."""
    frames, exps = [], []
    for y in range(ARCHIVE_YEARS[0], ARCHIVE_YEARS[1] + 1):
        for m in range(1, 13):
            raw = fetch(ARCHIVE_URL % (MONTH_CODES[m - 1], y % 100))
            if raw is None:
                continue
            df = parse_archive_csv(raw)
            if not df.empty:
                frames.append(df)
                exps.append(df["liquidacion"].iloc[0])
            time.sleep(PAUSE)
    log.info("Archivo CFE: %d contratos", len(frames))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), exps


def vix3m_extendido(vix3m: pd.Series) -> pd.Series:
    """VIX3M de CBOE (desde 09/2009) y, antes, el ^VIX3M de baro_history
    (yfinance, desde 07/2006). Solo rellena fechas anteriores al primer dato de CBOE."""
    baro = Path("data/baro_history.parquet")
    if vix3m.empty or not baro.exists():
        return vix3m
    try:
        b = pd.read_parquet(baro, columns=["VIX3M"])["VIX3M"].dropna()
    except Exception as e:                       # noqa: BLE001
        log.warning("baro_history ilegible (%s): VIX3M solo desde CBOE", e)
        return vix3m
    b.index = pd.DatetimeIndex(b.index).normalize()
    previo = b[b.index < vix3m.index.min()]
    return pd.concat([previo, vix3m]).sort_index()


def build_curve(contracts: pd.DataFrame) -> pd.DataFrame:
    """Panel diario M1..M8. El día de liquidación el contrato ya no es front."""
    d = contracts.dropna(subset=["fecha", "settle"])
    d = d[(d["settle"] > 0) & (d["liquidacion"] > d["fecha"])].copy()
    d["rank"] = d.groupby("fecha")["liquidacion"].rank(method="first").astype(int)
    d["dias"] = (d["liquidacion"] - d["fecha"]).dt.days
    d = d[d["rank"] <= N_MONTHS]

    out = pd.DataFrame(index=pd.DatetimeIndex(sorted(d["fecha"].unique()), name="fecha"))
    for k in range(1, N_MONTHS + 1):
        sub = d[d["rank"] == k].set_index("fecha")
        out[f"m{k}"] = sub["settle"]
        out[f"dias_m{k}"] = sub["dias"]
        out[f"oi_m{k}"] = sub["oi"]
        out[f"vol_m{k}"] = sub["vol"]
        out[f"sym_m{k}"] = sub["sym"]
        out[f"exp_m{k}"] = sub["liquidacion"]
    return out.sort_index()


def add_measures(c: pd.DataFrame) -> pd.DataFrame:
    """Medidas de la curva + flags de integridad."""
    c = c.copy()
    c["contango_pct"] = (c["m2"] / c["m1"] - 1.0) * 100.0
    c["basis_pct"] = (c["m1"] / c["VIX"] - 1.0) * 100.0
    gap = (c["dias_m2"] - c["dias_m1"]).astype(float)
    c["roll_ann"] = c["contango_pct"] * (365.0 / gap.where(gap > 0))
    c["ratio_m2m1"] = c["m2"] / c["m1"]                     # M2/M1 — >1 = contango
    c["ratio_vix3m"] = c["VIX3M"] / c["VIX"]                # VIX3M/VIX — >1 = contango
    c["gap_ok"] = (gap >= 20) & (gap <= 45)
    return c


def flag_front(c: pd.DataFrame, expirations: pd.DatetimeIndex) -> pd.Series:
    """True si el M1 de la fecha es el vencimiento mensual más cercano conocido."""
    exps = pd.DatetimeIndex(sorted(set(expirations.dropna())))
    pos = exps.searchsorted(c.index, side="right")
    expected = pd.Series(pd.NaT, index=c.index, dtype="datetime64[ns]")
    ok_pos = pos < len(exps)
    expected[ok_pos] = exps[pos[ok_pos]]
    got = pd.to_datetime(c["exp_m1"])
    return (got == expected).fillna(False)


def settlement_rows(dates: list[pd.Timestamp], vix: pd.Series, vix3m: pd.Series,
                    expected_front: dict[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    """Filas oficiales del día desde el CSV de settlement, validadas."""
    rows = {}
    for d in dates:
        try:
            st = cboe.fetch_settlement(d)
        except cboe.CboeError as e:
            log.info("settlement %s: %s", d.date(), e)
            continue
        if len(st) < 4 or st.iloc[0]["DTE"] > 40:
            log.warning("settlement %s rechazado: lista truncada (M1 a %s días)",
                        d.date(), st.iloc[0]["DTE"] if len(st) else "?")
            continue
        exp_front = expected_front.get(d)
        if exp_front is not None and pd.Timestamp(st.iloc[0]["Expiration"]) != exp_front:
            log.warning("settlement %s rechazado: M1 %s ≠ front esperado %s",
                        d.date(), st.iloc[0]["Expiration"].date(), exp_front.date())
            continue
        if d not in vix.index or d not in vix3m.index:
            log.info("settlement %s: índices aún sin cierre — se espera", d.date())
            continue
        rows[d] = cboe.settlement_to_curve_row(st, N_MONTHS)
        time.sleep(PAUSE)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index = pd.DatetimeIndex(out.index, name="fecha")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--archive", action="store_true",
                    help="añade 2004-2013 desde el archivo histórico de CFE (incluido en --full)")
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
    full = old.empty

    months = [(y, m) for y in range(FIRST_YEAR, today.year + 2) for m in range(1, 13)
              if standard_settlement(y, m) <= today + pd.Timedelta(days=400)]
    if not full:
        since = old.index.max() - pd.Timedelta(days=45)
        months = [(y, m) for (y, m) in months if standard_settlement(y, m) > since]
        log.info("INCREMENTAL · %d contratos vivos desde %s", len(months), since.date())
    else:
        log.info("BACKFILL completo · %d contratos", len(months))

    frames, found_exps, missing = [], [], []
    try:
        for i, (y, m) in enumerate(months, 1):
            df, exp = fetch_contract(y, m)
            if df is not None:
                frames.append(df)
                found_exps.append(exp)
                if exp != standard_settlement(y, m):
                    log.info("  %d-%02d vence %s (desplazado por festivo)", y, m, exp.date())
            elif standard_settlement(y, m) < today - pd.Timedelta(days=5):
                missing.append(f"{y}-{m:02d}")
            if i % 25 == 0:
                log.info("  %d/%d · %d con datos", i, len(months), len(frames))
            time.sleep(PAUSE)
        vix = cboe.fetch_index_history("VIX")
        vix3m = cboe.fetch_index_history("VIX3M")
    except (TransientError, cboe.CboeError) as e:
        log.error("Fallo de red persistente — abortando SIN escribir: %s", e)
        sys.exit(2)

    if missing:
        log.warning("Contratos vencidos sin fichero en el CDN: %s", ", ".join(missing))
    if not frames:
        log.error("Ningún contrato descargado — abortando sin tocar el parquet")
        sys.exit(1)

    new = build_curve(pd.concat(frames, ignore_index=True))

    # Vencimientos conocidos: los encontrados ahora + los ya presentes en el histórico
    exps_known = pd.DatetimeIndex(found_exps)
    if not old.empty and "exp_m1" in old.columns:
        exps_known = exps_known.append(pd.DatetimeIndex(pd.to_datetime(old["exp_m1"])))

    if not full:
        # En incremental solo se fusionan fechas con M1..M4 completos y M1 correcto.
        new = new.dropna(subset=["m1", "m2", "m3", "m4"])
        new = new[flag_front(new, exps_known)]
        curve_cols = [c for c in old.columns
                      if c.startswith(("dias_", "oi_", "vol_", "sym_", "exp_"))
                      or (c.startswith("m") and c[1:].isdigit())]
        combined = pd.concat([old[curve_cols], new])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new

    if a.full or a.archive:
        try:
            arch, arch_exps = fetch_archive()
        except TransientError as e:
            log.error("Fallo de red en el archivo CFE — abortando SIN escribir: %s", e)
            sys.exit(2)
        if arch.empty:
            log.error("Archivo CFE vacío — abortando sin tocar el parquet")
            sys.exit(1)
        arch_curve = build_curve(arch)
        inicio_cdn = combined.dropna(subset=["m1", "m2", "m3", "m4"]).index.min()
        arch_curve = arch_curve[arch_curve.index < inicio_cdn]
        combined = combined[combined.index >= inicio_cdn]
        combined = pd.concat([arch_curve, combined]).sort_index()
        exps_known = exps_known.append(pd.DatetimeIndex(arch_exps))
        log.info("Archivo CFE: %s fechas añadidas (%s → %s)", f"{len(arch_curve):,}",
                 arch_curve.index.min().date(), arch_curve.index.max().date())

    # Cierre oficial de los últimos días desde el CSV de settlement (misma tarde)
    exps_sorted = pd.DatetimeIndex(sorted(set(exps_known.dropna())))
    recent = pd.bdate_range(today - pd.Timedelta(days=CATCHUP_DAYS), today)
    exp_front = {}
    for d in recent:
        p = exps_sorted.searchsorted(d, side="right")
        if p < len(exps_sorted):
            exp_front[d] = exps_sorted[p]
    pending = [d for d in recent if d not in combined.index or pd.isna(combined.loc[d, "m1"])]
    st_rows = settlement_rows(pending, vix, vix3m, exp_front)
    if not st_rows.empty:
        log.info("Settlement oficial añadido para: %s",
                 ", ".join(str(d.date()) for d in st_rows.index))
        combined = pd.concat([combined, st_rows])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    combined["VIX"] = vix.reindex(combined.index)
    combined["VIX3M"] = vix3m_extendido(vix3m).reindex(combined.index)
    combined = add_measures(combined)
    combined["front_ok"] = flag_front(combined, exps_known)

    n_front = int((~combined["front_ok"]).sum())
    n_gap = int((~combined["gap_ok"]).sum())
    if n_front:
        bad = combined.index[~combined["front_ok"]]
        log.warning("INTEGRIDAD: %d fechas con M1 distinto del front esperado (%s … %s)",
                    n_front, bad.min().date(), bad.max().date())
    if n_gap:
        log.warning("INTEGRIDAD: %d fechas con hueco M1→M2 fuera de 20-45 días", n_gap)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(a.output, compression="snappy")
    last = combined.index.max()
    log.info("OK %s · %s filas · %s → %s · M2/M1=%.4f · VIX3M/VIX=%.4f",
             a.output, f"{len(combined):,}", combined.index.min().date(), last.date(),
             combined.loc[last, "ratio_m2m1"], combined.loc[last, "ratio_vix3m"])


if __name__ == "__main__":
    main()
