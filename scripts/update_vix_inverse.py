"""
update_vix_inverse.py — Avanza la historia derivada del modelo VIX Inverse.

Continúa data/vix_inverse_history.parquet desde su último estado con:
  · las dos medidas de data/vx_curve_history.parquet (CBOE, ya actualizada)
  · aperturas y cierres del VXX y cierres del SPY de yfinance (mismo ajuste
    que la Plataforma: ratio 1,000000 en el VXX, medido)

Una sesión solo se añade si llega COMPLETA (precios + las dos medidas). Si
falta algo, se espera a la siguiente pasada: nunca se rellena.

Uso:  python scripts/update_vix_inverse.py
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vix_controller.quant import vix_inverse_live as vl  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("update_vix_inverse")

HIST = Path("data/vix_inverse_history.parquet")
CURVE = Path("data/vx_curve_history.parquet")
ET = ZoneInfo("America/New_York")


def yf_daily(symbol: str, start: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf
    last_err = None
    for _ in range(3):
        try:
            h = yf.Ticker(symbol).history(start=start.strftime("%Y-%m-%d"), auto_adjust=True)
            if not h.empty:
                h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
                return h
        except Exception as e:                    # noqa: BLE001
            last_err = e
    raise RuntimeError(f"yfinance {symbol}: sin datos ({last_err})")


def main() -> int:
    if not HIST.exists():
        log.error("No existe %s — hay que sembrarla (scripts/seed_vix_inverse.py)", HIST)
        return 1
    hist = pd.read_parquet(HIST)
    hist.index = pd.DatetimeIndex(hist.index).normalize()
    curve = pd.read_parquet(CURVE)
    curve.index = pd.DatetimeIndex(curve.index).normalize()
    if "front_ok" in curve.columns:
        curve = curve[curve["front_ok"].astype(bool)]

    anchor = hist.index[-1]
    vxx = yf_daily("VXX", anchor - pd.Timedelta(days=20))
    spy = yf_daily("SPY", anchor - pd.Timedelta(days=20))

    # La sesión de hoy solo cuenta con el mercado ya cerrado
    now_et = datetime.now(ET)
    today_et = pd.Timestamp(now_et.date())
    if now_et.hour * 60 + now_et.minute < 16 * 60 + 30:
        vxx = vxx[vxx.index < today_et]
        spy = spy[spy.index < today_et]

    if anchor not in vxx.index or anchor not in spy.index:
        log.error("yfinance no trae el ancla %s — no se puede continuar", anchor.date())
        return 1

    fechas = vxx.index[vxx.index >= anchor]
    nuevos = pd.DataFrame({
        "open": vxx["Open"].reindex(fechas),
        "close": vxx["Close"].reindex(fechas),
        "spy_close": spy["Close"].reindex(fechas),
        "ratio_m2m1": curve["ratio_m2m1"].reindex(fechas),
        "ratio_vix3m": curve["ratio_vix3m"].reindex(fechas),
    }, index=fechas)

    out = vl.continuar_historia(hist, nuevos, vxx_close_lookback=vxx["Close"])
    added = len(out) - len(hist)
    pendientes = [d.date() for d in fechas[1:] if d not in out.index]
    if pendientes:
        log.warning("Sesiones a la espera de datos completos: %s", pendientes)
    if added == 0:
        log.info("Sin sesiones nuevas completas · última %s", anchor.date())
        return 0
    out.to_parquet(HIST, compression="snappy")
    u = out.iloc[-1]
    log.info("OK +%d sesiones · última %s · posición %s · contango %s · M2/M1 %.4f · VIX3M/VIX %.4f",
             added, out.index[-1].date(), "DENTRO" if u["pos"] else "FUERA",
             bool(u["contango"]), u["ratio_m2m1"], u["ratio_vix3m"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
