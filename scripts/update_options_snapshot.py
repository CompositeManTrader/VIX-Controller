"""
update_options_snapshot.py — Foto diaria del mercado de opciones.

Guarda una fila por sesión en data/options_history.parquet con lo que muestra
la página Opciones: IV a 30 días, sesgo y butterfly 25Δ del SPY, GEX total,
nivel de giro, muros, y forward / IV ATM / sesgo de calls del vencimiento
mensual del VIX. Sin historia, una medida suelta no dice si es alta o baja.

Se ejecuta a media sesión (GitHub Actions). Fuera de horario Yahoo deja las
cotizaciones a cero: si el mercado no está abierto no escribe nada (salvo
--force). Si falla el SPY no escribe; si solo falla el VIX, guarda el SPY.

Uso:
    python scripts/update_options_snapshot.py
    python scripts/update_options_snapshot.py --force    # aunque esté cerrado
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vix_controller import alerts as al  # noqa: E402
from vix_controller import calendario as cal  # noqa: E402
from vix_controller.data import yahoo_options as yo  # noqa: E402
from vix_controller.quant import options_desk as od  # noqa: E402
from vix_controller.rates import get_dividend_yield, get_risk_free_rate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("options_snapshot")
SALIDA = Path("data/options_history.parquet")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=SALIDA)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ahora = pd.Timestamp.now(tz="America/New_York")
    hoy = ahora.normalize().tz_localize(None)
    if not al.es_sesion(hoy.date()) and not a.force:
        log.info("%s no es sesión de NYSE: nada que guardar", hoy.date())
        return

    eventos = tuple(f.strftime("%Y-%m-%d") for _, _, f in cal.proximos(hoy, 30))
    try:
        spy, spot, estado = yo.descargar("SPY", eventos=eventos)
    except yo.YahooError as e:
        log.error("SPY: %s — no se escribe nada", e)
        sys.exit(2)
    if estado != "REGULAR" and not a.force:
        log.warning("Mercado en estado %s: cotizaciones poco fiables, no se guarda", estado)
        return
    try:
        vix, _, _ = yo.descargar("^VIX", cercanos=2, objetivos=(30, 60), rango=(0.50, 4.00))
    except yo.YahooError as e:
        log.warning("VIX: %s — se guarda solo el SPY", e)
        vix = None

    r, q = get_risk_free_rate(), get_dividend_yield("SPY")
    fila = od.foto(spy, spot, r, q, vix)
    fila.update({"r": r, "q": q, "hora_utc": pd.Timestamp.now(tz="UTC").strftime("%H:%M")})
    nueva = pd.DataFrame([fila], index=pd.DatetimeIndex([hoy], name="fecha"))

    if a.output.exists():
        h = pd.read_parquet(a.output)
        h.index = pd.DatetimeIndex(h.index).normalize()
        h = pd.concat([h[h.index != hoy], nueva]).sort_index()
    else:
        h = nueva
    a.output.parent.mkdir(parents=True, exist_ok=True)
    h.to_parquet(a.output)
    log.info("OK %s · %d filas · IV30 %.2f · RR25 %.2f · GEX %.2f · sesgo VIX %s",
             a.output, len(h), fila["iv30"], fila["rr25_30"], fila["gex"],
             f"{fila['vix_sesgo']:.1f}" if "vix_sesgo" in fila else "—")


if __name__ == "__main__":
    main()
