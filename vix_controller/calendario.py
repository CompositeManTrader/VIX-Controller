"""
calendario.py — Datos macro que mueven la volatilidad (fechas oficiales de la
Fed y del BLS). Actualizar cada diciembre con el calendario del año siguiente.
"""
from __future__ import annotations

import pandas as pd

EVENTOS = {
    "FOMC": ["2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17", "2026-07-29",
             "2026-09-16", "2026-10-28", "2026-12-16"],
    "CPI": ["2026-01-14", "2026-02-12", "2026-03-11", "2026-04-14", "2026-05-13", "2026-06-10",
            "2026-07-15", "2026-08-12", "2026-09-10", "2026-10-13", "2026-11-12", "2026-12-10"],
    "Empleo": ["2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03", "2026-05-08", "2026-06-05",
               "2026-07-02", "2026-08-07", "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04"],
}


def proximos(hoy: pd.Timestamp, dias: int = 14) -> list[tuple[int, str, pd.Timestamp]]:
    """(días que faltan, nombre, fecha) de los eventos entre hoy y hoy + `dias`."""
    hoy = pd.Timestamp(hoy).normalize()
    out = []
    for nombre, fechas in EVENTOS.items():
        for f in fechas:
            d = (pd.Timestamp(f) - hoy).days
            if 0 <= d <= dias:
                out.append((d, nombre, pd.Timestamp(f)))
    return sorted(out)


def ultimo_evento_conocido() -> pd.Timestamp:
    return max(pd.Timestamp(f) for fs in EVENTOS.values() for f in fs)
