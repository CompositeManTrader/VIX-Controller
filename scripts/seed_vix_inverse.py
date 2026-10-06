"""
seed_vix_inverse.py — Siembra la historia derivada del modelo VIX Inverse.

Se ejecuta UNA vez, en local, donde está la Plataforma. Lee sus datos en
SOLO LECTURA, corre el motor congelado y guarda la historia derivada
(rendimientos + estado del motor, sin precios brutos) en
data/vix_inverse_history.parquet. A partir de ahí la continúa la GitHub
Action con datos públicos (scripts/update_vix_inverse.py).

Comprueba al final que reproduce las cifras publicadas del informe.

Uso:  python scripts/seed_vix_inverse.py [--hasta AAAA-MM-DD]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vix_controller import config as cfg  # noqa: E402
from vix_controller.quant import vix_inverse as vi  # noqa: E402
from vix_controller.quant import vix_inverse_live as vl  # noqa: E402

OUT = Path("data/vix_inverse_history.parquet")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hasta", default=None, help="última fecha a sembrar")
    a = ap.parse_args()

    df, diag = vi.cargar_datos(cfg.VIX_INVERSE_CURATED)
    if a.hasta:
        df = df.loc[:a.hasta]
    vi.verificar_sin_anticipacion(df, vi.señal_posicion(df))
    hist = vl.construir_historia(df)

    # ── Validación contra el informe publicado ──────────────────────────
    hasta_informe = hist.loc[:"2026-07-30"]
    m = vi.metricas(vl.cartera(hasta_informe)["cartera"], hasta_informe["spy_ret"])
    ops = vl.operaciones(hasta_informe)
    epi = vl.episodios(hist)
    print(f"historia: {len(hist):,} sesiones · {hist.index[0].date()} → {hist.index[-1].date()}")
    print(f"hasta 2026-07-30 · CAGR {m['cagr']:.2f} % (informe 17,91) · "
          f"Sharpe {m['sharpe']:.3f} (0,96) · caída {m['dd']:.2f} % (−30,93) · "
          f"operaciones {len(ops)} (76)")
    print("episodios:", ", ".join(d.strftime("%d/%m/%Y") for d in epi["fecha"]))
    # El informe usa 5 pb de comisión; el seguimiento cobra 5 + 2 pb de
    # deslizamiento (config.json) → −0,05 puntos de CAGR, medido. Tolerancia 0,10.
    ok = (abs(m["cagr"] - 17.91) < 0.10 and abs(m["dd"] + 30.93) < 0.05
          and abs(m["sharpe"] - 0.96) < 0.01 and len(ops) == 76)
    if not ok:
        print("ERROR: no reproduce el informe — no se escribe nada")
        sys.exit(1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(OUT, compression="snappy")
    print(f"escrito {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
