"""Tests de options_desk (página Opciones), selección de vencimientos y calendario."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from vix_controller import calendario as cal
from vix_controller.data import yahoo_options as yo
from vix_controller.quant import options_desk as od
from vix_controller.quant.bs import bs_call, bs_put, bs_iv

R, Q = 0.04, 0.01


def _cadena(spot, dte, vol_fn, strikes, oi=1000, T=None):
    """Cadena sintética con precios BS exactos a partir de una función de volatilidad."""
    T = T if T is not None else dte / 365.0
    filas_c, filas_p = [], []
    for k in strikes:
        v = vol_fn(k)
        c, p = bs_call(spot, k, R, T, v, Q), bs_put(spot, k, R, T, v, Q)
        filas_c.append({"strike": k, "mid": c, "openInterest": oi})
        filas_p.append({"strike": k, "mid": p, "openInterest": oi})
    d = {"calls": pd.DataFrame(filas_c), "puts": pd.DataFrame(filas_p), "dte": dte}
    if T is not None:
        d["T"] = T
    return d


# ── IV vectorizada ───────────────────────────────────────────────────
def test_iv_vectorizada_igual_que_brent():
    spot, T = 100.0, 30 / 365
    K = np.array([80.0, 95.0, 100.0, 105.0, 120.0])
    vols = np.array([0.30, 0.22, 0.20, 0.18, 0.16])
    precios = np.array([bs_call(spot, k, R, T, v, Q) for k, v in zip(K, vols)])
    iv = od.iv_bs(spot, K, T, R, Q, precios, True)
    ref = [bs_iv(spot, k, R, T, p, "C", Q) for k, p in zip(K, precios)]
    np.testing.assert_allclose(iv, vols, atol=1e-6)
    np.testing.assert_allclose(iv, ref, atol=1e-6)


def test_iv_sin_solucion_es_nan():
    iv = od.iv_bs(100.0, np.array([90.0]), 0.1, R, Q, np.array([0.01]), True)  # bajo el intrínseco
    assert np.isnan(iv[0])


# ── Sonrisa, 25Δ y estructura ────────────────────────────────────────
def test_sonrisa_plana_da_rr_y_bf_cero():
    spot = 100.0
    dato = _cadena(spot, 30, lambda k: 0.20, np.arange(70, 131, 1.0), T=30 / 365)
    m = od.metricas_vencimiento(dato, spot, R, Q)
    assert m["atm"] == pytest.approx(0.20, abs=1e-4)
    assert m["rr25"] == pytest.approx(0.0, abs=1e-4)
    assert m["bf25"] == pytest.approx(0.0, abs=1e-4)


def test_sesgo_de_puts_da_rr_negativo():
    spot = 100.0
    dato = _cadena(spot, 30, lambda k: 0.20 + 0.5 * max(0.0, (100 - k) / 100), np.arange(70, 131, 1.0),
                   T=30 / 365)
    m = od.metricas_vencimiento(dato, spot, R, Q)
    assert m["rr25"] < -0.01
    assert m["put25"] > m["atm"] > m["call25"] - 1e-6


def test_a_30_dias_interpola_varianza_y_no_extrapola():
    est = pd.DataFrame({"dte": [20, 40], "atm": [0.10, 0.20], "rr25": [-0.01, -0.03]})
    w = (0.10 ** 2 * 20 + (0.20 ** 2 * 40 - 0.10 ** 2 * 20) * 0.5) / 30
    assert od.a_30_dias(est, "atm") == pytest.approx(np.sqrt(w))
    assert od.a_30_dias(est, "rr25") == pytest.approx(-0.02)
    assert np.isnan(od.a_30_dias(est, "atm", dias=60))


def test_movimiento_descontado_es_el_straddle():
    dato = {"calls": pd.DataFrame({"strike": [99.0, 100.0], "mid": [2.5, 2.0]}),
            "puts": pd.DataFrame({"strike": [99.0, 100.0], "mid": [1.5, 2.1]}), "dte": 7}
    assert od.movimiento_descontado(dato, 100.2) == pytest.approx(4.1 / 100.2 * 100)


# ── GEX ──────────────────────────────────────────────────────────────
def test_gex_signos_convencion_estandar():
    spot = 100.0
    cal_ = _cadena(spot, 10, lambda k: 0.2, [100.0], T=10 / 365)
    cal_["puts"]["openInterest"] = 0
    put_ = _cadena(spot, 10, lambda k: 0.2, [100.0], T=10 / 365)
    put_["calls"]["openInterest"] = 0
    assert od.gex_por_strike({"a": cal_}, spot, R, Q)["neto"].sum() > 0     # calls: amortiguan
    assert od.gex_por_strike({"a": put_}, spot, R, Q)["neto"].sum() < 0     # puts: amplifican


def test_nivel_de_giro_entre_muro_de_puts_y_de_calls():
    """Puts concentradas en 95 y calls en 105: la GEX cambia de signo entre ambos."""
    spot = 100.0
    d = _cadena(spot, 10, lambda k: 0.2, [95.0, 105.0], T=10 / 365)
    d["calls"].loc[d["calls"]["strike"] == 95.0, "openInterest"] = 0
    d["puts"].loc[d["puts"]["strike"] == 105.0, "openInterest"] = 0
    curva = od.gex_en_precios({"a": d}, spot, R, Q, np.linspace(90, 110, 201))
    giro = od.nivel_giro(curva, spot)
    assert 95.0 < giro < 105.0
    assert curva.loc[curva.index < 95].mean() < 0 < curva.loc[curva.index > 105].mean()


# ── VIX ──────────────────────────────────────────────────────────────
def test_forward_del_vix_por_paridad():
    F, T, r = 18.0, 20 / 365, R
    K = np.array([15.0, 17.0, 18.0, 20.0, 25.0])
    c = od._precio_b76(F, K, T, r, 0.9, True)
    p = od._precio_b76(F, K, T, r, 0.9, False)
    dato = {"calls": pd.DataFrame({"strike": K, "mid": c, "openInterest": 100}),
            "puts": pd.DataFrame({"strike": K, "mid": p, "openInterest": 100}), "dte": 20, "T": T}
    assert od.vix_forward(dato, r) == pytest.approx(F, abs=1e-9)
    s = od.sonrisa_vix(dato, r)
    np.testing.assert_allclose(s["iv"], 0.9, atol=1e-5)


def test_vix_principal_es_el_de_mas_interes_abierto():
    def d(dte, oi):
        x = pd.DataFrame({"strike": [20.0], "mid": [1.0], "openInterest": [oi]})
        return {"calls": x, "puts": x, "dte": dte}
    vix = {"semanal": d(10, 1_000), "mensual": d(14, 2_000_000), "lejano": d(90, 5_000_000)}
    assert od.vix_principal(vix) == "mensual"


# ── Selección de vencimientos y calendario ───────────────────────────
def test_elegir_vencimientos():
    hoy = date(2026, 10, 8)
    fechas = ["2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13", "2026-10-16", "2026-10-23",
              "2026-10-30", "2026-11-06", "2026-11-20", "2026-12-18", "2027-01-15"]
    sel = dict(yo.elegir_vencimientos(fechas, hoy, cercanos=3, objetivos=(30, 60, 90),
                                      eventos=("2026-10-28",)))
    assert "2026-10-08" not in sel                      # el del día no
    assert {"2026-10-09", "2026-10-12", "2026-10-13"} <= set(sel)
    assert "2026-10-30" in sel                          # primero tras el FOMC del 28
    assert "2026-11-06" in sel                          # ≈ 30 días
    assert "2027-01-15" in sel                          # ≈ 90 días


def test_limpiar_exige_mercado_real():
    df = pd.DataFrame({"strike": [100, 101, 102, 103], "bid": [1.0, 0.0, 1.0, 1.0],
                       "ask": [1.1, 1.0, 3.0, 1.1], "openInterest": [50, 50, 50, 5]})
    out = yo.limpiar(df, 100.0)
    assert out["strike"].tolist() == [100]               # sin bid, horquilla ancha, OI bajo: fuera


def test_calendario_proximos():
    ev = cal.proximos(pd.Timestamp("2026-10-08"), 14)
    assert [(n, f.strftime("%d/%m")) for _, n, f in ev] == [("CPI", "13/10")]
    assert cal.ultimo_evento_conocido() >= pd.Timestamp("2026-12-01")
