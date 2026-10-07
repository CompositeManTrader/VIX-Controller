"""Tests de vol_desk (página Volatilidad), del archivo CFE y de las medidas extendidas."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vix_controller.quant import vix_inverse_live as vl
from vix_controller.quant import vol_desk as vd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import update_vx_curve as uvc  # noqa: E402


def _bdays(n, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


# ── Volatilidad realizada y prima ────────────────────────────────────
def test_realizada_constante():
    """Rendimiento log constante de 1 % diario → √252 · 1 % = 15,87 %."""
    idx = _bdays(60)
    close = pd.Series(100 * np.exp(0.01 * np.arange(60)), index=idx)
    rv = vd.realizada(close, 21)
    assert rv.iloc[:21].isna().all()
    assert rv.iloc[21] == pytest.approx(np.sqrt(252), rel=1e-9)


def test_realizada_futura_no_mira_el_dia_t():
    """La realizada futura de t solo usa rendimientos de t+1 … t+n."""
    idx = _bdays(80)
    r = np.zeros(80)
    r[40] = 0.05                                  # un salto en t=40
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    fut = vd.realizada_futura(close, 21)
    assert fut.iloc[40] == pytest.approx(0.0)     # el salto de t no cuenta en t
    assert fut.iloc[39] > 0                       # sí en t−1
    assert fut.iloc[19] > 0                       # t+1…t+21 = 20…40: el salto entra
    assert fut.iloc[18] == pytest.approx(0.0)     # 19…39: aún no
    assert fut.iloc[-21:].isna().all()


def test_prima_por_nivel_cuenta_y_porcentaje():
    idx = _bdays(4)
    d = pd.DataFrame({"vix": [11.0, 13.0, 14.0, 31.0], "rv_fut": [10.0, 15.0, 12.0, 20.0]},
                     index=idx)
    d["prima"] = d["vix"] - d["rv_fut"]
    t = vd.prima_por_nivel(d)
    assert t.loc["< 12", "dias"] == 1
    assert t.loc["12–15", "dias"] == 2
    assert t.loc["12–15", "gana"] == pytest.approx(50.0)
    assert t.loc["≥ 30", "prima"] == pytest.approx(11.0)
    assert t["peso"].sum() == pytest.approx(100.0)


def test_rendimiento_siguiente_y_huecos():
    idx = _bdays(6)
    r = pd.Series([0.0, 0.10, 0.10, np.nan, 0.0, 0.0], index=idx)
    f = vd.rendimiento_siguiente(r, 2)
    assert f.iloc[0] == pytest.approx(1.1 * 1.1 - 1)   # t+1, t+2
    assert np.isnan(f.iloc[1])                         # ventana con un NaN → NaN, no 0
    assert np.isnan(f.iloc[2])
    assert f.iloc[3] == pytest.approx(0.0)
    assert f.iloc[-2:].isna().all()


# ── VXX reconstruido ─────────────────────────────────────────────────
def _curva_sintetica():
    """Dos vencimientos y curva plana que sube un 10 % el último día."""
    fechas = pd.bdate_range("2021-01-04", "2021-02-12")
    exps = [pd.Timestamp("2021-01-20"), pd.Timestamp("2021-02-17"), pd.Timestamp("2021-03-17"),
            pd.Timestamp("2021-04-21")]
    filas = []
    for i, f in enumerate(fechas):
        vivos = [e for e in exps if e > f][:3]
        px = 20.0 * (1.1 if i == len(fechas) - 1 else 1.0)
        fila = {}
        for k, e in enumerate(vivos, 1):
            fila[f"m{k}"] = px
            fila[f"exp_m{k}"] = e
        filas.append(fila)
    return pd.DataFrame(filas, index=fechas), exps


def test_indice_corto_plazo_curva_plana():
    c, _ = _curva_sintetica()
    r = vd.indice_corto_plazo(c)
    assert r.iloc[-1] == pytest.approx(0.10)
    assert (r.iloc[1:-1].dropna().abs() < 1e-12).all()


def test_indice_corto_plazo_pesos_del_roll():
    """Con el front a 10 y el siguiente a 20, el rendimiento refleja el peso fijado en t−1."""
    c, exps = _curva_sintetica()
    c["m1"], c["m2"], c["m3"] = 10.0, 20.0, 30.0
    t, t1 = pd.Timestamp("2021-01-25"), pd.Timestamp("2021-01-26")
    c.loc[t1, ["m1", "m2", "m3"]] = [11.0, 20.0, 30.0]           # solo sube el front
    r = vd.indice_corto_plazo(c)
    f, prev = exps[1], exps[0]
    dt = np.busday_count(prev.date(), f.date())
    w1 = (np.busday_count(t.date(), f.date()) - 1) / dt
    esperado = (w1 * 11 + (1 - w1) * 20) / (w1 * 10 + (1 - w1) * 20) - 1
    assert r.loc[t1] == pytest.approx(esperado)


# ── Régimen ──────────────────────────────────────────────────────────
def test_tabla_regimen_separa_entornos_y_aplica_regla():
    idx = _bdays(60)
    vix = pd.Series(14.0, index=idx)
    vix.iloc[30:] = 35.0
    contango = pd.Series(True, index=idx)
    contango.iloc[30:] = False
    vxx = pd.Series(-0.01, index=idx)                 # el VXX cae: el corto gana
    t = vd.tabla_regimen(vix, contango, vxx, n=5)
    regs = set(t["regimen"])
    assert regs == {"VIX < 15 · contango", "VIX ≥ 30 · backwardation"}
    back = t.set_index("regimen").loc["VIX ≥ 30 · backwardation"]
    assert back["regla_media"] == pytest.approx(0.0)  # fuera: la regla no está corta
    assert back["siempre_media"] > 0


# ── Archivo CFE ──────────────────────────────────────────────────────
def test_parse_archive_reescala_y_comas_de_mas():
    raw = ("Trade Date,Futures,Open,High,Low,Close,Settle,Change,Total Volume,EFP,Open Interest\n"
           "3/23/2007,K (May 07),136.9,139,136.5,138,138.7,1.5,561,0,19879,\n"
           "03/26/2007,K (May 07),13.80,14.30,13.70,13.80,13.79,-0.80,490,0,20141\n"
           "03/26/2007,K (May 07),13.80,14.30,13.70,13.80,13.79,-0.80,490,0,20141\n"
           "05/16/2007,K (May 07),0,0,0,0,13.63,-0.33,0,0,11742\n").encode()
    d = uvc.parse_archive_csv(raw)
    assert len(d) == 3                                       # duplicado fuera
    assert d.iloc[0]["settle"] == pytest.approx(13.87)       # antes del 26/03/2007: ÷10
    assert d.iloc[1]["settle"] == pytest.approx(13.79)
    assert (d["liquidacion"] == pd.Timestamp("2007-05-16")).all()


# ── Medidas extendidas ───────────────────────────────────────────────
def test_medidas_extendidas_respetan_la_regla_y_el_hueco():
    idx = _bdays(40, "2013-01-01")
    curva = pd.DataFrame({"ratio_m2m1": 1.05, "ratio_vix3m": 1.10, "gap_ok": True}, index=idx)
    curva.iloc[3, curva.columns.get_loc("gap_ok")] = False
    curva.iloc[10:13, [0, 1]] = 0.95                         # las dos invertidas 3 días
    hist_idx = _bdays(5, "2013-03-01")
    hist = pd.DataFrame({"ratio_m2m1": 1.1, "ratio_vix3m": 1.1, "pos": True}, index=hist_idx)
    m = vl.medidas_extendidas(hist, curva)
    previo = m[~m["backtest"]]
    assert previo.index[0] == idx[4]                         # tras el último hueco
    assert not previo.loc[idx[11], "pos"]                    # decide con el cierre anterior
    assert previo.loc[idx[10], "pos"]                        # el día 10 aún se ejecuta dentro
    assert not previo.loc[idx[13], "pos"]
    assert previo.loc[idx[14], "pos"]
    assert m["backtest"].sum() == len(hist)
