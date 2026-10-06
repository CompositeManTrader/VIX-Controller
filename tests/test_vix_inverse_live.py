"""Historia derivada del modelo: siembra, continuación y lecturas del panel."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vix_controller.quant import vix_inverse as vi
from vix_controller.quant import vix_inverse_live as vl


def _df(n=500, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    px = pd.Series(40 * np.exp(np.cumsum(rng.normal(-0.002, 0.03, n))), index=idx)
    r1 = pd.Series(1.03 + rng.normal(0, 0.03, n), index=idx)
    r2 = pd.Series(1.05 + rng.normal(0, 0.04, n), index=idx)
    spy = pd.Series(400 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n))), index=idx)
    return pd.DataFrame({"open": px * (1 + rng.normal(0, 0.005, n)), "close": px,
                         "ratio_m2m1": r1, "ratio_vix3m": r2,
                         "spy": spy, "r_spy": spy.pct_change()})


def _nuevos(df, desde):
    t = df.loc[desde:]
    return pd.DataFrame({"open": t["open"], "close": t["close"], "spy_close": t["spy"],
                         "ratio_m2m1": t["ratio_m2m1"], "ratio_vix3m": t["ratio_vix3m"]})


class TestContinuacion:
    def test_continuar_equivale_a_una_pasada(self):
        df = _df()
        full = vl.construir_historia(df)
        corte = df.index[300]
        part = vl.construir_historia(df.loc[:corte])
        cont = vl.continuar_historia(part, _nuevos(df, corte), vxx_close_lookback=df["close"])
        assert len(cont) == len(full)
        for col in ("pos", "contango"):
            assert (cont[col] == full[col]).all()
        for col in ("sleeve_ret", "always_ret", "cap", "exp"):
            np.testing.assert_allclose(cont[col].iloc[301:], full[col].iloc[301:], rtol=1e-12)

    def test_sesion_incompleta_no_se_anade(self):
        df = _df()
        corte = df.index[300]
        part = vl.construir_historia(df.loc[:corte])
        n = _nuevos(df, corte)
        n.iloc[3, n.columns.get_loc("ratio_vix3m")] = np.nan     # 3ª sesión nueva sin medida
        cont = vl.continuar_historia(part, n)
        assert len(cont) == len(part) + 2                          # para antes del hueco

    def test_ancla_obligatoria(self):
        df = _df()
        part = vl.construir_historia(df.iloc[:300])
        with pytest.raises(vl.HistoriaError):
            vl.continuar_historia(part, _nuevos(df, df.index[310]))

    def test_px_exec_solo_en_cambios(self):
        h = vl.construir_historia(_df())
        cambios = h["pos"] != h["pos"].shift(fill_value=False)
        assert h["px_exec"].notna().sum() == cambios.sum()


class TestLecturas:
    def test_spy_apalancado_iguala_volatilidad(self):
        c = vl.cartera(vl.construir_historia(_df()), 0.2)
        assert c["cartera"].std() == pytest.approx(c["spy_apalancado"].std(), rel=1e-9)

    def test_operaciones_coinciden_con_motor_canonico(self):
        df = _df()
        h = vl.construir_historia(df)
        pos = vi.señal_posicion(df)
        sl = vi.simular_corto(df["open"], df["close"], pos)
        canon = vi.extraer_operaciones(df, pos, sl)
        mias = vl.operaciones(h)
        assert len(mias) == len(canon)
        for a, b in zip(mias, canon):
            assert a["f_entrada"] == b["f_entrada"] and a["ret"] == pytest.approx(b["ret"])

    def test_episodio_un_dia_por_susto(self):
        idx = pd.bdate_range("2022-01-03", periods=60)
        close = pd.Series(20.0, index=idx)
        close.iloc[30:35] = [24, 26, 28, 30, 32]        # +60 % en 5 sesiones
        epi = vl._episodios(close)
        assert epi.sum() == 1 and epi.idxmax() == idx[33]

    def test_condiciones_y_veredicto(self):
        h = vl.construir_historia(_df(n=900))
        conds = vl.condiciones_parada(h)
        assert len(conds) == 4
        estado, _ = vl.veredicto_protocolo(conds)
        assert estado in ("OPERABLE", "REVISIÓN", "APAGADO")

    def test_veredicto_reglas(self):
        mk = lambda d: vl.Condicion("x", "", "", "", d, False)
        assert vl.veredicto_protocolo([mk(False)] * 4)[0] == "OPERABLE"
        assert vl.veredicto_protocolo([mk(True), mk(False)])[0] == "REVISIÓN"
        assert vl.veredicto_protocolo([mk(True), mk(True)])[0] == "APAGADO"


class TestEstado:
    def test_curva_un_dia_por_delante(self):
        df = _df()
        h = vl.construir_historia(df.iloc[:-1])
        ultimo = df.index[-1]
        curva = pd.DataFrame({"ratio_m2m1": [0.95], "ratio_vix3m": [0.97]}, index=[ultimo])
        e = vl.estado(h, curva)
        assert e["fecha_dato"] == ultimo
        assert e["dentro"] == bool(h["contango"].iloc[-1])    # posición vigente hoy
        assert e["prox_dentro"] is False                       # las dos invertidas
        assert e["cambio_pendiente"] == e["dentro"]

    def test_una_invertida(self):
        h = vl.construir_historia(_df())
        curva = pd.DataFrame({"ratio_m2m1": [1.006], "ratio_vix3m": [0.998]},
                             index=[h.index[-1] + pd.offsets.BDay(1)])
        e = vl.estado(h, curva)
        assert e["una_invertida"] and e["prox_dentro"]


@pytest.mark.skipif(not Path("data/vix_inverse_history.parquet").exists(),
                    reason="historia derivada no generada")
class TestHistoriaReal:
    def test_reproduce_informe(self):
        h = pd.read_parquet("data/vix_inverse_history.parquet").loc[:"2026-07-30"]
        m = vi.metricas(vl.cartera(h)["cartera"], h["spy_ret"])
        assert m["sharpe"] == pytest.approx(0.96, abs=0.01)
        assert m["dd"] == pytest.approx(-30.93, abs=0.05)
        assert len(vl.operaciones(h)) == 76

    def test_cinco_episodios_publicados(self):
        h = pd.read_parquet("data/vix_inverse_history.parquet")
        fechas = [d.strftime("%Y-%m-%d") for d in h.index[h["episodio"].astype(bool)]]
        assert fechas[:5] == ["2015-08-24", "2018-02-05", "2020-02-25", "2024-08-05", "2025-04-04"]

    def test_sin_precios_brutos(self):
        h = pd.read_parquet("data/vix_inverse_history.parquet")
        assert not {"open", "close", "spy"} & set(h.columns)
        assert h["px_exec"].notna().sum() < 300              # solo días de ejecución
