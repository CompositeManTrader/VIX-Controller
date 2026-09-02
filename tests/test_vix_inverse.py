"""
Tests del seguimiento del modelo VIX Inverse (congelado 2026-08-03).

El test que más importa es TestConvencionDeSigno: falla si alguien invierte
la señal a M1/M2, que es el error que el README del modelo induce.
"""
import numpy as np
import pandas as pd
import pytest

from vix_controller.quant import vix_inverse as vi


# ─────────────────────────────────────────────────────────────────────
# Datos sintéticos con la MISMA convención que curva.parquet
# ─────────────────────────────────────────────────────────────────────
def _df(n=300, seed=0, contango=True):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    m1 = pd.Series(18.0 + rng.normal(0, 0.3, n).cumsum() * 0.05, index=idx).clip(9, 60)
    # En contango M2 > M1; en backwardation al revés
    m2 = m1 * (1.05 if contango else 0.95)
    vix = m1 * 0.97
    vix3m = vix * (1.08 if contango else 0.93)
    px = pd.Series(40 * np.exp(np.cumsum(rng.normal(-0.002, 0.03, n))), index=idx)
    return pd.DataFrame({
        "open": px * 0.995,
        "close": px,
        "ratio_m2m1": m2 / m1,          # M2/M1 — la convención real
        "ratio_vix3m": vix3m / vix,     # VIX3M/VIX
        "VIX": vix, "VIX3M": vix3m, "m1": m1, "m2": m2,
        "dias_m1": 20.0,
        "contango": (m2 > m1),
        "spy": pd.Series(400 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n))), index=idx),
    }).assign(r_spy=lambda d: d["spy"].pct_change())


# ═════════════════════════════════════════════════════════════════════
# EL TEST OBLIGATORIO: si alguien invierte el signo, esto revienta
# ═════════════════════════════════════════════════════════════════════
class TestConvencionDeSigno:
    """`ratio_m2m1` es M2/M1. Contango = ratio_m2m1 > 1. NO al revés."""

    def test_ratio_m2m1_mayor_que_uno_es_contango(self):
        """Con M2 > M1 (contango) la señal debe estar DENTRO (corto)."""
        d = _df(contango=True)
        assert (d["ratio_m2m1"] > 1).all(), "fixture mal construido"
        assert vi.señal_contango(d).all(), (
            "Con M2/M1 > 1 (contango) la señal debe activarse. "
            "Si esto falla, la señal está INVERTIDA.")

    def test_backwardation_deja_fuera(self):
        """Con M2 < M1 y VIX3M < VIX (backwardation) la señal debe estar FUERA."""
        d = _df(contango=False)
        assert (d["ratio_m2m1"] < 1).all() and (d["ratio_vix3m"] < 1).all()
        assert not vi.señal_contango(d).any(), (
            "En backwardation la señal debe estar apagada. "
            "Si esto falla, la señal está INVERTIDA.")

    def test_señal_invertida_seria_distinta(self):
        """
        Prueba de contraste: la señal correcta y la invertida (M1/M2) NO
        pueden coincidir. Blinda contra un refactor que cambie el sentido.
        """
        d = _df(contango=True)
        correcta = vi.señal_contango(d)
        invertida = ((1 / d["ratio_m2m1"] > 1) | (1 / d["ratio_vix3m"] > 1))
        assert not correcta.equals(invertida.astype(bool)), (
            "La señal coincide con su versión invertida: revisa el signo.")
        assert correcta.sum() > invertida.sum(), (
            "En contango la señal correcta debe estar dentro MÁS días que la invertida.")

    def test_coincide_con_la_columna_contango_del_parquet(self):
        """`ratio_m2m1 > 1` debe coincidir con la columna `contango` del dato."""
        d = _df(contango=True)
        np.testing.assert_array_equal(
            (d["ratio_m2m1"] > 1).to_numpy(), d["contango"].to_numpy())

    def test_or_no_and(self):
        """Basta UNA medida en contango para estar dentro; salir exige las dos."""
        d = _df(contango=True).copy()
        d["ratio_vix3m"] = 0.95           # una invertida, la otra no
        assert vi.señal_contango(d).all(), "Debe ser OR, no AND"
        d["ratio_m2m1"] = 0.98            # ahora las dos
        assert not vi.señal_contango(d).any(), "Con las dos invertidas: fuera"


# ═════════════════════════════════════════════════════════════════════
# Anticipación
# ═════════════════════════════════════════════════════════════════════
class TestSinAnticipacion:
    def test_posicion_va_un_dia_por_detras(self):
        d = _df()
        cont = vi.señal_contango(d)
        pos = vi.señal_posicion(d)
        assert pos.iloc[0] == False              # noqa: E712 — primer día siempre fuera
        pd.testing.assert_series_equal(
            pos.iloc[1:].reset_index(drop=True),
            cont.iloc[:-1].reset_index(drop=True),
            check_names=False)

    def test_verificador_pasa_con_señal_correcta(self):
        d = _df()
        vi.verificar_sin_anticipacion(d, vi.señal_posicion(d))   # no lanza

    def test_verificador_detecta_anticipacion(self):
        """Si alguien quita el shift, el verificador debe cazarlo."""
        d = _df()
        pos_mala = vi.señal_contango(d)          # SIN shift → anticipación
        original = vi.señal_posicion
        try:
            vi.señal_posicion = vi.señal_contango      # inyecta el fallo
            with pytest.raises(AssertionError, match="ANTICIPACIÓN"):
                vi.verificar_sin_anticipacion(d, pos_mala)
        finally:
            vi.señal_posicion = original

    def test_solo_un_shift(self):
        """Un doble shift retrasaría la señal dos días: no debe ocurrir."""
        d = _df()
        pos = vi.señal_posicion(d)
        doble = vi.señal_contango(d).shift(2).fillna(False).astype(bool)
        assert not pos.equals(doble), "Hay un shift de más"


# ═════════════════════════════════════════════════════════════════════
# Simulación
# ═════════════════════════════════════════════════════════════════════
class TestSimulacion:
    def test_corto_gana_si_el_precio_baja(self):
        n = 60
        idx = pd.bdate_range("2021-01-04", periods=n)
        px = pd.Series(np.linspace(100, 50, n), index=idx)
        pos = pd.Series(True, index=idx)
        r = vi.simular_corto(px * 0.999, px, pos, prestamo=0.0, coste=0.0)
        assert (1 + r.dropna()).prod() > 1.0

    def test_corto_pierde_si_el_precio_sube(self):
        n = 60
        idx = pd.bdate_range("2021-01-04", periods=n)
        px = pd.Series(np.linspace(50, 100, n), index=idx)
        pos = pd.Series(True, index=idx)
        r = vi.simular_corto(px * 0.999, px, pos, prestamo=0.0, coste=0.0)
        assert (1 + r.dropna()).prod() < 1.0

    def test_fuera_no_mueve_capital(self):
        d = _df()
        pos = pd.Series(False, index=d.index)
        r = vi.simular_corto(d["open"], d["close"], pos)
        assert np.allclose(r.dropna(), 0.0)

    def test_el_coste_se_cobra_al_entrar_y_al_salir(self):
        """
        Con precio PLANO y una sola operación, el capital final debe caer
        exactamente dos veces el coste (una al abrir, otra al cerrar).

        Con el precio plano la exposición no se mueve, así que el cobro de
        entrada (sobre el capital) y el de salida (sobre la exposición) valen
        lo mismo: capital final = 1 − 2·coste.

        No se comprueba monotonía sobre muchas operaciones a propósito: el
        motor canónico fija `exp = cap` ANTES de descontar la comisión, así
        que un coste mayor eleva mínimamente el apalancamiento efectivo y,
        encadenando tramos ganadores, el resultado final no es monótono en
        el coste. Se replica tal cual para ser fiel al motor congelado.
        """
        n = 20
        idx = pd.bdate_range("2021-01-04", periods=n)
        px = pd.Series(np.full(n, 100.0), index=idx)
        pos = pd.Series(False, index=idx)
        pos.iloc[5:15] = True
        coste = 0.001
        r = vi.simular_corto(px, px, pos, prestamo=0.0, coste=coste)
        final = float((1 + r.dropna()).prod())
        assert final == pytest.approx(1 - 2 * coste, rel=1e-9)

    def test_sin_coste_precio_plano_no_cambia_nada(self):
        n = 20
        idx = pd.bdate_range("2021-01-04", periods=n)
        px = pd.Series(np.full(n, 100.0), index=idx)
        pos = pd.Series(False, index=idx)
        pos.iloc[5:15] = True
        r = vi.simular_corto(px, px, pos, prestamo=0.0, coste=0.0)
        assert float((1 + r.dropna()).prod()) == pytest.approx(1.0, rel=1e-12)

    def test_el_prestamo_resta(self):
        d = _df()
        pos = vi.señal_posicion(d)
        sin = vi.simular_corto(d["open"], d["close"], pos, prestamo=0.0)
        con = vi.simular_corto(d["open"], d["close"], pos, prestamo=0.10)
        assert (1 + con.dropna()).prod() < (1 + sin.dropna()).prod()


# ═════════════════════════════════════════════════════════════════════
# Cartera y métricas
# ═════════════════════════════════════════════════════════════════════
class TestCartera:
    def test_spy_apalancado_iguala_volatilidad(self):
        d = _df()
        sleeve = vi.simular_corto(d["open"], d["close"], vi.señal_posicion(d))
        s = vi.construir_cartera(d, sleeve, peso=0.20)
        assert s["cartera"].std() == pytest.approx(s["spy_apalancado"].std(), rel=1e-9), (
            "El SPY apalancado debe tener EXACTAMENTE la misma volatilidad "
            "que la cartera: si no, la comparación no es honesta.")

    def test_peso_cero_es_spy_puro(self):
        d = _df()
        sleeve = vi.simular_corto(d["open"], d["close"], vi.señal_posicion(d))
        s = vi.construir_cartera(d, sleeve, peso=0.0)
        pd.testing.assert_series_equal(s["cartera"], s["spy"], check_names=False)

    def test_metricas_basicas(self):
        idx = pd.bdate_range("2020-01-02", periods=252 * 3)
        r = pd.Series(np.full(len(idx), 0.0004), index=idx)
        m = vi.metricas(r)
        assert m["cagr"] == pytest.approx(((1.0004 ** 252) - 1) * 100, rel=1e-3)
        assert m["dd"] == pytest.approx(0.0, abs=1e-9)

    def test_metricas_con_benchmark(self):
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2020-01-02", periods=600)
        b = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
        r = 1.5 * b                                    # beta exacta 1,5
        m = vi.metricas(r, b)
        assert m["beta"] == pytest.approx(1.5, rel=1e-6)

    def test_serie_corta_devuelve_vacio(self):
        r = pd.Series([0.01, -0.01], index=pd.bdate_range("2020-01-02", periods=2))
        assert vi.metricas(r) == {}


# ═════════════════════════════════════════════════════════════════════
# Operaciones
# ═════════════════════════════════════════════════════════════════════
class TestOperaciones:
    def test_un_tramo_una_operacion(self):
        n = 40
        idx = pd.bdate_range("2021-01-04", periods=n)
        d = pd.DataFrame({"open": np.linspace(100, 60, n),
                          "close": np.linspace(100, 60, n)}, index=idx)
        pos = pd.Series(False, index=idx)
        pos.iloc[10:20] = True
        sleeve = vi.simular_corto(d["open"], d["close"], pos, prestamo=0.0, coste=0.0)
        ops = vi.extraer_operaciones(d, pos, sleeve)
        assert len(ops) == 1
        assert ops[0]["f_entrada"] == idx[10]
        assert ops[0]["f_salida"] == idx[20]
        assert not ops[0]["abierta"]
        assert ops[0]["ret"] > 0            # precio bajando y estamos cortos

    def test_operacion_abierta_al_final(self):
        n = 30
        idx = pd.bdate_range("2021-01-04", periods=n)
        d = pd.DataFrame({"open": np.full(n, 100.0), "close": np.full(n, 100.0)},
                         index=idx)
        pos = pd.Series(False, index=idx)
        pos.iloc[25:] = True
        sleeve = vi.simular_corto(d["open"], d["close"], pos)
        ops = vi.extraer_operaciones(d, pos, sleeve)
        assert ops[-1]["abierta"] is True
        assert ops[-1]["f_salida"] is None

    def test_racha(self):
        idx = pd.bdate_range("2021-01-04", periods=10)
        pos = pd.Series([False] * 6 + [True] * 4, index=idx)
        assert vi.racha_actual(pos) == 4
        assert vi.racha_actual(pd.Series([], dtype=bool)) == 0


# ═════════════════════════════════════════════════════════════════════
# Estado de hoy
# ═════════════════════════════════════════════════════════════════════
class TestEstadoActual:
    def test_usa_el_cierre_de_ayer(self):
        d = _df()
        pos = vi.señal_posicion(d)
        e = vi.estado_actual(d, pos)
        assert e["fecha_dato"] == d.index[-2]
        assert e["fecha_ejecucion"] == d.index[-1]
        assert e["ratio_m2m1"] == pytest.approx(float(d["ratio_m2m1"].iloc[-2]))

    def test_distancias_al_uno(self):
        d = _df(contango=True)
        e = vi.estado_actual(d, vi.señal_posicion(d))
        assert e["dist_m2m1"] == pytest.approx(e["ratio_m2m1"] - 1.0)
        assert e["dist_vix3m"] > 0          # en contango, por encima de 1

    def test_detecta_cambio_pendiente(self):
        """Si el último cierre invierte la curva, debe avisar del cambio."""
        d = _df(contango=True).copy()
        d.iloc[-1, d.columns.get_loc("ratio_m2m1")] = 0.97
        d.iloc[-1, d.columns.get_loc("ratio_vix3m")] = 0.98
        e = vi.estado_actual(d, vi.señal_posicion(d))
        assert e["dentro"] is True          # hoy seguimos dentro
        assert e["prox_dentro"] is False    # mañana se sale
        assert e["cambio_pendiente"] is True


# ═════════════════════════════════════════════════════════════════════
# NaN y datos que faltan — nada silencioso
# ═════════════════════════════════════════════════════════════════════
class TestDatosProblematicos:
    def test_nan_en_vix3m_no_activa_la_señal_por_ese_lado(self):
        d = _df(contango=True).copy()
        d["ratio_vix3m"] = np.nan
        d["ratio_m2m1"] = 0.98              # la otra, invertida
        assert not vi.señal_contango(d).any(), (
            "Un NaN no puede activar el OR: sin dato no hay contango.")

    def test_nan_en_vix3m_con_m2m1_en_contango_sigue_dentro(self):
        d = _df(contango=True).copy()
        d["ratio_vix3m"] = np.nan
        assert vi.señal_contango(d).all()

    def test_dataframe_corto_lanza_error(self):
        d = _df(n=1)
        with pytest.raises(vi.DatosVixInverse):
            vi.estado_actual(d, vi.señal_posicion(d))

    def test_ruta_inexistente_lanza_error_explicito(self):
        with pytest.raises(vi.DatosVixInverse, match="No existe"):
            vi.cargar_datos("C:/ruta/que/no/existe/jamas")


# ═════════════════════════════════════════════════════════════════════
# Formato español
# ═════════════════════════════════════════════════════════════════════
class TestFormatoEspanol:
    def test_coma_decimal_y_punto_de_millar(self):
        assert vi.num_es(1234.5) == "1.234,50"
        assert vi.num_es(1234567.891, 3) == "1.234.567,891"
        assert vi.num_es(-42.7) == "-42,70"
        assert vi.num_es(18.5, 2, signo=True) == "+18,50"

    def test_porcentaje(self):
        assert vi.pct_es(18.5) == "+18,50 %"
        assert vi.pct_es(-32.71) == "-32,71 %"
        assert vi.pct_es(13.91, signo=False) == "13,91 %"

    def test_nan_y_none(self):
        assert vi.num_es(None) == "—"
        assert vi.num_es(float("nan")) == "—"
        assert vi.pct_es(np.nan) == "—"


# ═════════════════════════════════════════════════════════════════════
# Las cifras de referencia no se tocan
# ═════════════════════════════════════════════════════════════════════
class TestReferenciasCongeladas:
    def test_parametros_congelados(self):
        assert vi.PESO_SLEEVE == 0.20
        assert vi.BANDA_PESO == (0.15, 0.20)
        assert vi.PRESTAMO_ANUAL == 0.06
        assert vi.COSTE_POR_LADO == pytest.approx(0.0007)
        assert vi.FECHA_CONGELACION == "2026-08-03"

    def test_referencias_del_motor_generico(self):
        r = vi.REF_MOTOR_GENERICO
        assert r["cagr"] == 18.50 and r["vol"] == 22.75
        assert r["sharpe"] == 0.862 and r["beta"] == 1.220
        assert r["dd"] == -32.71 and r["cagr_spy"] == 13.91

    def test_el_modelo_no_esta_validado(self):
        assert vi.ESTADO["validado"] is False
        assert vi.ESTADO["dsr_modelo"] < vi.ESTADO["dsr_umbral"]
        assert vi.ESTADO["prob_batir_spy"] < vi.ESTADO["prob_umbral"]

    def test_el_peso_recomendado_es_menor_que_kelly(self):
        assert vi.PESO_SLEEVE < vi.ESTADO["kelly_completo"]
        assert vi.PESO_SLEEVE == pytest.approx(vi.ESTADO["kelly_medio"], abs=0.04)
