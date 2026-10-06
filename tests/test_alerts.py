"""Alertas: eventos, calendario NYSE, correo y script de envío."""
import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from vix_controller import alerts as al


def _curva(pares, desde="2024-07-29"):
    idx = pd.bdate_range(desde, periods=len(pares))
    return pd.DataFrame(pares, index=idx, columns=["ratio_m2m1", "ratio_vix3m"])


class TestEventos:
    def test_agosto_2024(self):
        # 31/07, 01/08, 02/08 (una invertida), 05/08 (dos), 06/08, 07/08, 08/08 (vuelve)
        pares = [(1.031, 1.037), (1.022, 1.030), (1.006, 0.998), (0.947, 0.874),
                 (0.965, 0.966), (0.940, 0.970), (0.958, 1.014)]
        c = _curva(pares, "2024-07-31")
        tipos = [getattr(al.detectar_evento(c.iloc[:i]), "tipo", None) for i in range(2, 8)]
        assert tipos == [None, al.AVISO, al.SALIDA, None, None, al.ENTRADA]

    def test_salida_se_ejecuta_en_la_siguiente_sesion(self):
        # Cierre del jueves 02/04/2026: el viernes es Viernes Santo → ejecución el lunes 06/04
        c = _curva([(1.02, 1.03), (0.98, 0.99)], "2026-04-01")    # mié, jue
        ev = al.detectar_evento(c)
        assert ev.tipo == al.SALIDA and ev.fecha_ejecucion == date(2026, 4, 6)

    def test_salida_requiere_las_dos(self):
        c = _curva([(1.02, 1.03), (0.99, 1.01)])
        assert al.detectar_evento(c).tipo == al.AVISO

    def test_aviso_no_se_repite(self):
        c = _curva([(1.02, 1.03), (0.99, 1.01), (0.98, 1.005)])
        assert al.detectar_evento(c) is None

    def test_dato_faltante_no_es_salida(self):
        c = _curva([(1.02, 1.03), (float("nan"), 0.95)])
        assert al.detectar_evento(c) is None                      # sin las dos no se evalúa


class TestCalendario:
    def test_festivos_2026(self):
        f = al.nyse_holidays(2026)
        assert date(2026, 4, 3) in f            # Viernes Santo
        assert date(2026, 6, 19) in f           # Juneteenth
        assert date(2026, 7, 3) in f            # 4 de julio en sábado → viernes
        assert date(2026, 11, 26) in f          # Thanksgiving

    def test_siguiente_sesion_salta_fin_de_semana_y_festivo(self):
        assert al.siguiente_sesion(date(2026, 4, 2)) == date(2026, 4, 6)
        assert al.siguiente_sesion(date(2026, 10, 9)) == date(2026, 10, 12)   # Columbus abre
        assert al.sesion_anterior(date(2026, 1, 20)) == date(2026, 1, 16)      # MLK


class TestCorreo:
    def test_asunto_y_cuerpo(self):
        ev = al.Evento(al.SALIDA, date(2024, 8, 5), date(2024, 8, 6), 0.9468, 0.874, 1.006, 0.9979,
                       detalle="Las dos medidas han cerrado en backwardation.")
        assert "SALIDA" in al.asunto(ev) and "06/08/2024" in al.asunto(ev)
        txt = al.cuerpo_texto(ev)
        assert "0,9468" in txt and "0,8740" in txt          # convención española
        assert "No es asesoramiento" in txt
        assert "#EA3943" in al.cuerpo_html(ev)


class TestScript:
    @pytest.fixture
    def entorno(self, tmp_path, monkeypatch):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import vix_inverse_alert as va
        c = _curva([(1.03, 1.04), (1.02, 1.03), (0.97, 0.98)], "2024-08-01")
        c["front_ok"] = True
        curve = tmp_path / "curve.parquet"
        c.to_parquet(curve)
        monkeypatch.setattr(va, "CURVE", curve)
        monkeypatch.setattr(va, "STATE", tmp_path / "state.json")
        monkeypatch.setattr(sys, "argv", ["vix_inverse_alert.py"])
        return va, tmp_path / "state.json"

    def test_sin_smtp_registra_y_no_falla(self, entorno, monkeypatch):
        va, state = entorno
        for k in ("SMTP_USER", "SMTP_PASSWORD", "ALERT_TO"):
            monkeypatch.delenv(k, raising=False)
        assert va.main() == 0
        st = json.loads(state.read_text(encoding="utf-8"))
        assert st["eventos"][0]["tipo"] == "SALIDA" and not st["eventos"][0]["enviado"]
        assert st["correo"] == "sin configurar"

    def test_envia_una_sola_vez(self, entorno, monkeypatch):
        va, state = entorno
        monkeypatch.setenv("SMTP_USER", "x@example.com")
        monkeypatch.setenv("SMTP_PASSWORD", "app-pass")
        monkeypatch.setenv("ALERT_TO", "y@example.com")
        with patch.object(va, "enviar") as env:
            assert va.main() == 0
            assert va.main() == 0                                  # segunda pasada
        assert env.call_count == 1
        assert "SALIDA" in env.call_args[0][1]
        st = json.loads(state.read_text(encoding="utf-8"))
        assert st["eventos"][0]["enviado"] is True

    def test_fallo_smtp_hace_fallar_el_workflow(self, entorno, monkeypatch):
        va, state = entorno
        monkeypatch.setenv("SMTP_USER", "x@example.com")
        monkeypatch.setenv("SMTP_PASSWORD", "app-pass")
        monkeypatch.setenv("ALERT_TO", "y@example.com")
        with patch.object(va, "enviar", side_effect=OSError("smtp caído")):
            assert va.main() == 1
        st = json.loads(state.read_text(encoding="utf-8"))
        assert not st["eventos"][0]["enviado"] and "smtp caído" in st["eventos"][0]["motivo"]
