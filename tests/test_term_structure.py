"""Tests de la señal de curva (term_structure) y de su réplica histórica."""
import numpy as np
import pandas as pd
import pytest

from vix_controller.quant import term_structure as ts


class TestCurveSignal:
    def test_inverted_contango_is_long_vol(self):
        r = ts.curve_signal(front_ct=-2.0, spot_m1=3.0, ct_pctile=50, vix=18)
        assert r["signal"] == ts.SIGNAL_LONG and r["code"] == -1 and r["inverted"]

    def test_negative_basis_alone_is_spot_spike_not_long_vol(self):
        # Spot > M1 con la curva en contango: revierte (medido), no es inversión.
        r = ts.curve_signal(front_ct=1.0, spot_m1=-0.5, ct_pctile=50, vix=18)
        assert r["signal"] == ts.SIGNAL_NEUTRAL
        assert "spike" in r["desc"]
        assert not r["inverted"]

    def test_negative_basis_blocks_short_vol(self):
        r = ts.curve_signal(front_ct=6.0, spot_m1=-0.5, ct_pctile=60, vix=16)
        assert r["signal"] != ts.SIGNAL_SHORT

    def test_full_checklist_is_short_vol(self):
        r = ts.curve_signal(front_ct=8.0, spot_m1=5.0, ct_pctile=60, vix=16)
        assert r["signal"] == ts.SIGNAL_SHORT and r["n_ok"] == 4 and r["code"] == 1

    def test_high_vix_blocks_short_vol(self):
        r = ts.curve_signal(front_ct=8.0, spot_m1=5.0, ct_pctile=60, vix=30)
        assert r["signal"] == ts.SIGNAL_NEUTRAL

    def test_compressed_contango_is_neutral_with_specific_desc(self):
        r = ts.curve_signal(front_ct=0.5, spot_m1=1.0, ct_pctile=10, vix=15)
        assert r["signal"] == ts.SIGNAL_NEUTRAL
        assert "comprimido" in r["desc"]

    def test_needs_at_least_three_checks_for_short(self):
        # Solo dos checks disponibles (sin percentil ni VIX) → no puede ser SHORT
        r = ts.curve_signal(front_ct=8.0, spot_m1=5.0, ct_pctile=None, vix=None)
        assert r["signal"] == ts.SIGNAL_NEUTRAL and len(r["checks"]) == 2

    def test_negative_basis_near_expiry_is_convergence_not_signal(self):
        # A 2 días del vencimiento M1 < VIX es mecánico → no es LONG VOL
        r = ts.curve_signal(front_ct=6.0, spot_m1=-1.0, ct_pctile=55, vix=17, dte1=2)
        assert r["signal"] == ts.SIGNAL_SHORT          # la base se ignora: checklist 3/3
        assert all(name != "Prima M1 sobre VIX (basis > 0)" for name, _, _ in r["checks"])
        # Con DTE amplio la misma base negativa sí cuenta (y bloquea el short)
        r2 = ts.curve_signal(front_ct=6.0, spot_m1=-1.0, ct_pctile=55, vix=17, dte1=20)
        assert r2["signal"] == ts.SIGNAL_NEUTRAL and "spike" in r2["desc"]

    def test_pct_change(self):
        assert ts.pct_change(20.0, 22.0) == pytest.approx(10.0)
        assert ts.pct_change(0.0, 22.0) is None
        assert ts.pct_change(None, 22.0) is None


def _curve(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    vix = pd.Series(16 + rng.normal(0, 0.4, n).cumsum() * 0.2, index=idx).clip(10, 60)
    m1 = vix * 1.04
    m2 = m1 * 1.05
    return pd.DataFrame({"m1": m1, "m2": m2, "VIX": vix,
                         "dias_m1": 15.0, "dias_m2": 45.0})


class TestSignalHistory:
    def test_columns_and_values(self):
        h = ts.signal_history(_curve())
        assert {"contango_pct", "basis_pct", "ct_pctile", "vix", "roll_ann",
                "signal", "code"} <= set(h.columns)
        # contango constante 5% → roll_ann = 5 * 365/30
        assert h["contango_pct"].iloc[-1] == pytest.approx(5.0)
        assert h["roll_ann"].iloc[-1] == pytest.approx(5.0 * 365 / 30)

    def test_history_signal_matches_pointwise_rule(self):
        h = ts.signal_history(_curve())
        last = h.iloc[-1]
        r = ts.curve_signal(last["contango_pct"], last["basis_pct"],
                            last["ct_pctile"], last["vix"])
        assert last["signal"] == r["signal"]

    def test_percentile_is_trailing_only(self):
        """Cambiar el futuro no puede alterar la señal del pasado."""
        c = _curve()
        h1 = ts.signal_history(c)
        c2 = c.copy()
        c2.iloc[-50:, c2.columns.get_loc("m2")] *= 0.8      # backwardation al final
        h2 = ts.signal_history(c2)
        pd.testing.assert_series_equal(h1["signal"].iloc[:-50], h2["signal"].iloc[:-50])

    def test_inversion_shows_as_long_vol(self):
        c = _curve()
        c.iloc[-1, c.columns.get_loc("m2")] = c["m1"].iloc[-1] * 0.95
        h = ts.signal_history(c)
        assert h["signal"].iloc[-1] == ts.SIGNAL_LONG

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError, match="faltan columnas"):
            ts.signal_history(pd.DataFrame({"m1": [1.0]}))
