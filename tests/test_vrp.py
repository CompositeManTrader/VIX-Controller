"""Tests para el VRP tracker."""
import numpy as np
import pandas as pd
import pytest

from vix_controller.quant.vrp import compute_vrp_tracker


def _mk(n=600, vix_level=18.0, rv_level=14.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    vix = pd.Series(vix_level + rng.normal(0, 1.5, n).cumsum() * 0.05, index=idx).clip(9, 80)
    rv = pd.Series(rv_level + rng.normal(0, 1.2, n).cumsum() * 0.05, index=idx).clip(4, 90)
    return vix, rv


class TestVRPTracker:
    def test_basic_outputs(self):
        vix, rv = _mk()
        out = compute_vrp_tracker(vix, rv, window=252)
        assert out
        assert out["regime"] in ("COMPRIMIDA", "NORMAL", "RICA", "NEGATIVA")
        assert 0 <= out["percentile"] <= 100
        # consistencia: vrp_vol = vix - rv en el último punto
        assert out["vrp_vol"] == pytest.approx(
            float(vix.iloc[-1] - rv.iloc[-1]), abs=1e-9)
        # variance units = (vix² − rv²)/100
        assert out["vrp_var"] == pytest.approx(
            float((vix.iloc[-1] ** 2 - rv.iloc[-1] ** 2) / 100), abs=1e-9)

    def test_negative_vrp_flags_negativa(self):
        n = 400
        idx = pd.bdate_range("2023-01-02", periods=n)
        vix = pd.Series(np.full(n, 15.0), index=idx)
        rv = pd.Series(np.full(n, 14.0), index=idx)
        rv.iloc[-1] = 30.0   # RV explota por encima del VIX
        out = compute_vrp_tracker(vix, rv, window=252)
        assert out["regime"] == "NEGATIVA"
        assert out["vrp_vol"] < 0

    def test_insufficient_data_returns_empty(self):
        vix, rv = _mk(n=50)
        assert compute_vrp_tracker(vix, rv) == {}

    def test_none_inputs(self):
        assert compute_vrp_tracker(None, None) == {}

    def test_misaligned_indices_intersect(self):
        vix, rv = _mk(n=600)
        out = compute_vrp_tracker(vix.iloc[:500], rv.iloc[100:], window=252)
        # intersección = 400 filas > 120 mínimo → funciona
        assert out and len(out["df"]) == 400
