"""Tests para las señales de stress cross-asset."""
import numpy as np
import pandas as pd

from vix_controller.quant.cross_asset import compute_stress_signals


def _series(vals, start="2023-01-02"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)))


def _flat_then_spike(n=500, base=100.0, spike=1.5):
    """Serie plana con ruido leve que termina en spike (stress al final)."""
    rng = np.random.default_rng(1)
    vals = base + rng.normal(0, 0.5, n).cumsum() * 0.01
    vals[-15:] = vals[-15] * np.linspace(1, spike, 15)
    return _series(vals)


class TestStressSignals:
    def test_move_spike_goes_red(self):
        out = compute_stress_signals(move=_flat_then_spike())
        sig = next(s for s in out["signals"] if s["key"] == "move")
        assert sig["light"] == "red"
        assert out["light"] == "red"          # composite = max

    def test_credit_crash_goes_red(self):
        n = 500
        hyg = _flat_then_spike(n, base=80, spike=0.80)   # HYG cae 20%
        lqd = _series(np.full(n, 110.0))
        out = compute_stress_signals(hyg=hyg, lqd=lqd)
        sig = next(s for s in out["signals"] if s["key"] == "credit")
        assert sig["light"] == "red"
        assert "LQD" in sig["name"]

    def test_credit_falls_back_to_ief(self):
        n = 500
        hyg = _series(np.full(n, 80.0))
        ief = _series(np.full(n, 95.0))
        out = compute_stress_signals(hyg=hyg, ief=ief)
        sig = next(s for s in out["signals"] if s["key"] == "credit")
        assert "IEF" in sig["name"]

    def test_calm_market_green(self):
        rng = np.random.default_rng(3)
        n = 500
        move = _series(90 + rng.normal(0, 0.3, n))
        out = compute_stress_signals(move=move)
        sig = next(s for s in out["signals"] if s["key"] == "move")
        assert sig["light"] in ("green", "yellow")   # sin spike no hay rojo

    def test_missing_everything_returns_empty(self):
        assert compute_stress_signals() == {}

    def test_short_series_skipped(self):
        out = compute_stress_signals(move=_series(np.full(50, 100.0)))
        assert out == {}

    def test_composite_is_max(self):
        n = 500
        move = _flat_then_spike(n)                      # rojo
        rng = np.random.default_rng(5)
        dxy = _series(100 + rng.normal(0, 0.05, n))     # tranquilo
        out = compute_stress_signals(move=move, dxy=dxy)
        assert out["composite"] == max(s["stress_pct"] for s in out["signals"])
