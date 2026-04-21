"""
Tests SVI: verifica fit en datos sintéticos y constraint de no-arbitraje.
"""
import numpy as np
import pytest

from vix_controller.quant.svi import fit_svi_slice, svi_raw_w


def _synthetic_smile(F=100.0, T=30/365, a=None, b=0.08, rho=-0.4,
                    m=-0.02, sigma=0.12, n=25, seed=0):
    """Genera smile sintético desde parámetros SVI conocidos.

    `a` default = 0.04 * T (ATM variance ~ 20% vol para cualquier T).
    """
    if a is None:
        a = 0.04 * T
    rng = np.random.default_rng(seed)
    k = np.linspace(-0.15, 0.15, n)
    strikes = F * np.exp(k)
    w = svi_raw_w(k, a, b, rho, m, sigma)
    iv_true = np.sqrt(w / T)
    iv_noisy = iv_true * (1 + 0.005 * rng.standard_normal(n))  # 0.5% noise
    return strikes, iv_noisy, iv_true


class TestSVIFit:
    def test_fit_converges_on_clean_data(self):
        F, T = 100.0, 30 / 365
        strikes, ivs, iv_true = _synthetic_smile(F=F, T=T)
        fit = fit_svi_slice(strikes, ivs, F, T=T)
        assert fit is not None
        # IV fit ≈ IV true ± 1%
        np.testing.assert_allclose(fit["iv_fit"], iv_true, atol=0.01)
        assert fit["r2"] > 0.9

    def test_fit_butterfly_constraint(self):
        # Durrleman: b·(1+|ρ|) <= 2
        F, T = 100.0, 60 / 365
        strikes, ivs, _ = _synthetic_smile(F=F, T=T)
        fit = fit_svi_slice(strikes, ivs, F, T=T)
        assert fit is not None
        assert fit["b"] * (1 + abs(fit["rho"])) <= 2 + 1e-6

    def test_fit_returns_none_on_insufficient_data(self):
        strikes = np.array([95, 100, 105], dtype=float)
        ivs = np.array([0.22, 0.20, 0.22], dtype=float)
        assert fit_svi_slice(strikes, ivs, 100.0, T=0.1) is None

    def test_fit_returns_none_on_zero_T(self):
        strikes = np.linspace(90, 110, 20)
        ivs = np.full(20, 0.20)
        assert fit_svi_slice(strikes, ivs, 100.0, T=0.0) is None

    def test_different_T_give_same_IV_shape(self):
        """
        Fix B3 verificación: un slice con T real corto y otro largo para el
        mismo nivel de IV deben producir distintos `a` pero fits comparables.
        """
        F = 100.0
        for T in [7 / 365, 90 / 365]:
            strikes, ivs, iv_true = _synthetic_smile(F=F, T=T)
            fit = fit_svi_slice(strikes, ivs, F, T=T)
            assert fit is not None, f"Failed fit for T={T}"
            np.testing.assert_allclose(fit["iv_fit"], iv_true, atol=0.01)
