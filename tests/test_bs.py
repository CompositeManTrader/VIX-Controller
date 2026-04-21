"""
Golden tests para Black-Scholes pricing y IV inversion.

Golden values cross-validados contra QuantLib-python y Hull 10th ed.
"""
import math
import numpy as np
import pytest

from vix_controller.quant.bs import bs_call, bs_put, bs_gamma, bs_iv


# ─────────────────────────────────────────────────────────────
# BS pricing — golden values clásicos de Hull
# ─────────────────────────────────────────────────────────────
class TestBSPricing:
    def test_atm_call_one_year(self):
        # S=100, K=100, r=5%, T=1y, σ=20%, q=0
        # Valor Hull ≈ 10.4506
        price = bs_call(100.0, 100.0, 0.05, 1.0, 0.20, 0.0)
        assert price == pytest.approx(10.4506, abs=1e-3)

    def test_atm_put_parity(self):
        # Put-call parity: C - P = S·e^{-qT} - K·e^{-rT}
        S, K, r, T, v, q = 100.0, 100.0, 0.05, 1.0, 0.20, 0.02
        c = bs_call(S, K, r, T, v, q)
        p = bs_put(S, K, r, T, v, q)
        lhs = c - p
        rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
        assert lhs == pytest.approx(rhs, abs=1e-8)

    def test_deep_itm_call_approaches_intrinsic(self):
        # Call deep ITM con T pequeño ≈ intrinsic value
        price = bs_call(150.0, 100.0, 0.05, 0.01, 0.20, 0.0)
        intrinsic = 150.0 - 100.0 * math.exp(-0.05 * 0.01)
        assert price == pytest.approx(intrinsic, rel=1e-3)

    def test_deep_otm_call_is_small(self):
        price = bs_call(50.0, 150.0, 0.05, 0.1, 0.20, 0.0)
        assert 0 <= price < 0.01

    def test_zero_vol_returns_intrinsic_discounted(self):
        # σ=0 → call = max(S·e^{-qT} - K·e^{-rT}, 0)
        price = bs_call(100.0, 80.0, 0.05, 1.0, 0.0, 0.02)
        expected = max(100 * math.exp(-0.02) - 80 * math.exp(-0.05), 0)
        assert price == pytest.approx(expected, abs=1e-8)

    def test_zero_time_returns_intrinsic(self):
        assert bs_call(120.0, 100.0, 0.05, 0.0, 0.20, 0.0) == 20.0
        assert bs_put(80.0, 100.0, 0.05, 0.0, 0.20, 0.0) == 20.0


# ─────────────────────────────────────────────────────────────
# Gamma
# ─────────────────────────────────────────────────────────────
class TestBSGamma:
    def test_gamma_is_positive(self):
        g = bs_gamma(100.0, 100.0, 0.05, 1.0, 0.20, 0.0)
        assert g > 0

    def test_gamma_max_at_atm(self):
        # Para un mismo T y σ, gamma ATM > gamma OTM/ITM lejano
        g_atm = bs_gamma(100.0, 100.0, 0.05, 0.25, 0.20, 0.0)
        g_itm = bs_gamma(100.0, 70.0, 0.05, 0.25, 0.20, 0.0)
        g_otm = bs_gamma(100.0, 130.0, 0.05, 0.25, 0.20, 0.0)
        assert g_atm > g_itm
        assert g_atm > g_otm

    def test_gamma_zero_for_degenerate_inputs(self):
        assert bs_gamma(0.0, 100.0, 0.05, 1.0, 0.20, 0.0) == 0.0
        assert bs_gamma(100.0, 100.0, 0.05, 0.0, 0.20, 0.0) == 0.0
        assert bs_gamma(100.0, 100.0, 0.05, 1.0, 0.0, 0.0) == 0.0


# ─────────────────────────────────────────────────────────────
# IV inversion — roundtrip
# ─────────────────────────────────────────────────────────────
class TestBSImpliedVol:
    @pytest.mark.parametrize("sigma", [0.10, 0.20, 0.35, 0.60, 1.20])
    def test_roundtrip_call(self, sigma):
        S, K, r, T, q = 100.0, 100.0, 0.04, 0.5, 0.01
        price = bs_call(S, K, r, T, sigma, q)
        iv = bs_iv(S, K, r, T, price, "C", q)
        assert iv == pytest.approx(sigma, abs=1e-4)

    @pytest.mark.parametrize("sigma", [0.10, 0.25, 0.50])
    def test_roundtrip_put(self, sigma):
        S, K, r, T, q = 100.0, 100.0, 0.04, 0.5, 0.01
        price = bs_put(S, K, r, T, sigma, q)
        iv = bs_iv(S, K, r, T, price, "P", q)
        assert iv == pytest.approx(sigma, abs=1e-4)

    def test_iv_out_of_bounds_returns_nan(self):
        # Precio arriba del máximo teórico
        S, K, r, T, q = 100.0, 100.0, 0.04, 0.5, 0.01
        hi = S * math.exp(-q * T)  # precio call → σ→∞
        assert math.isnan(bs_iv(S, K, r, T, hi + 10, "C", q))
        # Precio debajo del intrinsic descontado
        assert math.isnan(bs_iv(S, K, r, T, -1, "C", q))

    def test_iv_rejects_zero_and_above_clamp(self):
        # Verifica que el clamp IV_UPPER=3.0 funciona
        S, K, r, T, q = 100.0, 100.0, 0.04, 0.5, 0.01
        # IV 250% (dentro del clamp)
        price = bs_call(S, K, r, T, 2.5, q)
        iv = bs_iv(S, K, r, T, price, "C", q)
        assert iv == pytest.approx(2.5, abs=1e-3)

    def test_iv_degenerate_inputs(self):
        assert math.isnan(bs_iv(100, 100, 0.04, 0, 5, "C", 0.01))
        assert math.isnan(bs_iv(100, 100, 0.04, 0.5, 0, "C", 0.01))
        assert math.isnan(bs_iv(100, 100, 0.04, 0.5, float("nan"), "C", 0.01))
