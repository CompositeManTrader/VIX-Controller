"""Tests para el HMM de régimen de volatilidad."""
import numpy as np
import pandas as pd
import pytest

from vix_controller.quant.regime import (
    GaussianHMM1D, fit_volatility_regime, REGIME_LABELS,
)


def _simulate_hmm(n=3000, seed=42):
    """Simula un HMM 3 estados con σ bien separadas (diario decimal)."""
    rng = np.random.default_rng(seed)
    sigmas = np.array([0.005, 0.013, 0.035])   # calm, transition, panic
    A = np.array([[0.98, 0.018, 0.002],
                  [0.03, 0.95, 0.02],
                  [0.01, 0.07, 0.92]])
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(3, p=A[states[t - 1]])
    x = rng.normal(0, sigmas[states])
    return x, states, sigmas


class TestGaussianHMM:
    def test_recovers_sigmas_sorted(self):
        x, states, true_sigmas = _simulate_hmm()
        hmm = GaussianHMM1D(n_states=3).fit(x)
        # σ ordenadas ascendente y razonablemente cerca de las reales
        assert np.all(np.diff(hmm.sigma) > 0)
        np.testing.assert_allclose(hmm.sigma, true_sigmas, rtol=0.35)

    def test_transmat_rows_sum_to_one(self):
        x, _, _ = _simulate_hmm()
        hmm = GaussianHMM1D(n_states=3).fit(x)
        np.testing.assert_allclose(hmm.A.sum(axis=1), 1.0, atol=1e-9)

    def test_filtered_probs_sum_to_one(self):
        x, _, _ = _simulate_hmm(n=1500)
        hmm = GaussianHMM1D(n_states=3).fit(x)
        probs = hmm.filtered_probs(x)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)

    def test_state_recovery_accuracy(self):
        x, states, _ = _simulate_hmm()
        hmm = GaussianHMM1D(n_states=3).fit(x)
        pred = hmm.filtered_probs(x).argmax(axis=1)
        acc = (pred == states).mean()
        assert acc > 0.70, f"accuracy {acc:.2f} — el HMM no separa estados"

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            GaussianHMM1D(n_states=3).fit(np.random.default_rng(0).normal(0, 0.01, 100))


class TestFitVolatilityRegime:
    def test_full_pipeline(self):
        x, _, _ = _simulate_hmm()
        r = pd.Series(x, index=pd.bdate_range("2014-01-01", periods=len(x)))
        out = fit_volatility_regime(r)
        assert out["state"] in REGIME_LABELS
        assert set(out["probs"].columns) == set(REGIME_LABELS)
        assert sum(out["current"].values()) == pytest.approx(1.0, abs=1e-6)
        assert len(out["sigmas_ann"]) == 3
        assert all(d >= 1 for d in out["durations"])

    def test_short_series_returns_empty(self):
        r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 200))
        assert fit_volatility_regime(r) == {}
