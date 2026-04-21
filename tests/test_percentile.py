"""
Tests para rolling_percentile.
"""
import numpy as np
import pandas as pd
import pytest

from vix_controller.quant.percentile import rolling_percentile


class TestRollingPercentile:
    def test_empty_series(self):
        assert rolling_percentile(pd.Series(dtype=float)).empty

    def test_constant_series_returns_100(self):
        s = pd.Series([5.0] * 300)
        p = rolling_percentile(s, window=252)
        # Todos iguales → percentil = 100% (todos <= al actual)
        valid = p.dropna()
        assert len(valid) > 0
        assert (valid == 100.0).all()

    def test_increasing_series(self):
        # Valor actual es el máximo siempre → percentil = 100
        s = pd.Series(np.arange(300, dtype=float))
        p = rolling_percentile(s, window=100)
        valid = p.dropna()
        assert (valid == 100.0).all()

    def test_nan_input_produces_nan_output(self):
        s = pd.Series([1.0, 2.0, np.nan, 4.0] + [5.0] * 100)
        p = rolling_percentile(s, window=10)
        # Los NaN en entrada se propagan
        assert np.isnan(p.iloc[2])

    def test_min_obs_respected(self):
        # Serie corta: deben salir todos NaN si no alcanza min_obs
        s = pd.Series([1.0] * 30)
        p = rolling_percentile(s, window=252)
        # min_obs = max(60, 252/3) = 84, serie de 30 → todos NaN
        assert p.isna().all()

    def test_custom_min_obs(self):
        s = pd.Series(np.arange(100, dtype=float))
        p = rolling_percentile(s, window=50, min_obs=10)
        assert p.notna().sum() > 0
