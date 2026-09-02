"""
Tests para la señal BB × Contango — incluye verificación de no look-ahead.
"""
import numpy as np
import pandas as pd
import pytest

from vix_controller.quant.signals import bb_contango_signal, bb_position_state


def _reference_state(px, sma, upper):
    """Máquina de estados original con .iloc, conservada como oráculo."""
    out, pos = [], 0
    for p, s, u in zip(px, sma, upper):
        if np.isnan(p) or np.isnan(s) or np.isnan(u):
            out.append(pos); continue
        if pos == 0 and p < s:
            pos = 1
        elif pos == 1 and p > u:
            pos = 0
        out.append(pos)
    return np.array(out)


class TestStateMachineParity:
    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_numpy_matches_reference(self, seed):
        rng = np.random.default_rng(seed)
        n = 2000
        px = 30 + np.cumsum(rng.normal(0, 0.8, n))
        px[rng.random(n) < 0.02] = np.nan
        s = pd.Series(px)
        sma = s.rolling(20).mean().to_numpy()
        upper = (sma + 2 * s.rolling(20).std().to_numpy())
        np.testing.assert_array_equal(bb_position_state(px, sma, upper),
                                      _reference_state(px, sma, upper))


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 200
    # Serie mean-reverting
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.95 * x[i - 1] + rng.standard_normal()
    vxx = pd.Series(30 + x, index=pd.date_range("2023-01-01", periods=n))
    # Contango la mayor parte del tiempo
    ct = pd.Series(rng.random(n) > 0.3, index=vxx.index)
    return vxx, ct


class TestBBContango:
    def test_output_shape(self, sample_data):
        vxx, ct = sample_data
        out = bb_contango_signal(vxx, ct)
        expected_cols = {"sma", "bb_upper", "sig_raw", "sig_bb", "ct_filter", "sig_final"}
        assert set(out.columns) == expected_cols
        assert len(out) == len(vxx)

    def test_sig_final_is_binary(self, sample_data):
        vxx, ct = sample_data
        out = bb_contango_signal(vxx, ct)
        assert set(out["sig_final"].unique()).issubset({0, 1})

    def test_no_signal_without_contango(self, sample_data):
        vxx, ct = sample_data
        ct_zero = pd.Series(False, index=ct.index)
        out = bb_contango_signal(vxx, ct_zero)
        assert (out["sig_final"] == 0).all()

    def test_no_lookahead(self, sample_data):
        """
        Verifica shift(1): sig_bb[t] == sig_raw[t-1]. Un cambio en
        vxx[t] NO debe afectar sig_bb[t] (solo puede afectar sig_bb[t+1]).
        """
        vxx, ct = sample_data
        out1 = bb_contango_signal(vxx, ct)
        vxx_modified = vxx.copy()
        vxx_modified.iloc[100] = 999.0  # shock en t=100
        out2 = bb_contango_signal(vxx_modified, ct)
        # sig_bb hasta t=100 debe ser idéntico (no ve el shock)
        pd.testing.assert_series_equal(
            out1["sig_bb"].iloc[:101], out2["sig_bb"].iloc[:101],
            check_names=False,
        )
