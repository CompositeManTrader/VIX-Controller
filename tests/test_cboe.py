"""Parsers de CBOE sobre respuestas reales guardadas (sin red)."""
from pathlib import Path

import pandas as pd
import pytest

from vix_controller.data import cboe

FX = Path(__file__).parent / "fixtures"


class TestFuturesQuotes:
    def test_only_monthly_sorted_and_alive(self):
        df = cboe.parse_futures_quotes((FX / "cboe_futures.json").read_text(encoding="utf-8"),
                                       today=pd.Timestamp("2026-10-06"))
        assert len(df) == 8
        assert df["Symbol"].str.match(r"^VX/[A-Z]\d+$").all()      # sin semanales
        assert df["Expiration"].is_monotonic_increasing
        assert (df["DTE"] >= 0).all()
        assert df.iloc[0]["Symbol"] == "VX/V6"

    def test_price_uses_last_when_traded(self):
        df = cboe.parse_futures_quotes((FX / "cboe_futures.json").read_text(encoding="utf-8"),
                                       today=pd.Timestamp("2026-10-06"))
        r = df.iloc[0]
        assert r["Price"] == r["Last"] == pytest.approx(17.03)

    def test_price_falls_back_to_settlement(self):
        payload = {"data": [{"symbol": "VX/X6", "expiration": "11/18/2026", "last_price": 0.0,
                             "settlement": 18.1393, "prev_settlement": 18.0, "change": None,
                             "volume": 0, "prev_open_int": 10, "high": 0, "low": 0}]}
        df = cboe.parse_futures_quotes(payload, today=pd.Timestamp("2026-10-06"))
        assert df.iloc[0]["Price"] == pytest.approx(18.1393)

    def test_bad_payload_raises(self):
        with pytest.raises(cboe.CboeError):
            cboe.parse_futures_quotes({"nope": []})


class TestSettlement:
    def test_monthly_rows_and_ratio(self):
        s = cboe.parse_settlement_csv((FX / "cboe_settlement_2026-10-05.csv").read_text(),
                                      "2026-10-05")
        assert s.iloc[0]["Symbol"] == "VX/V6" and s.iloc[1]["Symbol"] == "VX/X6"
        row = cboe.settlement_to_curve_row(s)
        assert row["m2"] / row["m1"] == pytest.approx(18.1393 / 17.4555)
        assert row["dias_m1"] == 16

    def test_contract_expiring_today_is_not_front(self):
        csv = ("Product,Symbol,Expiration Date,Price\n"
               "VX,VX/U6,2026-09-16,17.0\nVX,VX/V6,2026-10-21,18.5\n")
        s = cboe.parse_settlement_csv(csv, "2026-09-16")
        assert list(s["Symbol"]) == ["VX/V6"]


class TestIndex:
    def test_quote(self):
        q = cboe.parse_index_quote((FX / "cboe_vix_quote.json").read_text())
        assert q["price"] > 0 and q["prev_close"] > 0

    def test_history(self):
        h = cboe.parse_index_history((FX / "cboe_vix3m_history.csv").read_text(), "VIX3M")
        assert h.index.is_monotonic_increasing and len(h) == 30
        assert h.name == "VIX3M"
