"""Regression tests for decoded zero activity entering trading windows."""

import sqlite3
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from tdxpy.helper import get_volume

from scripts.data.market_data import TDX_DECODED_ZERO, TDXSource, normalize_tdx_decoded_zeros
from scripts.data.market_data_service import MarketDataService
from scripts.data.market_data_store import create_schema


def bars(volume_column="volume"):
    return pd.DataFrame({
        "datetime": ["2026-09-21", "2026-09-22", "2026-09-23"],
        "open": [18.0, 18.39, 18.4], "high": [18.5, 18.39, 18.8],
        "low": [17.9, 18.39, 18.3], "close": [18.39, 18.39, 18.7],
        volume_column: [100.0, get_volume(0), 0.01],
        "amount": [184000.0, get_volume(0), get_volume(0)],
    })


class TDXZeroActivityTests(unittest.TestCase):
    def test_actual_decoder_zero_is_the_observed_sentinel(self):
        self.assertEqual(get_volume(0), TDX_DECODED_ZERO)

    def test_source_frame_and_small_real_quantity_are_preserved(self):
        raw = bars()
        before = raw.copy(deep=True)
        fixed = normalize_tdx_decoded_zeros(raw)
        pd.testing.assert_frame_equal(raw, before)
        self.assertEqual(fixed.loc[1, "volume"], 0)
        self.assertEqual(fixed.loc[2, "volume"], .01)
        self.assertEqual(fixed.loc[2, "amount"], 0)
        pd.testing.assert_frame_equal(raw[["open", "high", "low", "close"]], fixed[["open", "high", "low", "close"]])

    def test_shared_ingestion_excludes_zero_activity_for_both_volume_names(self):
        for column in ("vol", "volume"):
            with self.subTest(column=column):
                rows = MarketDataService.normalized_bars(bars(column), "002860", "2026-09-23")
                self.assertEqual([row[1] for row in rows], ["2026-09-21", "2026-09-23"])
                self.assertEqual(rows[-1][6:9], (.01, 0.0, "tdx"))

    def test_fallback_does_not_use_tdx_decoder_rules(self):
        rows = MarketDataService.normalized_bars(bars(), "002860", "2026-09-23", source="miaoxiang")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1][6], TDX_DECODED_ZERO)
        self.assertEqual(rows[1][8], "miaoxiang")

    def test_legacy_adapter_also_excludes_zero_activity(self):
        source = TDXSource()
        source._client = Mock()
        source._client.bars.return_value = bars()
        result, error = source.fetch_bars("002860", "星帅尔")
        self.assertIsNone(error)
        self.assertEqual(result["date"].tolist(), ["2026-09-21", "2026-09-23"])

    def test_zero_activity_does_not_satisfy_target_date_coverage(self):
        frame = bars().iloc[:2]
        rows = MarketDataService.normalized_bars(frame, "002860", "2026-09-22")
        self.assertEqual(rows[-1][1], "2026-09-21")

    def test_failed_repair_does_not_publish_previous_day_as_target(self):
        conn = sqlite3.connect(":memory:")
        create_schema(conn)
        conn.executemany("INSERT INTO daily_bars VALUES(?,?,?,?,?,?,?,?,?,?)", [
            ("002860", "2026-09-21", 18, 19, 17, 18, 100, 180000, "tdx", "test"),
            ("000001", "2026-09-22", 10, 11, 9, 10, 100, 100000, "tdx", "test"),
        ])
        service = MarketDataService({"data": {}})
        with patch("scripts.data.market_data_service.connect_db", return_value=conn), patch.object(
            service, "ingest_bars", return_value={"failed_codes": {"002860": "115 无数据"}, "repaired_codes": []}
        ):
            frames, status = service.get_daily_bars([("002860", "星帅尔"), ("000001", "平安银行")], "2026-09-22", 1)
        self.assertNotIn("002860", frames)
        self.assertIn("000001", frames)
        self.assertEqual(status["002860"]["error"], "115 无数据")
        self.assertEqual(status["002860"]["last_trade_date"], "2026-09-21")

    def test_failed_force_refresh_keeps_existing_valid_target(self):
        conn = sqlite3.connect(":memory:")
        create_schema(conn)
        conn.execute("INSERT INTO daily_bars VALUES(?,?,?,?,?,?,?,?,?,?)", ("000001", "2026-09-22", 10, 11, 9, 10, 100, 100000, "tdx", "test"))
        service = MarketDataService({"data": {}})
        with patch("scripts.data.market_data_service.connect_db", return_value=conn), patch.object(
            service, "ingest_bars", return_value={"failed_codes": {"000001": "112 限流"}, "repaired_codes": []}
        ):
            frames, status = service.get_daily_bars([("000001", "平安银行")], "2026-09-22", 1, force_refresh=True)
        self.assertIn("000001", frames)
        self.assertTrue(status["000001"]["retryable"])


if __name__ == "__main__":
    unittest.main()
