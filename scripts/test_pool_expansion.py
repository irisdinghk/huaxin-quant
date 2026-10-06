import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.data.pool_expansion import build_expansion_pool
from scripts.data.pool_tracking import STRUCTURE, has_active_structure, prepare_tracking_rows


class PoolExpansionTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "market.sqlite"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.executescript(
            """
            CREATE TABLE securities (
                code TEXT PRIMARY KEY, name TEXT, is_st INTEGER
            );
            CREATE TABLE universe_members (
                trade_date TEXT, code TEXT, eligible INTEGER, exclusion_reason TEXT
            );
            CREATE TABLE daily_bars (
                code TEXT, trade_date TEXT, close REAL, amount REAL
            );
            CREATE TABLE stock_industries (
                snapshot_date TEXT, code TEXT, sw_industry_code TEXT
            );
            CREATE TABLE industry_definitions (
                snapshot_date TEXT, industry_system TEXT,
                industry_code TEXT, industry_name TEXT
            );
            """
        )
        self.dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-01-05", periods=15)]
        self.as_of = self.dates[-1]
        specs = {
            "000001": (1.00, 1.90, 1, 0, 1000.0),
            "000002": (1.00, 1.70, 0, 1, 1000.0),
            "000003": (1.00, 1.50, 0, 1, 10.0),
            "000004": (1.00, 1.30, 0, 1, 1000.0),
        }
        for code, (start, end, is_st, eligible, amount) in specs.items():
            self.conn.execute("INSERT INTO securities VALUES(?,?,?)", (code, f"N{code}", is_st))
            self.conn.execute(
                "INSERT INTO universe_members VALUES(?,?,?,?)",
                (self.as_of, code, eligible, "ST" if not eligible else ""),
            )
            for idx, trade_date in enumerate(self.dates):
                close = start + (end - start) * idx / (len(self.dates) - 1)
                self.conn.execute(
                    "INSERT INTO daily_bars VALUES(?,?,?,?)",
                    (code, trade_date, close, amount),
                )
        self.conn.commit()
        self.conn.close()

    def tearDown(self):
        self.tempdir.cleanup()

    def config(self, top_n):
        return {
            "enabled": True,
            "top_n_before_filters": top_n,
            "return_windows": {"short": 5, "long": 10},
            "weights": {"rs_short_percentile": 0.6, "rs_long_percentile": 0.4},
            "hard_filters": {
                "require_universe_eligible": True,
                "minimum_history_sessions": 11,
                "minimum_active_sessions_20": 5,
                "minimum_average_amount_20": 100.0,
            },
            "unverified_fundamental_tag": "FUNDAMENTAL_UNVERIFIED",
        }

    def test_filters_after_top_n_without_backfill(self):
        rows, summary = build_expansion_pool(
            self.as_of,
            self.config(3),
            core_codes={"000002"},
            db_path=self.db_path,
        )
        self.assertEqual([row["code"] for row in rows], ["000002"])
        self.assertEqual(summary["initial_top_total"], 3)
        self.assertEqual(summary["core_overlap_total"], 1)
        self.assertEqual(summary["expansion_only_total"], 0)
        self.assertEqual(summary["rejected"], {"LOW_LIQUIDITY": 1, "ST": 1})
        self.assertNotIn("000004", {row["code"] for row in rows})

    def test_channel_and_fundamental_status(self):
        rows, summary = build_expansion_pool(
            self.as_of,
            self.config(4),
            core_codes={"000002"},
            db_path=self.db_path,
        )
        by_code = {row["code"]: row for row in rows}
        self.assertEqual(by_code["000002"]["pool_channel"], "BOTH")
        self.assertEqual(by_code["000002"]["fundamental_status"], "CORE_VERIFIED")
        self.assertEqual(by_code["000004"]["pool_channel"], "EXPANSION_RS")
        self.assertEqual(by_code["000004"]["fundamental_status"], "FUNDAMENTAL_UNVERIFIED")
        self.assertEqual(summary["expansion_only_total"], 1)

    def test_tracked_stock_outside_top_n_keeps_metrics(self):
        rows, summary = build_expansion_pool(
            self.as_of,
            self.config(2),
            tracked_codes={"000004"},
            db_path=self.db_path,
        )
        by_code = {row["code"]: row for row in rows}
        self.assertTrue(by_code["000002"]["rs_current_eligible"])
        self.assertFalse(by_code["000004"]["rs_current_eligible"])
        self.assertEqual(by_code["000004"]["hard_filter_reason"], "")
        self.assertEqual(summary["initial_top_total"], 2)
        self.assertEqual(summary["tracked_metrics_total"], 1)

    def test_low_amount_structure_stays_in_pool_without_new_rs_eligibility(self):
        prior = {"000003": {
            "tracking_status": STRUCTURE, "quant_stage": "VCP_FORMING",
            "bloom_status": "FORMING", "first_seen_date": self.dates[0],
        }}
        exempt_codes = {code for code, state in prior.items() if has_active_structure(state)}
        for top_n in [2, 3]:
            with self.subTest(top_n=top_n):
                rows, summary = build_expansion_pool(
                    self.as_of, self.config(top_n), tracked_codes=set(prior),
                    db_path=self.db_path, liquidity_exempt_codes=exempt_codes,
                )
                row = next(row for row in rows if row["code"] == "000003")
                self.assertEqual(row["average_amount_20"], 10.0)
                self.assertEqual(row["hard_filter_reason"], "")
                self.assertTrue(row["liquidity_filter_exempt"])
                self.assertFalse(row["rs_current_eligible"])
                self.assertEqual(summary["liquidity_exempt_total"], 1)
                pending = prepare_tracking_rows(self.as_of, [row], prior, self.dates, 20)[0]
                self.assertEqual(pending["tracking_status"], STRUCTURE)
                self.assertTrue(pending["liquidity_filter_exempt"])

    def test_first_discovery_cannot_claim_structure_liquidity_exemption(self):
        rows, summary = build_expansion_pool(
            self.as_of, self.config(3), db_path=self.db_path,
            liquidity_exempt_codes={"000003"},
        )
        self.assertNotIn("000003", {row["code"] for row in rows})
        self.assertEqual(summary["rejected"]["LOW_LIQUIDITY"], 1)
        self.assertEqual(summary["liquidity_exempt_total"], 0)

    def test_structure_exemption_does_not_override_other_hard_filters(self):
        rows, _ = build_expansion_pool(
            self.as_of, self.config(3), tracked_codes={"000001"},
            db_path=self.db_path, liquidity_exempt_codes={"000001"},
        )
        st = next(row for row in rows if row["code"] == "000001")
        self.assertEqual(st["hard_filter_reason"], "ST")
        self.assertFalse(st["liquidity_filter_exempt"])

        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM daily_bars WHERE code='000003' AND trade_date=?", (self.as_of,))
        conn.commit()
        conn.close()
        rows, _ = build_expansion_pool(
            self.as_of, self.config(3), tracked_codes={"000003"},
            db_path=self.db_path, liquidity_exempt_codes={"000003"},
        )
        missing = next(row for row in rows if row["code"] == "000003")
        self.assertEqual(missing["hard_filter_reason"], "TARGET_DATE_MISSING")
        self.assertFalse(missing["liquidity_filter_exempt"])


if __name__ == "__main__":
    unittest.main()
