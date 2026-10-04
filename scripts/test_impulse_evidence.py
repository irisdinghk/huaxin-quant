"""Causal anchor, price-volume direction and compatibility checks for impulse research."""

import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import quant_filter as quant
from scripts.impulse_evidence import analyze_impulse_evidence
from scripts.data.strategy_data_store import connect, load_document, save_quant


def frame(closes):
    return pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=len(closes)).strftime("%Y-%m-%d"),
        "close": closes, "open": closes,
        "high": [price + 1 for price in closes], "low": [price - 1 for price in closes],
        "volume": [100.0] * len(closes),
    })


def structure(df, start, end):
    return {
        "state": "VCP_FORMING", "structure_valid": True, "volume_pattern": "mixed",
        "prior_breakout_context_tag": "之前已有突破并强势整理", "structure_score": 79,
        "contraction_group": [{
            "start_idx": start, "end_idx": end,
            "start_date": str(df.iloc[start]["date"]), "end_date": str(df.iloc[end]["date"]),
            "close_pullback_pct": (df.iloc[end]["close"] / df.iloc[start]["close"] - 1) * 100,
            "confirmation_status": "CONFIRMED",
        }],
    }


class ImpulseEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = copy.deepcopy(quant.IMPULSE_EVIDENCE_CFG)

    def analyze(self, df, start, end):
        return analyze_impulse_evidence(df, structure(df, start, end), self.cfg)

    def test_earlier_high_cannot_use_a_later_low_as_its_base(self):
        df = frame([100.0] * 25 + [140, 138, 135, 90, 95, 105, 115, 110, 108])
        evidence = self.analyze(df, 31, 33)
        selected = evidence["selected"]
        self.assertEqual(selected["anchor"]["base_idx"], 28)
        self.assertEqual(selected["anchor"]["peak_idx"], 31)
        self.assertAlmostEqual(selected["price"]["gain_pct"], (115 / 90 - 1) * 100, places=5)
        self.assertLess(selected["strongest_unlinked_candidate"]["bridge_drawdown_pct"], -12)

    def test_platform_pause_does_not_require_peak_within_two_days(self):
        df = frame([100.0] * 25 + [105, 115, 130, 128, 126, 129, 127, 123, 122])
        selected = self.analyze(df, 31, 33)["selected"]
        self.assertEqual(selected["status"], "IDENTIFIED")
        self.assertEqual(selected["price"]["peak_lead_days"], 4)
        self.assertAlmostEqual(selected["price"]["gain_pct"], 30)
        self.assertEqual(selected["anchor"]["base_idx"], 24)

    def test_down_day_peak_does_not_become_up_day_volume_confirmation(self):
        df = frame([100.0] * 25 + [110, 125, 115, 130, 125, 120])
        df.loc[25:28, "volume"] = [110, 140, 600, 120]
        df.loc[27, ["open", "high", "low"]] = [118, 145, 112]
        selected = self.analyze(df, 28, 30)["selected"]
        volume = selected["volume"]
        self.assertEqual(selected["status"], "IDENTIFIED")
        self.assertEqual(volume["peak_volume_ratio"], 6)
        self.assertEqual(volume["up_peak_volume_ratio"], 1.4)
        self.assertFalse(volume["up_peak_volume_confirmed"])
        self.assertIn("PEAK_VOLUME_ON_DOWN_DAY", volume["warnings"])
        self.assertIn("PEAK_VOLUME_LONG_UPPER_SHADOW", volume["warnings"])
        self.assertLess(selected["price"]["path_efficiency"], 1)

    def test_anchors_and_worst_path_survive_recovery_without_mutation(self):
        df = frame([100.0] * 25 + [110, 120, 130, 115, 90])
        df.loc[27, "low"] = 50
        original = structure(df, 27, 29)
        snapshot = copy.deepcopy(original)
        original_df = df.copy(deep=True)
        before = analyze_impulse_evidence(df, original, self.cfg)
        extended = frame(list(df["close"]) + [125, 129, 145])
        extended.loc[27, "low"] = 50
        after = analyze_impulse_evidence(extended, original, self.cfg)
        self.assertEqual(before["selected"]["anchor"], after["selected"]["anchor"])
        old_path, new_path = (x["selected"]["retention"]["post_peak"] for x in [before, after])
        self.assertEqual(old_path["close_retention_pct"], new_path["close_retention_pct"])
        self.assertLess(new_path["close_retention_pct"], 0)
        self.assertEqual(new_path["intraday_low"], 89)
        self.assertGreater(after["selected"]["retention"]["current_close_retention_pct"], 100)
        self.assertEqual(original, snapshot)
        pd.testing.assert_frame_equal(df, original_df)

    def test_distinct_rounds_keep_the_same_base_and_peak(self):
        df = frame([100.0] * 25 + [110, 120, 130, 120, 110, 125, 118, 115, 122])
        value = structure(df, 27, 29)
        value["contraction_group"] += structure(df, 30, 32)["contraction_group"]
        selected = analyze_impulse_evidence(df, value, self.cfg)["selected"]
        rounds = selected["retention"]["rounds"]
        self.assertAlmostEqual(rounds[0]["close_retention_pct"], 100 / 3, places=5)
        self.assertAlmostEqual(rounds[1]["close_retention_pct"], 50, places=5)
        self.assertEqual(rounds[0]["confirmation_status"], "CONFIRMED")

    def test_window_sensitivity_is_explicit_and_not_a_score(self):
        self.cfg.update(lookback_days=10, diagnostic_lookback_days=[2, 10, 30])
        df = frame([100.0] * 25 + [101, 104, 108, 112, 118, 123, 128, 130, 125, 120])
        evidence = self.analyze(df, 32, 34)
        self.assertEqual(evidence["windows"]["2"]["status"], "WEAK_ADVANCE")
        self.assertEqual(evidence["windows"]["10"]["status"], "IDENTIFIED")
        self.assertIs(evidence["selected"], evidence["windows"]["10"])
        self.assertTrue(evidence["research_only"])
        self.assertNotIn("score", evidence)

    def test_boundary_and_incomplete_baseline_are_not_complete_launch_confirmation(self):
        self.cfg.update(lookback_days=3, diagnostic_lookback_days=[3])
        df = frame([100.0] * 3 + [105, 115, 130, 120])
        selected = self.analyze(df, 5, 6)["selected"]
        self.assertEqual(selected["status"], "IDENTIFIED")
        self.assertIn("BASE_AT_SCAN_BOUNDARY", selected["reasons"])
        self.assertIn("BASELINE_INCOMPLETE", selected["reasons"])
        self.assertIsNone(selected["volume"]["up_peak_volume_confirmed"])

    def test_flat_or_declining_prices_have_no_positive_advance(self):
        df = frame([100.0] * 25 + [99, 98, 97, 96])
        self.assertEqual(self.analyze(df, 27, 28)["status"], "NO_ADVANCE")

    def test_disconnected_old_advance_is_retained_as_unlinked_evidence(self):
        df = frame([100.0] * 25 + [110, 130, 125, 105, 95, 90, 85, 87])
        selected = self.analyze(df, 30, 32)["selected"]
        self.assertEqual(selected["status"], "UNLINKED")
        self.assertEqual(selected["linked_candidate_count"], 0)
        self.assertIn("PLATFORM_LINK_BROKEN", selected["reasons"])
        self.assertEqual(selected["price"]["gain_pct"], 30)

    def test_current_stage_volume_and_context_do_not_filter_historical_price_fact(self):
        df = frame([100.0] * 25 + [110, 130, 120, 115])
        value = structure(df, 26, 28)
        value.update(state="NONE", volume_pattern="failed", structure_valid=False)
        before = copy.deepcopy(value)
        self.assertEqual(analyze_impulse_evidence(df, value, self.cfg)["status"], "IDENTIFIED")
        self.assertEqual(value, before)

    def test_missing_volume_preserves_price_and_retention_evidence(self):
        df = frame([100.0] * 25 + [110, 130, 120, 115]).drop(columns="volume")
        selected = self.analyze(df, 26, 28)["selected"]
        self.assertEqual(selected["status"], "IDENTIFIED")
        self.assertIn("VOLUME_DATA_INCOMPLETE", selected["reasons"])
        self.assertIsNone(selected["volume"]["up_peak_volume_confirmed"])
        self.assertIsNotNone(selected["retention"]["post_peak"]["close_low"])

    def test_missing_intraday_fields_do_not_break_close_based_research(self):
        df = frame([100.0] * 25 + [110, 130, 120, 115])
        df["high"] = None
        df["low"] = None
        selected = self.analyze(df, 26, 28)["selected"]
        self.assertEqual(selected["status"], "IDENTIFIED")
        self.assertIn("INTRADAY_DATA_INCOMPLETE", selected["reasons"])
        self.assertIsNone(selected["retention"]["post_peak"]["intraday_low"])
        self.assertIsNotNone(selected["retention"]["post_peak"]["close_low"])

    def test_invalid_dates_prices_and_group_anchors_are_explicit(self):
        df = frame([100.0] * 25 + [110, 130, 120, 115])
        value = structure(df, 26, 28)
        bad = copy.deepcopy(value)
        bad["contraction_group"][0]["start_idx"] = 25
        self.assertEqual(analyze_impulse_evidence(df, bad, self.cfg)["status"], "GROUP_ANCHOR_MISMATCH")
        bad = copy.deepcopy(value)
        bad["contraction_group"][0]["end_date"] = "2027-01-01"
        self.assertEqual(analyze_impulse_evidence(df, bad, self.cfg)["status"], "GROUP_ANCHOR_MISMATCH")
        df.loc[27, "close"] = 0
        self.assertEqual(analyze_impulse_evidence(df, value, self.cfg)["status"], "DATA_ISSUE")
        df.loc[27, "close"] = 120
        df.loc[27, "date"] = df.loc[26, "date"]
        self.assertEqual(analyze_impulse_evidence(df, value, self.cfg)["status"], "DATA_ISSUE")

    def test_research_output_does_not_change_any_screen_decision(self):
        df = quant.calc_indicators(frame([100.0] * 125 + [110, 120, 130, 125, 120, 115, 120, 128, 124, 120, 123]))
        enabled = quant.screen(df)
        with patch.dict(quant.IMPULSE_EVIDENCE_CFG, {"enabled": False}):
            disabled = quant.screen(df)
        self.assertEqual(disabled.pop("impulse_evidence")["status"], "DISABLED")
        enabled.pop("impulse_evidence")
        self.assertEqual(enabled, disabled)

    def test_csv_preserves_new_object_and_reads_records_without_it(self):
        df = quant.calc_indicators(frame([100.0] * 125 + [110, 120, 130, 125, 120]))
        result = quant.result_from_df("000001", "fixture", df, "2026-06-30")
        old = dict(result)
        old.pop("impulse_evidence")
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "quant.csv")
            quant.write_csv([result, old], path)
            with open(path, encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertTrue(all(None not in row for row in rows))
        self.assertEqual(json.loads(rows[0]["impulse_evidence"]), result["impulse_evidence"])
        self.assertEqual(json.loads(rows[1]["impulse_evidence"]), {})

    def test_strategy_store_preserves_the_full_research_object(self):
        df = frame([100.0] * 25 + [110, 130, 120, 115])
        evidence = self.analyze(df, 26, 28)
        payload = {"meta": {"run_date": "2026-09-15", "strategy_version": "model2_quant_v33"},
                   "results": [{"code": "000001", "name": "fixture", "structure_stage": "VCP_FORMING",
                                "structure_score": 79, "impulse_evidence": evidence}]}
        with tempfile.TemporaryDirectory() as directory:
            conn = connect(Path(directory) / "strategy.sqlite")
            try:
                save_quant(conn, payload)
                restored = load_document(conn, "quant", "2026-09-15")
                snapshot = json.loads(conn.execute("SELECT payload_json FROM vcp_structure_snapshots").fetchone()[0])
            finally:
                conn.close()
        self.assertEqual(restored, payload)
        self.assertEqual(snapshot["impulse_evidence"], evidence)


if __name__ == "__main__":
    unittest.main()
