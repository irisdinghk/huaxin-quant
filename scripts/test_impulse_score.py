"""Scoring contract tests for retention scope, incomplete data and isolated propagation."""

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import quant_filter as quant
from scripts.audit_impulse_score import rank, trial_module, main
from scripts.impulse_score import calibrate, impulse_metrics, score_impulse, score_structure, setup_conversion, validate_config

CFG = json.loads((ROOT / "strategies/02-quant-trial.json").read_text(encoding="utf-8"))
FROZEN = {"mappings": {"gain_pct": {"floor": 0, "cap": 50},
                       "up_volume_ratio": {"floor": 1, "cap": 2},
                       "up_down_volume_ratio": {"floor": 1, "cap": 2},
                       "speed_pct": {"floor": 0, "cap": 10}}}


def evidence():
    return {"status": "IDENTIFIED", "as_of": "2026-09-15", "reasons": [],
            "selected": {"anchor": {"base_close": 100, "peak_close": 125,
                                    "base_date": "2026-08-03", "peak_date": "2026-08-10", "anchor_id": "a"},
                         "price": {"advance_days": 5, "path_efficiency": .8},
                         "volume": {"baseline_days": 20, "up_mean_volume_ratio": 2,
                                    "down_mean_volume_ratio": 1, "down_days": 2, "warnings": []},
                         "retention": {"rounds": [{"close_low": 120, "close_low_date": "2026-08-20"}],
                                       "post_peak": {"close_low": 90}, "current_close_retention_pct": 150}}}


class ImpulseScoreTests(unittest.TestCase):
    def test_retention_uses_all_current_rounds_not_bridge_or_current_recovery(self):
        item = evidence()
        item["selected"]["retention"]["rounds"].append({"close_low": 110, "close_low_date": "2026-09-10"})
        item["selected"]["episodes"] = [{"retention": {"rounds": [{"close_low": 50}]}}]
        result = impulse_metrics(item, CFG)
        self.assertAlmostEqual(result["retention"], .4)
        self.assertEqual(result["retention_low_date"], "2026-09-10")

    def test_quality_and_retention_multiply_linearly(self):
        original = evidence()
        low = copy.deepcopy(original)
        low["selected"]["retention"]["rounds"][0]["close_low"] = 110
        first, second = (score_impulse(x, CFG, FROZEN) for x in [original, low])
        self.assertEqual(first["quality"], second["quality"])
        self.assertAlmostEqual(first["contribution"], second["contribution"] * 2)

    def test_negative_and_over_one_retention_only_clamp_contribution(self):
        item = evidence()
        for price, retained in [(90, 0), (100, 0), (140, 1)]:
            item["selected"]["retention"]["rounds"][0]["close_low"] = price
            result = score_impulse(item, CFG, FROZEN)
            self.assertEqual(result["retention"], retained)
            self.assertAlmostEqual(result["retention_raw"], (price - 100) / 25)
            self.assertAlmostEqual(result["contribution"], 30 * result["quality"] / 100 * retained)

    def test_no_down_days_reallocates_volume_without_infinite_ratio(self):
        item = evidence()
        item["selected"]["volume"].update(down_days=0, down_mean_volume_ratio=None)
        result = score_impulse(item, CFG, FROZEN)
        self.assertIsNone(result["metrics"]["up_down_volume_ratio"])
        self.assertEqual(result["quality_components"]["up_volume_ratio"], 40)
        self.assertEqual(result["quality_components"]["up_down_volume_ratio"], 0)

    def test_weak_advance_not_filtered_by_twelve_percent_label(self):
        item = evidence()
        item["status"] = "WEAK_ADVANCE"
        item["selected"]["anchor"]["peak_close"] = 110
        item["selected"]["retention"]["rounds"][0]["close_low"] = 108
        self.assertGreater(score_impulse(item, CFG, FROZEN)["contribution"], 0)

    def test_volume_missing_and_baseline_short_produce_null_total(self):
        for key, value in [("baseline_days", 19), ("up_mean_volume_ratio", None), ("down_mean_volume_ratio", 0)]:
            item = evidence()
            item["selected"]["volume"][key] = value
            result = score_structure({"structure": 55, "volume": 30, "trend": 15, "position": 22, "contraction_extensions": 12}, item, CFG, FROZEN)
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertIsNone(result["total"])
            self.assertEqual(result["extension"], 12)

    def test_invalid_price_days_efficiency_and_low(self):
        for field, value in [("advance_days", 0), ("advance_days", 1.5), ("path_efficiency", float("nan")), ("path_efficiency", 1.1)]:
            item = evidence()
            item["selected"]["price"][field] = value
            self.assertEqual(impulse_metrics(item, CFG)["status"], "INCOMPLETE")
        item = evidence()
        item["selected"]["retention"]["rounds"][0]["close_low"] = None
        self.assertEqual(impulse_metrics(item, CFG)["status"], "INCOMPLETE")

    def test_maximum_is_basic_100_plus_extension_12(self):
        item = evidence()
        item["selected"]["anchor"]["peak_close"] = 150
        item["selected"]["price"].update(advance_days=1, path_efficiency=1)
        item["selected"]["retention"]["rounds"][0]["close_low"] = 160
        result = score_structure({"structure": 55, "volume": 30, "trend": 15, "position": 22, "contraction_extensions": 12}, item, CFG, FROZEN)
        self.assertEqual(result["basic"], 100)
        self.assertEqual(result["total"], 112)

    def test_no_group_zero_impulse_and_original_volume_penalty_preserved(self):
        result = score_structure({"structure": 0, "volume": -10, "trend": 0, "position": 0, "contraction_extensions": 6}, {"status": "NO_GROUP"}, CFG, FROZEN)
        self.assertEqual(result["components"]["volume"], -20 / 3)
        self.assertEqual(result["basic"], 0)
        self.assertEqual(result["total"], 6)
        self.assertEqual(result["status"], "COMPLETE")

    def test_missing_original_component_is_not_zero(self):
        result = score_structure({"structure": 0, "volume": 0, "trend": 0, "contraction_extensions": 0}, {"status": "NO_GROUP"}, CFG, FROZEN)
        self.assertIsNone(result["total"])
        self.assertIn("LEGACY_COMPONENT_MISSING:position", result["reasons"])

    def test_inputs_and_frozen_parameters_not_mutated(self):
        inputs = [evidence(), copy.deepcopy(CFG), copy.deepcopy(FROZEN)]
        before = copy.deepcopy(inputs)
        score_impulse(*inputs)
        self.assertEqual(inputs, before)

    def test_calibration_deduplicates_code_and_anchor(self):
        cfg = copy.deepcopy(CFG)
        cfg["calibration"]["min_samples"] = 1
        duplicate = evidence()
        calibration = calibrate([("1", evidence()), ("1", duplicate), ("2", evidence())], cfg, "2026-09-15")
        self.assertEqual(len(calibration["samples"]), 2)
        self.assertEqual(calibration["distributions"]["gain_pct"]["count"], 2)

    def test_calibration_rejects_other_dates_insufficient_and_uninformative_data(self):
        with self.assertRaises(ValueError):
            calibrate([("1", evidence())], CFG, "2026-09-16")
        with self.assertRaises(ValueError):
            calibrate([("1", evidence())], CFG, "2026-09-15")
        cfg = copy.deepcopy(CFG)
        cfg["calibration"]["min_samples"] = 1
        item = evidence()
        item["selected"]["volume"]["up_mean_volume_ratio"] = .5
        with self.assertRaises(ValueError):
            calibrate([("1", item)], cfg, "2026-09-15")

    def test_setup_conversion_does_not_consume_bonus_at_old_cap(self):
        self.assertEqual(setup_conversion(112, {})["normalized_base"], 60)
        result = setup_conversion(106, {})
        self.assertEqual(result["normalized_base"], 57)
        self.assertEqual(result["legacy_capped_base"], 60)
        self.assertTrue(result["legacy_would_saturate"])

    def test_invalid_budget_and_non_research_configuration_rejected(self):
        for field, value in [("research_only", False), ("setup_conversion", "old")]:
            cfg = copy.deepcopy(CFG)
            cfg[field] = value
            with self.assertRaises(ValueError):
                validate_config(cfg)
        cfg = copy.deepcopy(CFG)
        cfg["budget"]["impulse"] = 31
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_rank_ties_use_code_and_missing_score_is_excluded(self):
        self.assertEqual(rank({"2": {"s": 50}, "1": {"s": 50}, "3": {"s": None}}, "s"), {"1": 1, "2": 2})

    def test_existing_output_directory_is_rejected_before_database_reads(self):
        args = ["audit", "--dates", "260915", "--calibration-date", "260915", "--evidence-dir", "reports/research", "--output-dir", "reports/research"]
        with patch.object(sys, "argv", args), patch("scripts.audit_impulse_score.sqlite3.connect") as connect:
            with self.assertRaises(SystemExit):
                main()
            connect.assert_not_called()

    def test_trial_namespace_and_setup_context_are_isolated(self):
        original_functions = (quant.score_setup, quant.finalize_setup_score)
        shadow, _ = trial_module(CFG, FROZEN)
        structure = {"state": "VCP_FORMING", "setup_score_context": {"PULLBACK_BUY": {"structure_score": 106, "anchor_date": "2026-09-10"}}}
        before = copy.deepcopy(structure)
        overheat = {"risk_score": 0, "risk_flags": []}
        result = shadow.finalize_setup_score("PULLBACK_BUY", 10, [], [], structure, overheat)
        self.assertEqual(structure, before)
        self.assertEqual(result[-1]["setup_structure_base"], 57)
        self.assertEqual(result[-1]["setup_structure_score"], 106)
        self.assertEqual((quant.score_setup, quant.finalize_setup_score), original_functions)

    def test_trial_recomputes_impulse_on_each_frozen_anchor_prefix(self):
        shadow, calls = trial_module(CFG, FROZEN)
        df = pd.DataFrame({"date": ["2026-09-10", "2026-09-11"], "MA20": [100, 100], "MA60": [90, 90],
                           "distance_ma20": [0, 0], "volume_dry_up": [.5, .5], "vol_ma20": [100, 100], "vol_ma60": [200, 200]})
        received = []
        def analyze(frame, structure, cfg):
            received.append(str(frame.iloc[-1]["date"]))
            item = evidence()
            item["as_of"] = received[-1]
            return item
        structure = {"state": "VCP_FORMING", "volume_pattern": "decreasing", "pivot_distance": -2}
        with patch.object(shadow, "analyze_impulse_evidence", side_effect=analyze):
            shadow.score_setup(df.iloc[:1], structure, {}, {}, {"risk_score": 0})
            shadow.score_setup(df, structure, {}, {}, {"risk_score": 0})
        self.assertEqual(received, ["2026-09-10", "2026-09-11"])
        self.assertEqual([x["as_of"] for x in calls], received)


if __name__ == "__main__":
    unittest.main()
