"""Qualification, causal bases and isolated impulse selection contracts."""

import copy
import json
import unittest
from pathlib import Path

import pandas as pd

from scripts import quant_filter as quant
from scripts.audit_impulse_score import trial_module
from scripts.impulse_evidence import analyze_impulse_evidence
from scripts.impulse_selection import analyze_selected_impulse, validate_selection_config
from scripts.impulse_score import impulse_metrics
from scripts.test_impulse_score import FROZEN


CFG = json.loads((Path(__file__).resolve().parents[1] / "strategies/02-quant-trial.json").read_text())


def fixture(same_peak=False):
    tail = [90, 95, 100, 105, 110, 115, 113, 110, 108, 105, 100, 105, 115, 125, 120, 118, 116]
    if not same_peak:
        tail[:6] = [90, 95, 105, 115, 125, 135]
    closes = [float(value) for value in [100] * 24 + tail]
    df = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=len(closes)).strftime("%Y-%m-%d"),
                       "open": closes, "close": closes,
                       "high": [v + 1 for v in closes], "low": [v - 1 for v in closes],
                       "volume": [100.] * 24 + [200.] * len(tail)})
    group = [{"start_idx": 37, "end_idx": 40, "start_date": str(df.iloc[37]["date"]),
              "end_date": str(df.iloc[40]["date"]), "confirmation_status": "CONFIRMED",
              "required_right_confirm_days": 0}]
    structure = {"state": "VCP_FORMING", "structure_valid": True, "volume_pattern": "decreasing",
                 "post_breakout_state": "PRE_BREAKOUT", "contraction_group": group,
                 "contraction_extensions": [], "contraction_extension_score": 0, "pivot_distance": -2}
    return df, structure


def analyze(df, structure, cfg=CFG):
    return analyze_selected_impulse(df, structure, quant.IMPULSE_EVIDENCE_CFG, cfg)


class ImpulseSelectionTests(unittest.TestCase):
    def test_shrinking_recent_bounce_is_rejected_for_earlier_expansion(self):
        df, st = fixture()
        df.loc[35:37, "volume"] = 70
        result = analyze(df, st)
        self.assertEqual(result["selected"]["anchor"]["base_idx"], 24)
        self.assertEqual(result["selected"]["anchor"]["peak_idx"], 29)
        recent = next(x for x in result["selected"]["candidate_summaries"] if x["base_date"] == df.iloc[34]["date"])
        self.assertEqual(recent["rejection_reasons"], ["NO_RELATIVE_VOLUME_EXPANSION"])
        self.assertFalse(recent["eligible"])

    def test_same_peak_prefers_nearest_valid_low_not_largest_gain(self):
        df, st = fixture(same_peak=True)
        result = analyze(df, st)
        self.assertEqual(result["selected"]["anchor"]["base_idx"], 34)
        self.assertEqual(result["selected"]["anchor"]["peak_idx"], 37)
        earlier = next(x for x in result["selected"]["candidate_summaries"] if x["base_date"] == df.iloc[24]["date"])
        self.assertTrue(earlier["eligible"])
        self.assertGreater(earlier["gain_pct"], result["selected"]["price"]["gain_pct"])

    def test_equal_or_lower_mean_volume_has_no_qualified_fallback(self):
        for volume in [100., 80.]:
            df, st = fixture()
            df.loc[24:, "volume"] = volume
            result = analyze(df, st)
            self.assertEqual(result["status"], "NO_ADVANCE")
            self.assertIsNone(result["selected"]["anchor"])
            self.assertEqual(impulse_metrics(result, CFG)["status"], "NO_IMPULSE")

    def test_weak_price_candidate_remains_diagnostic_only(self):
        df, st = fixture()
        df.loc[24:, "close"] = 90 + (df.loc[24:, "close"] - 90) * .1
        df["open"], df["high"], df["low"] = df["close"], df["close"] + 1, df["close"] - 1
        result = analyze(df, st)
        self.assertEqual(result["status"], "NO_ADVANCE")
        self.assertTrue(any("PRICE_GAIN_BELOW_RESEARCH_THRESHOLD" in x["rejection_reasons"]
                            for x in result["selected"]["candidate_summaries"]))

    def test_missing_volume_is_incomplete_not_a_smaller_anchor(self):
        df, st = fixture(same_peak=True)
        df.loc[25, "volume"] = float("nan")
        result = analyze(df, st)
        self.assertEqual(result["status"], "DATA_ISSUE")
        self.assertIsNone(result["selected"]["anchor"])
        self.assertEqual(impulse_metrics(result, CFG)["status"], "INCOMPLETE")

    def test_missing_ohlc_cannot_bypass_one_price_checks(self):
        df, st = fixture()
        result = analyze(df.drop(columns="open"), st)
        self.assertEqual(result["status"], "DATA_ISSUE")

    def test_excluded_one_price_volume_cannot_create_expansion(self):
        df, st = fixture()
        df["volume"] = 100.
        df.loc[25, ["open", "high", "low"]] = df.loc[25, "close"]
        df.loc[25, "volume"] = 10000.
        result = analyze(df, st)
        self.assertEqual(result["status"], "NO_ADVANCE")
        first = next(x for x in result["selected"]["candidate_summaries"] if x["base_date"] == df.iloc[24]["date"])
        self.assertEqual(first["volume"]["up_mean_volume_ratio"], 1.)
        self.assertEqual(first["volume"]["advance_excluded_dates"], [str(df.iloc[25]["date"])])

    def test_two_day_base_is_provisional_without_extra_minimum_duration(self):
        df, st = fixture(same_peak=True)
        st["contraction_group"][0].update(start_idx=36, start_date=str(df.iloc[36]["date"]))
        result = analyze(df, st)
        self.assertEqual(result["selected"]["anchor"]["base_idx"], 34)
        self.assertEqual(result["selected"]["price"]["advance_days"], 2)
        self.assertEqual(result["selected"]["selected_base_confirmation"]["confirmation_status"], "PROVISIONAL")

    def test_later_bars_do_not_confirm_or_reselect_a_stage_start_base(self):
        df, st = fixture(same_peak=True)
        st["contraction_group"][0].update(start_idx=36, start_date=str(df.iloc[36]["date"]))
        before = analyze(df.iloc[:37], {**st, "contraction_group": [{**st["contraction_group"][0],
                            "end_idx": 36, "end_date": str(df.iloc[36]["date"]), "confirmation_status": "PROVISIONAL"}]})
        df.loc[37:, "close"] = 80
        after = analyze(df, st)
        self.assertEqual(before["selected"]["anchor"], after["selected"]["anchor"])
        self.assertEqual(before["selected"]["candidate_summaries"], after["selected"]["candidate_summaries"])
        self.assertEqual(after["selected"]["selected_base_confirmation"]["confirmation_status"], "PROVISIONAL")

    def test_rebuild_respects_exhaustion_floor_and_round_ownership(self):
        df, st = fixture()
        df.loc[34, ["close", "open"]] = 80
        df.loc[34, ["high", "low"]] = [81, 79]
        old = {"start_idx": 29, "end_idx": 34, "start_date": str(df.iloc[29]["date"]),
               "end_date": str(df.iloc[34]["date"]), "confirmation_status": "CONFIRMED", "required_right_confirm_days": 0}
        st["contraction_group"].insert(0, old)
        result = analyze(df, st)
        self.assertEqual(len(result["selected"]["episodes"]), 2)
        self.assertEqual(result["selected"]["scan_floor_idx"], 34)
        self.assertEqual(result["selected"]["anchor"]["base_idx"], 34)
        self.assertEqual(result["selected"]["group_round_numbers"], [2])

    def test_monotonic_window_has_no_invented_boundary_low(self):
        df, st = fixture()
        df["close"] = [100 + i for i in range(len(df))]
        df["open"], df["high"], df["low"] = df["close"], df["close"] + 1, df["close"] - 1
        self.assertEqual(analyze(df, st)["status"], "NO_ADVANCE")

    def test_missing_left_confirmation_context_is_data_issue(self):
        df, st = fixture()
        cfg = copy.deepcopy(CFG)
        cfg["impulse_selection"].update(lookback_days=10, diagnostic_lookback_days=[])
        df.loc[24, "close"] = float("nan")
        self.assertEqual(analyze(df, st, cfg)["status"], "DATA_ISSUE")

    def test_disabled_rule_preserves_exact_legacy_evidence(self):
        df, st = fixture()
        cfg = copy.deepcopy(CFG)
        cfg["impulse_selection"]["enabled"] = False
        self.assertEqual(analyze(df, st, cfg), analyze_impulse_evidence(df, st, quant.IMPULSE_EVIDENCE_CFG))

    def test_inputs_and_production_functions_remain_unchanged(self):
        df, st = fixture()
        before, original_cfg, original_st = df.copy(deep=True), copy.deepcopy(CFG), copy.deepcopy(st)
        production = quant.analyze_impulse_evidence
        shadow, _ = trial_module(CFG, FROZEN)
        direct = analyze(df, st)
        self.assertEqual(shadow.analyze_impulse_evidence(df, st, quant.IMPULSE_EVIDENCE_CFG), direct)
        self.assertIs(quant.analyze_impulse_evidence, production)
        pd.testing.assert_frame_equal(df, before)
        self.assertEqual(CFG, original_cfg)
        self.assertEqual(st, original_st)

    def test_invalid_selection_rules_are_rejected(self):
        for key, value in [("lookback_days", True), ("base_swing_window", 0),
                           ("min_up_mean_volume_ratio", .8), ("same_peak_base", "max_gain")]:
            rule = copy.deepcopy(CFG["impulse_selection"])
            rule[key] = value
            with self.assertRaises(ValueError):
                validate_selection_config(rule)


if __name__ == "__main__":
    unittest.main()
