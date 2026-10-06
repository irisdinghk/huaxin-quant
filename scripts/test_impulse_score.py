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
from scripts.test_contraction_quality import fixture as quality_fixture

LIVE_CFG = json.loads((ROOT / "strategies/02-quant-trial.json").read_text(encoding="utf-8"))
CFG = copy.deepcopy(LIVE_CFG)
CFG.pop("stage_scores")
CFG.pop("contraction_quality_bonus")
CFG.pop("terminal_micro_bonus")
CFG["budget"].pop("contraction_quality")
CFG["setup_conversion"] = "normalize_112_to_100"
FROZEN = {"mappings": {"gain_pct": {"floor": 0, "cap": 50},
                       "up_volume_ratio": {"floor": 1, "cap": 2},
                       "up_down_volume_ratio": {"floor": 1, "cap": 2},
                       "speed_pct": {"floor": 0, "cap": 10}}}
EXTENSIONS = [{"type": "CONFIRMED_RESET_CONTRACTION", "score": 6},
              {"type": "TERMINAL_MICRO_CONTRACTION", "score": 6}]


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
        old_cfg = copy.deepcopy(CFG)
        old_cfg.pop("quality_curve")
        old_cfg.pop("retention_curve")
        first, second = (score_impulse(x, old_cfg, FROZEN) for x in [original, low])
        self.assertEqual(first["quality"], second["quality"])
        self.assertAlmostEqual(first["contribution"], second["contribution"] * 2)
        self.assertEqual(first["quality_curve"], "linear")
        self.assertEqual(first["retention_curve"], "linear")

    def test_sqrt_quality_preserves_efficiency_and_zero_volume_score(self):
        item = evidence()
        item["selected"]["volume"].update(up_mean_volume_ratio=1.25, down_mean_volume_ratio=2)
        result = score_impulse(item, CFG, FROZEN)
        self.assertEqual(result["linear_indicator_scores"]["up_volume_ratio"], 25)
        self.assertEqual(result["indicator_scores"]["up_volume_ratio"], 50)
        self.assertEqual(result["quality_components"]["up_volume_ratio"], 10)
        self.assertEqual(result["indicator_scores"]["up_down_volume_ratio"], 0)
        self.assertEqual(result["indicator_scores"]["path_efficiency"], 80)
        self.assertEqual(result["quality_components"]["path_efficiency"], 12)

    def test_retention_discount_accelerates_and_separates_raw_ratio(self):
        item = evidence()
        previous = None
        drops = []
        for price in [125, 120, 115, 110, 105, 100]:
            item["selected"]["retention"]["rounds"][0]["close_low"] = price
            result = score_impulse(item, CFG, FROZEN)
            self.assertAlmostEqual(result["retention_raw"], (price - 100) / 25)
            self.assertAlmostEqual(result["retention_discount"], (1 - result["retention"]) ** 2)
            self.assertAlmostEqual(result["contribution"], .3 * result["quality"] * result["retention_coefficient"])
            if previous is not None:
                drops.append(previous - result["contribution"])
            previous = result["contribution"]
        self.assertTrue(all(a < b for a, b in zip(drops, drops[1:])))

    def test_curves_can_be_compared_independently(self):
        item = evidence()
        results = {}
        for qcurve in ["linear", "sqrt"]:
            for rcurve in ["linear", "quadratic_drawback"]:
                cfg = {**CFG, "quality_curve": qcurve, "retention_curve": rcurve}
                results[qcurve, rcurve] = score_impulse(item, cfg, FROZEN)
        self.assertEqual(results['linear', 'linear']['quality'], results['linear', 'quadratic_drawback']['quality'])
        self.assertEqual(results['sqrt', 'linear']['quality'], results['sqrt', 'quadratic_drawback']['quality'])
        self.assertEqual(results['linear', 'quadratic_drawback']['retention_coefficient'], .96)
        self.assertEqual(results['sqrt', 'linear']['retention_coefficient'], .8)
        self.assertGreater(results['sqrt', 'quadratic_drawback']['contribution'], results['linear', 'quadratic_drawback']['contribution'])

    def test_unknown_curves_rejected_even_for_no_group(self):
        for key in ['quality_curve', 'retention_curve']:
            cfg = {**CFG, key: 'unsupported'}
            with self.assertRaises(ValueError):
                score_impulse({'status': 'NO_GROUP'}, cfg, FROZEN)

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
            result = score_structure({"structure": 55, "volume": 30, "trend": 15, "position": 22, "contraction_extensions": 12}, item, CFG, FROZEN, EXTENSIONS)
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertIsNone(result["total"])
            self.assertIsNone(result["impulse"]["quality"])
            self.assertIsNone(result["impulse"]["retention_coefficient"])
            self.assertIsNone(result["impulse"]["contribution"])
            self.assertEqual(result["extension"], 9)

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
        legacy_cfg = copy.deepcopy(CFG)
        legacy_cfg.pop("extension_score_overrides")
        result = score_structure({"structure": 55, "volume": 30, "trend": 15, "position": 22, "contraction_extensions": 12}, item, legacy_cfg, FROZEN)
        self.assertEqual(result["basic"], 100)
        self.assertEqual(result["total"], 112)

    def test_no_group_zero_impulse_and_original_volume_penalty_preserved(self):
        result = score_structure({"structure": 0, "volume": -10, "trend": 0, "position": 0, "contraction_extensions": 6}, {"status": "NO_GROUP"}, CFG, FROZEN, EXTENSIONS[1:])
        self.assertEqual(result["components"]["volume"], -20 / 3)
        self.assertEqual(result["basic"], 0)
        self.assertEqual(result["total"], 6)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertIsNone(result["impulse"]["retention_coefficient"])

    def test_missing_original_component_is_not_zero(self):
        result = score_structure({"structure": 0, "volume": 0, "trend": 0, "contraction_extensions": 0}, {"status": "NO_GROUP"}, CFG, FROZEN)
        self.assertIsNone(result["total"])
        self.assertIn("LEGACY_COMPONENT_MISSING:position", result["reasons"])

    def test_reset_bonus_reduced_only_for_its_own_type(self):
        components = {"structure": 55, "volume": 30, "trend": 15, "position": 22}
        for details, expected in [(EXTENSIONS[:1], 3), (EXTENSIONS[1:], 6), (EXTENSIONS, 9), ([], 0)]:
            before = copy.deepcopy(details)
            result = score_structure({**components, 'contraction_extensions': sum(d['score'] for d in details)},
                                     evidence(), CFG, FROZEN, details)
            self.assertEqual(result['status'], 'COMPLETE')
            self.assertEqual(result['extension'], expected)
            self.assertEqual(details, before)
        legacy_cfg = copy.deepcopy(CFG)
        legacy_cfg.pop('extension_score_overrides')
        legacy = score_structure({**components, 'contraction_extensions': 12}, evidence(), legacy_cfg, FROZEN)
        current = score_structure({**components, 'contraction_extensions': 12}, evidence(), CFG, FROZEN, EXTENSIONS)
        self.assertEqual(legacy['total'] - current['total'], 3)
        self.assertEqual(legacy['impulse'], current['impulse'])

    def test_missing_or_conflicting_extension_details_do_not_guess_score(self):
        components = {'structure': 55, 'volume': 30, 'trend': 15, 'position': 22, 'contraction_extensions': 6}
        for details in [None, [], EXTENSIONS, [EXTENSIONS[0], EXTENSIONS[0]], [{'type': 'CONFIRMED_RESET_CONTRACTION'}]]:
            result = score_structure(components, evidence(), CFG, FROZEN, details)
            self.assertEqual(result['status'], 'INCOMPLETE')
            self.assertIsNone(result['total'])
            self.assertIsNone(result['extension'])

    def test_new_bonus_adds_outside_basic_and_old_extension(self):
        components = {'structure': 55, 'volume': 30, 'trend': 15, 'position': 22, 'contraction_extensions': 12}
        base = score_structure(components, evidence(), LIVE_CFG, FROZEN, EXTENSIONS,
                               {'status': 'COMPLETE', 'score': 0, 'hit': False},
                               {'status': 'COMPLETE', 'score': 6, 'hit': True}, structure_stage='VCP_TIGHT')
        improved = score_structure(components, evidence(), LIVE_CFG, FROZEN, EXTENSIONS,
                                   {'status': 'COMPLETE', 'score': 6, 'hit': True},
                                   {'status': 'COMPLETE', 'score': 6, 'hit': True}, structure_stage='VCP_TIGHT')
        self.assertEqual(base['basic'], improved['basic'])
        self.assertEqual(base['extension'], improved['extension'])
        self.assertEqual(base['extension'], 9)
        self.assertEqual(improved['total'] - base['total'], 6)
        self.assertEqual(improved['score_maximum'], 118)
        for quality in [None, {'status': 'INCOMPLETE', 'score': None}, {'status': 'COMPLETE', 'score': 3}]:
            result = score_structure(components, evidence(), LIVE_CFG, FROZEN, EXTENSIONS, quality,
                                     structure_stage='VCP_TIGHT')
            self.assertEqual(result['status'], 'INCOMPLETE')
            self.assertIsNone(result['total'])

    def test_new_conversion_uses_118_and_legacy_112_still_works(self):
        self.assertEqual(setup_conversion(118, {}, 118)['normalized_base'], 60)
        self.assertEqual(setup_conversion(106, {}, 118)['normalized_base'], 54)
        self.assertEqual(setup_conversion(106, {})['normalized_base'], 57)
        shadow, _ = trial_module(LIVE_CFG, FROZEN)
        structure = {'state': 'VCP_FORMING', 'setup_score_context': {'PULLBACK_BUY': {'structure_score': 106}}}
        result = shadow.finalize_setup_score('PULLBACK_BUY', 10, [], [], structure,
                                             {'risk_score': 0, 'risk_flags': []})
        self.assertEqual(result[-1]['setup_structure_base'], 54)
        self.assertEqual(result[-1]['setup_score_components']['trial_conversion']['structure_maximum'], 118)

    def test_direct_vcp_stages_do_not_depend_on_legacy_stage_points(self):
        cfg = {**CFG, 'stage_scores': LIVE_CFG['stage_scores']}
        for stage, expected in [('VCP_EARLY', 15), ('VCP_FORMING', 25),
                                ('VCP_MATURE', 35), ('VCP_TIGHT', 40)]:
            for legacy_stage in [None, 0, 55]:
                components = {'volume': 0, 'trend': 0, 'position': 0, 'contraction_extensions': 0}
                if legacy_stage is not None:
                    components['structure'] = legacy_stage
                result = score_structure(components, {'status': 'NO_GROUP'}, cfg, FROZEN,
                                         structure_stage=stage)
                self.assertEqual(result['status'], 'COMPLETE')
                self.assertEqual(result['total'], expected)
                self.assertEqual(result['stage_score_policy'], 'direct_stage_scores')

    def test_missing_or_unknown_stage_does_not_guess_from_original_points(self):
        cfg = {**CFG, 'stage_scores': LIVE_CFG['stage_scores']}
        components = {'structure': 32, 'volume': 0, 'trend': 0, 'position': 0, 'contraction_extensions': 0}
        for stage in [None, 'VCP_UNKNOWN', ['VCP_FORMING']]:
            result = score_structure(components, {'status': 'NO_GROUP'}, cfg, FROZEN,
                                     structure_stage=stage)
            self.assertEqual(result['status'], 'INCOMPLETE')
            self.assertIsNone(result['total'])
            self.assertIn('STRUCTURE_STAGE_MISSING_OR_UNKNOWN', result['reasons'])

    def test_stage_table_requires_complete_finite_increasing_budgeted_values(self):
        for key, value in [('VCP_EARLY', None), ('VCP_EARLY', True), ('VCP_EARLY', '15'),
                           ('VCP_EARLY', float('nan')), ('VCP_FORMING', -1),
                           ('VCP_FORMING', 35), ('VCP_MATURE', 41), ('VCP_TIGHT', 39)]:
            cfg = copy.deepcopy(LIVE_CFG)
            cfg['stage_scores'][key] = value
            with self.assertRaises(ValueError):
                validate_config(cfg)
        for missing in [True, False]:
            cfg = copy.deepcopy(LIVE_CFG)
            if missing:
                cfg['stage_scores'].pop('VCP_EARLY')
            else:
                cfg['stage_scores']['VCP_OTHER'] = 0
            with self.assertRaises(ValueError):
                validate_config(cfg)

    def test_stage_change_preserves_every_other_score_and_legacy_path(self):
        cfg = {**CFG, 'stage_scores': LIVE_CFG['stage_scores']}
        components = {'structure': 32, 'volume': 20, 'trend': 8, 'position': 0, 'contraction_extensions': 0}
        old = score_structure(components, evidence(), CFG, FROZEN)
        new = score_structure(components, evidence(), cfg, FROZEN, structure_stage='VCP_FORMING')
        self.assertEqual(old['components']['structure'], 32 * 40 / 55)
        self.assertAlmostEqual(new['total'] - old['total'], 25 - 32 * 40 / 55)
        for key in ['volume', 'trend', 'position', 'impulse', 'contraction_extensions', 'contraction_quality']:
            self.assertEqual(old['components'][key], new['components'][key])
        self.assertEqual(old['impulse'], new['impulse'])
        self.assertEqual(old['score_maximum'], new['score_maximum'])

    def test_non_vcp_scores_preserve_collapsed_archive_states(self):
        cfg = {**CFG, 'stage_scores': LIVE_CFG['stage_scores']}
        for stage, raw in [('TREND_WATCH', 12), ('TREND_REBUILD', 8), ('POST_BREAKOUT', 8),
                           ('POST_BREAKOUT', 0), ('POST_BREAKOUT_FAILED', 0),
                           ('POST_BREAKOUT_EXPIRED', 0), ('REJECT', 0), ('NONE', 0)]:
            components = {'structure': raw, 'volume': 0, 'trend': 0, 'position': 0, 'contraction_extensions': 0}
            old = score_structure(components, {'status': 'NO_GROUP'}, CFG, FROZEN)
            new = score_structure(components, {'status': 'NO_GROUP'}, cfg, FROZEN, structure_stage=stage)
            self.assertEqual(old['total'], new['total'])
            self.assertEqual(new['stage_score_policy'], 'legacy_proportional')

    def test_historical_prefix_uses_its_own_final_stage_and_keeps_production_isolated(self):
        functions = (quant.score_setup, quant.finalize_setup_score, quant.detect_vcp_structure)
        shadow, calls = trial_module(LIVE_CFG, FROZEN)
        df = pd.DataFrame({'date': ['2026-09-10', '2026-09-11'], 'MA20': [100, 100], 'MA60': [90, 90],
                           'distance_ma20': [0, 0], 'volume_dry_up': [.5, .5],
                           'vol_ma20': [100, 100], 'vol_ma60': [200, 200]})
        structure = {'state': 'VCP_EARLY', 'volume_pattern': 'decreasing', 'pivot_distance': -2,
                     'contraction_extension_score': 0, 'contraction_extensions': []}
        snapshot = copy.deepcopy(structure)
        with patch.object(shadow, 'analyze_impulse_evidence', return_value=evidence()), \
                patch('scripts.audit_impulse_score.analyze_contraction_quality', return_value={'status': 'COMPLETE', 'score': 0}), \
                patch('scripts.audit_impulse_score.analyze_terminal_micro', return_value={'status': 'COMPLETE', 'score': 0}):
            shadow.score_setup(df.iloc[:1], structure, {}, {}, {'risk_score': 0})
            shadow.score_setup(df, {**structure, 'state': 'VCP_MATURE'}, {}, {}, {'risk_score': 0})
        self.assertEqual([x['as_of'] for x in calls], ['2026-09-10', '2026-09-11'])
        self.assertEqual([x['score']['components']['structure'] for x in calls], [15, 35])
        self.assertEqual([x['score']['structure_stage'] for x in calls], ['VCP_EARLY', 'VCP_MATURE'])
        self.assertEqual(structure, snapshot)
        self.assertEqual(functions, (quant.score_setup, quant.finalize_setup_score, quant.detect_vcp_structure))

    def test_forming_pullback_theoretical_ceiling_reaches_a_boundary_without_threshold_change(self):
        raw = LIVE_CFG['stage_scores']['VCP_FORMING'] + 60 + 6 + 6
        shadow, _ = trial_module(LIVE_CFG, FROZEN)
        structure = {'state': 'VCP_FORMING', 'setup_score_context': {'PULLBACK_BUY': {'structure_score': raw}}}
        result = shadow.finalize_setup_score('PULLBACK_BUY', 15, [], [], structure,
                                             {'risk_score': 0, 'risk_flags': []})
        self.assertEqual(raw, 97)
        self.assertEqual(result[-1]['setup_structure_base'], 49)
        self.assertEqual(result[0], 80)
        self.assertEqual(shadow.SETUP_SCORING_CFG, quant.SETUP_SCORING_CFG)

    def test_bonus_is_recomputed_on_each_historical_prefix(self):
        shadow, calls = trial_module(LIVE_CFG, FROZEN)
        frame, structure = quality_fixture()
        frame = frame.assign(open=frame['close'], high=frame['close'] + 1, low=frame['close'] - 1)
        frame = quant.calc_indicators(frame)
        structure.update(volume_pattern='decreasing', pivot_distance=-2, contraction_extensions=[],
                         contraction_extension_score=0)
        before = copy.deepcopy(structure)
        with patch.object(shadow, 'analyze_impulse_evidence', return_value=evidence()):
            shadow.score_setup(frame.iloc[:4], structure, {}, {}, {'risk_score': 0})
            shadow.score_setup(frame, structure, {}, {}, {'risk_score': 0})
        self.assertIsNone(calls[0]['score']['total'])
        self.assertEqual(calls[0]['score']['contraction_quality']['status'], 'INCOMPLETE')
        self.assertEqual(calls[1]['score']['contraction_quality']['score'], 6)
        self.assertEqual(calls[1]['score']['components']['contraction_quality'], 6)
        self.assertEqual(structure, before)

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
        structure = {"state": "VCP_FORMING", "volume_pattern": "decreasing", "pivot_distance": -2,
                     "contraction_extension_score": 12, "contraction_extensions": copy.deepcopy(EXTENSIONS)}
        with patch.object(shadow, "analyze_impulse_evidence", side_effect=analyze):
            shadow.score_setup(df.iloc[:1], structure, {}, {}, {"risk_score": 0})
            shadow.score_setup(df, structure, {}, {}, {"risk_score": 0})
        self.assertEqual(received, ["2026-09-10", "2026-09-11"])
        self.assertEqual([x["as_of"] for x in calls], received)
        self.assertEqual([x['score']['extension'] for x in calls], [9, 9])
        self.assertEqual(structure['contraction_extension_score'], 12)


if __name__ == "__main__":
    unittest.main()
