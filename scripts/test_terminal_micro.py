"""Boundary and integration checks for the research terminal support bonus."""

import copy
import unittest
from unittest.mock import patch

import pandas as pd

from scripts import quant_filter as quant
from scripts.terminal_micro import analyze_terminal_micro
from scripts.audit_impulse_score import trial_module
from scripts.impulse_score import score_structure, validate_config
from scripts.test_impulse_score import LIVE_CFG, FROZEN, EXTENSIONS, evidence


def fixture():
    frame = pd.DataFrame({
        "date": pd.bdate_range("2026-09-01", periods=8).strftime("%Y-%m-%d"),
        "close": [110, 102, 100, 104, 103, 102, 102, 102],
        "high": [111, 104, 101, 105, 104, 103, 103, 103],
        "low": [108, 100, 99, 103, 102, 101, 101, 101],
        "volume": [100, 100, 100, 90, 90, 90, 90, 90],
    })
    frame[['close', 'high', 'low', 'volume']] = frame[['close', 'high', 'low', 'volume']].astype(float)
    structure = {"state": "VCP_FORMING", "structure_valid": True,
                 "post_breakout_state": "PRE_BREAKOUT", "structure_pivot": 110,
                 "contraction_group": [{"start_idx": 0, "end_idx": 2,
                     "start_date": frame.iloc[0]['date'], "end_date": frame.iloc[2]['date']}]}
    return frame, structure


class TerminalMicroTests(unittest.TestCase):
    def test_flat_two_day_support_with_only_ten_percent_volume_reduction_gets_six(self):
        frame, structure = fixture()
        result = analyze_terminal_micro(frame, structure, LIVE_CFG)
        self.assertTrue(result['hit'])
        self.assertEqual(result['score'], 6)
        self.assertEqual(result['selected']['duration_days'], 5)
        self.assertEqual(result['confirmation']['closes'], [102, 102])

    def test_either_falling_low_or_falling_close_blocks_all_windows(self):
        for column, value, reason in [('low', 100.9, 'low_stopped'), ('close', 101.9, 'close_stopped')]:
            frame, structure = fixture()
            frame.loc[7, column] = value
            result = analyze_terminal_micro(frame, structure, LIVE_CFG)
            self.assertFalse(result['hit'])
            self.assertIn(reason, result['reasons'])

    def test_support_volume_and_range_each_remain_necessary(self):
        for column, value, reason in [('low', 98, 'support_held'),
                                      ('volume', 100, 'volume_decreasing'),
                                      ('high', 130, 'range_shrinking')]:
            frame, structure = fixture()
            frame.loc[3:, column] = value
            result = analyze_terminal_micro(frame, structure, LIVE_CFG)
            self.assertFalse(result['hit'])
            self.assertFalse(result['checks'][reason])

    def test_small_improvement_and_early_structure_have_no_extra_gate(self):
        frame, structure = fixture()
        structure['state'] = 'VCP_EARLY'
        frame.loc[3:, 'volume'] = 99.999
        frame.loc[7, 'low'] += .00001
        frame.loc[7, 'close'] += .00001
        self.assertEqual(analyze_terminal_micro(frame, structure, LIVE_CFG)['score'], 6)

    def test_one_day_after_reference_is_insufficient_and_old_finished_tail_not_borrowed(self):
        frame, structure = fixture()
        self.assertEqual(analyze_terminal_micro(frame.iloc[:4], structure, LIVE_CFG)['score'], 0)
        frame.loc[7, 'close'] = 101.5
        self.assertEqual(analyze_terminal_micro(frame.iloc[:7], structure, LIVE_CFG)['score'], 6)
        self.assertEqual(analyze_terminal_micro(frame, structure, LIVE_CFG)['score'], 0)

    def test_no_future_confirmation_and_inputs_unchanged(self):
        frame, structure = fixture()
        frame.loc[6, 'close'] = 101.8
        saved, original = frame.copy(deep=True), copy.deepcopy(structure)
        self.assertEqual(analyze_terminal_micro(frame.iloc[:7], structure, LIVE_CFG)['score'], 0)
        self.assertEqual(analyze_terminal_micro(frame, structure, LIVE_CFG)['score'], 6)
        pd.testing.assert_frame_equal(frame, saved)
        self.assertEqual(structure, original)

    def test_invalid_or_post_breakout_structure_and_pivot_distance_do_not_get_bonus(self):
        for key, value in [('structure_valid', False), ('post_breakout_state', 'POST_BREAKOUT_HOT'),
                            ('state', 'TREND_REBUILD'), ('structure_pivot', 90)]:
            frame, structure = fixture()
            structure[key] = value
            self.assertEqual(analyze_terminal_micro(frame, structure, LIVE_CFG)['score'], 0)

    def test_missing_prices_dates_or_wrong_boundaries_are_incomplete(self):
        frame, structure = fixture()
        for change in ['volume', 'high', 'pivot', 'index', 'date']:
            f, s = frame.copy(), copy.deepcopy(structure)
            if change in ['volume', 'high']:
                f.loc[7, change] = float('nan')
            elif change == 'pivot':
                s.pop('structure_pivot')
            elif change == 'index':
                s['contraction_group'][0]['end_idx'] = 3
            else:
                f.loc[7, 'date'] = f.loc[6, 'date']
            result = analyze_terminal_micro(f, s, LIVE_CFG)
            self.assertEqual(result['status'], 'INCOMPLETE')
            self.assertIsNone(result['score'])

    def test_disabled_legacy_rule_and_config_validation(self):
        cfg = copy.deepcopy(LIVE_CFG)
        cfg.pop('terminal_micro_bonus')
        self.assertEqual(analyze_terminal_micro(*fixture(), cfg)['reasons'], ['DISABLED'])
        for key, value in [('score', 5), ('min_days', 1), ('max_days', 6), ('enabled', 'yes')]:
            cfg = copy.deepcopy(LIVE_CFG)
            cfg['terminal_micro_bonus'][key] = value
            with self.assertRaises(ValueError):
                validate_config(cfg)

    def test_replaces_old_six_or_adds_new_six_without_duplicate_and_preserves_reset(self):
        components = {'structure': 55, 'volume': 30, 'trend': 15, 'position': 22}
        quality = {'status': 'COMPLETE', 'score': 0}
        for details, terminal_score, expected in [(EXTENSIONS, 6, 9), (EXTENSIONS, 0, 3),
                                                   (EXTENSIONS[:1], 6, 9), ([], 6, 6)]:
            legacy = {**components, 'contraction_extensions': sum(x['score'] for x in details)}
            result = score_structure(legacy, evidence(), LIVE_CFG, FROZEN, details, quality,
                                     {'status': 'COMPLETE', 'score': terminal_score}, structure_stage='VCP_TIGHT')
            self.assertEqual(result['extension'], expected)
            self.assertEqual(result['extension_original'], legacy['contraction_extensions'])
        result = score_structure({**components, 'contraction_extensions': 0}, evidence(),
                                 LIVE_CFG, FROZEN, [], quality, structure_stage='VCP_TIGHT')
        self.assertIsNone(result['total'])
        self.assertIn('TERMINAL_MICRO_INCOMPLETE', result['reasons'])

    def test_shadow_recomputes_terminal_on_each_prefix_without_production_mutation(self):
        frame, structure = fixture()
        frame.loc[6, 'close'] = 101.8
        frame['open'] = frame['close']
        frame = quant.calc_indicators(frame)
        structure.update(volume_pattern='decreasing', pivot_distance=-5,
                         contraction_extensions=[], contraction_extension_score=0)
        saved = copy.deepcopy(structure)
        shadow, calls = trial_module(LIVE_CFG, FROZEN)
        function = quant.detect_terminal_micro_contraction
        with patch.object(shadow, 'analyze_impulse_evidence', return_value=evidence()), \
                patch('scripts.audit_impulse_score.analyze_contraction_quality', return_value={'status': 'COMPLETE', 'score': 0}):
            shadow.score_setup(frame.iloc[:7], structure, {}, {}, {'risk_score': 0})
            shadow.score_setup(frame, structure, {}, {}, {'risk_score': 0})
        self.assertEqual([x['score']['extension'] for x in calls], [0, 6])
        self.assertEqual(structure, saved)
        self.assertIs(quant.detect_terminal_micro_contraction, function)


if __name__ == '__main__':
    unittest.main()
