"""Contract tests for the fixed research bonus and its time/phase boundaries."""

import copy
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.contraction_quality import analyze_contraction_quality

CFG = {"contraction_quality_bonus": {"enabled": True, "score": 6}}


def fixture(closes=None, volumes=None):
    closes = closes or [120, 110, 100, 115, 111, 108]
    frame = pd.DataFrame({"date": pd.bdate_range("2026-09-01", periods=len(closes)).strftime("%Y-%m-%d"),
                          "close": closes, "volume": volumes or [100, 100, 100, 60, 60, 60]})
    frame[['close', 'volume']] = frame[['close', 'volume']].astype(float)
    group = [{"start_date": frame.iloc[i]["date"], "end_date": frame.iloc[i + 2]["date"],
              "start_idx": i, "end_idx": i + 2,
              "confirmation_status": "PROVISIONAL" if i else "CONFIRMED"}
             for i in range(0, len(closes), 3)]
    structure = {"state": "VCP_FORMING", "structure_valid": True,
                 "post_breakout_state": "PRE_BREAKOUT", "contraction_group": group}
    return frame, structure


class ContractionQualityTests(unittest.TestCase):
    def test_all_three_checks_are_required_without_partial_points(self):
        for closes, volumes, failed in [
            ([120, 110, 100, 115, 111, 99], None, 'low_rising'),
            ([120, 110, 100, 140, 111, 108], None, 'drawdown_shrinking'),
            (None, [100] * 6, 'volume_decreasing'),
        ]:
            frame, structure = fixture(closes, volumes)
            result = analyze_contraction_quality(frame, structure, CFG)
            self.assertEqual(result['status'], 'COMPLETE')
            self.assertEqual(result['score'], 0)
            self.assertFalse(result['checks'][failed])
            self.assertEqual(sum(result['checks'].values()), 2)
        result = analyze_contraction_quality(*fixture(), CFG)
        self.assertTrue(result['hit'])
        self.assertEqual(result['score'], 6)

    def test_falling_median_does_not_veto_improving_low_depth_and_volume(self):
        frame, structure = fixture([120, 110, 100, 115, 108, 108])
        result = analyze_contraction_quality(frame, structure, CFG)
        self.assertFalse(result['observations']['center_rising'])
        self.assertNotIn('center_rising', result['checks'])
        self.assertTrue(result['hit'])
        self.assertEqual(result['score'], 6)

    def test_improvement_has_no_extra_magnitude_threshold(self):
        frame, structure = fixture([120, 110, 100, 120, 110.0001, 100.0001],
                                   [100, 100, 100, 99.999, 99.999, 99.999])
        self.assertEqual(analyze_contraction_quality(frame, structure, CFG)['score'], 6)

    def test_middle_round_may_have_minor_reversals(self):
        frame, structure = fixture([120, 110, 100, 119, 109.9, 99.9, 115, 111, 108],
                                   [100] * 3 + [101] * 3 + [60] * 3)
        self.assertEqual(analyze_contraction_quality(frame, structure, CFG)['score'], 6)

    def test_rebuilt_phase_cannot_borrow_old_contractions(self):
        frame, structure = fixture()
        result = analyze_contraction_quality(frame, structure, CFG,
                                             {'selected': {'group_round_numbers': [2]}})
        self.assertEqual(result['score'], 0)
        self.assertIn('INSUFFICIENT_ROUNDS', result['reasons'])

    def test_ineligible_states_and_post_breakout_do_not_get_bonus(self):
        frame, structure = fixture()
        for state in ['NONE', 'VCP_EARLY', 'TREND_REBUILD', 'POST_BREAKOUT', 'STRUCTURE_INVALID']:
            structure['state'] = state
            self.assertEqual(analyze_contraction_quality(frame, structure, CFG)['score'], 0)
        structure['state'] = 'VCP_FORMING'
        structure['post_breakout_state'] = 'POST_BREAKOUT_FAILED'
        self.assertEqual(analyze_contraction_quality(frame, structure, CFG)['score'], 0)
        structure['post_breakout_state'] = 'PRE_BREAKOUT'
        structure['structure_valid'] = False
        self.assertEqual(analyze_contraction_quality(frame, structure, CFG)['score'], 0)

    def test_insufficient_data_and_conflicting_boundaries_are_not_zero(self):
        for field, value in [('close', float('nan')), ('volume', 0), ('volume', float('inf'))]:
            frame, structure = fixture()
            frame.loc[3, field] = value
            result = analyze_contraction_quality(frame, structure, CFG)
            self.assertEqual(result['status'], 'INCOMPLETE')
            self.assertIsNone(result['score'])
        frame, structure = fixture()
        structure['contraction_group'][1]['end_idx'] = 4
        self.assertEqual(analyze_contraction_quality(frame, structure, CFG)['status'], 'INCOMPLETE')

    def test_no_future_bars_or_intraday_lows_are_used(self):
        frame, structure = fixture()
        frame['low'] = 1
        before = analyze_contraction_quality(frame, structure, CFG)
        extended = pd.concat([frame, pd.DataFrame([{'date': '2026-09-09', 'close': 1, 'volume': 999999, 'low': .1}])], ignore_index=True)
        self.assertEqual(before, analyze_contraction_quality(extended, structure, CFG))
        self.assertEqual(analyze_contraction_quality(frame.iloc[:4], structure, CFG)['status'], 'INCOMPLETE')

    def test_disabled_legacy_config_and_inputs_are_preserved(self):
        frame, structure = fixture()
        saved_frame, saved_structure = frame.copy(deep=True), copy.deepcopy(structure)
        self.assertEqual(analyze_contraction_quality(frame, structure, {})['score'], 0)
        analyze_contraction_quality(frame, structure, CFG)
        pd.testing.assert_frame_equal(frame, saved_frame)
        self.assertEqual(structure, saved_structure)


if __name__ == '__main__':
    unittest.main()
