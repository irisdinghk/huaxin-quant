"""Production score integration, persistence, missing data and version continuity."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import bloom, quant_filter as quant
from scripts.audit_impulse_score import trial_module
from scripts.data.strategy_data_store import connect, load_document, save_quant
from scripts.dashboard_vcp import compact_candidate
from scripts.quant_scoring import IncompleteStructureScore, load_profile, presentation_fields
from scripts.test_impulse_selection import fixture


class ProductionScoreTests(unittest.TestCase):
    def test_live_profile_is_frozen_and_matches_validated_trial_algorithm(self):
        cfg = quant.STRUCTURE_SCORING_CFG
        self.assertFalse(cfg['research_only'])
        self.assertEqual(quant.STRATEGY_VERSION, 'model2_quant_v36')
        trial = json.loads((Path(__file__).resolve().parents[1] / 'strategies/02-quant-trial.json').read_text())
        for key in trial.keys() - {'strategy_version', 'research_only'}:
            self.assertEqual(cfg[key], trial[key])
        self.assertEqual(quant.STRUCTURE_CALIBRATION['sample_count'], 333)
        self.assertEqual(quant.STRUCTURE_CALIBRATION['calibration_date'], '2026-09-15')
        self.assertTrue(quant.STRUCTURE_SCORE_POLICY_ID.startswith(cfg['parameter_id'] + ':'))

    def test_production_and_isolated_trial_share_score_and_118_conversion(self):
        df, structure = fixture()
        df = quant.calc_indicators(df)
        trial_cfg = json.loads((Path(__file__).resolve().parents[1] / 'strategies/02-quant-trial.json').read_text())
        trial, _ = trial_module(trial_cfg, quant.STRUCTURE_CALIBRATION)
        overheat = {'risk_score': 0, 'risk_flags': []}
        actual = quant.score_setup(df, structure, {}, {}, overheat)
        expected = trial.score_setup(df, structure, {}, {}, overheat)
        self.assertEqual(actual['structure_score'], expected['structure_score'])
        self.assertEqual(actual['components'], expected['components'])
        structure['setup_score_context'] = {'PULLBACK_BUY': {
            'structure_score': actual['structure_score'], 'anchor_date': str(df.iloc[-1]['date'])}}
        a = quant.finalize_setup_score('PULLBACK_BUY', 12, [], [], structure, overheat)
        b = trial.finalize_setup_score('PULLBACK_BUY', 12, [], [], structure, overheat)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[-1]['setup_structure_score'], actual['structure_score'])
        self.assertEqual(a[-1]['setup_structure_base'], round(actual['structure_score'] / 118 * 100 * .6))
        self.assertFalse(actual['details']['research_only'])

    def test_incomplete_score_blocks_entire_stock_without_legacy_fallback(self):
        df, _ = fixture()
        df = quant.calc_indicators(df)
        issue = {'status': 'INCOMPLETE', 'reasons': ['QUALIFICATION_VOLUME_INCOMPLETE'], 'total': None}
        with patch.object(quant, 'compute_structure_score', side_effect=IncompleteStructureScore(issue)):
            result = quant.screen(df)
        self.assertEqual(result['structure_stage'], 'DATA_ISSUE')
        self.assertEqual(result['setup_signal'], 'NONE')
        self.assertFalse(result['model2_include'])
        self.assertEqual(result['structure_scoring_status'], 'INCOMPLETE')
        self.assertIsNone(result['structure_score_details']['total'])

    def test_invalid_frozen_mapping_refuses_production_configuration(self):
        invalid = copy.deepcopy(quant.STRUCTURE_SCORING_CFG)
        invalid['frozen_calibration']['mappings'].pop('gain_pct')
        with patch('scripts.quant_scoring.load_strategy_config', return_value=(invalid, 'test')):
            with self.assertRaises(ValueError):
                load_profile(quant.QUANT_STRATEGY, quant.IMPULSE_EVIDENCE_CFG)

    def test_presentation_replaces_tail_and_does_not_double_count_sequence(self):
        old = {'contraction_extensions': [{'type': 'CONFIRMED_RESET_CONTRACTION', 'score': 6},
                                          {'type': 'TERMINAL_MICRO_CONTRACTION', 'score': 6}],
               'conditions': ['确认型重置收缩', '末端微收缩']}
        score = {'extension_score_adjustments': [{'type': 'CONFIRMED_RESET_CONTRACTION', 'trial_score': 3}],
                 'terminal_micro': {'hit': False, 'score': 0}, 'extension': 3,
                 'contraction_quality': {'hit': True, 'score': 6}}
        fields = presentation_fields(old, score)
        self.assertEqual(fields['contraction_extension_tags'], ['CONFIRMED_RESET_CONTRACTION'])
        self.assertEqual(sum(x['score'] for x in fields['contraction_extensions']), 3)
        self.assertEqual(fields['contraction_quality_score'], 6)
        self.assertNotIn('末端微收缩', fields['structure_conditions'])
        self.assertEqual(old['contraction_extensions'][0]['score'], 6)

    def test_quant_csv_database_and_dashboard_preserve_same_score_facts(self):
        df, _ = fixture()
        result = quant.result_from_df('000001', 'test', quant.calc_indicators(df), '2026-02-26')
        payload = {'meta': {'run_date': result['run_date'], 'strategy_version': quant.STRATEGY_VERSION}, 'results': [result]}
        conn = connect(':memory:')
        try:
            save_quant(conn, payload)
            self.assertEqual(load_document(conn, 'quant', result['run_date']), payload)
        finally:
            conn.close()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'quant.csv'
            quant.write_csv([result], str(path))
            with path.open(encoding='utf-8-sig') as handle:
                rows = list(csv.DictReader(handle))
            row = rows[0]
            self.assertNotIn(None, row)
            self.assertEqual(json.loads(row['structure_score_details']), result['structure_score_details'])
            self.assertEqual(row['structure_score_policy_id'], result['structure_score_policy_id'])
        displayed = compact_candidate({}, result, {})
        self.assertEqual(displayed['structure_score_details'], result['structure_score_details'])


class ScoreVersionContinuityTests(unittest.TestCase):
    def test_same_market_version_change_does_not_generate_false_downgrade(self):
        previous = {'structure_score': '79', 'best_score': '100', 'first_seen': '2026-09-15',
                    'days_tracked': '9', 'bloom_status': 'FORMING'}
        row = {'structure_score': 57.05, 'structure_score_policy_id': quant.STRUCTURE_SCORE_POLICY_ID,
               'strategy_version': quant.STRATEGY_VERSION, 'structure_stage': 'VCP_FORMING',
               'structure_risk_score': 0, 'post_breakout_state': 'PRE_BREAKOUT'}
        self.assertIsNone(bloom.score_change(previous, row))
        current = bloom.state_row(previous, row, '2026-09-30', 'FORMING')
        self.assertEqual(current['bloom_signal'], 'CONTINUED')
        self.assertEqual(current['structure_score_policy_changed'], 'true')
        self.assertEqual(current['best_score'], '57.05')
        self.assertEqual(current['first_seen'], previous['first_seen'])
        self.assertEqual(current['quant_strategy_version'], quant.STRATEGY_VERSION)
        self.assertEqual(bloom.row_event(current, '2026-09-30')['structure_score_policy_id'], quant.STRUCTURE_SCORE_POLICY_ID)

    def test_same_policy_still_detects_score_decline_and_zero_is_valid(self):
        previous = {'structure_score': 0, 'last_score': 100, 'structure_score_policy_id': 'same'}
        self.assertEqual(bloom.score_change(previous, {'structure_score': 3, 'structure_score_policy_id': 'same'}), 3)
        previous['structure_score'] = 79
        delta = bloom.score_change(previous, {'structure_score': 57.05, 'structure_score_policy_id': 'same'})
        self.assertEqual(bloom.bloom_signal({}, 'FORMING', 'CONTINUED', delta), 'DOWNGRADE')

    def test_real_failure_is_not_hidden_by_score_policy_change(self):
        previous = {'structure_score': 90, 'structure_score_policy_id': 'old', 'bloom_status': 'COOLDOWN'}
        row = {'structure_score': 60, 'structure_score_policy_id': 'new',
               'post_breakout_state': 'POST_BREAKOUT_FAILED', 'structure_stage': 'NONE'}
        current = bloom.state_row(previous, row, '2026-09-30', 'COOLDOWN')
        self.assertEqual(current['bloom_status'], 'EXIT')
        self.assertEqual(current['bloom_signal'], 'EXIT')

    def test_missing_quote_preserves_score_policy_and_tracking(self):
        previous = {'structure_score': 75, 'structure_score_policy_id': 'same',
                    'quant_strategy_version': 'v36', 'days_tracked': 9, 'first_seen': '2026-09-15'}
        current = bloom.missing_data_row(previous, '2026-09-30')
        self.assertEqual(current['structure_score_policy_id'], 'same')
        self.assertEqual(current['structure_score_policy_changed'], 'false')
        self.assertEqual(current['quant_strategy_version'], 'v36')
        self.assertEqual(current['bloom_signal'], 'DATA_HOLD')

    def test_explicit_scoring_issue_does_not_install_zero_as_a_new_score(self):
        previous = {'structure_score': '75', 'structure_score_policy_id': 'old',
                    'quant_strategy_version': 'v35', 'bloom_status': 'FORMING', 'best_score': '80'}
        invalid = {'structure_score': 0, 'structure_score_policy_id': 'new',
                   'strategy_version': 'v36', 'structure_stage': 'DATA_ISSUE'}
        held = bloom.state_row(previous, invalid, '2026-09-30', 'DATA_ISSUE')
        self.assertEqual(held['structure_score'], '75')
        self.assertEqual(held['structure_score_policy_id'], 'old')
        self.assertEqual(held['score_change'], '')
        self.assertEqual(held['best_score'], '80')
        resumed = bloom.state_row(held, {**invalid, 'structure_stage': 'VCP_FORMING', 'structure_score': 57},
                                 '2026-10-08', 'FORMING')
        self.assertEqual(resumed['score_change'], '')
        self.assertNotEqual(resumed['bloom_signal'], 'DOWNGRADE')


if __name__ == '__main__':
    unittest.main()
