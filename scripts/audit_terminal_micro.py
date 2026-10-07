"""Read-only comparison of archived terminal cases and predetermined controls."""

import argparse
import copy
import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import quant_filter as quant
from scripts.audit_impulse_score import canonical, digest, source_hashes, write_json
from scripts.contraction_quality import analyze_contraction_quality
from scripts.data.corporate_actions import apply_point_in_time_qfq, load_actions
from scripts.impulse_score import score_structure, validate_config
from scripts.terminal_micro import analyze_terminal_micro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2026-08-01')
    parser.add_argument('--end', default='2026-09-30')
    parser.add_argument('--control-dates', default='2026-08-14,2026-08-31,2026-09-15,2026-09-16,2026-09-24,2026-09-30')
    parser.add_argument('--controls-per-date', type=int, default=12)
    parser.add_argument('--calibration', default='reports/research/impulse_score_v1_260915_260916_20261005/calibration.json')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    out = (ROOT / args.output_dir).resolve()
    research = (ROOT / 'reports/research').resolve()
    if out.exists() or out == research or not out.is_relative_to(research):
        parser.error('Output must be a fresh directory below reports/research')
    if args.controls_per_date < 2 or args.start > args.end:
        parser.error('Invalid selection range or control count')
    cfg = json.loads((ROOT / 'strategies/02-quant-trial.json').read_text())
    validate_config(cfg)
    if not cfg.get('terminal_micro_bonus', {}).get('enabled'):
        parser.error('Terminal trial must be enabled')
    previous_cfg = copy.deepcopy(cfg)
    previous_cfg.pop('terminal_micro_bonus')
    previous_cfg['strategy_version'] = 'quant_impulse_score_research_v5'
    calibration_path = ROOT / args.calibration
    calibration = json.loads(calibration_path.read_text())
    hashes_before = source_hashes()
    hashes_before['scripts/audit_terminal_micro.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    selected, controls = [], []
    market = sqlite3.connect(f'file:{ROOT / "cache/market_data/market_data.sqlite"}?mode=ro', uri=True)
    strategy = sqlite3.connect(f'file:{ROOT / "cache/strategy/strategy_data.sqlite"}?mode=ro', uri=True)
    query = """SELECT s.trade_date,s.code,s.payload_json FROM vcp_structure_snapshots s
               JOIN current_documents d ON s.document_id=d.document_id AND d.module='quant'
               WHERE s.trade_date BETWEEN ? AND ? ORDER BY s.trade_date,s.code"""
    try:
        records = strategy.execute(query, (args.start, args.end)).fetchall()
        authority_hash = digest(records)
        by_day = {}
        for day, code, payload in records:
            row = json.loads(payload)
            if 'TERMINAL_MICRO_CONTRACTION' in row.get('contraction_extension_tags', []):
                selected.append((day, code, row, 'ARCHIVED_TERMINAL'))
            elif (row.get('structure_valid') is True
                    and row.get('structure_stage') in {'VCP_EARLY', 'VCP_FORMING', 'VCP_MATURE', 'VCP_TIGHT'}
                    and row.get('post_breakout_state') == 'PRE_BREAKOUT'):
                by_day.setdefault(day, []).append((day, code, row, 'PRESELECTED_CONTROL'))
        # Evenly spaced codes, selected before running either detector; never select by future returns.
        for day in args.control_dates.split(','):
            candidates = by_day.get(day, [])
            if len(candidates) <= args.controls_per_date:
                controls.extend(candidates)
            else:
                indexes = [round(i * (len(candidates) - 1) / (args.controls_per_date - 1))
                           for i in range(args.controls_per_date)]
                controls.extend(candidates[i] for i in indexes)
        selected += controls
        selected.sort(key=lambda item: (item[0], item[1]))
        details, rows = {}, []
        for ordinal, (day, code, original, cohort) in enumerate(selected, 1):
            if ordinal % 40 == 1:
                print(f'Comparing {ordinal}/{len(selected)}: {day} {code}', flush=True)
            raw = pd.read_sql_query('SELECT trade_date AS date,open,high,low,close,volume,amount,source '
                                    'FROM daily_bars WHERE code=? AND trade_date<=? ORDER BY trade_date',
                                    market, params=(code, day))
            if raw.empty or str(raw.iloc[-1]['date']) != day:
                raise ValueError(f'Target bars missing: {day} {code}')
            actions = load_actions(market, code, day)
            adjusted, applied = apply_point_in_time_qfq(raw, actions, day)
            frame = quant.calc_indicators(adjusted.reset_index(drop=True))
            if len(frame) != original['data_days'] or abs(float(frame.iloc[-1]['close']) - original['close']) >= .011:
                raise ValueError(f'Authority price/history conflict: {day} {code}')
            original_hash = digest(original)
            frame_before = frame.copy(deep=True)
            old = quant.detect_terminal_micro_contraction(frame, original['contraction_group'], original['structure_pivot'])
            new = analyze_terminal_micro(frame, original, cfg)
            evidence = quant.analyze_impulse_evidence(frame, original, quant.IMPULSE_EVIDENCE_CFG)
            quality = analyze_contraction_quality(frame, original, cfg, evidence)
            before = score_structure(original['score_components'], evidence, previous_cfg, calibration,
                                     original.get('contraction_extensions'), quality,
                                     structure_stage=original.get('structure_stage'))
            after = score_structure(original['score_components'], evidence, cfg, calibration,
                                    original.get('contraction_extensions'), quality, new,
                                    structure_stage=original.get('structure_stage'))
            if digest(original) != original_hash:
                raise AssertionError('Original structure was mutated')
            pd.testing.assert_frame_equal(frame, frame_before)
            if canonical(old) != canonical(quant.detect_terminal_micro_contraction(
                    frame, original['contraction_group'], original['structure_pivot'])):
                raise AssertionError('Production detector output changed')
            archived = next((item for item in original.get('contraction_extensions', [])
                             if item['type'] == 'TERMINAL_MICRO_CONTRACTION'), None)
            selected_window = new.get('selected') or {}
            expected_delta = new['score'] - (archived['score'] if archived else 0) if new['score'] is not None else None
            score_delta = after['total'] - before['total'] if before['total'] is not None and after['total'] is not None else None
            if score_delta is not None and abs(score_delta - expected_delta) > 1e-8:
                raise AssertionError('Other scoring components changed')
            item = {'date': day, 'code': code, 'name': original['name'], 'cohort': cohort,
                    'stage': original['structure_stage'], 'adjustment_status': original.get('adjustment_status', 'UNKNOWN'),
                    'archive_hit': bool(archived), 'same_input_old_hit': bool(old), 'new_status': new['status'],
                    'new_hit': new['hit'], 'score_delta': score_delta, 'expected_delta': expected_delta,
                    'before_total': before['total'], 'after_total': after['total'],
                    'window_start': selected_window.get('start_date'), 'window_end': selected_window.get('end_date'),
                    'window_days': selected_window.get('duration_days'), 'reasons': ','.join(new['reasons'])}
            ref = new.get('reference') or {}
            item['range_ratio'] = selected_window.get('range_pct', 0) / ref['range_pct'] if ref.get('range_pct') else None
            item['volume_ratio'] = selected_window.get('avg_volume', 0) / ref['avg_volume'] if ref.get('avg_volume') else None
            rows.append(item)
            details[f'{day}:{code}'] = {'summary': item, 'original_quant': original,
                                      'same_input_old': old, 'new_terminal': new, 'before_score': before,
                                      'after_score': after, 'impulse_evidence': evidence,
                                      'applied_actions': applied, 'input_sha256': digest(raw.to_dict(orient='records'))}
        if digest(strategy.execute(query, (args.start, args.end)).fetchall()) != authority_hash:
            raise AssertionError('Authority changed during audit')
    finally:
        market.close()
        strategy.close()
    hashes_after = source_hashes()
    hashes_after['scripts/audit_terminal_micro.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if hashes_before != hashes_after:
        raise AssertionError('Source changed during audit')
    summaries = {}
    for cohort in ['ARCHIVED_TERMINAL', 'PRESELECTED_CONTROL']:
        subset = [row for row in rows if row['cohort'] == cohort]
        summaries[cohort] = {'rows': len(subset), 'unique_stocks': len({row['code'] for row in subset}),
                             'retained': sum(row['archive_hit'] and row['new_hit'] for row in subset),
                             'removed': sum(row['archive_hit'] and not row['new_hit'] and row['new_status'] == 'COMPLETE' for row in subset),
                             'added': sum(not row['archive_hit'] and row['new_hit'] for row in subset),
                             'incomplete': sum(row['new_status'] != 'COMPLETE' for row in subset),
                             'old_replay_disagreements': sum(row['archive_hit'] != row['same_input_old_hit'] for row in subset),
                             'full_score_complete': sum(row['score_delta'] is not None for row in subset),
                             'unhit_reason_counts': dict(Counter(reason for row in subset if not row['new_hit']
                                                                for reason in row['reasons'].split(',')))}
    out.mkdir(parents=True)
    write_json(out / 'details.json', details)
    write_json(out / 'summary.json', summaries)
    write_json(out / 'manifest.json', {'strategy_version': cfg['strategy_version'], 'config': cfg,
                                     'source_hashes': hashes_before, 'authority_sha256': authority_hash,
                                     'calibration_sha256': hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
                                     'selection': vars(args), 'production_output_changes': 0,
                                     'adjustment_status_counts': dict(Counter(row['adjustment_status'] for row in rows))})
    with (out / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 尾段收缩隔离规则对比（research_v6）', '',
             '原标签案例全部纳入，不按后续收益筛选；未命中对照按预先指定日期、代码均匀抽样。使用原权威组选组、本地点时前复权日线及各自截止日，无外部取数。', '',
             '| 样本 | 记录 | 股票 | 保留 | 取消 | 新增 | 数据不完整 | 旧函数与档案不一致 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for cohort, summary in summaries.items():
        lines.append(f'| {cohort} | {summary["rows"]} | {summary["unique_stocks"]} | {summary["retained"]} | {summary["removed"]} | {summary["added"]} | {summary["incomplete"]} | {summary["old_replay_disagreements"]} |')
    lines += ['', '## 代表案例', '', '| 日期 | 股票 | 原标签 | 新尾段 | 结构分变化（仅尾段） | 原因 |',
              '|---|---|---:|---:|---:|---|']
    displayed = set()
    for condition in [lambda r: r['archive_hit'] and r['new_hit'],
                      lambda r: r['archive_hit'] and not r['new_hit'],
                      lambda r: not r['archive_hit'] and r['new_hit']]:
        count = 0
        for row in reversed(rows):
            if condition(row) and row['code'] not in displayed:
                displayed.add(row['code'])
                delta = f'{row["score_delta"]:+.0f}' if row['score_delta'] is not None else '不完整'
                lines.append(f'| {row["date"]} | {row["code"]} {row["name"]} | {int(row["archive_hit"])} | {int(row["new_hit"])} | {delta} | {row["reasons"]} |')
                count += 1
                if count == 6:
                    break
    lines += ['', '## 口径与限制', '',
              '新尾段为截至分析日最后2—5日，包含最后两天止跌确认；旧窗口仅为2—5日下跌段，再检查曾经反弹3%。允许低点及收盘持平，量能与相对振幅只需较上一轮减小，守住上一轮日内低点；枢轴距离范围沿用。此处识别迹象，不构成买入建议。',
              'before/after结构总分使用同一原权威分项、同一v35推进证据、同一优质序列加分与冻结标定，仅替换尾段6分。重置3分及118预算不变；不完整总分保留null。未重放整个买点动作/评级。',
              '历史评分套用9/15冻结标定仅为敏感性比较，不声称8月时参数已可得。PARTIAL/PENDING/UNKNOWN状态按原档案保留，没有升级复权核验。相邻日期同股重复记录不能当作独立样本；未命中对照为抽样，不代表全量新增覆盖率。',
              '旧函数与档案不一致须单列为版本/输入差异，不能归因于新规则。原策略/数据库只读、源指纹前后一致，原检测输出复算一致；旧报告未覆盖。', '',
              '完整逐项检查及新旧窗口见details.json；CSV可按日期/代码筛选。']
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(out)


if __name__ == '__main__':
    main()
