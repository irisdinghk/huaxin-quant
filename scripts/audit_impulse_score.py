"""Isolated 100+12 impulse scoring with read-only inputs and frozen calibration."""

import argparse
import copy
import csv
import hashlib
import json
import os
import sqlite3
import sys
import types
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(os.path.abspath(__file__)).parents[1]
sys.path.insert(0, str(ROOT))
from scripts import quant_filter as quant
from scripts.data.corporate_actions import apply_point_in_time_qfq, load_actions
from scripts.data.strategy_data_store import load_document
from scripts.impulse_score import calibrate, score_structure, setup_conversion, validate_config


def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in [
        "scripts/quant_filter.py", "scripts/impulse_evidence.py", "strategies/02-quant.json",
        "scripts/impulse_score.py", "scripts/audit_impulse_score.py", "strategies/02-quant-trial.json"]}


def trial_module(cfg, calibration):
    """Load an independent module namespace; never replace production globals."""
    module = types.ModuleType("quant_impulse_score_trial")
    module.__file__ = str(ROOT / "scripts/quant_filter.py")
    source = (ROOT / "scripts/quant_filter.py").read_text(encoding="utf-8")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    original_score, original_finalize = module.score_setup, module.finalize_setup_score
    calls = []

    def score(df, structure, pullback, retest, overheat):
        old = original_score(df, structure, pullback, retest, overheat)
        evidence = module.analyze_impulse_evidence(df, structure, module.IMPULSE_EVIDENCE_CFG)
        scored = score_structure(old["components"], evidence, cfg, calibration)
        day = str(df.iloc[-1]["date"])
        anchor = ((evidence.get("selected") or {}).get("anchor") or {})
        if anchor.get("peak_date") and anchor["peak_date"] > day:
            raise AssertionError("Trial used a future price anchor")
        calls.append({"as_of": day, "stage": structure.get("state"),
                      "legacy_score": old["structure_score"], "score": scored,
                      "impulse_anchor": anchor})
        if scored["total"] is None:
            # The complete replay is later marked unusable; this only lets it collect diagnostics.
            return old
        return {**old, "structure_score": round(scored["total"], 2), "components": scored["components"]}

    def finalize(signal, action, reasons, misses, structure, overheat):
        temporary = copy.deepcopy(structure)
        context = (temporary.get("setup_score_context") or {}).get(signal, {})
        raw = context.get("structure_score")
        conversion = None
        if raw is not None:
            conversion = setup_conversion(raw, module.SETUP_SCORING_CFG.get("structure_quality", {}))
            context["structure_score"] = conversion["normalized_structure"]
        result = original_finalize(signal, action, reasons, misses, temporary, overheat)
        if conversion is not None:
            result[-1]["setup_structure_score"] = raw
            result[-1]["setup_score_components"]["trial_conversion"] = conversion
        return result

    module.score_setup, module.finalize_setup_score = score, finalize
    return module, calls


def rank(records, score_key):
    selected = [(code, item[score_key]) for code, item in records.items() if item[score_key] is not None]
    return {code: index for index, (code, _) in enumerate(sorted(selected, key=lambda x: (-x[1], x[0])), 1)}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


def summarize(day, details, document):
    available = [x for x in details.values() if x["archive_trial"]["total"] is not None]
    valid_replays = [x for x in details.values() if x["replay_usable"]]
    transitions = Counter(f'{x["baseline_decision"]["setup_signal"]}->{x["trial_decision"]["setup_signal"]}'
                          for x in valid_replays if x["baseline_decision"]["setup_signal"] != x["trial_decision"]["setup_signal"])
    baseline_quality = Counter(x["baseline_decision"]["setup_quality"] for x in valid_replays
                               if x["baseline_decision"]["setup_signal"] != "NONE")
    trial_quality = Counter(x["trial_decision"]["setup_quality"] for x in valid_replays
                            if x["trial_decision"]["setup_signal"] != "NONE")
    frozen_calls = [c for x in details.values() for c in x["score_calls"] if c["as_of"] < day]
    conversions = [x["trial_decision"].get("setup_score_components", {}).get("trial_conversion")
                   for x in valid_replays]
    conversions = [x for x in conversions if x]
    return {
        "day": day, "rows": len(details), "original_failed_count": document["meta"]["total"] - len(details),
        "archive_score_complete": len(available),
        "impulse_status_counts": dict(Counter(x["archive_trial"]["impulse"]["status"] for x in details.values())),
        "adjustment_status_counts": dict(Counter(x["original_quant"].get("adjustment_status", "UNKNOWN") for x in details.values())),
        "replay_usable": len(valid_replays), "replay_incomplete": len(details) - len(valid_replays),
        "production_output_changes": 0,
        "archive_vs_same_input_difference_rows": sum(bool(x["archive_baseline_difference_fields"]) for x in details.values()),
        "trial_stage_changes": sum(x["baseline_decision"]["structure_stage"] != x["trial_decision"]["structure_stage"] for x in valid_replays),
        "trial_group_changes": sum(digest(x["baseline_decision"]["contraction_group"]) != digest(x["trial_decision"]["contraction_group"]) for x in valid_replays),
        "archive_score_increases": sum(x["archive_trial"]["total"] > x["original_quant"]["structure_score"] for x in available),
        "archive_score_decreases": sum(x["archive_trial"]["total"] < x["original_quant"]["structure_score"] for x in available),
        "trial_basic_saturated": sum(x["archive_trial"]["basic"] >= 100 for x in available),
        "trial_total_above_100": sum(x["archive_trial"]["total"] > 100 for x in available),
        "trial_max_score": max((x["archive_trial"]["total"] for x in available), default=None),
        "historical_score_call_count": len(frozen_calls),
        "historical_call_dates": sorted(set(c["as_of"] for c in frozen_calls)),
        "signal_transitions": dict(transitions), "baseline_signal_grades": dict(baseline_quality),
        "trial_signal_grades": dict(trial_quality),
        "same_signal_anchor_date_changes": sum(
            x["baseline_decision"]["setup_structure_anchor_date"] != x["trial_decision"]["setup_structure_anchor_date"]
            for x in valid_replays if x["baseline_decision"]["setup_signal"] == x["trial_decision"]["setup_signal"] != "NONE"),
        "conversion_legacy_saturation_count": sum(x["legacy_would_saturate"] for x in conversions),
        "conversion_base_difference_count": sum(x["normalized_base"] != x["legacy_capped_base"] for x in conversions),
    }


def report(out, metas, calibration):
    lines = ["# 推进评分独立试算：基础100＋扩展12", "",
             "仅研究，生产评分未接入。预算：收缩阶段40、推进30、整理量能20、趋势5、位置5；原扩展另加0—12。",
             f"标定日：{calibration['calibration_date']}，完整去重锚点{len(calibration['samples'])}个。参数冻结后应用后续日；相邻日不是独立样本外验证。", "",
             "| 指标 | 起分值 | 满分值（标定p90） | 样本数 |", "|---|---:|---:|---:|"]
    for key, mapping in calibration["mappings"].items():
        lines.append(f"| {key} | {mapping['floor']:.4f} | {mapping['cap']:.4f} | {calibration['distributions'][key]['count']} |")
    lines.extend(["", "| 分析日 | 原成功记录 | 档案试算完整 | 重放可用 | 原输出变化 | 信号变化 | 新分数上限 |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for meta in metas:
        lines.append(f"| {meta['day']} | {meta['rows']} | {meta['archive_score_complete']} | {meta['replay_usable']} | 0 | {sum(meta['signal_transitions'].values())} | {meta['trial_max_score']:.2f} |")
    lines.extend(["", "## 阅读与边界", "",
                  "comparison CSV 的 archive_trial 是原权威分项的试算，baseline 为同输入原算法重算，replay 为独立内存新评分重放。排名按分数降序、代码升序破同分；全量排名与原有原组范围内排名分别记录。",
                  "112分先归一到100再按原0.6换算买点结构基础；旧0.6直接乘后封顶仅列诊断。原数值买点门槛固定，评级变化不等于已批准的新交易规则。",
                  "历史评分调用用对应锚点截止的行情重新计算推进与最低收盘保留；没有套用分析日当前推进。历史锚点套用标定日参数仅作敏感性研究，不代表参数当时已可得。",
                  "缺失评分的内部兼容占位仅为收集诊断；trial_decision=null并标明不可用，不输出有效新信号。原失败、复权PARTIAL/PENDING不补造或提升。",
                  "未取外部数据、未运行daily、未写生产库/JSON/CSV、未计算后续收益。生产参数和原函数保持不变，重放前后原输出逐条一致。",
                  "", "明细见 calibration.json、scores_*.json、comparison_*.csv、meta_*.json、manifest.json。"])
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", required=True)
    parser.add_argument("--calibration-date", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    parse = lambda value: datetime.strptime(value.strip(), "%y%m%d" if len(value.strip()) == 6 else "%Y-%m-%d").strftime("%Y-%m-%d")
    days = sorted(set(parse(value) for value in args.dates.split(",")))
    calibration_day = parse(args.calibration_date)
    if calibration_day != days[0]:
        parser.error("Calibration date must be the earliest requested analysis date")
    research_root = (ROOT / "reports/research").resolve()
    evidence_dir, out = (ROOT / args.evidence_dir).resolve(), (ROOT / args.output_dir).resolve()
    if out.exists() or out == research_root or not out.is_relative_to(research_root):
        parser.error("Output must be a fresh directory below reports/research")
    if not evidence_dir.is_relative_to(research_root) or not evidence_dir.is_dir():
        parser.error("Evidence input must be a reports/research directory")
    cfg = json.loads((ROOT / "strategies/02-quant-trial.json").read_text(encoding="utf-8"))
    validate_config(cfg)
    hashes = source_hashes()
    verification = json.loads((evidence_dir / "final_verification.json").read_text(encoding="utf-8"))
    for name, expected in verification["final_source_sha256"].items():
        if hashes[name] != expected:
            raise ValueError(f"Verified evidence producer changed: {name}")
    production_parameters = digest(quant.QUANT_STRATEGY)
    production_functions = (quant.score_setup, quant.finalize_setup_score)
    market = sqlite3.connect(f"file:{ROOT / 'cache/market_data/market_data.sqlite'}?mode=ro", uri=True)
    strategy = sqlite3.connect(f"file:{ROOT / 'cache/strategy/strategy_data.sqlite'}?mode=ro", uri=True)
    artifacts, manifest_inputs = {}, {}
    try:
        documents, evidence_by_day = {}, {}
        for day in days:
            stamp = day[2:].replace("-", "")
            path = evidence_dir / f"final_evidence_{stamp}.json"
            saved = json.loads(path.read_text(encoding="utf-8"))
            documents[day] = load_document(strategy, "quant", day)
            if documents[day] is None:
                raise ValueError(f"Missing authority Quant document: {day}")
            originals = {row["code"]: row for row in documents[day]["results"]}
            if set(saved) != set(originals):
                raise ValueError(f"Evidence universe differs from authority: {day}")
            for code, item in saved.items():
                if canonical(item["original_quant"]) != canonical(originals[code]):
                    raise ValueError(f"Authority record differs from evidence: {day} {code}")
            evidence_by_day[day] = saved
            manifest_inputs[day] = {"evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "authority_sha256": digest(documents[day])}
        calibration = calibrate([(code, x["impulse_evidence"]) for code, x in evidence_by_day[calibration_day].items()], cfg, calibration_day)
        calibration["config"] = cfg
        calibration["config_sha256"] = hashes["strategies/02-quant-trial.json"]
        frozen_calibration_hash = digest(calibration)
        shadow, calls = trial_module(cfg, calibration)
        for day in days:
            details = {}
            for ordinal, (code, saved) in enumerate(evidence_by_day[day].items(), 1):
                original = saved["original_quant"]
                scored = score_structure(original["score_components"], saved["impulse_evidence"], cfg, calibration)
                raw = pd.read_sql_query("SELECT trade_date AS date,open,high,low,close,volume,amount,source FROM daily_bars WHERE code=? AND trade_date<=? ORDER BY trade_date", market, params=(code, day))
                if raw.empty or str(raw.iloc[-1]["date"]) != day:
                    raise ValueError(f"Target bars missing: {day} {code}")
                actions = load_actions(market, code, day)
                adjusted, applied = apply_point_in_time_qfq(raw, actions, day)
                df = quant.calc_indicators(adjusted.reset_index(drop=True))
                if len(df) != original["data_days"] or abs(float(df.iloc[-1]["close"]) - original["close"]) >= .011:
                    raise ValueError(f"Authority price or history conflict: {day} {code}")
                df_before = df.copy(deep=True)
                baseline = quant.screen(df, code=code)
                calls.clear()
                trial = shadow.screen(df, code=code)
                baseline_after = quant.screen(df, code=code)
                if canonical(baseline) != canonical(baseline_after):
                    raise AssertionError(f"Production output mutated: {day} {code}")
                pd.testing.assert_frame_equal(df, df_before)
                baseline.pop("impulse_evidence", None)
                trial.pop("impulse_evidence", None)
                calls_saved = copy.deepcopy(calls)
                usable = all(call["score"]["status"] == "COMPLETE" for call in calls_saved)
                archive_differences = [key for key in baseline if key in original and canonical(baseline[key]) != canonical(original[key])]
                details[code] = {"original_quant": original, "archive_trial": scored,
                                 "baseline_decision": baseline, "trial_decision": trial if usable else None,
                                 "replay_usable": usable, "score_calls": calls_saved,
                                 "archive_baseline_difference_fields": archive_differences,
                                 "input_df_sha256": digest(df.to_json(orient="split", double_precision=15)),
                                 "applied_actions": applied,
                                 "original_score": original["structure_score"], "trial_score": scored["total"]}
                if ordinal % 50 == 0:
                    print(f"{day}: {ordinal}/{len(evidence_by_day[day])}, original output changes=0", flush=True)
            old_ranks, new_ranks = rank(details, "original_score"), rank(details, "trial_score")
            grouped = {code: x for code, x in details.items() if x["original_quant"].get("contraction_group")}
            group_old, group_new = rank(grouped, "original_score"), rank(grouped, "trial_score")
            rows = []
            for code, item in details.items():
                old, new, score = item["original_quant"], item["trial_decision"] or {}, item["archive_trial"]
                row = {"code": code, "name": old["name"], "original_stage": old["structure_stage"],
                       "original_score": item["original_score"], "archive_trial": item["trial_score"],
                       "basic": score["basic"], "extension": score["extension"],
                       "old_rank": old_ranks[code], "new_rank": new_ranks.get(code),
                       "rank_improvement": old_ranks[code] - new_ranks[code] if code in new_ranks else None,
                       "old_group_rank": group_old.get(code), "new_group_rank": group_new.get(code),
                       "quality": score["impulse"]["quality"], "retention": score["impulse"].get("retention"),
                       "base_date": score["impulse"].get("base_date"), "peak_date": score["impulse"].get("peak_date"),
                       "status": score["status"], "reasons": ";".join(score["reasons"]),
                       "adjustment_status": old.get("adjustment_status"), "replay_usable": item["replay_usable"],
                       "baseline_score": item["baseline_decision"]["structure_score"],
                       "replay_score": new.get("structure_score"), "archive_signal": old["setup_signal"],
                       "baseline_signal": item["baseline_decision"]["setup_signal"], "replay_signal": new.get("setup_signal"),
                       "baseline_grade": item["baseline_decision"]["setup_quality"], "replay_grade": new.get("setup_quality"),
                       "baseline_setup_score": item["baseline_decision"]["setup_score"], "replay_setup_score": new.get("setup_score"),
                       "replay_anchor_date": new.get("setup_structure_anchor_date"),
                       **{f"component_{key}": value for key, value in score["components"].items()}}
                rows.append(row)
            meta = summarize(day, details, documents[day])
            artifacts[day] = (details, rows, meta)
            print(canonical(meta), flush=True)
        if frozen_calibration_hash != digest(calibration):
            raise AssertionError("Calibration changed during validation")
        if (hashes != source_hashes() or production_parameters != digest(quant.QUANT_STRATEGY)
                or production_functions != (quant.score_setup, quant.finalize_setup_score)):
            raise AssertionError("Production source, parameters or functions changed during trial")
    finally:
        market.close()
        strategy.close()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "calibration.json", calibration)
    for day, (details, rows, meta) in artifacts.items():
        stamp = day[2:].replace("-", "")
        write_json(out / f"scores_{stamp}.json", details)
        write_json(out / f"meta_{stamp}.json", meta)
        with (out / f"comparison_{stamp}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(out / "manifest.json", {"source_sha256": hashes, "inputs": manifest_inputs,
                                       "config": cfg, "calibration_sha256": frozen_calibration_hash,
                                       "generated_at": datetime.now().isoformat(timespec="seconds"),
                                       "production_writes": False, "external_fetches": False})
    report(out, [value[2] for value in artifacts.values()], calibration)
    print(f"Isolated scoring output: {out}", flush=True)


if __name__ == "__main__":
    main()
