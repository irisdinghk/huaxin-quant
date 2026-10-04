"""Read-only full Quant replay for research-only impulse evidence and legacy compatibility."""

import argparse
import copy
import csv
import hashlib
import json
import os
import sqlite3
import subprocess
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
from scripts.impulse_evidence import analyze_impulse_evidence, empty_impulse_evidence

SOURCE = Path(quant.__file__).resolve().parents[1]


def _git_text(ref, name):
    return subprocess.check_output(["git", "-C", str(SOURCE), "show", f"{ref}:{name}"], text=True)


def _baseline(ref):
    old_config = json.loads(_git_text(ref, "strategies/02-quant.json"))
    current_config = copy.deepcopy(quant.QUANT_STRATEGY)
    current_config["vcp"].pop("impulse_evidence", None)
    for config in [old_config, current_config]:
        config.pop("strategy_version", None)
        config.pop("description", None)
    if old_config != current_config:
        raise ValueError("Existing strategy parameters differ from baseline; isolate those changes first")
    source = _git_text(ref, "scripts/quant_filter.py")
    module = types.ModuleType("quant_impulse_baseline")
    module.__file__ = str(ROOT / "scripts" / "quant_filter.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, hashlib.sha256(source.encode()).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _hashes():
    names = ["scripts/quant_filter.py", "scripts/impulse_evidence.py", "strategies/02-quant.json"]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def _validate(evidence, df, group):
    if not group or evidence["status"] in {"DATA_ISSUE", "GROUP_ANCHOR_MISMATCH"}:
        return
    if evidence["as_of"] != str(df.iloc[-1]["date"]):
        raise AssertionError("Evidence date differs from target")
    for window in evidence["windows"].values():
        anchor = window["anchor"]
        if anchor is None:
            continue
        b, h, s = (anchor[key] for key in ["base_idx", "peak_idx", "group_start_idx"])
        if not 0 <= b < h <= s < len(df):
            raise AssertionError("Impulse anchors are not ordered")
        if float(df.iloc[h]["close"]) <= float(df.iloc[b]["close"]):
            raise AssertionError("Impulse is not a positive price advance")
        baseline_end = window["volume"]["baseline_end_date"]
        if baseline_end and baseline_end >= anchor["base_date"]:
            raise AssertionError("Volume baseline overlaps the impulse")
        for path in window["retention"]["rounds"]:
            if path["end_date"] > evidence["as_of"]:
                raise AssertionError("Retention uses future data")
        # Removing all bars after S must leave every selected anchor unchanged.
        prefix_structure = {"contraction_group": [{
            "start_idx": s, "end_idx": s, "start_date": anchor["group_start_date"],
            "end_date": anchor["group_start_date"],
        }]}
        prefix = analyze_impulse_evidence(df.iloc[:s + 1], prefix_structure, quant.IMPULSE_EVIDENCE_CFG)
        if prefix["windows"][str(window["lookback_days"])]["anchor"] != anchor:
            raise AssertionError("Later bars changed the impulse anchor")


def _row(code, name, original, evidence, legacy_differences, archive_differences):
    selected = evidence.get("selected") or {}
    anchor, price, volume, retention = (selected.get(key) or {} for key in ["anchor", "price", "volume", "retention"])
    row = {
        "code": code, "name": name, "original_stage": original["structure_stage"],
        "original_score": original["structure_score"], "original_setup": original["setup_signal"],
        "original_tag": original.get("prior_breakout_context_tag", ""),
        "status": evidence["status"], "group_start": evidence["group_start_date"],
        "base_date": anchor.get("base_date"), "base_close": anchor.get("base_close"),
        "peak_date": anchor.get("peak_date"), "peak_close": anchor.get("peak_close"),
        "gain_pct": price.get("gain_pct"), "efficiency": price.get("path_efficiency"),
        "peak_lead_days": price.get("peak_lead_days"),
        "up_peak_volume_ratio": volume.get("up_peak_volume_ratio"),
        "up_peak_volume_confirmed": volume.get("up_peak_volume_confirmed"),
        "down_peak_volume_ratio": volume.get("down_peak_volume_ratio"),
        "peak_volume_date": volume.get("peak_volume_date"),
        "peak_volume_day_return_pct": volume.get("peak_volume_day_return_pct"),
        "worst_close_retention_pct": (retention.get("post_peak") or {}).get("close_retention_pct"),
        "current_close_retention_pct": retention.get("current_close_retention_pct"),
        "rounds": _json(retention.get("rounds", [])), "reasons": ";".join(evidence["reasons"]),
        "legacy_field_differences": ";".join(legacy_differences),
        "archive_baseline_differences": ";".join(archive_differences),
    }
    for n, value in evidence.get("windows", {}).items():
        a, p = value.get("anchor") or {}, value.get("price") or {}
        row.update({f"w{n}_status": value["status"], f"w{n}_base": a.get("base_date"),
                    f"w{n}_peak": a.get("peak_date"), f"w{n}_gain_pct": p.get("gain_pct")})
    return row


def verify_report(report_dir):
    """Verify final helper output against a saved full replay without rewriting that replay."""
    if not report_dir.is_relative_to((ROOT / "reports/research").resolve()) or not report_dir.is_dir():
        raise ValueError("Verification requires an existing reports/research directory")
    target = report_dir / "final_verification.json"
    if target.exists():
        raise ValueError("Final verification already exists; do not overwrite audit history")
    hashes, checked = _hashes(), []
    market = sqlite3.connect(f"file:{ROOT / 'cache/market_data/market_data.sqlite'}?mode=ro", uri=True)
    try:
        for meta_path in sorted(report_dir.glob("meta_*.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            day = meta["day"]
            for name in ["scripts/quant_filter.py", "strategies/02-quant.json"]:
                if hashes[name] != meta["current_source_sha256"][name]:
                    raise AssertionError("Legacy screen/config changed; a fresh full replay is required")
            if meta["config"] != quant.IMPULSE_EVIDENCE_CFG or meta["legacy_changed"] or meta["legacy_checked"] != meta["rows"]:
                raise AssertionError("Initial replay was not complete or configuration changed")
            details = json.loads((report_dir / meta_path.name.replace("meta_", "evidence_")).read_text(encoding="utf-8"))
            for ordinal, (code, saved) in enumerate(details.items(), 1):
                original = saved["original_quant"]
                raw = pd.read_sql_query(
                    "SELECT trade_date AS date,open,high,low,close,volume,amount,source FROM daily_bars "
                    "WHERE code=? AND trade_date<=? ORDER BY trade_date", market, params=(code, day))
                adjusted, _ = apply_point_in_time_qfq(raw, load_actions(market, code, day), day)
                df = quant.calc_indicators(adjusted.reset_index(drop=True))
                actual = analyze_impulse_evidence(df, original, quant.IMPULSE_EVIDENCE_CFG)
                if _json(actual) != _json(saved["impulse_evidence"]):
                    raise AssertionError(f"Final helper evidence changed: {day} {code}")
                if hashlib.sha256(_json(saved["baseline_decision"]).encode()).hexdigest() != saved["legacy_output_sha256"]:
                    raise AssertionError("Saved legacy comparison fingerprint mismatch")
                if ordinal % 100 == 0:
                    print(f"Final helper verification {day}: {ordinal}/{len(details)}", flush=True)
            checked.append({"day": day, "rows": len(details), "research_output_changed": 0,
                            "legacy_changed": 0, "initial_source_sha256": meta["current_source_sha256"]})
    finally:
        market.close()
    if not checked or hashes != _hashes():
        raise AssertionError("No audit inputs or source changed while verifying")
    payload = {"checked": checked, "final_source_sha256": hashes,
               "generated_at": datetime.now().isoformat(timespec="seconds"),
               "method": "Final helper independently recalculated from read-only bars; all objects equal to saved replay; legacy screen/config hashes unchanged"}
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(_json(payload), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", help="Comma-separated YYMMDD or YYYY-MM-DD dates")
    parser.add_argument("--output-dir", help="New, unused directory below reports/research")
    parser.add_argument("--verify-report", help="Verify final helper against an existing isolated replay")
    parser.add_argument("--baseline-ref", default="4509413", help="Git source/config baseline")
    args = parser.parse_args()
    if args.verify_report:
        if args.dates or args.output_dir:
            parser.error("Use --verify-report separately from --dates/--output-dir")
        verify_report((ROOT / args.verify_report).resolve())
        return
    if not args.dates or not args.output_dir:
        parser.error("A fresh replay requires both --dates and --output-dir")
    dates = list(dict.fromkeys(datetime.strptime(x.strip(), "%y%m%d" if len(x.strip()) == 6 else "%Y-%m-%d").strftime("%Y-%m-%d")
                               for x in args.dates.split(",")))
    out = (ROOT / args.output_dir).resolve()
    if not out.is_relative_to((ROOT / "reports" / "research").resolve()) or out.exists():
        parser.error("Output must be a new directory below reports/research; existing data is never overwritten")
    baseline, baseline_hash = _baseline(args.baseline_ref)
    hashes = _hashes()
    artifacts = {}
    market = sqlite3.connect(f"file:{ROOT / 'cache/market_data/market_data.sqlite'}?mode=ro", uri=True)
    strategy = sqlite3.connect(f"file:{ROOT / 'cache/strategy/strategy_data.sqlite'}?mode=ro", uri=True)
    try:
        for day in dates:
            document = load_document(strategy, "quant", day)
            if document is None:
                raise ValueError(f"No original Quant document for {day}")
            rows, details = [], {}
            archive_counts = Counter()
            for ordinal, original in enumerate(document["results"], 1):
                code = original["code"]
                raw = pd.read_sql_query(
                    "SELECT trade_date AS date,open,high,low,close,volume,amount,source FROM daily_bars "
                    "WHERE code=? AND trade_date<=? ORDER BY trade_date", market, params=(code, day))
                if raw.empty or str(raw.iloc[-1]["date"]) != day:
                    evidence = empty_impulse_evidence("DATA_ISSUE", ["TARGET_BAR_MISSING"])
                    rows.append(_row(code, original["name"], original, evidence, [], []))
                    details[code] = {"original_quant": original, "impulse_evidence": evidence}
                    continue
                adjusted, applied = apply_point_in_time_qfq(raw, load_actions(market, code, day), day)
                df = quant.calc_indicators(adjusted.reset_index(drop=True))
                if len(df) != original["data_days"] or abs(float(df.iloc[-1]["close"]) - original["close"]) >= .011:
                    raise ValueError(f"Original Quant price/history conflict: {day} {code}")
                old_group = copy.deepcopy(original)
                evidence = analyze_impulse_evidence(df, original, quant.IMPULSE_EVIDENCE_CFG)
                if original != old_group:
                    raise AssertionError("Research analysis mutated original Quant")
                _validate(evidence, df, original.get("contraction_group") or [])
                current = quant.screen(df, code=code)
                screen_evidence = current.pop("impulse_evidence")
                old_screen = baseline.screen(df, code=code)
                differences = [key for key in sorted(set(current) | set(old_screen))
                               if _json(current.get(key)) != _json(old_screen.get(key))]
                if differences:
                    raise AssertionError(f"Legacy output changed: {day} {code} {differences}")
                archive_differences = [key for key, value in old_screen.items()
                                       if key in original and _json(value) != _json(original[key])]
                archive_counts.update(archive_differences)
                rows.append(_row(code, original["name"], original, evidence, differences, archive_differences))
                details[code] = {
                    "original_quant": original, "impulse_evidence": evidence,
                    "baseline_decision": old_screen, "current_screen_evidence": screen_evidence,
                    "legacy_field_differences": differences,
                    "legacy_output_sha256": hashlib.sha256(_json(current).encode()).hexdigest(),
                    "archive_baseline_differences": archive_differences, "applied_actions": applied,
                }
                if ordinal % 50 == 0:
                    print(f"{day}: {ordinal}/{len(document['results'])}, legacy differences=0", flush=True)
            grouped = [row for row in rows if row["status"] not in {"NO_GROUP", "DATA_ISSUE", "GROUP_ANCHOR_MISMATCH"}]
            meta = {
                "day": day, "original_meta": document["meta"], "original_stats": document["stats"],
                "rows": len(rows), "failed_input_count": document["meta"]["total"] - len(rows),
                "status_counts": dict(Counter(row["status"] for row in rows)),
                "legacy_checked": sum("baseline_decision" in x for x in details.values()),
                "legacy_changed": 0, "archive_baseline_difference_fields": dict(archive_counts),
                "archive_baseline_difference_rows": sum(bool(x.get("archive_baseline_differences")) for x in details.values()),
                "up_volume_confirmed_identified": sum(row["status"] == "IDENTIFIED" and row["up_peak_volume_confirmed"] is True for row in rows),
                "scan_boundary_rows": sum("BASE_AT_SCAN_BOUNDARY" in row["reasons"] for row in rows),
                "peak_down_day_rows": sum((row.get("peak_volume_day_return_pct") or 0) < 0 for row in grouped),
                "config": quant.IMPULSE_EVIDENCE_CFG, "baseline_ref": args.baseline_ref,
                "baseline_source_sha256": baseline_hash, "current_source_sha256": hashes,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "method": "Read-only original Quant groups; ordered research metrics; full screen comparison against frozen source with identical legacy parameters",
            }
            artifacts[day] = (rows, details, meta)
            print(_json(meta), flush=True)
    finally:
        market.close()
        strategy.close()
    if hashes != _hashes():
        raise AssertionError("Source/config changed while audit was running")
    out.mkdir(parents=True, exist_ok=False)
    for day, (rows, details, meta) in artifacts.items():
        stamp = day[2:].replace("-", "")
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with (out / f"fields_{stamp}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        for name, payload in [("evidence", details), ("meta", meta)]:
            (out / f"{name}_{stamp}.json").write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    print(f"Isolated research output: {out}")


if __name__ == "__main__":
    main()
