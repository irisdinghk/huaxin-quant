"""Shared price-volume qualification and nearby closing-low impulse bases."""

import copy
import math
from functools import partial

import numpy as np

from scripts.impulse_evidence import analyze_impulse_evidence, _number, _volume_evidence


def validate_selection_config(rule):
    if not isinstance(rule, dict) or type(rule.get("enabled", False)) is not bool:
        raise ValueError("Invalid impulse selection configuration")
    if not rule.get("enabled", False):
        return
    if (type(rule.get("lookback_days")) is not int or rule["lookback_days"] <= 0
            or type(rule.get("base_swing_window")) is not int or rule["base_swing_window"] <= 0
            or rule.get("same_peak_base") not in ("nearest", "max_gain", "coverage_nearest")
            or not isinstance(rule.get("diagnostic_lookback_days"), list)
            or any(type(value) is not int or value <= 0 for value in rule["diagnostic_lookback_days"])):
        raise ValueError("Invalid impulse search or base selection rule")
    ratio = rule.get("min_up_mean_volume_ratio")
    if (isinstance(ratio, bool) or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio) or ratio < 1):
        raise ValueError("Impulse qualification must require relative volume expansion")
    if rule["same_peak_base"] == "coverage_nearest":
        coverage = rule.get("min_advance_coverage")
        if (isinstance(coverage, bool) or not isinstance(coverage, (int, float))
                or not math.isfinite(coverage) or not 0 < coverage <= 1):
            raise ValueError("Impulse coverage must be finite and in (0, 1]")


def _base_lows(df, scan_start, start, window):
    # Import at call time: production Quant imports the legacy evidence module.
    from scripts.quant_filter import find_close_swings

    prefix = df.iloc[:start + 1]
    lows = {
        item["idx"]: {"confirmation_status": "CONFIRMED", "right_confirm_days": window,
                      "confirmed_date": str(prefix.iloc[item["idx"] + window]["date"])}
        for item in find_close_swings(prefix, lookback=start - scan_start + window + 1, window=window)
        if item["type"] == "low" and scan_start <= item["idx"] < start
    }
    closes = prefix["close"].to_numpy(dtype=float)
    for index in range(max(scan_start, window, start - window + 1), start):
        left, right = closes[index - window:index], closes[index + 1:start + 1]
        if (len(left) == window and len(right) > 0
                and (closes[index] < left).all() and (closes[index] < right).all()):
            lows[index] = {"confirmation_status": "PROVISIONAL", "right_confirm_days": len(right),
                           "confirmed_date": None}
    return lows


def select_candidates(df, candidates, scan_start, start, cfg, rule):
    """Return a qualified anchor and diagnostics using only the stage-start prefix."""
    prefix = df.iloc[:start + 1]
    lows = _base_lows(prefix, scan_start, start, rule["base_swing_window"])
    eligible, summaries, incomplete = [], [], False
    if rule["same_peak_base"] == "max_gain":
        rank = lambda item: (item["peak_idx"], item["gain_pct"], item["base_idx"])
    else:
        rank = lambda item: (item["peak_idx"], item["base_idx"])
    for item in sorted(candidates, key=rank, reverse=True):
        b, h = item["base_idx"], item["peak_idx"]
        low = lows.get(b)
        reasons = []
        volume = None
        if low is None:
            reasons.append("NOT_CLOSE_SWING_LOW")
        if item["gain_pct"] < cfg["min_gain_pct"]:
            reasons.append("PRICE_GAIN_BELOW_RESEARCH_THRESHOLD")
        if item["exhausted"]:
            reasons.append("ADVANCE_FULLY_RETRACED_BEFORE_GROUP")
        if not reasons:
            volume = _volume_evidence(prefix, b, h, cfg)
            warnings = set(volume["warnings"])
            ratio = volume["up_mean_volume_ratio"]
            if (volume["baseline_days"] != cfg["baseline_volume_days"] or ratio is None
                    or warnings.intersection({"VOLUME_DATA_INCOMPLETE", "BASELINE_INCOMPLETE",
                                              "ONE_PRICE_CHECK_INCOMPLETE"})):
                reasons.append("QUALIFICATION_VOLUME_INCOMPLETE")
                incomplete = True
            elif ratio <= rule["min_up_mean_volume_ratio"]:
                reasons.append("NO_RELATIVE_VOLUME_EXPANSION")
            else:
                eligible.append(item)
        base, peak = float(prefix.iloc[b]["close"]), float(prefix.iloc[h]["close"])
        bridge = float(prefix.iloc[h:start + 1]["close"].min())
        summaries.append({
            "base_date": str(prefix.iloc[b]["date"]), "peak_date": str(prefix.iloc[h]["date"]),
            "gain_pct": _number(item["gain_pct"]), "bridge_drawdown_pct": _number(item["bridge_drawdown_pct"]),
            "exhausted": item["exhausted"], "bridge_close_retention_pct": _number((bridge - base) / (peak - base) * 100),
            "base_confirmation": low, "volume": volume, "eligible": not reasons,
            "rejection_reasons": reasons,
        })
    coverage_diagnostics = {}
    if rule["same_peak_base"] == "coverage_nearest":
        threshold = rule["min_advance_coverage"]
        summary_by_base = {item["base_date"]: item for item in summaries}
        for summary in summaries:
            summary.update(price_volume_eligible=summary["eligible"], advance_coverage=None,
                           advance_price_span=None, max_qualified_advance_span=None)
        coverage_diagnostics = {"min_advance_coverage": threshold,
                                "coverage_status": "INCOMPLETE" if incomplete else "COMPLETE",
                                "price_volume_eligible_candidate_count": len(eligible),
                                "coverage_reference_by_peak": []}
        if not incomplete:
            references = {}
            for item in eligible:
                b, h = item["base_idx"], item["peak_idx"]
                span = float(prefix.iloc[h]["close"] - prefix.iloc[b]["close"])
                if h not in references or span > references[h]["span"]:
                    references[h] = {"span": span, "item": item, "count": 0}
            for item in eligible:
                b, h = item["base_idx"], item["peak_idx"]
                span = float(prefix.iloc[h]["close"] - prefix.iloc[b]["close"])
                reference = references[h]
                reference["count"] += 1
                summary = summary_by_base[str(prefix.iloc[b]["date"])]
                passes = span >= threshold * reference["span"]
                summary.update(advance_price_span=_number(span),
                               max_qualified_advance_span=_number(reference["span"]),
                               advance_coverage=_number(span / reference["span"]), eligible=passes)
                if not passes:
                    summary["rejection_reasons"].append("ADVANCE_COVERAGE_BELOW_THRESHOLD")
            eligible = [item for item in eligible
                        if summary_by_base[str(prefix.iloc[item["base_idx"]]["date"])]["eligible"]]
            for h, reference in sorted(references.items(), reverse=True):
                b = reference["item"]["base_idx"]
                coverage_diagnostics["coverage_reference_by_peak"].append({
                    "peak_date": str(prefix.iloc[h]["date"]), "peak_close": _number(prefix.iloc[h]["close"]),
                    "reference_base_date": str(prefix.iloc[b]["date"]),
                    "reference_base_close": _number(prefix.iloc[b]["close"]),
                    "max_advance_span": _number(reference["span"]), "qualified_count": reference["count"],
                })
    # Unknown eligibility must not silently become a rejection or a weaker anchor.
    selected = None if incomplete else (max(eligible, key=rank) if eligible else None)
    return {
        "selected": selected,
        "status": "DATA_ISSUE" if incomplete else ("IDENTIFIED" if selected else "NO_ADVANCE"),
        "reasons": (["QUALIFICATION_VOLUME_INCOMPLETE"] if incomplete else
                    ([] if selected else ["NO_QUALIFIED_PRICE_VOLUME_ADVANCE"])),
        "diagnostics": {"candidate_summaries": summaries, "eligible_candidate_count": len(eligible),
                        "base_candidate_count": len(lows),
                        "selection_rule": f"qualified_{rule['same_peak_base']}_close_low_v1",
                        "selected_base_confirmation": lows.get(selected["base_idx"]) if selected else None,
                        **coverage_diagnostics},
    }


def analyze_selected_impulse(df, structure, evidence_cfg, trial_cfg):
    rule = trial_cfg.get("impulse_selection", {})
    validate_selection_config(rule)
    if not rule.get("enabled", False):
        return analyze_impulse_evidence(df, structure, evidence_cfg)
    config = copy.deepcopy(evidence_cfg)
    config["lookback_days"] = rule["lookback_days"]
    config["diagnostic_lookback_days"] = list(rule["diagnostic_lookback_days"])
    config["base_context_days"] = rule["base_swing_window"]
    selector = partial(select_candidates, rule=rule)
    result = analyze_impulse_evidence(df, structure, config, candidate_selector=selector)
    result["selection_config"] = copy.deepcopy(rule)
    return result
