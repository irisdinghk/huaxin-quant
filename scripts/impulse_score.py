"""Shared deterministic impulse scoring for explicit research or production use."""

import math

import numpy as np


def number(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def clamp(value, low, high):
    return max(low, min(high, value))


def validate_config(cfg):
    research = cfg.get("research_only")
    if research is not True and not (
            research is False and cfg.get("execution_purpose") == "production"):
        raise ValueError("Scoring use must be explicitly research or production")
    if cfg.get("quality_curve", "linear") not in {"linear", "sqrt"}:
        raise ValueError("Unsupported quality curve")
    if cfg.get("retention_curve", "linear") not in {"linear", "quadratic_drawback"}:
        raise ValueError("Unsupported retention curve")
    overrides = cfg.get("extension_score_overrides", {})
    if (not isinstance(overrides, dict)
            or any(key not in {"CONFIRMED_RESET_CONTRACTION", "TERMINAL_MICRO_CONTRACTION"}
                   or number(value) is None or not 0 <= value <= cfg["budget"]["extension"]
                   for key, value in overrides.items())):
        raise ValueError("Invalid extension score override")
    budget = cfg["budget"]
    if (sum(budget[key] for key in ["structure", "volume", "impulse", "trend", "position"]) != 100
            or budget["extension"] != 12 or any(number(v) is None or v <= 0 for v in budget.values())):
        raise ValueError("Budget must be 100 basic points plus 12 extension points")
    if "stage_scores" in cfg:
        stages = cfg["stage_scores"]
        vcp = ["VCP_EARLY", "VCP_FORMING", "VCP_MATURE", "VCP_TIGHT"]
        if (not isinstance(stages, dict) or set(stages) != set(vcp)
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or number(v) is None or not 0 <= v <= budget["structure"] for v in stages.values())
                or any(stages[a] >= stages[b] for a, b in zip(vcp, vcp[1:]))
                or stages["VCP_TIGHT"] != budget["structure"]):
            raise ValueError("Invalid direct stage scores")
    bonus = cfg.get("contraction_quality_bonus", {})
    if (not isinstance(bonus, dict) or type(bonus.get("enabled", False)) is not bool
            or budget.get("contraction_quality", 0) not in {0, 6}
            or (bonus.get("enabled") and (bonus.get("score") != 6 or budget.get("contraction_quality") != 6))):
        raise ValueError("Invalid contraction quality bonus")
    terminal = cfg.get("terminal_micro_bonus", {})
    if (not isinstance(terminal, dict) or type(terminal.get("enabled", False)) is not bool
            or (terminal.get("enabled") and (
                terminal.get("score") != 6 or terminal.get("min_days") != 2
                or terminal.get("max_days") != 5
                or number(terminal.get("min_pivot_distance_pct")) is None
                or number(terminal.get("max_pivot_distance_pct")) is None
                or terminal["min_pivot_distance_pct"] > terminal["max_pivot_distance_pct"]))):
        raise ValueError("Invalid terminal micro bonus")
    keys = {"gain_pct", "up_volume_ratio", "up_down_volume_ratio", "path_efficiency", "speed_pct"}
    weights = cfg["quality_weights"]
    if (set(weights) != keys or abs(sum(weights.values()) - 1) > 1e-9
            or any(number(v) is None or v < 0 for v in weights.values())):
        raise ValueError("Quality weights must sum to one")
    if set(cfg["calibration"]["floors"]) != keys - {"path_efficiency"}:
        raise ValueError("Calibration metric set is invalid")
    if not 0 < cfg["calibration"]["quantile"] < 1 or cfg["calibration"]["min_samples"] < 1:
        raise ValueError("Invalid calibration quantile or sample count")
    if any(number(v) is None for v in cfg["calibration"]["floors"].values()):
        raise ValueError("Invalid mapping floor")
    if any(number(cfg["legacy_caps"].get(k)) is None or cfg["legacy_caps"][k] <= 0
           for k in ["structure", "volume", "trend", "position"]):
        raise ValueError("Invalid legacy cap")
    if (cfg["setup_conversion"] not in {"normalize_112_to_100", "normalize_total_to_100"}
            or (cfg["setup_conversion"] == "normalize_112_to_100" and budget.get("contraction_quality", 0))):
        raise ValueError("Unsupported setup conversion")


def impulse_metrics(evidence, cfg):
    status = evidence.get("status")
    selected = evidence.get("selected") or {}
    if status in {"NO_GROUP", "NO_ADVANCE"}:
        return {"status": "NO_IMPULSE", "metrics": {}, "reasons": [status]}
    anchor = selected.get("anchor") or {}
    price = selected.get("price") or {}
    volume = selected.get("volume") or {}
    reasons = []
    base, peak = number(anchor.get("base_close")), number(anchor.get("peak_close"))
    days, efficiency = number(price.get("advance_days")), number(price.get("path_efficiency"))
    if (status in {"DISABLED", "DATA_ISSUE", "GROUP_ANCHOR_MISMATCH"}
            or base is None or peak is None or base <= 0 or peak <= base
            or days is None or days < 1 or days != int(days)
            or efficiency is None or not 0 <= efficiency <= 1):
        return {"status": "INCOMPLETE", "metrics": {}, "reasons": ["PRICE_OR_ANCHOR_INVALID"]}
    up, down = number(volume.get("up_mean_volume_ratio")), number(volume.get("down_mean_volume_ratio"))
    down_days = number(volume.get("down_days"))
    if (volume.get("baseline_days") != cfg["baseline_volume_days"]
            or "VOLUME_DATA_INCOMPLETE" in volume.get("warnings", [])
            or up is None or up <= 0 or down_days is None or down_days < 0
            or down_days != int(down_days)):
        reasons.append("VOLUME_OR_BASELINE_INCOMPLETE")
    elif down_days > 0 and (down is None or down <= 0):
        reasons.append("DOWN_VOLUME_INCOMPLETE")
    rounds = (selected.get("retention") or {}).get("rounds") or []
    lows = [number(item.get("close_low")) for item in rounds]
    if not lows or any(v is None or v <= 0 for v in lows):
        reasons.append("CONTRACTION_CLOSE_LOW_INCOMPLETE")
    if reasons:
        return {"status": "INCOMPLETE", "metrics": {}, "reasons": reasons}
    low_item = min(rounds, key=lambda item: float(item["close_low"]))
    raw_retention = (float(low_item["close_low"]) - base) / (peak - base)
    metrics = {
        "gain_pct": (peak / base - 1) * 100,
        "up_volume_ratio": up,
        "up_down_volume_ratio": up / down if down_days > 0 else None,
        "path_efficiency": efficiency,
        "speed_pct": math.expm1(math.log(peak / base) / days) * 100,
    }
    if any(v is not None and not math.isfinite(v) for v in metrics.values()):
        return {"status": "INCOMPLETE", "metrics": {}, "reasons": ["NONFINITE_METRIC"]}
    return {
        "status": "COMPLETE", "metrics": metrics, "reasons": [],
        "anchor_id": anchor.get("anchor_id"),
        "base_date": anchor.get("base_date"), "peak_date": anchor.get("peak_date"),
        "base_close": base, "peak_close": peak,
        "retention_raw": raw_retention, "retention": clamp(raw_retention, 0, 1),
        "retention_low_close": low_item["close_low"],
        "retention_low_date": low_item["close_low_date"],
        "no_down_days": down_days == 0,
        "warnings": list(evidence.get("reasons", [])),
    }


def calibrate(samples, cfg, day):
    """Use only the supplied calibration date, with one observation per code/B/H."""
    validate_config(cfg)
    selected, seen = [], set()
    for code, evidence in samples:
        if evidence.get("as_of") != day:
            raise ValueError("Calibration evidence must be from the declared date")
        values = impulse_metrics(evidence, cfg)
        if values["status"] != "COMPLETE":
            continue
        key = (code, values["base_date"], values["peak_date"])
        if key in seen:
            continue
        seen.add(key)
        selected.append({"code": code, **values})
    mappings, distributions = {}, {}
    for key, floor in cfg["calibration"]["floors"].items():
        values = [x["metrics"][key] for x in selected if x["metrics"][key] is not None]
        if len(values) < cfg["calibration"]["min_samples"]:
            raise ValueError(f"Insufficient calibration samples: {key} ({len(values)})")
        cap = float(np.quantile(values, cfg["calibration"]["quantile"], method="linear"))
        if cap <= floor:
            raise ValueError(f"Calibration cap does not exceed floor: {key}")
        mappings[key] = {"floor": floor, "cap": cap}
        distributions[key] = {"count": len(values), "min": min(values), "max": max(values),
                              **{f"p{p}": float(np.quantile(values, p / 100, method="linear"))
                                 for p in [10, 25, 50, 75, 90, 95]}}
    return {"schema": "impulse_score_calibration_v1", "calibration_date": day,
            "mappings": mappings, "distributions": distributions, "samples": selected,
            "method": "Fixed linear floors to calibration-day p90; no return labels"}


def score_impulse(evidence, cfg, calibration):
    validate_config(cfg)
    values = impulse_metrics(evidence, cfg)
    quality_curve = cfg.get("quality_curve", "linear")
    retention_curve = cfg.get("retention_curve", "linear")
    result = {**values, "quality": None, "contribution": None, "quality_components": {},
              "strategy_version": cfg.get("strategy_version"),
              "quality_curve": quality_curve, "retention_curve": retention_curve,
              "retention_coefficient": None, "retention_discount": None}
    if values["status"] == "NO_IMPULSE":
        result.update(quality=0, contribution=0)
        return result
    if values["status"] != "COMPLETE":
        return result
    scores = {"path_efficiency": values["metrics"]["path_efficiency"] * 100}
    for key, mapping in calibration["mappings"].items():
        floor, cap = mapping["floor"], mapping["cap"]
        if number(floor) is None or number(cap) is None or cap <= floor:
            raise ValueError(f"Invalid frozen mapping: {key}")
        value = values["metrics"][key]
        scores[key] = clamp((value - floor) / (cap - floor), 0, 1) * 100 if value is not None else None
    linear_scores = dict(scores)
    if quality_curve == "sqrt":
        scores = {key: (100 * math.sqrt(value / 100)
                        if key != "path_efficiency" and value is not None else value)
                  for key, value in scores.items()}
    weights = dict(cfg["quality_weights"])
    if values["no_down_days"]:
        weights["up_volume_ratio"] += weights["up_down_volume_ratio"]
        weights["up_down_volume_ratio"] = 0
        result["warnings"] = [*values["warnings"], "NO_DOWN_DAYS_VOLUME_WEIGHT_REALLOCATED"]
    components = {key: (scores[key] or 0) * weight for key, weight in weights.items()}
    quality = sum(components.values())
    retained = values["retention"]
    coefficient = 1 - (1 - retained) ** 2 if retention_curve == "quadratic_drawback" else retained
    result.update(quality=quality, quality_components=components, indicator_scores=scores,
                  linear_indicator_scores=linear_scores,
                  retention_coefficient=coefficient, retention_discount=1 - coefficient,
                  contribution=cfg["budget"]["impulse"] * quality / 100 * coefficient)
    return result


def score_structure(legacy_components, evidence, cfg, calibration, extension_details=None,
                    contraction_quality=None, terminal_micro=None, structure_stage=None):
    validate_config(cfg)
    impulse = score_impulse(evidence, cfg, calibration)
    components, reasons = {}, list(impulse["reasons"])
    stage_policy = "legacy_proportional"
    for key in ["structure", "volume", "trend", "position"]:
        if key == "structure" and "stage_scores" in cfg:
            non_vcp = {"TREND_WATCH", "TREND_REBUILD", "POST_BREAKOUT", "POST_BREAKOUT_FAILED",
                       "POST_BREAKOUT_EXPIRED", "REJECT", "NONE", "DATA_INSUFFICIENT", "DATA_ISSUE"}
            if not isinstance(structure_stage, str) or structure_stage not in set(cfg["stage_scores"]) | non_vcp:
                components[key] = None
                reasons.append("STRUCTURE_STAGE_MISSING_OR_UNKNOWN")
                continue
            if structure_stage in cfg["stage_scores"]:
                components[key] = cfg["stage_scores"][structure_stage]
                stage_policy = "direct_stage_scores"
                continue
        value = number(legacy_components.get(key))
        if value is None:
            reasons.append(f"LEGACY_COMPONENT_MISSING:{key}")
            components[key] = None
        else:
            components[key] = value * cfg["budget"][key] / cfg["legacy_caps"][key]
    extension = number(legacy_components.get("contraction_extensions"))
    extension_original = extension
    adjustments = []
    if extension is None:
        reasons.append("LEGACY_EXTENSION_MISSING")
    elif cfg.get("extension_score_overrides") or cfg.get("terminal_micro_bonus", {}).get("enabled"):
        if extension_details is None:
            if extension > 0:
                reasons.append("EXTENSION_DETAILS_MISSING")
                extension = None
        elif not isinstance(extension_details, list):
            reasons.append("EXTENSION_DETAILS_INVALID")
            extension = None
        else:
            seen, old_sum, new_sum = set(), 0, 0
            for item in extension_details:
                tag = item.get("type") if isinstance(item, dict) else None
                old_score = number(item.get("score")) if isinstance(item, dict) else None
                if not isinstance(tag, str) or tag in seen or old_score is None or old_score < 0:
                    reasons.append("EXTENSION_DETAILS_INVALID")
                    extension = None
                    break
                seen.add(tag)
                new_score = cfg["extension_score_overrides"].get(tag, old_score)
                old_sum += old_score
                new_sum += new_score
                adjustments.append({"type": tag, "original_score": old_score, "trial_score": new_score})
            else:
                if not math.isclose(clamp(old_sum, 0, cfg["budget"]["extension"]), extension):
                    reasons.append("EXTENSION_TOTAL_MISMATCH")
                    extension = None
                else:
                    extension = new_sum
    if cfg.get("terminal_micro_bonus", {}).get("enabled"):
        terminal_score = number((terminal_micro or {}).get("score"))
        if (not isinstance(terminal_micro, dict) or terminal_micro.get("status") != "COMPLETE"
                or terminal_score not in {0, cfg["terminal_micro_bonus"]["score"]}):
            reasons.append("TERMINAL_MICRO_INCOMPLETE")
            extension = None
        elif extension is not None:
            prior = next((item for item in adjustments if item["type"] == "TERMINAL_MICRO_CONTRACTION"), None)
            extension = extension - (prior["trial_score"] if prior else 0) + terminal_score
            if prior:
                prior["trial_score"] = terminal_score
            elif terminal_score:
                adjustments.append({"type": "TERMINAL_MICRO_CONTRACTION", "original_score": 0,
                                    "trial_score": terminal_score})
    components["impulse"] = impulse["contribution"]
    components["contraction_extensions"] = clamp(extension, 0, cfg["budget"]["extension"]) if extension is not None else None
    quality_score = 0
    if cfg.get("contraction_quality_bonus", {}).get("enabled"):
        quality_score = number((contraction_quality or {}).get("score"))
        if (not isinstance(contraction_quality, dict) or contraction_quality.get("status") != "COMPLETE"
                or quality_score not in {0, cfg["contraction_quality_bonus"]["score"]}):
            quality_score = None
            reasons.append("CONTRACTION_QUALITY_INCOMPLETE")
    components["contraction_quality"] = quality_score
    complete = not reasons or impulse["status"] == "NO_IMPULSE" and len(reasons) == 1
    complete = complete and all(v is not None for v in components.values())
    basic_raw = sum(components[k] for k in ["structure", "volume", "trend", "position", "impulse"]) if complete else None
    basic = clamp(basic_raw, 0, 100) if complete else None
    return {"schema": "impulse_structure_score_v1", "research_only": cfg["research_only"],
            "strategy_version": cfg.get("strategy_version"),
            "structure_stage": structure_stage,
            "stage_score_policy": stage_policy,
            "status": "COMPLETE" if complete else "INCOMPLETE", "reasons": reasons,
            "components": components, "basic_raw": basic_raw, "basic": basic,
            "extension": components["contraction_extensions"],
            "extension_original": extension_original, "extension_score_adjustments": adjustments,
            "contraction_quality": contraction_quality,
            "terminal_micro": terminal_micro,
            "score_maximum": 100 + cfg["budget"]["extension"] + cfg["budget"].get("contraction_quality", 0),
            "total": basic + components["contraction_extensions"] + quality_score if complete else None,
            "impulse": impulse}


def setup_conversion(total, quality_cfg, structure_max=112):
    if (number(structure_max) is None or structure_max <= 0
            or number(total) is None or not 0 <= total <= structure_max):
        raise ValueError("Expected complete structure score within its configured maximum")
    weight, low, high = quality_cfg.get("weight", .6), quality_cfg.get("min", 0), quality_cfg.get("max", 60)
    return {"raw_total": total, "structure_maximum": structure_max,
            "normalized_structure": total * 100 / structure_max,
            "normalized_base": clamp(int(round(total * 100 / structure_max * weight)), low, high),
            "legacy_capped_base": clamp(int(round(total * weight)), low, high),
            "legacy_would_saturate": total * weight > high}
