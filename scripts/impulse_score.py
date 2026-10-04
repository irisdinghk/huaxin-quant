"""Pure research scoring; deliberately not imported by the production Quant path."""

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
    if cfg.get("research_only") is not True:
        raise ValueError("Trial configuration must be research-only")
    budget = cfg["budget"]
    if (sum(budget[key] for key in ["structure", "volume", "impulse", "trend", "position"]) != 100
            or budget["extension"] != 12 or any(number(v) is None or v <= 0 for v in budget.values())):
        raise ValueError("Budget must be 100 basic points plus 12 extension points")
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
    if cfg["setup_conversion"] != "normalize_112_to_100":
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
    values = impulse_metrics(evidence, cfg)
    result = {**values, "quality": None, "contribution": None, "quality_components": {}}
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
    weights = dict(cfg["quality_weights"])
    if values["no_down_days"]:
        weights["up_volume_ratio"] += weights["up_down_volume_ratio"]
        weights["up_down_volume_ratio"] = 0
        result["warnings"] = [*values["warnings"], "NO_DOWN_DAYS_VOLUME_WEIGHT_REALLOCATED"]
    components = {key: (scores[key] or 0) * weight for key, weight in weights.items()}
    quality = sum(components.values())
    result.update(quality=quality, quality_components=components, indicator_scores=scores,
                  contribution=cfg["budget"]["impulse"] * quality / 100 * values["retention"])
    return result


def score_structure(legacy_components, evidence, cfg, calibration):
    validate_config(cfg)
    impulse = score_impulse(evidence, cfg, calibration)
    components, reasons = {}, list(impulse["reasons"])
    for key in ["structure", "volume", "trend", "position"]:
        value = number(legacy_components.get(key))
        if value is None:
            reasons.append(f"LEGACY_COMPONENT_MISSING:{key}")
            components[key] = None
        else:
            components[key] = value * cfg["budget"][key] / cfg["legacy_caps"][key]
    extension = number(legacy_components.get("contraction_extensions"))
    if extension is None:
        reasons.append("LEGACY_EXTENSION_MISSING")
    components["impulse"] = impulse["contribution"]
    components["contraction_extensions"] = clamp(extension, 0, cfg["budget"]["extension"]) if extension is not None else None
    complete = not reasons or impulse["status"] == "NO_IMPULSE" and len(reasons) == 1
    complete = complete and all(v is not None for v in components.values())
    basic_raw = sum(components[k] for k in ["structure", "volume", "trend", "position", "impulse"]) if complete else None
    basic = clamp(basic_raw, 0, 100) if complete else None
    return {"schema": "impulse_structure_score_v1", "research_only": True,
            "status": "COMPLETE" if complete else "INCOMPLETE", "reasons": reasons,
            "components": components, "basic_raw": basic_raw, "basic": basic,
            "extension": components["contraction_extensions"],
            "total": basic + components["contraction_extensions"] if complete else None,
            "impulse": impulse}


def setup_conversion(total, quality_cfg):
    if number(total) is None or not 0 <= total <= 112:
        raise ValueError("Expected complete structure score within 0..112")
    weight, low, high = quality_cfg.get("weight", .6), quality_cfg.get("min", 0), quality_cfg.get("max", 60)
    return {"raw_total": total, "normalized_structure": total * 100 / 112,
            "normalized_base": clamp(int(round(total * 100 / 112 * weight)), low, high),
            "legacy_capped_base": clamp(int(round(total * weight)), low, high),
            "legacy_would_saturate": total * weight > high}
