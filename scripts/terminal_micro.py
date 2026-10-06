"""Shared terminal tightening with current two-day support evidence."""

import numpy as np
import pandas as pd


def analyze_terminal_micro(df, structure, cfg):
    rule = cfg.get("terminal_micro_bonus", {})
    result = {"status": "COMPLETE", "hit": False, "score": 0,
              "checks": {}, "candidates": [], "selected": None, "reasons": []}
    if not rule.get("enabled", False):
        result["reasons"] = ["DISABLED"]
        return result

    def incomplete(reason):
        return {**result, "status": "INCOMPLETE", "score": None, "reasons": [reason]}

    stage = structure.get("state", structure.get("structure_stage"))
    if (stage not in {"VCP_EARLY", "VCP_FORMING", "VCP_MATURE", "VCP_TIGHT"}
            or structure.get("structure_valid") is False
            or structure.get("post_breakout_state", "PRE_BREAKOUT") != "PRE_BREAKOUT"):
        result["reasons"] = ["NOT_ELIGIBLE"]
        return result
    if structure.get("structure_valid") is not True:
        return incomplete("STRUCTURE_VALIDITY_MISSING")
    group = structure.get("contraction_group")
    if not isinstance(group, list) or not group:
        return incomplete("CONTRACTION_GROUP_MISSING")
    if df.empty or "date" not in df:
        return incomplete("TERMINAL_DATA_MISSING")
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        return incomplete("TERMINAL_DATE_INVALID")
    date_values = dates.dt.strftime("%Y-%m-%d").tolist()
    positions = {date: idx for idx, date in enumerate(date_values)}
    reference = group[-1]
    if not isinstance(reference, dict):
        return incomplete("REFERENCE_BOUNDARY_INVALID")
    start = positions.get(str(reference.get("start_date")))
    end = positions.get(str(reference.get("end_date")))
    if (start is None or end is None or start > end
            or ("start_idx" in reference and reference["start_idx"] != start)
            or ("end_idx" in reference and reference["end_idx"] != end)):
        return incomplete("REFERENCE_BOUNDARY_INVALID")
    available = len(df) - end - 1
    if available < rule["min_days"]:
        result["reasons"] = ["INSUFFICIENT_TERMINAL_DAYS"]
        return result
    columns = ["high", "low", "close", "volume"]
    if not all(key in df for key in columns):
        return incomplete("TERMINAL_DATA_MISSING")
    values = df.iloc[start:][columns].apply(pd.to_numeric, errors="coerce")
    if (not np.isfinite(values).all().all() or not values.gt(0).all().all()
            or (values["low"] > values["close"]).any()
            or (values["close"] > values["high"]).any()):
        return incomplete("TERMINAL_PRICE_OR_VOLUME_INVALID")
    try:
        pivot = float(structure.get("structure_pivot"))
    except (TypeError, ValueError, OverflowError):
        return incomplete("PIVOT_INVALID")
    if not np.isfinite(pivot) or pivot <= 0:
        return incomplete("PIVOT_INVALID")

    def metrics(segment):
        low = float(segment["low"].min())
        high = float(segment["high"].max())
        return {"low": low, "high": high,
                "range_pct": (high - low) / float(segment["close"].iloc[0]) * 100,
                "avg_volume": float(segment["volume"].mean())}

    # Reference and terminal use the same actual bars, rather than rounded archive means.
    ref = metrics(values.iloc[:end - start + 1])
    result["reference"] = {"start_date": date_values[start], "end_date": date_values[end], **ref}
    recent = values.iloc[-2:]
    distance = (float(recent["close"].iloc[-1]) / pivot - 1) * 100
    checks = {"low_stopped": bool(recent["low"].iloc[-1] >= recent["low"].iloc[-2]),
              "close_stopped": bool(recent["close"].iloc[-1] >= recent["close"].iloc[-2]),
              "pivot_in_range": rule["min_pivot_distance_pct"] <= distance <= rule["max_pivot_distance_pct"]}
    result["confirmation"] = {"dates": date_values[-2:], "lows": recent["low"].tolist(),
                              "closes": recent["close"].tolist(), "pivot_distance_pct": distance}
    for length in range(rule["min_days"], min(rule["max_days"], available) + 1):
        window = metrics(values.iloc[-length:])
        candidate_checks = {**checks, "support_held": window["low"] >= ref["low"],
                            "range_shrinking": window["range_pct"] < ref["range_pct"],
                            "volume_decreasing": window["avg_volume"] < ref["avg_volume"]}
        result["candidates"].append({"start_date": date_values[-length], "end_date": date_values[-1],
                                     "duration_days": length, **window, "checks": candidate_checks,
                                     "hit": all(candidate_checks.values())})
    hits = [item for item in result["candidates"] if item["hit"]]
    selected = hits[-1] if hits else result["candidates"][-1]
    result["checks"] = selected["checks"]
    result["selected"] = selected
    result["hit"] = bool(hits)
    result["score"] = rule["score"] if hits else 0
    result["reasons"] = ["TERMINAL_SUPPORT_TIGHTENING"] if hits else [
        key for key, passed in selected["checks"].items() if not passed]
    return result
