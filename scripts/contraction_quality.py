"""Shared fixed bonus for improving contractions in one active phase."""

import numpy as np
import pandas as pd


def analyze_contraction_quality(df, structure, cfg, impulse_evidence=None):
    rule = cfg.get("contraction_quality_bonus", {})
    result = {"status": "COMPLETE", "hit": False, "score": 0,
              "checks": {}, "observations": {}, "rounds": [], "reasons": [], "group_round_numbers": []}
    if not rule.get("enabled", False):
        result["reasons"] = ["DISABLED"]
        return result
    stage = structure.get("state", structure.get("structure_stage"))
    result["stage"] = stage
    if (stage not in {"VCP_FORMING", "VCP_MATURE", "VCP_TIGHT"}
            or structure.get("structure_valid") is False
            or structure.get("post_breakout_state", "PRE_BREAKOUT") != "PRE_BREAKOUT"):
        result["reasons"] = ["NOT_ELIGIBLE"]
        return result

    def incomplete(reason):
        return {**result, "status": "INCOMPLETE", "score": None, "reasons": [reason]}

    if structure.get("structure_valid") is not True:
        return incomplete("STRUCTURE_VALIDITY_MISSING")
    group = structure.get("contraction_group")
    if not isinstance(group, list):
        return incomplete("CONTRACTION_GROUP_MISSING")
    selected = ((impulse_evidence or {}).get("selected") or {})
    numbers = selected.get("group_round_numbers", list(range(1, len(group) + 1)))
    if (not isinstance(numbers, list)
            or any(type(n) is not int or not 1 <= n <= len(group) for n in numbers)
            or numbers != sorted(set(numbers))):
        return incomplete("RESEARCH_PHASE_INVALID")
    result["group_round_numbers"] = numbers
    if len(numbers) < 2:
        result["reasons"] = ["INSUFFICIENT_ROUNDS"]
        return result
    if df.empty or not all(key in df for key in ("date", "close", "volume")):
        return incomplete("CONTRACTION_DATA_MISSING")
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        return incomplete("CONTRACTION_DATE_INVALID")
    date_values = dates.dt.strftime("%Y-%m-%d").tolist()
    positions = {date: index for index, date in enumerate(date_values)}
    previous_end = -1
    for number in numbers:
        item = group[number - 1]
        if not isinstance(item, dict):
            return incomplete("CONTRACTION_BOUNDARY_INVALID")
        start = positions.get(str(item.get("start_date")))
        end = positions.get(str(item.get("end_date")))
        if (start is None or end is None or not previous_end < start <= end
                or ("start_idx" in item and item["start_idx"] != start)
                or ("end_idx" in item and item["end_idx"] != end)):
            return incomplete("CONTRACTION_BOUNDARY_INVALID")
        previous_end = end
        segment = df.iloc[start:end + 1]
        values = segment[["close", "volume"]].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(values).all().all() or not values.gt(0).all().all():
            return incomplete("CONTRACTION_PRICE_OR_VOLUME_INVALID")
        low = float(values["close"].min())
        opened = float(values["close"].iloc[0])
        result["rounds"].append({"round": number, "start_date": date_values[start],
                                 "end_date": date_values[end], "sample_days": len(segment),
                                 "low_close": low, "median_close": float(values["close"].median()),
                                 "drawdown_pct": (opened - low) / opened * 100,
                                 "avg_volume": float(values["volume"].mean()),
                                 "confirmation_status": item.get("confirmation_status")})
    first, last = result["rounds"][0], result["rounds"][-1]
    result["checks"] = {"low_rising": last["low_close"] > first["low_close"],
                         "drawdown_shrinking": last["drawdown_pct"] < first["drawdown_pct"],
                         "volume_decreasing": last["avg_volume"] < first["avg_volume"]}
    result["observations"] = {"center_rising": last["median_close"] > first["median_close"]}
    result["hit"] = all(result["checks"].values())
    result["score"] = rule["score"] if result["hit"] else 0
    result["reasons"] = ["QUALITY_CONTRACTION_SEQUENCE"] if result["hit"] else [
        key for key, value in result["checks"].items() if not value]
    return result
