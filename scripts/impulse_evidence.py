"""Research-only ordered impulse and retention measurements for an existing VCP group."""

import math

import numpy as np
import pandas as pd


def empty_impulse_evidence(status, reasons=None):
    return {
        "schema": "quant_impulse_evidence_v1", "research_only": True,
        "as_of": "", "group_start_date": "", "lookback_days": None,
        "status": status, "reasons": reasons or [], "selected": None, "windows": {},
    }


def _number(value):
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 6)


def _position(group_item, key, positions):
    date = group_item.get(f"{key}_date")
    index = group_item.get(f"{key}_idx")
    if date is not None:
        found = positions.get(str(date))
        if found is None or (index is not None and int(index) != found):
            return None
        return found
    return int(index) if index is not None else None


def _candidate_summary(candidate, df):
    if candidate is None:
        return None
    return {
        "base_date": str(df.iloc[candidate["base_idx"]]["date"]),
        "peak_date": str(df.iloc[candidate["peak_idx"]]["date"]),
        "gain_pct": _number(candidate["gain_pct"]),
        "bridge_drawdown_pct": _number(candidate["bridge_drawdown_pct"]),
    }


def _retention_path(df, start, end, base, peak):
    segment = df.iloc[start:end + 1]
    result = {
        "start_date": str(segment.iloc[0]["date"]) if not segment.empty else "",
        "end_date": str(segment.iloc[-1]["date"]) if not segment.empty else "",
        "close_low_date": "", "close_low": None, "close_retention_pct": None,
        "intraday_low_date": "", "intraday_low": None, "intraday_retention_pct": None,
    }
    if segment.empty:
        return result
    idx = int(segment["close"].idxmin())
    low = float(df.iloc[idx]["close"])
    result.update(close_low_date=str(df.iloc[idx]["date"]), close_low=_number(low),
                  close_retention_pct=_number((low - base) / (peak - base) * 100))
    if "low" in segment:
        lows = pd.to_numeric(segment["low"], errors="coerce")
        if np.isfinite(lows).all() and (lows > 0).all():
            idx = int(lows.idxmin())
            low = float(df.iloc[idx]["low"])
            result.update(intraday_low_date=str(df.iloc[idx]["date"]), intraday_low=_number(low),
                          intraday_retention_pct=_number((low - base) / (peak - base) * 100))
    return result


def _volume_evidence(df, base_idx, peak_idx, cfg):
    count = int(cfg["baseline_volume_days"])
    baseline = df.iloc[max(0, base_idx - count):base_idx]
    leg = df.iloc[base_idx + 1:peak_idx + 1]
    result = {
        "baseline_start_date": str(baseline.iloc[0]["date"]) if not baseline.empty else "",
        "baseline_end_date": str(baseline.iloc[-1]["date"]) if not baseline.empty else "",
        "baseline_days": len(baseline), "baseline_mean": None,
        "advance_days": len(leg), "up_days": 0, "down_days": 0, "flat_days": 0,
        "mean_volume_ratio": None, "up_mean_volume_ratio": None, "down_mean_volume_ratio": None,
        "up_peak_volume_ratio": None, "down_peak_volume_ratio": None,
        "up_volume_share_pct": None, "up_peak_volume_confirmed": None,
        "peak_volume_date": "", "peak_volume_ratio": None,
        "peak_volume_day_return_pct": None, "peak_volume_upper_shadow_ratio": None,
        "warnings": [],
    }
    if len(baseline) < count:
        result["warnings"].append("BASELINE_INCOMPLETE")
    if "volume" not in df or baseline.empty:
        result["warnings"].append("VOLUME_DATA_INCOMPLETE")
        return result
    baseline_volumes = pd.to_numeric(baseline["volume"], errors="coerce")
    volumes = pd.to_numeric(leg["volume"], errors="coerce")
    if not (np.isfinite(baseline_volumes).all() and (baseline_volumes > 0).all()
            and np.isfinite(volumes).all() and (volumes > 0).all()):
        result["warnings"].append("VOLUME_DATA_INCOMPLETE")
        return result
    mean = float(baseline_volumes.mean())
    changes = df["close"].diff().iloc[base_idx + 1:peak_idx + 1]
    up = volumes[changes > 0]
    down = volumes[changes < 0]
    result.update(
        baseline_mean=_number(mean), mean_volume_ratio=_number(volumes.mean() / mean),
        up_days=len(up), down_days=len(down), flat_days=int((changes == 0).sum()),
        up_mean_volume_ratio=_number(up.mean() / mean) if not up.empty else None,
        down_mean_volume_ratio=_number(down.mean() / mean) if not down.empty else None,
        up_peak_volume_ratio=_number(up.max() / mean) if not up.empty else None,
        down_peak_volume_ratio=_number(down.max() / mean) if not down.empty else None,
        up_volume_share_pct=_number(up.sum() / volumes.sum() * 100),
        up_peak_volume_confirmed=(bool(up.max() / mean >= cfg["min_up_peak_volume_ratio"])
                                  if len(baseline) == count and not up.empty else None),
    )
    volume_idx = int(volumes.idxmax())
    bar = df.iloc[volume_idx]
    daily_return = (float(bar["close"]) / float(df.iloc[volume_idx - 1]["close"]) - 1) * 100
    result.update(peak_volume_date=str(bar["date"]), peak_volume_ratio=_number(volumes.max() / mean),
                  peak_volume_day_return_pct=_number(daily_return))
    if daily_return < 0:
        result["warnings"].append("PEAK_VOLUME_ON_DOWN_DAY")
    if all(key in df for key in ["open", "high", "low"]):
        ohlc = pd.to_numeric(bar[["open", "high", "low", "close"]], errors="coerce").to_numpy(dtype=float)
        opened, high, low, closed = ohlc
        if (all(math.isfinite(value) and value > 0 for value in ohlc)
                and high > low and low <= min(opened, closed) <= max(opened, closed) <= high):
            upper = (high - max(opened, closed)) / (high - low)
            result["peak_volume_upper_shadow_ratio"] = _number(upper)
            if upper >= cfg["upper_shadow_warning_ratio"]:
                result["warnings"].append("PEAK_VOLUME_LONG_UPPER_SHADOW")
        elif not (high == low == opened == closed and high > 0):
            result["warnings"].append("INTRADAY_DATA_INCOMPLETE")
    else:
        result["warnings"].append("INTRADAY_DATA_INCOMPLETE")
    return result


def _window_evidence(df, group, positions, start, lookback, cfg):
    scan_start = max(0, start - lookback)
    closes = df["close"].to_numpy(dtype=float)
    candidates = []
    for base_idx in range(scan_start, start):
        # The maximum is searched strictly after B; later lows cannot become its base.
        peak_idx = base_idx + 1 + int(np.argmax(closes[base_idx + 1:start + 1]))
        base, peak = closes[base_idx], closes[peak_idx]
        if peak <= base:
            continue
        bridge_low = float(closes[peak_idx:start + 1].min())
        candidates.append({
            "base_idx": base_idx, "peak_idx": peak_idx,
            "gain_pct": (peak / base - 1) * 100,
            "bridge_drawdown_pct": (bridge_low / peak - 1) * 100,
        })
    rank = lambda item: (item["gain_pct"], item["base_idx"])
    linked = [item for item in candidates if item["bridge_drawdown_pct"] >= -cfg["max_bridge_drawdown_pct"]]
    rejected = [item for item in candidates if item["bridge_drawdown_pct"] < -cfg["max_bridge_drawdown_pct"]]
    selected = max(linked or candidates, key=rank) if candidates else None
    result = {
        "lookback_days": lookback, "scan_start_date": str(df.iloc[scan_start]["date"]),
        "available_lookback_days": start - scan_start, "candidate_count": len(candidates),
        "linked_candidate_count": len(linked), "status": "NO_ADVANCE", "reasons": [],
        "strongest_unlinked_candidate": _candidate_summary(max(rejected, key=rank) if rejected else None, df),
        "anchor": None, "price": None, "volume": None, "retention": None,
    }
    if start - scan_start < lookback:
        result["reasons"].append("SCAN_HISTORY_INCOMPLETE")
    if selected is None:
        result["reasons"].append("NO_ORDERED_PRICE_ADVANCE")
        return result
    base_idx, peak_idx = selected["base_idx"], selected["peak_idx"]
    base, peak = closes[base_idx], closes[peak_idx]
    if not linked:
        result["status"] = "UNLINKED"
        result["reasons"].append("PLATFORM_LINK_BROKEN")
    elif selected["gain_pct"] >= cfg["min_gain_pct"]:
        result["status"] = "IDENTIFIED"
    else:
        result["status"] = "WEAK_ADVANCE"
        result["reasons"].append("PRICE_GAIN_BELOW_RESEARCH_THRESHOLD")
    if base_idx == scan_start:
        result["reasons"].append("BASE_AT_SCAN_BOUNDARY")
    base_date, peak_date, start_date = (str(df.iloc[idx]["date"]) for idx in [base_idx, peak_idx, start])
    path_length = float(np.abs(np.diff(closes[base_idx:peak_idx + 1])).sum())
    result["anchor"] = {
        "anchor_id": f"{lookback}:{start_date}:{base_date}:{peak_date}",
        "base_idx": base_idx, "base_date": base_date, "base_close": _number(base),
        "peak_idx": peak_idx, "peak_date": peak_date, "peak_close": _number(peak),
        "group_start_idx": start, "group_start_date": start_date,
    }
    result["price"] = {
        "gain_pct": _number(selected["gain_pct"]), "advance_days": peak_idx - base_idx,
        "peak_lead_days": start - peak_idx,
        "bridge_low_close": _number(closes[peak_idx:start + 1].min()),
        "bridge_drawdown_pct": _number(selected["bridge_drawdown_pct"]),
        "path_efficiency": _number((peak - base) / path_length),
    }
    result["volume"] = _volume_evidence(df, base_idx, peak_idx, cfg)
    result["reasons"].extend(result["volume"]["warnings"])
    rounds = []
    for number, item in enumerate(group, 1):
        a, b = (_position(item, key, positions) for key in ["start", "end"])
        path = _retention_path(df, a, b, base, peak)
        rounds.append({"round": number, **path,
                       "original_close_pullback_pct": item.get("close_pullback_pct", item.get("pullback_pct")),
                       "confirmation_status": item.get("confirmation_status", "UNKNOWN")})
    result["retention"] = {
        "rounds": rounds,
        # A closing-price peak occurs at the close; that day's low is not a later retracement.
        "post_peak": _retention_path(df, peak_idx + 1, len(df) - 1, base, peak),
        "post_group": _retention_path(df, start, len(df) - 1, base, peak),
        "current_close": _number(closes[-1]),
        "current_close_retention_pct": _number((closes[-1] - base) / (peak - base) * 100),
    }
    if (any(path["intraday_low"] is None for path in rounds)
            and "INTRADAY_DATA_INCOMPLETE" not in result["reasons"]):
        result["reasons"].append("INTRADAY_DATA_INCOMPLETE")
    return result


def analyze_impulse_evidence(df, structure, cfg):
    """Return evidence without mutating the frame, group, existing tags or decisions."""
    if not cfg.get("enabled"):
        return empty_impulse_evidence("DISABLED")
    result = empty_impulse_evidence("DATA_ISSUE")
    if df.empty or not all(key in df for key in ["date", "close"]):
        result["reasons"] = ["PRICE_OR_DATE_MISSING"]
        return result
    frame = df.reset_index(drop=True).copy()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        result["reasons"] = ["DATE_ORDER_OR_DUPLICATE"]
        return result
    frame["date"] = dates.dt.strftime("%Y-%m-%d")
    result["as_of"] = str(frame.iloc[-1]["date"])
    result["lookback_days"] = int(cfg["lookback_days"])
    group = structure.get("contraction_group") or []
    if not group:
        result.update(status="NO_GROUP", reasons=["NO_ORIGINAL_CONTRACTION_GROUP"])
        return result
    positions = {date: idx for idx, date in enumerate(frame["date"])}
    try:
        bounds = [(_position(item, "start", positions), _position(item, "end", positions)) for item in group]
    except (TypeError, ValueError, OverflowError):
        bounds = []
    if (not bounds or any(a is None or b is None or not 0 <= a <= b < len(frame) for a, b in bounds)
            or any(bounds[idx][0] <= bounds[idx - 1][1] for idx in range(1, len(bounds)))):
        result.update(status="GROUP_ANCHOR_MISMATCH", reasons=["GROUP_DATE_INDEX_OR_ORDER_CONFLICT"])
        return result
    start = bounds[0][0]
    result["group_start_date"] = str(frame.iloc[start]["date"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    windows = sorted(set([int(cfg["lookback_days"])] + [int(n) for n in cfg["diagnostic_lookback_days"]]))
    if any(n <= 0 for n in windows) or int(cfg["baseline_volume_days"]) <= 0:
        raise ValueError("Impulse evidence windows must be positive")
    needed = frame.iloc[max(0, start - max(windows)):]["close"]
    if not np.isfinite(needed).all() or not (needed > 0).all():
        result["reasons"] = ["CLOSE_DATA_INVALID"]
        return result
    for lookback in windows:
        result["windows"][str(lookback)] = _window_evidence(frame, group, positions, start, lookback, cfg)
    result["selected"] = result["windows"][str(cfg["lookback_days"])]
    result["status"] = result["selected"]["status"]
    result["reasons"] = list(result["selected"]["reasons"])
    return result
