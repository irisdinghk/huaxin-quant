"""Research-only ordered impulse and retention measurements for an existing VCP group."""

import math

import numpy as np
import pandas as pd


def empty_impulse_evidence(status, reasons=None):
    return {
        "schema": "quant_impulse_evidence_v2", "research_only": True,
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


def _one_price_up_mask(df):
    if not all(key in df for key in ("open", "high", "low", "close")):
        return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    prices = df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    previous = prices["close"].shift()
    valid = (np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
             & np.isfinite(previous) & previous.gt(0))
    one_price = prices.eq(prices["close"], axis=0).all(axis=1)
    return valid & one_price & prices["close"].gt(previous), valid


def _volume_evidence(df, base_idx, peak_idx, cfg):
    count = int(cfg["baseline_volume_days"])
    baseline = df.iloc[max(0, base_idx - count):base_idx]
    leg = df.iloc[base_idx + 1:peak_idx + 1]
    result = {
        "volume_policy": "exclude_one_price_up_v1",
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
    excluded, checked = _one_price_up_mask(df)
    baseline_excluded = excluded.loc[baseline.index]
    leg_excluded = excluded.loc[leg.index]
    result.update(
        baseline_effective_days=int((~baseline_excluded).sum()),
        advance_effective_days=int((~leg_excluded).sum()),
        baseline_excluded_dates=baseline.loc[baseline_excluded, "date"].astype(str).tolist(),
        advance_excluded_dates=leg.loc[leg_excluded, "date"].astype(str).tolist(),
    )
    if not checked.loc[baseline.index.union(leg.index)].all():
        result["warnings"].append("ONE_PRICE_CHECK_INCOMPLETE")
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
    baseline_volumes = baseline_volumes.loc[~baseline_excluded]
    volumes = volumes.loc[~leg_excluded]
    if baseline_volumes.empty or volumes.empty:
        result["warnings"].extend(["VOLUME_SAMPLE_EMPTY", "VOLUME_DATA_INCOMPLETE"])
        return result
    mean = float(baseline_volumes.mean())
    # Direction still uses the actual preceding close, including excluded days.
    changes = df["close"].diff().loc[volumes.index]
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


def _episode_evidence(df, group, positions, start, lookback, cfg, floor=0, candidate_selector=None):
    window_start = max(0, start - lookback)
    scan_start = max(window_start, floor)
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
            "exhausted": bool(bridge_low <= base),
        })
    rank = lambda item: (item["peak_idx"], item["gain_pct"], item["base_idx"])
    linked = [item for item in candidates if not item["exhausted"]]
    substantial = [item for item in linked if item["gain_pct"] >= cfg["min_gain_pct"]]
    weak = [item for item in candidates if item["gain_pct"] < cfg["min_gain_pct"]]
    exhausted = [item for item in candidates if item["exhausted"]]
    selected = max(substantial or linked or exhausted, key=rank) if candidates else None
    selection = None
    if candidate_selector is not None:
        selection = candidate_selector(df.iloc[:start + 1], candidates, scan_start, start, cfg)
        selected = selection["selected"]
    result = {
        "lookback_days": lookback, "scan_start_date": str(df.iloc[scan_start]["date"]),
        "available_lookback_days": start - scan_start, "candidate_count": len(candidates),
        "linked_candidate_count": len(linked), "status": "NO_ADVANCE", "reasons": [],
        "candidate_summaries": [
            {**_candidate_summary(item, df), "exhausted": item["exhausted"],
             "bridge_close_retention_pct": _number(
                 (closes[item["peak_idx"]] * (1 + item["bridge_drawdown_pct"] / 100)
                  - closes[item["base_idx"]]) / (closes[item["peak_idx"]] - closes[item["base_idx"]]) * 100)}
            for item in sorted(candidates, key=rank, reverse=True)],
        "latest_weak_candidate": _candidate_summary(max(weak, key=rank) if weak else None, df),
        "strongest_exhausted_candidate": _candidate_summary(
            max(exhausted, key=lambda item: (item["gain_pct"], item["base_idx"])) if exhausted else None, df),
        "anchor": None, "price": None, "volume": None, "retention": None,
    }
    if selection is not None:
        result.update(selection["diagnostics"], status=selection["status"])
        result["reasons"].extend(selection["reasons"])
    if start - window_start < lookback:
        result["reasons"].append("SCAN_HISTORY_INCOMPLETE")
    if selected is None:
        if selection is None:
            result["reasons"].append("NO_ORDERED_PRICE_ADVANCE")
        return result
    base_idx, peak_idx = selected["base_idx"], selected["peak_idx"]
    base, peak = closes[base_idx], closes[peak_idx]
    if not linked:
        result["status"] = "EXHAUSTED"
        result["reasons"].append("ADVANCE_FULLY_RETRACED_BEFORE_GROUP")
    elif selected["gain_pct"] >= cfg["min_gain_pct"]:
        result["status"] = "IDENTIFIED"
    else:
        result["status"] = "WEAK_ADVANCE"
        result["reasons"].append("PRICE_GAIN_BELOW_RESEARCH_THRESHOLD")
    if base_idx == window_start:
        result["reasons"].append("BASE_AT_SCAN_BOUNDARY")
    if floor and base_idx == floor:
        result["reasons"].append("BASE_AT_REBUILD_BOUNDARY")
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


def _exhaustion_event(df, item, positions, anchor, round_number):
    if anchor is None or item.get("confirmation_status") != "CONFIRMED":
        return None, None
    a, b = (_position(item, key, positions) for key in ["start", "end"])
    base = float(df.iloc[anchor["base_idx"]]["close"])
    if float(df.iloc[b]["close"]) > base:
        return None, None
    required = item.get("required_right_confirm_days")
    if required is None:
        return None, "CONFIRMATION_TIMING_UNAVAILABLE"
    try:
        days = float(required)
        if not math.isfinite(days) or days < 0 or not days.is_integer():
            return None, "CONFIRMATION_TIMING_UNAVAILABLE"
    except (TypeError, ValueError, OverflowError):
        return None, "CONFIRMATION_TIMING_UNAVAILABLE"
    confirmed_idx = b + int(days)
    if confirmed_idx >= len(df):
        return None, "EXHAUSTION_CONFIRMATION_NOT_YET_AVAILABLE"
    path = df.iloc[a:b + 1]
    breach_idx = int(path[path["close"] <= base].index[0])
    return {
        "round": round_number, "first_breach_idx": breach_idx,
        "first_breach_date": str(df.iloc[breach_idx]["date"]),
        "end_idx": b, "end_date": str(df.iloc[b]["date"]),
        "end_close": _number(df.iloc[b]["close"]),
        "confirmed_idx": confirmed_idx, "confirmed_date": str(df.iloc[confirmed_idx]["date"]),
    }, None


def _window_evidence(df, group, positions, start, lookback, cfg, candidate_selector=None):
    episodes = []
    pending = None
    for number, item in enumerate(group, 1):
        s = _position(item, "start", positions)
        if not episodes or (pending and pending["confirmed_idx"] <= s):
            floor = pending["end_idx"] if pending else 0
            phase = _episode_evidence(df.iloc[:s + 1], [], positions, s, lookback, cfg, floor, candidate_selector)
            phase.update(
                episode_id=f"{lookback}:{df.iloc[s]['date']}:{floor}",
                group_start_date=str(df.iloc[s]["date"]),
                group_round_numbers=[], lifecycle="OPEN", exhaustion=None,
                rebuild_from=dict(pending) if pending else None,
                scan_floor_idx=floor,
            )
            if phase["status"] == "EXHAUSTED":
                phase["lifecycle"] = "EXHAUSTED"
            if pending and phase["status"] != "IDENTIFIED":
                phase["lifecycle"] = "REBUILD_PENDING"
                phase["reasons"].append("NO_SUBSTANTIAL_REBUILD_ADVANCE")
            else:
                pending = None
            episodes.append(phase)
        phase = episodes[-1]
        phase["group_round_numbers"].append(number)
        # A pending weak bounce cannot erase or replace the preceding failed impulse.
        if phase["lifecycle"] != "REBUILD_PENDING" and phase["exhaustion"] is None:
            event, warning = _exhaustion_event(df, item, positions, phase["anchor"], number)
            if warning and warning not in phase["reasons"]:
                phase["reasons"].append(warning)
            if event:
                phase.update(lifecycle="EXHAUSTED", exhaustion=event)
                pending = event
    for idx, phase in enumerate(episodes):
        next_start = (_position(group[episodes[idx + 1]["group_round_numbers"][0] - 1], "start", positions)
                      if idx + 1 < len(episodes) else len(df))
        horizon = next_start - 1
        assigned = [group[n - 1] for n in phase["group_round_numbers"]]
        measurements = _episode_evidence(
            df.iloc[:horizon + 1], assigned, positions,
            _position(assigned[0], "start", positions), lookback, cfg, phase["scan_floor_idx"], candidate_selector)
        for key in ["anchor", "price", "volume", "retention"]:
            phase[key] = measurements[key]
        if measurements["anchor"]:
            for path, number in zip(phase["retention"]["rounds"], phase["group_round_numbers"]):
                path["round"] = number
        phase["reasons"] = list(dict.fromkeys(phase["reasons"] + measurements["reasons"]))
        phase["observed_through"] = str(df.iloc[horizon]["date"])
    current = dict(episodes[-1])
    current.update(episodes=episodes, active_episode_id=current["episode_id"])
    return current


def analyze_impulse_evidence(df, structure, cfg, candidate_selector=None):
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
    needed = frame.iloc[max(0, start - max(windows) - int(cfg.get("base_context_days", 0))):]["close"]
    if not np.isfinite(needed).all() or not (needed > 0).all():
        result["reasons"] = ["CLOSE_DATA_INVALID"]
        return result
    for lookback in windows:
        result["windows"][str(lookback)] = _window_evidence(frame, group, positions, start, lookback, cfg, candidate_selector)
    result["selected"] = result["windows"][str(cfg["lookback_days"])]
    result["original_group_start_date"] = result["group_start_date"]
    result["group_start_date"] = result["selected"]["group_start_date"]
    result["status"] = result["selected"]["status"]
    result["reasons"] = list(result["selected"]["reasons"])
    return result
