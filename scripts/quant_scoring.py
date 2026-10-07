"""Production profile validation, shared V10 calculation and consistent output."""

import copy
import hashlib
import json

from scripts.contraction_quality import analyze_contraction_quality
from scripts.impulse_score import number, score_structure, validate_config
from scripts.impulse_selection import analyze_selected_impulse, validate_selection_config
from scripts.strategy_config import load_strategy_config
from scripts.terminal_micro import analyze_terminal_micro


class IncompleteStructureScore(ValueError):
    def __init__(self, details):
        self.details = details
        super().__init__(";".join(details.get("reasons", [])) or "STRUCTURE_SCORE_INCOMPLETE")


def load_profile(strategy, evidence_cfg):
    selection = strategy.get("structure_scoring", {})
    if not selection or selection.get("mode") == "legacy":
        return None, None, ""
    if selection.get("mode") != "impulse_v10":
        raise ValueError("Unsupported production structure scoring mode")
    cfg, _ = load_strategy_config(selection["profile_file"])
    if cfg.get("research_only") is not False or cfg.get("execution_purpose") != "production":
        raise ValueError("Production scoring requires a production profile")
    validate_config(cfg)
    validate_selection_config(cfg.get("impulse_selection"))
    calibration = cfg.get("frozen_calibration", {})
    mappings = calibration.get("mappings", {})
    floors = cfg["calibration"]["floors"]
    if (not cfg.get("parameter_id") or set(mappings) != set(floors)
            or type(calibration.get("sample_count")) is not int
            or calibration["sample_count"] < cfg["calibration"]["min_samples"]
            or not calibration.get("calibration_date")
            or not calibration.get("source_calibration_sha256")):
        raise ValueError("Incomplete frozen scoring profile")
    for key, mapping in mappings.items():
        if (mapping.get("floor") != floors[key] or number(mapping.get("cap")) is None
                or mapping["cap"] <= mapping["floor"]):
            raise ValueError(f"Invalid frozen mapping: {key}")
    fingerprint = hashlib.sha256(json.dumps(
        {"profile": cfg, "evidence": evidence_cfg, "legacy_conditions": strategy["scores"]},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    return cfg, calibration, f"{cfg['parameter_id']}:{fingerprint}"


def compute_structure_score(df, structure, legacy, cfg, calibration, evidence_cfg):
    evidence = analyze_selected_impulse(df, structure, evidence_cfg, cfg)
    evidence["research_only"] = False
    quality = analyze_contraction_quality(df, structure, cfg, evidence)
    terminal = analyze_terminal_micro(df, structure, cfg)
    scored = score_structure(
        legacy["components"], evidence, cfg, calibration,
        extension_details=structure.get("contraction_extensions"),
        contraction_quality=quality, terminal_micro=terminal,
        structure_stage=structure.get("state"),
    )
    scored["impulse_evidence"] = evidence
    if scored["status"] != "COMPLETE":
        raise IncompleteStructureScore(scored)
    return scored


def presentation_fields(structure, scored):
    """Serialize score facts after standard structure detection has finished."""
    details = []
    for item in structure.get("contraction_extensions", []):
        if item["type"] == "CONFIRMED_RESET_CONTRACTION":
            adjustment = next(x for x in scored["extension_score_adjustments"]
                              if x["type"] == item["type"])
            details.append({**copy.deepcopy(item), "score": adjustment["trial_score"]})
    terminal = scored["terminal_micro"]
    if terminal["hit"]:
        details.append({"type": "TERMINAL_MICRO_CONTRACTION", "score": terminal["score"],
                        **copy.deepcopy(terminal["selected"])})
    conditions = [x for x in structure.get("conditions", []) if x != "末端微收缩"]
    if terminal["hit"]:
        conditions.append("末端承接微收缩")
    quality = scored["contraction_quality"]
    if quality["hit"]:
        conditions.append("优质收缩序列")
    return {
        "contraction_extensions": details,
        "contraction_extension_tags": [x["type"] for x in details],
        "contraction_extension_score": scored["extension"],
        "contraction_quality_score": quality["score"],
        "contraction_quality_tags": ["QUALITY_CONTRACTION_SEQUENCE"] if quality["hit"] else [],
        "structure_conditions": conditions,
    }
