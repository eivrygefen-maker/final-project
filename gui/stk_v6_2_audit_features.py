#!/usr/bin/env python3
"""Load normalized features from STK V6 physical DOF audit JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

DEFAULT_AUDIT_JSON = (
    Path(__file__).resolve().parents[1] / "audio" / "debug_reports" / "stk_v6_physical_dof_audit.json"
)

_FEATURE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "helmholtz_like_frequency_proxy": ("helmholtz_like_frequency_proxy", "helmholtz_like_frequency_hz"),
    "cavity_decay_proxy": ("cavity_decay_proxy", "cavity_decay_s"),
    "cavity_q_proxy": ("cavity_q_proxy", "cavity_q"),
    "high_frequency_absorption_proxy": (
        "high_frequency_absorption_proxy",
        "high_frequency_absorption",
    ),
}


def load_audit_report(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path or DEFAULT_AUDIT_JSON)
    if not p.is_file():
        raise FileNotFoundError(f"audit report not found: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("audit JSON must be an object")
    return doc


def get_sample_record(audit: Mapping[str, Any], sample_id: str) -> Dict[str, Any]:
    for s in audit.get("samples") or []:
        if str(s.get("sample_id")) == str(sample_id):
            return dict(s)
    raise KeyError(f"sample {sample_id!r} not in audit report")


def get_feature(
    sample: Mapping[str, Any],
    feature_name: str,
    *,
    audit: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Search geometry → material → modal → derived_features for a provenance record.

    Returns dict with value, status, source_path, confidence, per_sample, intended_v6_use.
    """
    names = _FEATURE_ALIASES.get(feature_name, (feature_name,))
    for name in names:
        for section in ("geometry", "material", "modal", "derived_features"):
            block = sample.get(section) or {}
            if name in block and isinstance(block[name], dict):
                rec = dict(block[name])
                rec.setdefault("feature_name", name)
                return rec

    if audit is not None:
        ref_agg = (audit.get("reference_modal_catalog") or {}).get("reference_aggregates") or {}
        for name in names:
            if name in ref_agg and isinstance(ref_agg[name], dict):
                rec = dict(ref_agg[name])
                rec.setdefault("feature_name", name)
                return rec

    return {
        "feature_name": feature_name,
        "value": None,
        "status": "missing",
        "source_path": "",
        "confidence": "low",
        "per_sample": False,
        "intended_v6_use": "",
    }


def feature_value(
    sample: Mapping[str, Any],
    feature_name: str,
    *,
    audit: Optional[Mapping[str, Any]] = None,
    default: Any = None,
) -> Any:
    rec = get_feature(sample, feature_name, audit=audit)
    val = rec.get("value")
    return default if val is None else val


def collect_features_for_synthesis(
    audit: Mapping[str, Any],
    sample_id: str,
) -> Dict[str, Dict[str, Any]]:
    """Gather all features used by V6.2 synthesis with provenance."""
    sample = get_sample_record(audit, sample_id)
    names = (
        "body_depth",
        "body_length",
        "body_width",
        "body_area_proxy",
        "body_volume_proxy",
        "soundhole_radius",
        "soundhole_area",
        "helmholtz_like_frequency_proxy",
        "cavity_decay_proxy",
        "cavity_q_proxy",
        "top_wood_id",
        "back_wood_id",
        "top_density_proxy",
        "back_density_proxy",
        "top_damping_coeff_proxy",
        "back_damping_coeff_proxy",
        "mass_loading_proxy",
        "high_frequency_absorption_proxy",
        "bridge_mobility_proxy",
        "low_body_mode_frequency",
        "bridge_to_radiation_strength",
        "air_to_structural_ratio",
        "top_to_back_ratio",
        "aperture_to_top_radiation_ratio",
        "top_radiation_gain_proxy",
        "soundhole_radiation_gain_proxy",
    )
    return {n: get_feature(sample, n, audit=audit) for n in names}
