#!/usr/bin/env python3
"""Shape-aware physical validation profiles for post-aggregation acceptance (advisory layer)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from m4_shape_registry import normalize_shape_key, registered_shape_keys  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILES_CONFIG_PATH = REPO_ROOT / "configs" / "shape_validation_profiles.json"

# Frozen reference baseline — do not change thresholds without a new profile_id revision.
CLASSIC_LOCKED_PROFILE_ID = "classical_guitar_reference_v1_locked"
CLASSIC_LEGACY_PROFILE_ALIAS = "classical_guitar_reference_v1"

_LOCKED_PROFILE_FIELD_KEYS = frozenset(
    {
        "profile_type",
        "mode_count_min",
        "mode_count_warn_below",
        "deduped_mode_count_min",
        "allow_worker_pass_with_warning_after_aggregation",
        "bridge_coupling_policy",
        "radiation_policy",
        "mic_output_policy",
        "top_back_air_share_policy",
        "air_share_policy",
        "frequency_band_policy",
    }
)

_BUILTIN_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "classic": {
        "shape_name": "classic",
        "profile_id": CLASSIC_LOCKED_PROFILE_ID,
        "profile_type": "classical_reference",
        "locked": True,
        "reference_baseline": "classic_lhs_6_simulations",
        "mode_count_min": 8,
        "mode_count_warn_below": 10,
        "deduped_mode_count_min": 8,
        "allow_worker_pass_with_warning_after_aggregation": False,
        "bridge_coupling_policy": "stricter_guitar_like",
        "radiation_policy": "guitar_like",
        "mic_output_policy": "guitar_like",
        "top_back_air_share_policy": "classical_reference",
        "air_share_policy": "classical_reference",
        "frequency_band_policy": "classical_body_modes",
        "notes": (
            "Locked classical guitar reference baseline (6 existing CLASSIC simulations). "
            "Thresholds are frozen; advisory validator only — does not alter production acceptance."
        ),
    },
    "box": {
        "shape_name": "box",
        "profile_id": "box_body_plausibility_v1",
        "profile_type": "shape_relative_body_validation",
        "mode_count_min": 6,
        "mode_count_warn_below": 8,
        "deduped_mode_count_min": 6,
        "allow_worker_pass_with_warning_after_aggregation": True,
        "bridge_coupling_policy": "relative_relaxed",
        "radiation_policy": "relative_quantile",
        "mic_output_policy": "relative_quantile",
        "top_back_air_share_policy": "shape_relative",
        "air_share_policy": "volume_scaled",
        "frequency_band_policy": "geometry_scaled",
        "notes": (
            "Box body should be judged relative to simple cavity/plate behavior, "
            "not classical guitar guitar-like criteria."
        ),
    },
    "acoustic": {
        "shape_name": "acoustic",
        "profile_id": "acoustic_guitar_reference_v1",
        "profile_type": "acoustic_reference",
        "mode_count_min": 8,
        "mode_count_warn_below": 10,
        "deduped_mode_count_min": 8,
        "allow_worker_pass_with_warning_after_aggregation": True,
        "bridge_coupling_policy": "guitar_like",
        "radiation_policy": "guitar_like",
        "mic_output_policy": "guitar_like",
        "top_back_air_share_policy": "acoustic_reference",
        "air_share_policy": "acoustic_reference",
        "frequency_band_policy": "geometry_scaled",
        "notes": "Acoustic/dreadnought body reference with geometry-scaled frequency expectations.",
    },
}

_SHAPE_DEFAULT_PROFILE_ID: Dict[str, str] = {
    key: str(_BUILTIN_DEFAULTS[key]["profile_id"]) for key in _BUILTIN_DEFAULTS
}


@dataclass(frozen=True)
class ShapeValidationProfile:
    shape_name: str
    profile_id: str
    profile_type: str
    mode_count_min: int
    mode_count_warn_below: int
    deduped_mode_count_min: int
    allow_worker_pass_with_warning_after_aggregation: bool
    bridge_coupling_policy: str
    radiation_policy: str
    mic_output_policy: str
    top_back_air_share_policy: str
    air_share_policy: str
    frequency_band_policy: str
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def locked(self) -> bool:
        return bool(self.extra.get("locked"))

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "shape_name": self.shape_name,
            "profile_id": self.profile_id,
            "profile_type": self.profile_type,
            "mode_count_min": self.mode_count_min,
            "mode_count_warn_below": self.mode_count_warn_below,
            "deduped_mode_count_min": self.deduped_mode_count_min,
            "allow_worker_pass_with_warning_after_aggregation": (
                self.allow_worker_pass_with_warning_after_aggregation
            ),
            "bridge_coupling_policy": self.bridge_coupling_policy,
            "radiation_policy": self.radiation_policy,
            "mic_output_policy": self.mic_output_policy,
            "top_back_air_share_policy": self.top_back_air_share_policy,
            "air_share_policy": self.air_share_policy,
            "frequency_band_policy": self.frequency_band_policy,
            "notes": self.notes,
        }
        if self.locked:
            body["locked"] = True
        if self.extra:
            body.update({k: v for k, v in self.extra.items() if k not in body})
        return body


def _load_profiles_config() -> Dict[str, Any]:
    if not PROFILES_CONFIG_PATH.is_file():
        return {}
    try:
        doc = json.loads(PROFILES_CONFIG_PATH.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _merge_profile_body(
    *,
    shape_name: str,
    profile_id: str,
    base: Mapping[str, Any],
    overrides: Optional[Mapping[str, Any]] = None,
) -> ShapeValidationProfile:
    merged: Dict[str, Any] = dict(base)
    locked = bool(merged.get("locked"))
    if overrides:
        if locked:
            safe = {
                k: v
                for k, v in overrides.items()
                if k not in _LOCKED_PROFILE_FIELD_KEYS and k not in ("shape_name", "profile_id", "locked")
            }
            merged.update(safe)
        else:
            merged.update({k: v for k, v in overrides.items() if k not in ("shape_name", "profile_id")})
    merged["shape_name"] = shape_name
    merged["profile_id"] = profile_id
    extra = {
        k: v
        for k, v in merged.items()
        if k
        not in {
            "shape_name",
            "profile_id",
            "profile_type",
            "mode_count_min",
            "mode_count_warn_below",
            "deduped_mode_count_min",
            "allow_worker_pass_with_warning_after_aggregation",
            "bridge_coupling_policy",
            "radiation_policy",
            "mic_output_policy",
            "top_back_air_share_policy",
            "air_share_policy",
            "frequency_band_policy",
            "notes",
        }
    }
    if merged.get("locked"):
        extra["locked"] = True
    if merged.get("reference_baseline"):
        extra["reference_baseline"] = merged.get("reference_baseline")
    return ShapeValidationProfile(
        shape_name=shape_name,
        profile_id=profile_id,
        profile_type=str(merged.get("profile_type") or "custom"),
        mode_count_min=int(merged.get("mode_count_min") or 0),
        mode_count_warn_below=int(merged.get("mode_count_warn_below") or merged.get("mode_count_min") or 0),
        deduped_mode_count_min=int(merged.get("deduped_mode_count_min") or merged.get("mode_count_min") or 0),
        allow_worker_pass_with_warning_after_aggregation=bool(
            merged.get("allow_worker_pass_with_warning_after_aggregation")
        ),
        bridge_coupling_policy=str(merged.get("bridge_coupling_policy") or "default"),
        radiation_policy=str(merged.get("radiation_policy") or "default"),
        mic_output_policy=str(merged.get("mic_output_policy") or "default"),
        top_back_air_share_policy=str(merged.get("top_back_air_share_policy") or "default"),
        air_share_policy=str(merged.get("air_share_policy") or "default"),
        frequency_band_policy=str(merged.get("frequency_band_policy") or "default"),
        notes=str(merged.get("notes") or ""),
        extra=extra,
    )


def resolve_shape_validation_profile(
    shape_key: str,
    *,
    profile_id: Optional[str] = None,
) -> ShapeValidationProfile:
    """Resolve validation profile for a shape (config overrides + code defaults)."""
    shape_name = normalize_shape_key(shape_key)
    cfg = _load_profiles_config()
    shape_defaults = cfg.get("shape_defaults") or {}
    profiles_doc = cfg.get("profiles") if isinstance(cfg.get("profiles"), dict) else {}

    if shape_name == "classic" and not profile_id:
        resolved_id = CLASSIC_LOCKED_PROFILE_ID
    else:
        default_id = _SHAPE_DEFAULT_PROFILE_ID.get(shape_name)
        resolved_id = str(
            profile_id or shape_defaults.get(shape_name) or default_id or f"{shape_name}_custom_v1"
        )
    if resolved_id == CLASSIC_LEGACY_PROFILE_ALIAS:
        resolved_id = CLASSIC_LOCKED_PROFILE_ID

    if shape_name == "classic":
        builtin = dict(_BUILTIN_DEFAULTS["classic"])
    else:
        builtin = dict(_BUILTIN_DEFAULTS.get(shape_name) or _BUILTIN_DEFAULTS["box"])
    builtin["shape_name"] = shape_name
    builtin["profile_id"] = resolved_id

    config_entry: Dict[str, Any] = {}
    if isinstance(profiles_doc, dict):
        raw_entry = profiles_doc.get(resolved_id)
        if isinstance(raw_entry, dict):
            config_entry = dict(raw_entry)
            alias = config_entry.get("alias_of")
            if alias and isinstance(profiles_doc.get(alias), dict):
                config_entry = {**dict(profiles_doc[alias]), **{k: v for k, v in config_entry.items() if k != "alias_of"}}
        elif resolved_id == CLASSIC_LOCKED_PROFILE_ID:
            legacy = profiles_doc.get(CLASSIC_LEGACY_PROFILE_ALIAS)
            if isinstance(legacy, dict):
                config_entry = dict(legacy)
                alias = config_entry.get("alias_of")
                if alias and isinstance(profiles_doc.get(alias), dict):
                    config_entry = dict(profiles_doc[alias])

    return _merge_profile_body(
        shape_name=shape_name,
        profile_id=resolved_id,
        base=builtin,
        overrides=config_entry,
    )


def classic_locked_profile_snapshot() -> Dict[str, Any]:
    """Immutable threshold snapshot for regression tests (CLASSIC reference baseline)."""
    return resolve_shape_validation_profile("classic").to_dict()


def register_custom_shape_validation_profile(
    shape_key: str,
    profile_body: Mapping[str, Any],
    *,
    profile_id: Optional[str] = None,
) -> ShapeValidationProfile:
    """
    Resolve a profile for a future/custom shape without changing the shared pipeline.

    Caller supplies profile_body fields; unknown shape keys use box-relative defaults as template.
    """
    try:
        shape_name = normalize_shape_key(shape_key)
    except ValueError:
        shape_name = str(shape_key or "custom").strip().lower()
    pid = str(profile_id or profile_body.get("profile_id") or f"{shape_name}_custom_v1")
    template_key = shape_name if shape_name in _BUILTIN_DEFAULTS else "box"
    base = dict(_BUILTIN_DEFAULTS[template_key])
    base.update(profile_body)
    base["shape_name"] = shape_name
    return _merge_profile_body(shape_name=shape_name, profile_id=pid, base=base)


def list_registered_validation_shapes() -> tuple[str, ...]:
    return registered_shape_keys()
