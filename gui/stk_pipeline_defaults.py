#!/usr/bin/env python3
"""Website/pipeline STK defaults — frozen Stage 5.1H final candidate."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from body_hybrid_v4_1_identity_space import (
    STK_BODY_TRANSFER_FINAL_V1,
    STK_BODY_TRANSFER_FINAL_V1_DE_THUMP,
    STK_FINAL_CANDIDATE_CANONICAL,
    STK_FINAL_DE_THUMP_CANONICAL,
    STK_FINAL_GUI_LABEL,
    canonical_stk_final_mode,
    g_config_for_mode,
    requires_identity_contrast_context,
)
from sample_parameters import normalize_sample_parameters

# Bump when precompute payload semantics change (invalidates body_signature cache).
STK_PRECOMPUTE_SCHEMA_VERSION = "stk_final_v1_precompute_v1"
STK_PIPELINE_MODEL_VERSION = "stk_final_v1_30_70"

DEFAULT_WEBSITE_STK_MODE = STK_BODY_TRANSFER_FINAL_V1
DEFAULT_WEBSITE_STK_LABEL = STK_FINAL_GUI_LABEL
DEBUG_WEBSITE_STK_MODE = STK_BODY_TRANSFER_FINAL_V1_DE_THUMP
DEBUG_WEBSITE_STK_LABEL = "Physical Body Identity v1 (de-thump debug)"

WEBSITE_SAMPLE_ID = "website"

STK_MODE_REGISTRY: Dict[str, Dict[str, str]] = {
    STK_BODY_TRANSFER_FINAL_V1: {
        "canonical_mode": STK_FINAL_CANDIDATE_CANONICAL,
        "user_label": STK_FINAL_GUI_LABEL,
        "exposure": "default",
    },
    STK_BODY_TRANSFER_FINAL_V1_DE_THUMP: {
        "canonical_mode": STK_FINAL_DE_THUMP_CANONICAL,
        "user_label": DEBUG_WEBSITE_STK_LABEL,
        "exposure": "debug_only",
    },
}

DEBUG_STK_MODE_CHOICES: Tuple[Tuple[str, str], ...] = (
    (STK_BODY_TRANSFER_FINAL_V1, STK_FINAL_GUI_LABEL),
    (STK_BODY_TRANSFER_FINAL_V1_DE_THUMP, DEBUG_WEBSITE_STK_LABEL),
)


def resolve_pipeline_stk_mode_alias(
    *,
    override: Optional[str] = None,
    developer_debug: bool = False,
) -> str:
    """Return the STK alias used by Generate (never canonical mode names)."""
    if override:
        alias = str(override)
        if alias in STK_MODE_REGISTRY:
            if STK_MODE_REGISTRY[alias]["exposure"] == "debug_only" and not developer_debug:
                return DEFAULT_WEBSITE_STK_MODE
            return alias
    return DEFAULT_WEBSITE_STK_MODE


def resolved_canonical_mode(stk_mode_alias: str) -> str:
    return canonical_stk_final_mode(stk_mode_alias)


def user_label_for_stk_mode(stk_mode_alias: str) -> str:
    entry = STK_MODE_REGISTRY.get(stk_mode_alias)
    if entry:
        return str(entry["user_label"])
    return STK_FINAL_GUI_LABEL


def debug_stk_mode_options() -> Sequence[Tuple[str, str]]:
    """(alias, user_label) pairs for developer UI only."""
    return DEBUG_STK_MODE_CHOICES


def lhs_params_to_sample_parameters(
    lhs_params: Mapping[str, Any],
    *,
    sample_id: str = WEBSITE_SAMPLE_ID,
) -> Dict[str, Any]:
    """Convert ROM/LHS dict from the website into synthesis sample_parameters."""
    raw = dict(lhs_params)
    out = normalize_sample_parameters(raw)
    out["sample_id"] = sample_id
    if "top_wood_id" not in out and raw.get("materials.top.wood_id"):
        out["top_wood_id"] = raw["materials.top.wood_id"]
    if "back_wood_id" not in out and raw.get("materials.back.wood_id"):
        out["back_wood_id"] = raw["materials.back.wood_id"]
    for key in (
        "geometry.length",
        "geometry.width",
        "geometry.depth",
        "geometry.top_thickness",
        "geometry.hole_radius",
    ):
        if key in raw and key not in out:
            out[key] = raw[key]
    if "geometry.shape_type" in raw:
        out["geometry.shape_type"] = raw["geometry.shape_type"]
    return out


def resolved_g_config(stk_mode_alias: str) -> Dict[str, Any]:
    return dict(g_config_for_mode(stk_mode_alias))


def enrich_sample_parameters_for_note(
    *,
    base_parameters: Mapping[str, Any],
    modal_data: Any,
    frequency_hz: float,
    stk_mode_alias: str,
    sample_id: str = WEBSITE_SAMPLE_ID,
    repo_root: Optional[Any] = None,
) -> Dict[str, Any]:
    """Attach per-note identity contrast context when the STK mode requires it."""
    from body_hybrid_v4_1_identity_space import (
        build_batch_contrast_context,
        build_body_identity_vector,
    )

    params = dict(normalize_sample_parameters(base_parameters))
    params["sample_id"] = sample_id
    if not requires_identity_contrast_context(stk_mode_alias):
        return params
    z_body = build_body_identity_vector(
        parameters=params,
        modal_data=modal_data,
        frequency_hz=float(frequency_hz),
        repo_root=repo_root,
        sample_id=sample_id,
    )
    contrast_ctx = build_batch_contrast_context({sample_id: z_body}).get(sample_id) or {}
    params["identity_contrast_context"] = contrast_ctx
    return params
