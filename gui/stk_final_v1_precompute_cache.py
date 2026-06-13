#!/usr/bin/env python3
"""
Conservative STK final-v1 precompute/cache (note-independent body signatures).

Cache root: ``audio_cache/stk_final_v1/body_signatures/``
Report: ``audio/debug_reports/stk_precompute_cache_report.json``

Safe behaviour: version/hash mismatch → recompute; read failure → recompute;
precompute failure → caller falls back to direct synthesis path.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from build_note_cache import file_sha256, optional_geometry_fingerprint
from stk_pipeline_defaults import (
    STK_PIPELINE_MODEL_VERSION,
    STK_PRECOMPUTE_SCHEMA_VERSION,
    WEBSITE_SAMPLE_ID,
    lhs_params_to_sample_parameters,
    resolve_pipeline_stk_mode_alias,
    resolved_canonical_mode,
    resolved_g_config,
    user_label_for_stk_mode,
)

STK_FINAL_CACHE_ROOT_NAME = "stk_final_v1"
BODY_SIGNATURES_SUBDIR = "body_signatures"
PRECOMPUTE_REPORT_REL = Path("audio") / "debug_reports" / "stk_precompute_cache_report.json"

REFERENCE_Z_BODY_HZ = 220.0


def stk_final_cache_root(repo_root: Path) -> Path:
    return Path(repo_root) / "audio_cache" / STK_FINAL_CACHE_ROOT_NAME


def body_signatures_dir(repo_root: Path) -> Path:
    return stk_final_cache_root(repo_root) / BODY_SIGNATURES_SUBDIR


def precompute_report_path(repo_root: Path) -> Path:
    return Path(repo_root) / PRECOMPUTE_REPORT_REL


def _sorted_sample_parameters_payload(sample_parameters: Mapping[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key in sorted(sample_parameters.keys()):
        val = sample_parameters[key]
        if isinstance(val, float):
            payload[key] = round(val, 8)
        else:
            payload[key] = val
    return payload


def compute_guitar_signature_hash(
    *,
    modal_json_sha256: str,
    geometry_fingerprint: Optional[str],
    sample_parameters: Mapping[str, Any],
    stk_model_alias: str,
    stk_pipeline_model_version: str = STK_PIPELINE_MODEL_VERSION,
    schema_version: str = STK_PRECOMPUTE_SCHEMA_VERSION,
) -> str:
    payload = {
        "schema_version": schema_version,
        "stk_pipeline_model_version": stk_pipeline_model_version,
        "stk_model_alias": str(stk_model_alias),
        "modal_json_sha256": str(modal_json_sha256),
        "geometry_fingerprint": geometry_fingerprint or "",
        "sample_parameters": _sorted_sample_parameters_payload(sample_parameters),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def body_signature_cache_path(repo_root: Path, guitar_signature_hash: str) -> Path:
    return body_signatures_dir(repo_root) / f"{guitar_signature_hash}.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_precompute_report(repo_root: Path, report: Mapping[str, Any]) -> Path:
    path = precompute_report_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2), encoding="utf-8")
    return path


def _validate_cached_bundle(
    cached: Mapping[str, Any],
    *,
    expected_hash: str,
    modal_json_sha256: str,
    stk_model_alias: str,
) -> bool:
    if str(cached.get("schema_version") or "") != STK_PRECOMPUTE_SCHEMA_VERSION:
        return False
    if str(cached.get("stk_pipeline_model_version") or "") != STK_PIPELINE_MODEL_VERSION:
        return False
    if str(cached.get("guitar_signature_hash") or "") != expected_hash:
        return False
    if str(cached.get("modal_json_sha256") or "") != modal_json_sha256:
        return False
    if str(cached.get("model_alias") or "") != stk_model_alias:
        return False
    if not cached.get("sample_parameters"):
        return False
    return True


def load_body_signature_cache(
    repo_root: Path,
    guitar_signature_hash: str,
    *,
    modal_json_sha256: str,
    stk_model_alias: str,
) -> Optional[Dict[str, Any]]:
    path = body_signature_cache_path(repo_root, guitar_signature_hash)
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(cached, dict):
        return None
    if not _validate_cached_bundle(
        cached,
        expected_hash=guitar_signature_hash,
        modal_json_sha256=modal_json_sha256,
        stk_model_alias=stk_model_alias,
    ):
        return None
    cached = dict(cached)
    cached["_cache_path"] = str(path)
    cached["_cache_hit"] = True
    return cached


def _build_reference_z_body(
    *,
    sample_parameters: Mapping[str, Any],
    modal_data: Any,
    repo_root: Path,
) -> Optional[Dict[str, Any]]:
    try:
        from body_hybrid_v4_1_identity_space import build_body_identity_vector

        return build_body_identity_vector(
            parameters=sample_parameters,
            modal_data=modal_data,
            frequency_hz=REFERENCE_Z_BODY_HZ,
            repo_root=repo_root,
            sample_id=WEBSITE_SAMPLE_ID,
        )
    except Exception:
        return None


def save_body_signature_cache(
    repo_root: Path,
    bundle: Mapping[str, Any],
) -> Path:
    sig_dir = body_signatures_dir(repo_root)
    sig_dir.mkdir(parents=True, exist_ok=True)
    path = body_signature_cache_path(repo_root, str(bundle["guitar_signature_hash"]))
    path.write_text(json.dumps(dict(bundle), indent=2), encoding="utf-8")
    return path


def ensure_stk_precompute_cache(
    *,
    repo_root: Path,
    modal_json: Path,
    modal_data: Any,
    lhs_params: Optional[Mapping[str, Any]] = None,
    sample_parameters: Optional[Mapping[str, Any]] = None,
    geometry_config: Optional[Path] = None,
    stk_mode_alias: Optional[str] = None,
    developer_debug: bool = False,
    force_recompute: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load or build note-independent STK precompute bundle.

    Returns ``(bundle, report)``. On failure the bundle still contains resolved
    mode/params so Generate can proceed via the direct synthesis path.
    """
    repo_root = Path(repo_root).resolve()
    modal_json = Path(modal_json).resolve()
    t0 = time.perf_counter()

    alias = resolve_pipeline_stk_mode_alias(
        override=stk_mode_alias,
        developer_debug=developer_debug,
    )
    canonical = resolved_canonical_mode(alias)
    g_config = resolved_g_config(alias)

    if sample_parameters is None:
        sample_parameters = lhs_params_to_sample_parameters(lhs_params or {})
    else:
        sample_parameters = dict(sample_parameters)

    modal_sha = file_sha256(modal_json) if modal_json.is_file() else ""
    geom_fp = optional_geometry_fingerprint(geometry_config)
    guitar_hash = compute_guitar_signature_hash(
        modal_json_sha256=modal_sha,
        geometry_fingerprint=geom_fp,
        sample_parameters=sample_parameters,
        stk_model_alias=alias,
    )
    cache_path = body_signature_cache_path(repo_root, guitar_hash)

    report: Dict[str, Any] = {
        "report_version": "stk_precompute_cache_report_v1",
        "timestamp_utc": _utc_now(),
        "guitar_signature_hash": guitar_hash,
        "model_alias": alias,
        "model_user_label": user_label_for_stk_mode(alias),
        "resolved_canonical_mode": canonical,
        "cache_hit": False,
        "cache_miss": True,
        "precompute_time_sec": 0.0,
        "cached_items": [],
        "cache_path": str(cache_path),
        "errors": [],
        "warnings": [],
        "fallback_used": False,
    }

    if not force_recompute:
        cached = load_body_signature_cache(
            repo_root,
            guitar_hash,
            modal_json_sha256=modal_sha,
            stk_model_alias=alias,
        )
        if cached is not None:
            report["cache_hit"] = True
            report["cache_miss"] = False
            report["precompute_time_sec"] = round(time.perf_counter() - t0, 4)
            report["cached_items"] = list(cached.get("cached_items") or [])
            _write_precompute_report(repo_root, report)
            return cached, report

    bundle: Dict[str, Any] = {
        "schema_version": STK_PRECOMPUTE_SCHEMA_VERSION,
        "stk_pipeline_model_version": STK_PIPELINE_MODEL_VERSION,
        "guitar_signature_hash": guitar_hash,
        "model_alias": alias,
        "model_user_label": user_label_for_stk_mode(alias),
        "resolved_canonical_mode": canonical,
        "g_config": g_config,
        "sample_parameters": dict(sample_parameters),
        "modal_json_sha256": modal_sha,
        "geometry_fingerprint": geom_fp,
        "website_sample_id": WEBSITE_SAMPLE_ID,
        "created_utc": _utc_now(),
        "cached_items": [
            "sample_parameters",
            "g_config",
            "model_resolution",
            "reference_z_body",
        ],
    }

    ref_z = _build_reference_z_body(
        sample_parameters=sample_parameters,
        modal_data=modal_data,
        repo_root=repo_root,
    )
    if ref_z is not None:
        bundle["reference_z_body"] = ref_z
    else:
        report["warnings"].append("reference_z_body_precompute_skipped")
        bundle["cached_items"] = [x for x in bundle["cached_items"] if x != "reference_z_body"]

    try:
        save_body_signature_cache(repo_root, bundle)
        bundle["_cache_path"] = str(cache_path)
        bundle["_cache_hit"] = False
    except OSError as exc:
        report["errors"].append(f"cache_write_failed: {exc}")
        report["fallback_used"] = True
        bundle["_cache_hit"] = False
        bundle["_cache_path"] = str(cache_path)

    report["precompute_time_sec"] = round(time.perf_counter() - t0, 4)
    report["cached_items"] = list(bundle.get("cached_items") or [])
    _write_precompute_report(repo_root, report)
    return bundle, report


def trigger_stk_precompute_if_ready(
    *,
    repo_root: Path,
    modal_json: Path,
    lhs_params: Mapping[str, Any],
    geometry_config: Optional[Path] = None,
    stk_mode_alias: Optional[str] = None,
    developer_debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Best-effort precompute when ROM body JSON is ready (Save & Sync path).
    Never raises — returns report dict or None on total skip.
    """
    modal_json = Path(modal_json)
    if not modal_json.is_file():
        return None
    try:
        from body_response_synth import load_modal_data_from_path

        modal_data = load_modal_data_from_path(modal_json)
        _bundle, report = ensure_stk_precompute_cache(
            repo_root=repo_root,
            modal_json=modal_json,
            modal_data=modal_data,
            lhs_params=lhs_params,
            geometry_config=geometry_config,
            stk_mode_alias=stk_mode_alias,
            developer_debug=developer_debug,
        )
        return report
    except Exception as exc:
        try:
            report = {
                "report_version": "stk_precompute_cache_report_v1",
                "timestamp_utc": _utc_now(),
                "cache_hit": False,
                "cache_miss": True,
                "errors": [f"precompute_trigger_failed: {exc}"],
                "fallback_used": True,
            }
            _write_precompute_report(repo_root, report)
            return report
        except OSError:
            return None
