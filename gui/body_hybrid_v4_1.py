#!/usr/bin/env python3
"""
Stage 5.1 — modal_body_hybrid_v4_1: strict f0 hybrid of baseline + radiation v1 only.

Low f0 delegates to baseline_current exactly; high f0 delegates to v1 exactly.
Transition zone uses equal-loudness crossfade on dry mixes, then unified finalize.
"""
from __future__ import annotations

import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from body_response_synth import (
    ModalInput,
    _rms,
    apply_anti_click_taper,
    apply_loudness_finalize,
)
from string_body_balance import _smoothstep

V4_1_F_LOW = 160.0
V4_1_F_HIGH = 320.0
V4_1_ENDPOINT_EPS = 1e-6


@dataclass(frozen=True)
class V4_1Ablation:
    """Reserved for future controlled additions; core and full are identical in 5.1."""

    reserved: bool = False


V4_1_MODE_ABLATIONS: Dict[str, V4_1Ablation] = {
    "modal_body_hybrid_v4_1": V4_1Ablation(),
    "modal_body_hybrid_v4_1_core": V4_1Ablation(),
    "modal_body_hybrid_v4_1_full": V4_1Ablation(),
    "stk_body_transfer_v4_1": V4_1Ablation(),
}


def get_v4_1_ablation(mode_name: Optional[str]) -> Optional[V4_1Ablation]:
    if not mode_name:
        return None
    return V4_1_MODE_ABLATIONS.get(mode_name)


def is_v4_1_family_mode(mode_name: Optional[str]) -> bool:
    return get_v4_1_ablation(mode_name) is not None


def radiation_blend_weight_f0(
    f0: float,
    *,
    f_low: float = V4_1_F_LOW,
    f_high: float = V4_1_F_HIGH,
) -> float:
    return _smoothstep(f_low, f_high, max(40.0, float(f0)))


def equal_loudness_crossfade(
    baseline: np.ndarray,
    v1: np.ndarray,
    w_rad: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """RMS-match branches only for transition blend; endpoints use pure branch."""
    w = max(0.0, min(1.0, float(w_rad)))
    b = np.asarray(baseline, dtype=np.float64)
    v = np.asarray(v1, dtype=np.float64)
    n = min(len(b), len(v))
    b, v = b[:n], v[:n]
    rb, rv = _rms(b), _rms(v)
    if w <= V4_1_ENDPOINT_EPS:
        return b.copy(), {"w_rad": w, "crossfade_mode": "baseline_only"}
    if w >= 1.0 - V4_1_ENDPOINT_EPS:
        return v.copy(), {"w_rad": w, "crossfade_mode": "v1_only"}
    target = math.sqrt(max(rb * rv, 1e-18))
    b_s = b * target / max(rb, 1e-12)
    v_s = v * target / max(rv, 1e-12)
    hybrid = (1.0 - w) * b_s + w * v_s
    return hybrid, {
        "w_rad": w,
        "crossfade_mode": "blended",
        "branch_baseline_rms": rb,
        "branch_v1_rms": rv,
        "matched_target_rms": target,
    }


def _relative_rms_error(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    if n < 8:
        return 0.0
    diff = float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))
    ref = max(_rms(b[:n]), 1e-12)
    return diff / ref


def _render_dry_branch(
    *,
    mode: str,
    frequency_hz: float,
    note_name: str,
    duration_s: float,
    sample_rate: int,
    modal_data: ModalInput,
    velocity: float,
    sample_parameters: Optional[Mapping[str, Any]],
    modal_source: Optional[str],
    synthesis_preset: Optional[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from body_response_synth import _synthesize_note_with_body_response_core
    from diagnostic_synthesis import use_diagnostic_mode
    from synthesis_presets import use_synthesis_preset

    with use_synthesis_preset(synthesis_preset):
        with use_diagnostic_mode(mode, sample_parameters=sample_parameters):
            meta = _synthesize_note_with_body_response_core(
                frequency_hz=frequency_hz,
                note_name=note_name,
                duration_s=duration_s,
                sample_rate=sample_rate,
                modal_data=modal_data,
                output_wav=Path("_dry_unused.wav"),
                velocity=velocity,
                modal_source=modal_source,
                dry_mix_only=True,
            )
    return np.asarray(meta["dry_mix"], dtype=np.float64), meta


def _finalize_and_write(
    hybrid: np.ndarray,
    *,
    output_wav: Path,
    sample_rate: int,
    duration_s: float,
    preserve: float = 0.50,
    loudness_strength: float = 0.32,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    tapered, taper_info = apply_anti_click_taper(hybrid, sample_rate, duration_s=duration_s)
    final, loudness_info = apply_loudness_finalize(
        tapered,
        sample_rate,
        raw_body_variation_preserve=preserve,
        loudness_normalization_strength=loudness_strength,
    )
    loudness_info.update(taper_info)
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(final * 32767.0, -32767, 32767).astype(np.int16)
    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return final, loudness_info


def synthesize_hybrid_v4_1_note(
    *,
    frequency_hz: float,
    note_name: str,
    duration_s: float,
    sample_rate: int,
    modal_data: ModalInput,
    output_wav: Path,
    output_metadata_json: Optional[Path],
    velocity: float,
    sample_parameters: Optional[Mapping[str, Any]],
    modal_source: Optional[str],
    diagnostic_mode: str,
    synthesis_preset: Optional[str],
    repo_root: Path,
    sample_id: str,
    f_low: float = V4_1_F_LOW,
    f_high: float = V4_1_F_HIGH,
) -> Dict[str, Any]:
    from body_response_synth import _synthesize_note_with_body_response_core
    from diagnostic_synthesis import get_diagnostic_mode, use_diagnostic_mode
    from synthesis_presets import use_synthesis_preset

    w_rad = radiation_blend_weight_f0(frequency_hz, f_low=f_low, f_high=f_high)
    cfg = get_diagnostic_mode(diagnostic_mode)

    # Exact endpoints: delegate to single-mode synthesis path (identical to baseline/v1).
    if w_rad <= V4_1_ENDPOINT_EPS:
        with use_synthesis_preset(synthesis_preset):
            with use_diagnostic_mode("baseline_current", sample_parameters=sample_parameters):
                meta = _synthesize_note_with_body_response_core(
                    frequency_hz=frequency_hz,
                    note_name=note_name,
                    duration_s=duration_s,
                    sample_rate=sample_rate,
                    modal_data=modal_data,
                    output_wav=output_wav,
                    output_metadata_json=output_metadata_json,
                    velocity=velocity,
                    modal_source=modal_source,
                )
        meta["body_hybrid_v4_1_active"] = True
        meta["radiation_blend_weight_w_rad"] = round(w_rad, 8)
        meta["v4_1_endpoint"] = "baseline_delegate"
        meta["endpoint_equivalence"] = {
            "w_rad": round(w_rad, 8),
            "endpoint_equivalence_error_to_baseline": 0.0,
            "endpoint_equivalence_error_to_v1": None,
            "difference_source": "none_baseline_delegate",
        }
        return meta

    if w_rad >= 1.0 - V4_1_ENDPOINT_EPS:
        with use_synthesis_preset(synthesis_preset):
            with use_diagnostic_mode("modal_radiation_color_v1", sample_parameters=sample_parameters):
                meta = _synthesize_note_with_body_response_core(
                    frequency_hz=frequency_hz,
                    note_name=note_name,
                    duration_s=duration_s,
                    sample_rate=sample_rate,
                    modal_data=modal_data,
                    output_wav=output_wav,
                    output_metadata_json=output_metadata_json,
                    velocity=velocity,
                    modal_source=modal_source,
                )
        meta["body_hybrid_v4_1_active"] = True
        meta["radiation_blend_weight_w_rad"] = round(w_rad, 8)
        meta["v4_1_endpoint"] = "v1_delegate"
        meta["endpoint_equivalence"] = {
            "w_rad": round(w_rad, 8),
            "endpoint_equivalence_error_to_baseline": None,
            "endpoint_equivalence_error_to_v1": 0.0,
            "difference_source": "none_v1_delegate",
        }
        return meta

    baseline_dry, meta_b = _render_dry_branch(
        mode="baseline_current",
        frequency_hz=frequency_hz,
        note_name=note_name,
        duration_s=duration_s,
        sample_rate=sample_rate,
        modal_data=modal_data,
        velocity=velocity,
        sample_parameters=sample_parameters,
        modal_source=modal_source,
        synthesis_preset=synthesis_preset,
    )
    v1_dry, meta_v1 = _render_dry_branch(
        mode="modal_radiation_color_v1",
        frequency_hz=frequency_hz,
        note_name=note_name,
        duration_s=duration_s,
        sample_rate=sample_rate,
        modal_data=modal_data,
        velocity=velocity,
        sample_parameters=sample_parameters,
        modal_source=modal_source,
        synthesis_preset=synthesis_preset,
    )
    hybrid_dry, xf_info = equal_loudness_crossfade(baseline_dry, v1_dry, w_rad)
    pre_norm_rms = _rms(hybrid_dry)

    final, loudness_info = _finalize_and_write(
        hybrid_dry,
        output_wav=output_wav,
        sample_rate=sample_rate,
        duration_s=duration_s,
        preserve=cfg.effective_loudness_preserve(),
        loudness_strength=float(cfg.final_loudness_normalization_strength),
    )
    post_norm_rms = _rms(final)

    endpoint_eq = {
        "branch_baseline_rms": round(float(xf_info.get("branch_baseline_rms") or _rms(baseline_dry)), 8),
        "branch_v1_rms": round(float(xf_info.get("branch_v1_rms") or _rms(v1_dry)), 8),
        "hybrid_pre_norm_rms": round(pre_norm_rms, 8),
        "hybrid_post_norm_rms": round(post_norm_rms, 8),
        "w_rad": round(w_rad, 8),
        "crossfade_mode": xf_info.get("crossfade_mode"),
        "endpoint_equivalence_error_to_baseline": round(_relative_rms_error(hybrid_dry, baseline_dry), 8),
        "endpoint_equivalence_error_to_v1": round(_relative_rms_error(hybrid_dry, v1_dry), 8),
        "difference_source": "transition_crossfade",
        "f_low_hz": f_low,
        "f_high_hz": f_high,
    }

    metadata: Dict[str, Any] = {
        **meta_b,
        "diagnostic_mode": diagnostic_mode,
        "body_hybrid_v4_1_active": True,
        "v4_1_endpoint": "transition_blend",
        "radiation_blend_weight_w_rad": round(w_rad, 8),
        "hybrid_formula": "(1-w_rad)*baseline_dry + w_rad*v1_dry after equal_loudness_crossfade",
        "endpoint_equivalence": endpoint_eq,
        "normalization_diagnostics": {
            "rms_gain_applied": loudness_info.get("rms_gain_applied"),
            "output_rms_dbfs": loudness_info.get("output_rms_dbfs"),
            "output_peak_dbfs": loudness_info.get("output_peak_dbfs"),
            "limiter_used": loudness_info.get("limiter_used"),
        },
    }
    if output_metadata_json:
        output_metadata_json.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return metadata
