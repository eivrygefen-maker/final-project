#!/usr/bin/env python3
"""
PGSM STK parameter export — Python orchestration only.

Exports physical synthesis parameters for the C++/STK renderer.
No audio generation, no FEM/ROM heavy calls, no WAV synthesis.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pgsm_emergency_guitar_demo_engine import (
    NOTE_PEAK_TARGET_DBFS,
    PHYSICAL_FACTOR_KEYS,
    TARGET_RMS_DBFS,
    V11_VOICING,
    _load_readonly_reference_modes,
    build_sample_synthesis_state,
)
from pgsm_step3a_numerical_ir_testbench import NUMERIC_SR
from pgsm_step5a_limited_note_set_diagnostic_audio import NOTE_FREQUENCY_HZ
from pgsm_step5l_limited_multiguitar_differentiation import (
    REFERENCE_SAMPLE_ID,
    extract_per_sample_physical_parameters,
)
from stk_v6_2_audit_features import load_audit_report

EXPORT_VERSION = "pgsm_stk_parameter_export_v1"
RENDERER_TARGET = "stk_cpp"
PYTHON_ROLE = "parameter_export_only"
DURATION_S = 2.5
SAMPLE_SET: Tuple[str, ...] = ("sample_000", "sample_001", "sample_002")
NOTE_SET: Tuple[str, ...] = ("A2", "A4", "E5")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_stk_demo_parameters.json"
STK_AUDIO_DIR = REPO_ROOT / "audio" / "pgsm_stk_guitar_demo"

REQUIRED_RENDER_GROUPS: Tuple[str, ...] = (
    "string_model",
    "bridge_model",
    "body_model",
    "material_model",
    "radiation_model",
    "output_model",
)

SAMPLE_PROFILES: Dict[str, str] = {
    "sample_000": "balanced_neutral",
    "sample_001": "bright_light_fast",
    "sample_002": "warm_deep_heavy",
}

# Lightweight fallback when stk_v6 audit JSON is absent (no FEM/ROM load).
FALLBACK_PHYSICAL: Dict[str, Dict[str, Any]] = {
    "sample_000": {
        "sample_id": "sample_000",
        "body_depth_m": 0.100,
        "body_volume_proxy": 0.0130,
        "helmholtz_like_frequency_proxy": 118.0,
        "bridge_mobility_proxy": 1.00,
        "top_damping_coeff_proxy": 1.00,
        "back_damping_coeff_proxy": 1.00,
        "soundhole_area": 0.00636,
        "mass_proxies": {"mixed_body_mass_proxy": 1.00},
    },
    "sample_001": {
        "sample_id": "sample_001",
        "body_depth_m": 0.092,
        "body_volume_proxy": 0.0116,
        "helmholtz_like_frequency_proxy": 126.0,
        "bridge_mobility_proxy": 1.14,
        "top_damping_coeff_proxy": 1.12,
        "back_damping_coeff_proxy": 0.96,
        "soundhole_area": 0.00585,
        "mass_proxies": {"mixed_body_mass_proxy": 0.88},
    },
    "sample_002": {
        "sample_id": "sample_002",
        "body_depth_m": 0.112,
        "body_volume_proxy": 0.0148,
        "helmholtz_like_frequency_proxy": 108.0,
        "bridge_mobility_proxy": 0.88,
        "top_damping_coeff_proxy": 0.90,
        "back_damping_coeff_proxy": 1.08,
        "soundhole_area": 0.00700,
        "mass_proxies": {"mixed_body_mass_proxy": 1.16},
    },
}

NOTE_BODY_SUPPORT: Dict[str, Dict[str, float]] = {
    "A2": {"body_modal_mult": 1.00, "low_mid_mode_mult": 1.00, "high_rad_mult": 1.00, "tau_mult": 1.00},
    "A4": {"body_modal_mult": 1.08, "low_mid_mode_mult": 1.12, "high_rad_mult": 0.95, "tau_mult": 1.10},
    "E5": {"body_modal_mult": 1.12, "low_mid_mode_mult": 1.15, "high_rad_mult": 0.90, "tau_mult": 1.14},
}

PER_SAMPLE_NOTE_SUPPORT: Dict[str, Dict[str, Dict[str, float]]] = {
    "A4": {
        "sample_000": {"low_mid_mode_mult": 1.10},
        "sample_001": {"low_mid_mode_mult": 1.06},
        "sample_002": {"low_mid_mode_mult": 1.16},
    },
    "E5": {
        "sample_000": {"low_mid_mode_mult": 1.13},
        "sample_001": {"low_mid_mode_mult": 1.08},
        "sample_002": {"low_mid_mode_mult": 1.20},
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def expected_wav_filename(sample_id: str, note_name: str) -> str:
    return f"{sample_id}_{note_name}_stk_guitar.wav"


def expected_wav_paths(repo_root: Optional[Path] = None) -> List[Path]:
    root = Path(repo_root or REPO_ROOT)
    out_dir = root / "audio" / "pgsm_stk_guitar_demo"
    return [
        out_dir / expected_wav_filename(sample_id, note)
        for sample_id in SAMPLE_SET
        for note in NOTE_SET
    ]


def load_physical_parameters(
    sample_id: str,
    *,
    audit: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Load per-sample physical proxies from audit JSON or lightweight fallback."""
    if audit is not None:
        return extract_per_sample_physical_parameters(sample_id, audit)
    try:
        audit_doc = load_audit_report()
        return extract_per_sample_physical_parameters(sample_id, audit_doc)
    except (FileNotFoundError, KeyError, ValueError):
        fb = FALLBACK_PHYSICAL.get(sample_id)
        if fb is None:
            raise KeyError(f"no physical fallback for {sample_id!r}")
        return dict(fb)


def _note_support(sample_id: str, note_name: str) -> Dict[str, float]:
    base = dict(NOTE_BODY_SUPPORT.get(note_name, NOTE_BODY_SUPPORT["A2"]))
    extra = (PER_SAMPLE_NOTE_SUPPORT.get(note_name) or {}).get(sample_id) or {}
    base.update(extra)
    return base


def _string_decay(factors: Mapping[str, float], note_name: str) -> float:
    top_damp = float(factors.get("top_damping_factor") or 1.0)
    brightness = float(factors.get("radiation_brightness_factor") or 1.0)
    # Higher notes decay faster; bright guitars slightly faster.
    note_scale = {"A2": 0.92, "A4": 1.00, "E5": 1.08}.get(note_name, 1.0)
    sustain = 0.68 / (top_damp ** 0.22 * brightness ** 0.08 * note_scale)
    return round(_clamp(sustain, 0.42, 0.88), 6)


def _harmonic_brightness(factors: Mapping[str, float], mix: Mapping[str, Any]) -> float:
    rad = float(factors.get("radiation_brightness_factor") or 1.0)
    stiff = float(factors.get("top_stiffness_to_weight_factor") or 1.0)
    string_share = float(mix.get("string_bridge") or 0.25)
    return round(_clamp(0.55 + 0.22 * rad + 0.10 * stiff + 0.18 * string_share, 0.35, 1.25), 6)


def _modes_for_stk(
    modes: Sequence[Mapping[str, Any]],
    *,
    note_support: Mapping[str, float],
    factors: Mapping[str, float],
) -> List[Dict[str, Any]]:
    body_mult = float(note_support.get("body_modal_mult") or 1.0)
    low_mid_mult = float(note_support.get("low_mid_mode_mult") or 1.0)
    tau_mult = float(note_support.get("tau_mult") or 1.0)
    soundhole = float(factors.get("soundhole_radiation_factor") or 1.0)
    out: List[Dict[str, Any]] = []
    for row in modes:
        f_hz = float(row["frequency_hz"])
        gain = float(row.get("gain") or 0.0)
        component = str(row.get("component") or "top")
        tau = float(row.get("tau_s") or 0.08) * tau_mult
        q = float(row.get("q") or max(8.0, math.pi * f_hz * tau))
        if component in ("back", "air") and f_hz < 260.0:
            gain *= low_mid_mult
        if component == "air":
            gain *= soundhole
        gain *= body_mult
        out.append(
            {
                "frequency_hz": round(f_hz, 4),
                "gain": round(gain, 6),
                "tau_or_q": round(tau, 6),
                "q": round(q, 4),
                "component": component,
                "role": str(row.get("role") or component),
            }
        )
    return out


def _radiation_weights(mix: Mapping[str, Any], factors: Mapping[str, float]) -> Dict[str, float]:
    air_share = float(mix.get("air_share") or 0.10)
    body_share = float(mix.get("body_modal") or 0.65)
    string_share = float(mix.get("string_bridge") or 0.25)
    warmth = float(factors.get("back_density_warmth_factor") or 1.0)
    brightness = float(factors.get("radiation_brightness_factor") or 1.0)
    top_w = round(_clamp(0.42 * body_share * brightness, 0.15, 0.75), 6)
    back_w = round(_clamp(0.34 * body_share * warmth, 0.12, 0.65), 6)
    air_w = round(_clamp(air_share * float(factors.get("air_helmholtz_factor") or 1.0), 0.04, 0.35), 6)
    total = top_w + back_w + air_w + string_share
    if total > 1e-9:
        scale = (1.0 - string_share) / (top_w + back_w + air_w)
        top_w *= scale
        back_w *= scale
        air_w *= scale
    return {
        "radiation_brightness": round(brightness, 6),
        "top_weight": round(top_w, 6),
        "back_weight": round(back_w, 6),
        "air_weight": round(air_w, 6),
        "string_direct_weight": round(string_share, 6),
    }


def build_render_entry(
    sample_id: str,
    note_name: str,
    *,
    physical: Mapping[str, Any],
    reference_physical: Mapping[str, Any],
    sample_rate: int = NUMERIC_SR,
    duration_s: float = DURATION_S,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build one STK render parameter block (no audio)."""
    root = Path(repo_root or REPO_ROOT)
    readonly_modes = _load_readonly_reference_modes(root)
    state = build_sample_synthesis_state(
        sample_id,
        physical=dict(physical),
        reference_physical=dict(reference_physical),
        readonly_modes=readonly_modes,
        voicing=V11_VOICING,
    )
    factors = state.factors
    mix = state.mix_ratios
    note_support = _note_support(sample_id, note_name)
    modes = _modes_for_stk(state.modes, note_support=note_support, factors=factors)
    bridge = state.bridge_transfer
    wav_name = expected_wav_filename(sample_id, note_name)
    wav_path = root / "audio" / "pgsm_stk_guitar_demo" / wav_name

    string_to_body = round(
        _clamp(1.0 - float(mix.get("string_bridge") or 0.25) + 0.12 * float(factors.get("bridge_mobility_factor") or 1.0) - 0.12, 0.35, 0.92),
        6,
    )

    return {
        "sample_id": sample_id,
        "note_name": note_name,
        "frequency_hz": float(NOTE_FREQUENCY_HZ[note_name]),
        "duration_s": float(duration_s),
        "sample_rate": int(sample_rate),
        "profile": SAMPLE_PROFILES.get(sample_id, state.voicing_profile),
        "physical_factors": {k: factors[k] for k in PHYSICAL_FACTOR_KEYS if k in factors},
        "string_model": {
            "pluck_position": state.pluck_position_ratio,
            "string_decay": _string_decay(factors, note_name),
            "harmonic_brightness": _harmonic_brightness(factors, mix),
            "excitation_strength": round(_clamp(0.82 + 0.10 * float(factors.get("bridge_mobility_factor") or 1.0), 0.65, 1.15), 6),
        },
        "bridge_model": {
            "bridge_mobility": round(float(factors.get("bridge_mobility_factor") or 1.0), 6),
            "bridge_damping": round(0.045 / max(float(bridge.get("attack_scale") or 1.0), 0.5), 6),
            "string_to_body_send": string_to_body,
            "highpass_hz": float(bridge.get("highpass_hz") or 55.0),
            "low_coupling_scale": float(bridge.get("low_coupling_scale") or 1.0),
        },
        "body_model": {
            "effective_mass_loading": round(float(factors.get("effective_mass_loading_factor") or 1.0), 6),
            "body_size_cavity_factor": round(float(factors.get("body_size_cavity_factor") or 1.0), 6),
            "depth_factor": round(float(physical.get("body_depth_m") or 0.10) / 0.10, 6),
            "soundhole_radiation_factor": round(float(factors.get("soundhole_radiation_factor") or 1.0), 6),
            "low_mid_body_support": round(float(note_support.get("low_mid_mode_mult") or 1.0), 6),
            "modes": modes,
        },
        "material_model": {
            "top_damping": round(float(factors.get("top_damping_factor") or 1.0), 6),
            "back_warmth": round(float(factors.get("back_density_warmth_factor") or 1.0), 6),
            "material_loss": round(float(factors.get("top_damping_factor") or 1.0) * 0.92, 6),
            "stiffness_to_weight": round(float(factors.get("top_stiffness_to_weight_factor") or 1.0), 6),
        },
        "radiation_model": _radiation_weights(mix, factors),
        "output_model": {
            "peak_target_dbfs": float(NOTE_PEAK_TARGET_DBFS.get(note_name, -6.0)),
            "loudness_target": float(TARGET_RMS_DBFS),
            "output_wav_path": str(wav_path.relative_to(root)).replace("\\", "/"),
        },
    }


def build_parameter_export(
    *,
    repo_root: Optional[Path] = None,
    audit: Optional[Mapping[str, Any]] = None,
    sample_rate: int = NUMERIC_SR,
    duration_s: float = DURATION_S,
) -> Dict[str, Any]:
    """Assemble full demo parameter JSON for C++/STK renderer."""
    root = Path(repo_root or REPO_ROOT)
    reference_physical = load_physical_parameters(REFERENCE_SAMPLE_ID, audit=audit)
    renders: List[Dict[str, Any]] = []
    per_sample_summary: Dict[str, Any] = {}

    for sample_id in SAMPLE_SET:
        physical = load_physical_parameters(sample_id, audit=audit)
        per_sample_summary[sample_id] = {
            "profile": SAMPLE_PROFILES.get(sample_id),
            "physical_source": "audit_json" if audit is not None else "audit_or_fallback",
            "body_depth_m": physical.get("body_depth_m"),
            "bridge_mobility_proxy": physical.get("bridge_mobility_proxy"),
        }
        for note_name in NOTE_SET:
            renders.append(
                build_render_entry(
                    sample_id,
                    note_name,
                    physical=physical,
                    reference_physical=reference_physical,
                    sample_rate=sample_rate,
                    duration_s=duration_s,
                    repo_root=root,
                )
            )

    return {
        "export_version": EXPORT_VERSION,
        "generated_at": _utc_now(),
        "renderer": RENDERER_TARGET,
        "python_role": PYTHON_ROLE,
        "repo_root": str(root),
        "sample_set": list(SAMPLE_SET),
        "note_set": list(NOTE_SET),
        "sample_rate": int(sample_rate),
        "duration_s": float(duration_s),
        "physical_factor_keys": list(PHYSICAL_FACTOR_KEYS),
        "per_sample_summary": per_sample_summary,
        "per_sample_differences": _per_sample_difference_summary(renders),
        "renders": renders,
        "expected_render_count": len(SAMPLE_SET) * len(NOTE_SET),
        "expected_wav_files": [expected_wav_filename(s, n) for s in SAMPLE_SET for n in NOTE_SET],
        "limitations": [
            "Python exports parameters only; WAV synthesis is C++/STK on VM.",
            "Modal catalog is read-only PGSM reference — not live FEM/ROM at export time.",
            "Body response in STK must be driven by bridge force, not an independent pluck.",
        ],
    }


def _per_sample_difference_summary(renders: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Highlight explicit physical deltas between demo guitars (A2 anchor)."""
    by_sample: Dict[str, Dict[str, Any]] = {}
    for row in renders:
        if row.get("note_name") != "A2":
            continue
        sid = str(row["sample_id"])
        by_sample[sid] = {
            "profile": row.get("profile"),
            "string_to_body_send": (row.get("bridge_model") or {}).get("string_to_body_send"),
            "bridge_mobility": (row.get("bridge_model") or {}).get("bridge_mobility"),
            "soundhole_radiation_factor": (row.get("body_model") or {}).get("soundhole_radiation_factor"),
            "effective_mass_loading": (row.get("body_model") or {}).get("effective_mass_loading"),
            "radiation_brightness": (row.get("radiation_model") or {}).get("radiation_brightness"),
            "top_damping": (row.get("material_model") or {}).get("top_damping"),
            "first_mode_hz": ((row.get("body_model") or {}).get("modes") or [{}])[0].get("frequency_hz"),
        }
    ref = by_sample.get("sample_000") or {}
    deltas: Dict[str, Dict[str, float]] = {}
    for sid, row in by_sample.items():
        if sid == "sample_000":
            continue
        deltas[sid] = {
            k: round(float(row.get(k) or 0.0) - float(ref.get(k) or 0.0), 6)
            for k in row
            if k != "profile" and isinstance(row.get(k), (int, float))
        }
    return {"A2_anchor": by_sample, "delta_vs_sample_000": deltas}


def write_parameter_export(
    output_path: Optional[Path] = None,
    *,
    repo_root: Optional[Path] = None,
    audit: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write parameter JSON to disk."""
    root = Path(repo_root or REPO_ROOT)
    out = Path(output_path or (root / "audio" / "debug_reports" / "pgsm_stk_demo_parameters.json"))
    doc = build_parameter_export(repo_root=root, audit=audit)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export PGSM physical parameters for STK/C++ renderer.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path (default: audio/debug_reports/pgsm_stk_demo_parameters.json)",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    path = write_parameter_export(args.output, repo_root=args.repo_root)
    print(f"Wrote STK parameter export: {path}")
    print(f"Renders: {len(SAMPLE_SET) * len(NOTE_SET)} (no audio generated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
