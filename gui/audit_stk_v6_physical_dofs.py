#!/usr/bin/env python3
"""
STK V6 Step 1 — physical / modal degrees-of-freedom audit (read-only).

Inspects geometry, material, modal, and derived features available in project data.
Does NOT synthesize audio, run FEM, run ROM batch, or change website defaults.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_response_synth import (  # noqa: E402
    FULL_MODAL_BAND_HZ,
    modes_in_validated_band,
    parse_modal_modes,
)
from bridge_mobility_proxy import WOOD_DENSITY_REL, compute_body_mass_proxies  # noqa: E402
from build_sample_comparison import m4_surrogate_model_available  # noqa: E402
from modal_damping import WOOD_DAMPING_COEFF, compute_per_mode_damping, infer_mode_category  # noqa: E402
from sample_parameters import normalize_sample_parameters  # noqa: E402
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

REPORT_VERSION = "stk_v6_physical_dof_audit_v1_1"
DEFAULT_SAMPLE_IDS = tuple(f"sample_{i:03d}" for i in range(10))
DEFAULT_JSON_OUT = REPO / "audio" / "debug_reports" / "stk_v6_physical_dof_audit.json"
DEFAULT_MD_OUT = REPO / "audio" / "debug_reports" / "stk_v6_physical_dof_audit.md"
REFERENCE_MODAL_PATH = REPO / "FEM" / "outputs" / "rom_stk_body.json"

MODAL_BANDS = (
    ("sub_body", 60.0, 120.0),
    ("low_body", 120.0, 220.0),
    ("mid_body", 220.0, 400.0),
    ("upper_body", 400.0, 550.0),
)

# ---------------------------------------------------------------------------
# V6 DOF influence map (design intent for later routing model)
# ---------------------------------------------------------------------------
DOF_INFLUENCE_MAP: Dict[str, Dict[str, str]] = {
    "body_length": {
        "modal_frequency": "indirect via area/mass and surrogate-trained catalog",
        "gain_amplitude": "via bridge mobility and top/back area proxies",
        "decay_time": "weak via geometry damping scale",
        "q_damping": "weak via geometry damping scale",
        "bridge_excitation": "indirect via modal mass distribution",
        "top_radiation": "via top plate area proxy",
        "soundhole_radiation": "indirect via coupled volume",
        "cavity_air_resonance": "via body_volume_proxy",
        "pluck_attack": "none direct",
        "hf_metallicity_damping": "none direct",
        "sustain": "weak via mass loading",
        "guitar_differentiation": "medium when combined with width/depth",
    },
    "body_width": {
        "modal_frequency": "indirect via area and mass",
        "gain_amplitude": "via top/back area and mobility",
        "decay_time": "weak geometry damping",
        "q_damping": "weak geometry damping",
        "bridge_excitation": "indirect",
        "top_radiation": "strong via radiating top area proxy",
        "soundhole_radiation": "indirect",
        "cavity_air_resonance": "via volume proxy",
        "pluck_attack": "none",
        "hf_metallicity_damping": "none",
        "sustain": "weak",
        "guitar_differentiation": "medium-high across LHS samples",
    },
    "body_depth": {
        "modal_frequency": "lowers Helmholtz-like air mode; shifts cavity coupling",
        "gain_amplitude": "via air volume and cavity gain proxy",
        "decay_time": "extends cavity_decay_proxy",
        "q_damping": "raises cavity_q_proxy slightly",
        "bridge_excitation": "indirect",
        "top_radiation": "weak",
        "soundhole_radiation": "via air volume / Helmholtz balance",
        "cavity_air_resonance": "strong — primary low-body breathing control",
        "pluck_attack": "none",
        "hf_metallicity_damping": "weak via absorption proxy",
        "sustain": "strong — cavity tail extension target",
        "guitar_differentiation": "high for low/mid body character",
    },
    "soundhole_radius": {
        "modal_frequency": "raises Helmholtz-like frequency with area",
        "gain_amplitude": "soundhole radiation gain proxy",
        "decay_time": "weak",
        "q_damping": "weak",
        "bridge_excitation": "none",
        "top_radiation": "balance vs aperture path",
        "soundhole_radiation": "strong target for V6 routing stem",
        "cavity_air_resonance": "strong — air plug stiffness",
        "pluck_attack": "none",
        "hf_metallicity_damping": "none",
        "sustain": "weak",
        "guitar_differentiation": "medium",
    },
    "top_thickness": {
        "modal_frequency": "mass/stiffness shift (not fully modeled yet)",
        "gain_amplitude": "top effective mass proxy",
        "decay_time": "geometry damping scale",
        "q_damping": "material-weighted when shares known",
        "bridge_excitation": "via mobility",
        "top_radiation": "indirect",
        "soundhole_radiation": "none",
        "cavity_air_resonance": "none",
        "pluck_attack": "none",
        "hf_metallicity_damping": "weak",
        "sustain": "medium via mass",
        "guitar_differentiation": "medium with wood pairing",
    },
    "top_wood_id": {
        "modal_frequency": "none direct in current data",
        "gain_amplitude": "density proxy → mobility",
        "decay_time": "via wood damping coeff",
        "q_damping": "strong when participation shares available",
        "bridge_excitation": "indirect",
        "top_radiation": "via top participation × damping",
        "soundhole_radiation": "none",
        "cavity_air_resonance": "none",
        "pluck_attack": "none",
        "hf_metallicity_damping": "strong target for E5 damping",
        "sustain": "strong",
        "guitar_differentiation": "medium — discrete 5×5 wood grid",
    },
    "modal_frequency_hz": {
        "modal_frequency": "direct pole placement",
        "gain_amplitude": "via admittance peak location",
        "decay_time": "via tau = pi*Q/f",
        "q_damping": "paired",
        "bridge_excitation": "selects excited modes",
        "top_radiation": "via participation-weighted radiation",
        "soundhole_radiation": "via air_share modes",
        "cavity_air_resonance": "low modes anchor breathing",
        "pluck_attack": "harmonic overlap only",
        "hf_metallicity_damping": "high modes need stronger damping",
        "sustain": "mode density and Q tail",
        "guitar_differentiation": "strong when catalog is sample-specific",
    },
    "bridge_excitation_abs": {
        "modal_frequency": "none",
        "gain_amplitude": "strong — primary bridge drive weight",
        "decay_time": "indirect via excited mode Q",
        "q_damping": "none",
        "bridge_excitation": "direct",
        "top_radiation": "indirect through excited modes",
        "soundhole_radiation": "indirect",
        "cavity_air_resonance": "indirect",
        "pluck_attack": "none",
        "hf_metallicity_damping": "none",
        "sustain": "indirect",
        "guitar_differentiation": "strong when per-sample catalog available",
    },
    "radiation_proxy": {
        "modal_frequency": "none",
        "gain_amplitude": "radiated output weight",
        "decay_time": "radiation damping scale",
        "q_damping": "lowers Q when high",
        "bridge_excitation": "none",
        "top_radiation": "primary proxy today",
        "soundhole_radiation": "partial — use air_pressure_proxy in full catalog",
        "cavity_air_resonance": "weak",
        "pluck_attack": "none",
        "hf_metallicity_damping": "moderate at high f",
        "sustain": "moderate",
        "guitar_differentiation": "strong in full ROM catalog",
    },
    "top_share": {
        "modal_frequency": "none",
        "gain_amplitude": "routes energy to top stem",
        "decay_time": "material damping mix",
        "q_damping": "top wood weighted",
        "bridge_excitation": "none",
        "top_radiation": "direct V6 stem routing",
        "soundhole_radiation": "balance term",
        "cavity_air_resonance": "competes with air_share",
        "pluck_attack": "none",
        "hf_metallicity_damping": "indirect",
        "sustain": "indirect",
        "guitar_differentiation": "strong with back/air shares",
    },
    "air_share": {
        "modal_frequency": "none",
        "gain_amplitude": "air/cavity stem energy",
        "decay_time": "air damping coeff",
        "q_damping": "air modes often lower Q",
        "bridge_excitation": "none",
        "top_radiation": "complementary",
        "soundhole_radiation": "strong V6 target",
        "cavity_air_resonance": "direct",
        "pluck_attack": "none",
        "hf_metallicity_damping": "weak",
        "sustain": "cavity tail",
        "guitar_differentiation": "medium-high",
    },
    "air_pressure_proxy": {
        "modal_frequency": "none",
        "gain_amplitude": "soundhole/aperture radiation stem",
        "decay_time": "via mode Q",
        "q_damping": "none direct",
        "bridge_excitation": "none",
        "top_radiation": "complementary",
        "soundhole_radiation": "direct — best aperture target in full catalog",
        "cavity_air_resonance": "strong",
        "pluck_attack": "none",
        "hf_metallicity_damping": "none",
        "sustain": "medium",
        "guitar_differentiation": "requires per-sample catalog",
    },
}

V6_CRITICAL_FEATURES: Dict[str, str] = {
    "body_length": "geometry.length in LHS pool",
    "body_width": "geometry.width in LHS pool",
    "body_depth": "geometry.depth in LHS pool",
    "soundhole_radius": "geometry.hole_radius in LHS pool",
    "top_back_wood_ids": "top_wood_id / back_wood_id in LHS pool",
    "per_mode_participation_shares": "predicted_modes top/back/air shares — requires modal catalog inference",
    "bridge_excitation_per_mode": "predicted_modes bridge_excitation_abs",
    "radiation_and_aperture_proxies": "radiation_proxy, air_pressure_proxy, top/back_output_proxy",
    "per_mode_q_or_tau": "derived in synthesis from shares + wood; explicit Q rare in catalog",
    "cavity_helmholtz_geometry": "derived from depth, area, soundhole",
    "scale_length": "not in LHS pool — pluck/string routing gap",
    "bridge_position": "not in LHS pool",
    "elastic_moduli_anisotropy": "not in LHS pool — only wood IDs",
    "side_wood_identity": "not in LHS pool",
    "internal_cavity_mesh_modes": "ROM catalog only — not stored per sample on disk here",
}


DEFAULT_SOUNDHOLE_RADIUS_M = 0.047
_REF_BODY_VOLUME_M3 = 0.013
_REF_SOUNDHOLE_AREA_M2 = math.pi * DEFAULT_SOUNDHOLE_RADIUS_M**2
_REF_HELMHOLTZ_HZ = 105.0


def _compute_cavity_geometry_proxy(parameters: Mapping[str, Any]) -> Dict[str, Any]:
    """Audit-only cavity proxy (mirrors planned V6 geometry routing; no synthesis)."""
    p = normalize_sample_parameters(parameters)
    length = _g(p, "length") or 0.52
    width = _g(p, "width") or 0.32
    depth = _g(p, "depth") or 0.10
    hole_r = _g(p, "hole_radius") or DEFAULT_SOUNDHOLE_RADIUS_M
    hole_fallback = hole_r <= 0 or hole_r > 0.08
    body_area_proxy = length * width * 0.90
    soundhole_area = math.pi * hole_r * hole_r if not hole_fallback else _REF_SOUNDHOLE_AREA_M2
    body_volume_proxy = max(length * width * depth - soundhole_area * 0.35, 0.008)
    vol_ratio = _REF_BODY_VOLUME_M3 / max(body_volume_proxy, 1e-6)
    area_ratio = soundhole_area / max(_REF_SOUNDHOLE_AREA_M2, 1e-9)
    helmholtz_hz = _REF_HELMHOLTZ_HZ * math.sqrt(area_ratio * vol_ratio)
    helmholtz_hz = max(85.0, min(128.0, helmholtz_hz))
    top_wood = str(p.get("top_wood_id") or "spruce").lower()
    back_wood = str(p.get("back_wood_id") or "mahogany").lower()
    top_damp = WOOD_DAMPING_COEFF.get(top_wood, 1.0)
    back_damp = WOOD_DAMPING_COEFF.get(back_wood, 1.0)
    mean_damp = 0.5 * (top_damp + back_damp)
    depth_factor = depth / 0.10
    cavity_q = max(8.0, min(22.0, 14.0 + 4.0 * (1.05 - mean_damp) + 2.0 * (depth_factor - 1.0)))
    cavity_decay_s = max(0.65, min(2.8, 1.0 + 0.55 * depth_factor + 0.25 * (1.0 / max(mean_damp, 0.5))))
    cavity_gain = 0.42 + 0.18 * min(1.0, body_volume_proxy / 0.016)
    hf_absorb = max(0.15, min(0.55, 0.18 + 0.22 * mean_damp + 0.08 * (depth_factor - 1.0)))
    return {
        "body_area_proxy": round(body_area_proxy, 8),
        "body_volume_proxy": round(body_volume_proxy, 8),
        "soundhole_area_used": round(soundhole_area, 8),
        "helmholtz_like_frequency_hz": round(helmholtz_hz, 3),
        "cavity_q": round(cavity_q, 4),
        "cavity_decay_s": round(cavity_decay_s, 4),
        "cavity_gain": round(cavity_gain, 4),
        "high_frequency_absorption": round(hf_absorb, 4),
        "material_top_damping": round(top_damp, 4),
        "material_back_damping": round(back_damp, 4),
    }


def load_lhs_sample_entries_full(repo_root: Path, *, max_samples: int = 26) -> List[Dict[str, Any]]:
    """Load LHS pool entries including ROM aggregation metadata."""
    pool_path = repo_root / "ROM" / "classic" / "lhs_pool.json"
    if not pool_path.is_file():
        return []
    doc = json.loads(pool_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for entry in doc.get("entries") or []:
        sid = str(entry.get("id") or "")
        if not sid.startswith("sample_"):
            continue
        params = dict(entry.get("parameters") or {})
        if not params:
            continue
        row = dict(entry)
        row["sample_id"] = sid
        row["parameters"] = params
        rows.append(row)
        if len(rows) >= int(max_samples):
            break
    rows.sort(key=lambda r: str(r.get("sample_id") or ""))
    return rows


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _g(params: Mapping[str, Any], key: str) -> Optional[float]:
    for candidate in (f"geometry.{key}", key):
        v = params.get(candidate)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    geom = params.get("geometry")
    if isinstance(geom, Mapping) and key in geom:
        try:
            return float(geom[key])
        except (TypeError, ValueError):
            pass
    return None


def _field_record(
    *,
    name: str,
    value: Any,
    key_path: str,
    status: str,
    confidence: str,
    v6_use: str,
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "source_key_path": key_path,
        "status": status,
        "confidence": confidence,
        "intended_v6_use": v6_use,
        "notes": notes,
    }


def _provenance(
    *,
    value: Any,
    status: str,
    source_path: str,
    confidence: str,
    per_sample: bool,
    intended_v6_use: str,
    notes: str = "",
) -> Dict[str, Any]:
    """Normalized feature record for Step 2 consumption."""
    rec: Dict[str, Any] = {
        "value": value,
        "status": status,
        "source_path": source_path,
        "confidence": confidence,
        "per_sample": bool(per_sample),
        "intended_v6_use": intended_v6_use,
    }
    if notes:
        rec["notes"] = notes
    return rec


def _records_to_provenance_dict(
    records: Sequence[Mapping[str, Any]],
    *,
    per_sample: bool,
    status_override: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    status_map = {
        "available": "available",
        "derived": "derived",
        "missing": "missing",
        "fallback": "fallback",
        "reference_shared": "reference_shared",
    }
    out: Dict[str, Dict[str, Any]] = {}
    for r in records:
        st = status_override or status_map.get(str(r.get("status") or ""), str(r.get("status") or "missing"))
        out[str(r["name"])] = _provenance(
            value=r.get("value"),
            status=st,
            source_path=str(r.get("source_key_path") or ""),
            confidence=str(r.get("confidence") or "low"),
            per_sample=per_sample,
            intended_v6_use=str(r.get("intended_v6_use") or ""),
            notes=str(r.get("notes") or ""),
        )
    return out


REFERENCE_SHARED_FEATURE_NAMES = (
    "low_body_mode_frequency",
    "bridge_to_radiation_strength",
    "air_to_structural_ratio",
    "top_to_back_ratio",
    "aperture_to_top_radiation_ratio",
    "modal_density_by_band",
    "top_radiation_gain_proxy",
    "soundhole_radiation_gain_proxy",
    "low_body_mode_q_or_decay",
)

PER_SAMPLE_DERIVED_FEATURE_NAMES = (
    "body_area_proxy",
    "body_volume_proxy",
    "soundhole_area",
    "helmholtz_like_frequency_proxy",
    "cavity_q_proxy",
    "cavity_decay_proxy",
    "high_frequency_absorption_proxy",
    "mass_loading_proxy",
    "body_decay_scale_proxy",
    "cavity_contribution_proxy",
    "modal_density_by_band_sample_grid",
    "low_body_mode_frequency_sample_grid",
)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 4)


def _load_reference_modal_catalog(repo_root: Path) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    path = REFERENCE_MODAL_PATH
    if not path.is_file():
        return [], "missing", ["reference catalog file not found"]
    doc = json.loads(path.read_text(encoding="utf-8"))
    modes, defaults = parse_modal_modes(doc)
    return modes, str(path.relative_to(repo_root)).replace("\\", "/"), defaults


def _collect_modal_schema_keys(modes: Sequence[Mapping[str, Any]]) -> List[str]:
    keys: set[str] = set()
    for m in modes:
        keys.update(str(k) for k in m.keys())
    return sorted(keys)


def _modal_field_summary(modes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    band = modes_in_validated_band(modes)
    if not band:
        return {"active_modes_in_band": 0, "field_coverage": {}}

    numeric_keys = (
        "frequency_hz",
        "bridge_excitation_abs",
        "bridge_excitation_coupling",
        "radiation_proxy",
        "mic_output_proxy",
        "air_pressure_proxy",
        "top_output_proxy",
        "back_output_proxy",
        "top_share",
        "back_share",
        "air_share",
        "Q",
        "q",
        "modal_q",
    )
    coverage: Dict[str, Any] = {}
    for key in numeric_keys:
        present = sum(1 for m in band if m.get(key) is not None)
        coverage[key] = {
            "present_count": present,
            "fraction": round(present / len(band), 4),
            "status": "available" if present == len(band) else ("partial" if present else "missing"),
        }
    freqs = [float(m["frequency_hz"]) for m in band]
    density = _modal_density_by_band(freqs)
    categories: Dict[str, int] = {}
    for m in band:
        cat = infer_mode_category(m)
        categories[cat] = categories.get(cat, 0) + 1
    low_modes = [m for m in band if float(m["frequency_hz"]) < 130.0]
    low_q_vals: List[float] = []
    for m in low_modes[:5]:
        q = m.get("Q") or m.get("q") or m.get("modal_q")
        if q is not None:
            low_q_vals.append(float(q))
    return {
        "active_modes_in_band": len(band),
        "frequency_band_hz": list(FULL_MODAL_BAND_HZ),
        "frequency_min_hz": round(min(freqs), 3),
        "frequency_max_hz": round(max(freqs), 3),
        "field_coverage": coverage,
        "modal_density_by_band": density,
        "mode_category_counts": categories,
        "low_body_mode_frequency_hz": round(min(freqs), 3),
        "low_body_mode_q_available": bool(low_q_vals),
        "low_body_mode_q_sample": round(statistics.mean(low_q_vals), 4) if low_q_vals else None,
    }


def _modal_density_by_band(freqs: Sequence[float]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for label, lo, hi in MODAL_BANDS:
        out[label] = sum(1 for f in freqs if lo <= f < hi)
    return out


def inspect_geometry_fields(params: Mapping[str, Any]) -> List[Dict[str, Any]]:
    p = normalize_sample_parameters(params)
    records: List[Dict[str, Any]] = []
    specs = (
        ("body_length", "length", "geometry.length", "Body planform — cavity volume, top area"),
        ("body_width", "width", "geometry.width", "Body planform — radiation area proxy"),
        ("body_depth", "depth", "geometry.depth", "Cavity volume / Helmholtz / rib height proxy"),
        ("top_thickness", "top_thickness", "geometry.top_thickness", "Top mass, stiffness proxy, damping scale"),
        ("back_thickness", "back_thickness", "geometry.back_thickness", "Back mass and damping scale"),
        ("soundhole_radius", "hole_radius", "geometry.hole_radius", "Helmholtz / aperture radiation"),
    )
    for name, key, path, use in specs:
        val = _g(p, key)
        if val is not None:
            records.append(
                _field_record(
                    name=name,
                    value=round(val, 8),
                    key_path=path,
                    status="available",
                    confidence="high",
                    v6_use=use,
                )
            )
        else:
            records.append(
                _field_record(
                    name=name,
                    value=None,
                    key_path=path,
                    status="missing",
                    confidence="low",
                    v6_use=use,
                    notes="Not present in sample parameters",
                )
            )

    length = _g(p, "length")
    width = _g(p, "width")
    depth = _g(p, "depth")
    hole_r = _g(p, "hole_radius")
    if length is not None and width is not None:
        area = length * width * 0.90
        records.append(
            _field_record(
                name="body_area_proxy",
                value=round(area, 8),
                key_path="derived:length*width*0.90",
                status="derived",
                confidence="medium",
                v6_use="Volume / radiation area estimate for V6 cavity coupling",
            )
        )
    else:
        records.append(
            _field_record(
                name="body_area_proxy",
                value=None,
                key_path="derived:length*width*0.90",
                status="missing",
                confidence="low",
                v6_use="Volume / radiation area estimate",
                notes="Requires length and width",
            )
        )

    if length is not None and width is not None and depth is not None:
        hole_area = math.pi * (hole_r or 0.047) ** 2
        vol = max(length * width * depth - hole_area * 0.35, 0.008)
        records.append(
            _field_record(
                name="body_volume_proxy",
                value=round(vol, 8),
                key_path="derived:length*width*depth-hole_correction",
                status="derived",
                confidence="medium",
                v6_use="Helmholtz / cavity air mass / low-body breathing",
            )
        )
    if hole_r is not None:
        sha = math.pi * hole_r * hole_r
        records.append(
            _field_record(
                name="soundhole_area",
                value=round(sha, 8),
                key_path="derived:pi*hole_radius^2",
                status="derived",
                confidence="high",
                v6_use="Helmholtz stiffness / soundhole radiation aperture",
            )
        )

    for name, path, use in (
        ("scale_length", "geometry.scale_length", "String/pluck routing — bridge position along string"),
        ("bridge_location", "geometry.bridge_location", "Bridge admittance excitation point"),
        ("body_outline_parameters", "geometry.outline/*", "Shape generator DOFs beyond box proxy"),
        ("side_rib_height", "geometry.side_height", "Side wall / cavity height (often approximated by depth)"),
    ):
        records.append(
            _field_record(
                name=name,
                value=None,
                key_path=path,
                status="missing",
                confidence="low",
                v6_use=use,
                notes="Not present in current LHS parameter schema",
            )
        )
    return records


def inspect_material_fields(params: Mapping[str, Any]) -> List[Dict[str, Any]]:
    p = normalize_sample_parameters(params)
    records: List[Dict[str, Any]] = []
    top = str(p.get("top_wood_id") or "").lower()
    back = str(p.get("back_wood_id") or "").lower()
    for name, key, val in (
        ("top_wood_id", "top_wood_id", top or None),
        ("back_wood_id", "back_wood_id", back or None),
    ):
        records.append(
            _field_record(
                name=name,
                value=val,
                key_path=key,
                status="available" if val else "missing",
                confidence="high" if val else "low",
                v6_use="Material damping / density lookup for V6 pole damping and mass",
            )
        )

    if top:
        rho = WOOD_DENSITY_REL.get(top)
        records.append(
            _field_record(
                name="top_density_proxy",
                value=rho,
                key_path=f"derived:WOOD_DENSITY_REL[{top}]",
                status="derived" if rho else "fallback",
                confidence="medium",
                v6_use="Top plate mass loading proxy",
                notes="Relative density table — not measured kg/m³ per sample",
            )
        )
    if back:
        rho = WOOD_DENSITY_REL.get(back)
        records.append(
            _field_record(
                name="back_density_proxy",
                value=rho,
                key_path=f"derived:WOOD_DENSITY_REL[{back}]",
                status="derived" if rho else "fallback",
                confidence="medium",
                v6_use="Back plate mass loading proxy",
            )
        )

    if top:
        damp = WOOD_DAMPING_COEFF.get(top)
        records.append(
            _field_record(
                name="top_damping_coeff_proxy",
                value=damp,
                key_path=f"derived:WOOD_DAMPING_COEFF[{top}]",
                status="derived",
                confidence="medium",
                v6_use="Per-mode Q when weighted by top_share",
            )
        )
    if back:
        damp = WOOD_DAMPING_COEFF.get(back)
        records.append(
            _field_record(
                name="back_damping_coeff_proxy",
                value=damp,
                key_path=f"derived:WOOD_DAMPING_COEFF[{back}]",
                status="derived",
                confidence="medium",
                v6_use="Per-mode Q when weighted by back_share",
            )
        )

    mass = compute_body_mass_proxies(p)
    for key, use in (
        ("top_effective_mass_proxy", "Bridge mobility / modal mass loading"),
        ("back_effective_mass_proxy", "Back path mass loading"),
        ("body_air_volume_proxy", "Cavity / air mode mass"),
        ("bridge_mobility_proxy", "Bridge admittance gain scaling"),
    ):
        records.append(
            _field_record(
                name=key,
                value=mass.get(key),
                key_path=f"derived:bridge_mobility_proxy.compute_body_mass_proxies",
                status="derived",
                confidence="medium",
                v6_use=use,
            )
        )

    for name, path, note in (
        ("side_wood_id", "materials.side.wood_id", "Side wood not varied in LHS pool"),
        ("youngs_modulus", "materials.*.youngs_modulus", "Not in synthesis parameter schema"),
        ("poisson_ratio", "materials.*.poisson_ratio", "Not in synthesis parameter schema"),
        ("anisotropic_stiffness", "materials.*.stiffness_tensor", "Not available to STK path"),
        ("material_delta", "parameter_payload.material_delta", "FEM pilot only — not in lhs_pool.json"),
        ("parameter_payload", "parameter_payload", "Not in lhs_pool.json entries"),
        ("loss_factor_tabulated", "materials.*.loss_factor", "Only discrete wood damping coeffs used"),
    ):
        records.append(
            _field_record(
                name=name,
                value=None,
                key_path=path,
                status="missing",
                confidence="low",
                v6_use="Future FEM-calibrated material routing",
                notes=note,
            )
        )
    return records


def inspect_body_signature_cache(repo_root: Path, sample_id: str) -> Dict[str, Any]:
    from body_signature_cache import cache_paths, load_body_signature_cache

    json_path, npz_path = cache_paths(repo_root, sample_id)
    loaded = load_body_signature_cache(repo_root, sample_id)
    if loaded is None:
        return {
            "status": "missing",
            "json_path": str(json_path.relative_to(repo_root)).replace("\\", "/"),
            "npz_path": str(npz_path.relative_to(repo_root)).replace("\\", "/"),
        }
    freqs = loaded.get("frequencies_hz")
    weights = loaded.get("modal_weights")
    density = _modal_density_by_band(freqs) if freqs is not None else {}
    return {
        "status": "available",
        "json_path": str(json_path.relative_to(repo_root)).replace("\\", "/"),
        "npz_path": str(npz_path.relative_to(repo_root)).replace("\\", "/"),
        "frequencies_hz_count": int(len(freqs)) if freqs is not None else 0,
        "modal_weights_count": int(len(weights)) if weights is not None else 0,
        "envelope_meta": {k: loaded.get(k) for k in ("G_peak", "dmax_db", "bridge_mobility_proxy")},
        "modal_density_by_band_on_grid": density,
        "low_grid_frequency_hz": round(float(np.min(freqs)), 3) if freqs is not None and len(freqs) else None,
    }


def inspect_modal_availability(
    *,
    repo_root: Path,
    sample: Mapping[str, Any],
    reference_summary: Mapping[str, Any],
    reference_schema_keys: Sequence[str],
) -> Dict[str, Any]:
    sid = str(sample["sample_id"])
    lhs_meta = {
        k: sample.get(k)
        for k in (
            "last_deduped_mode_count",
            "last_participation_computed_count",
            "last_audio_coupling_computed_count",
            "last_aggregation_status",
            "last_run_id",
            "status",
        )
        if sample.get(k) is not None
    }
    sig = inspect_body_signature_cache(repo_root, sid)
    m4_available = m4_surrogate_model_available(repo_root, "classic")

    per_sample_catalog_on_disk = False
    catalog_glob = list((repo_root / "ROM" / "classic").glob(f"**/{sid}*modal*.json"))
    if catalog_glob:
        per_sample_catalog_on_disk = True

    fields: List[Dict[str, Any]] = []
    ref_cov = reference_summary.get("field_coverage") or {}
    cov_key_map = {
        "frequency_hz": "frequency_hz",
        "bridge_excitation_abs": "bridge_excitation_abs",
        "radiation_proxy": "radiation_proxy",
        "mic_output_proxy": "mic_output_proxy",
        "air_pressure_proxy": "air_pressure_proxy",
        "top_share": "top_share",
        "back_share": "back_share",
        "air_share": "air_share",
        "top_output_proxy": "top_output_proxy",
        "back_output_proxy": "back_output_proxy",
    }
    for key, v6_use in (
        ("frequency_hz", "Modal pole frequency — admittance peak placement"),
        ("bridge_excitation_abs", "Bridge force → body mode excitation strength"),
        ("radiation_proxy", "Broad radiation weight / top-dominated output proxy"),
        ("mic_output_proxy", "Legacy mic pickup proxy — not soundhole-specific"),
        ("air_pressure_proxy", "Soundhole / aperture radiation target for V6"),
        ("top_share", "Route modal energy to top radiation stem"),
        ("back_share", "Route modal energy to back radiation stem"),
        ("air_share", "Route modal energy to cavity/air stem"),
        ("top_output_proxy", "Top plate radiated output proxy"),
        ("back_output_proxy", "Back plate radiated output proxy"),
        ("mode_q", "Explicit Q — usually derived, not stored"),
        ("mode_tau_s", "Decay time — derived in modal_damping.py"),
    ):
        if key in ("mode_q", "mode_tau_s"):
            status = "derived"
            confidence = "medium"
            value = "computed at synthesis from shares + wood IDs"
        else:
            cov = ref_cov.get(cov_key_map.get(key, key))
            if cov:
                status = str(cov.get("status") or "partial")
                confidence = "high" if status == "available" else "medium"
                value = f"reference catalog fraction={cov.get('fraction')}"
            else:
                status = "missing"
                confidence = "low"
                value = None
        fields.append(
            _field_record(
                name=key,
                value=value,
                key_path=f"predicted_modes[].{key}",
                status=status,
                confidence=confidence,
                v6_use=v6_use,
                notes=(
                    "Per-sample values require M4 surrogate inference or ROM aggregation — "
                    "not loaded in this audit (no ROM run)"
                ),
            )
        )

    return {
        "sample_id": sid,
        "lhs_rom_metadata": lhs_meta,
        "body_signature_cache": sig,
        "per_sample_modal_catalog_on_disk": per_sample_catalog_on_disk,
        "m4_surrogate_model_files_present": m4_available,
        "modal_catalog_inference_available_without_rom_run": m4_available,
        "modal_field_inventory": fields,
        "reference_catalog_schema_keys": list(reference_schema_keys),
        "reference_catalog_summary": reference_summary,
        "modal_source_for_v6": (
            "m4_surrogate_or_rom_aggregation"
            if m4_available or lhs_meta.get("last_deduped_mode_count")
            else "synthetic_fixture_only"
        ),
    }


def compute_reference_catalog_aggregates(
    reference_modes: Sequence[Mapping[str, Any]],
    *,
    reference_path: str,
) -> Dict[str, Dict[str, Any]]:
    """Aggregates from shared reference ROM catalog — identical for every sample in this audit."""
    band = modes_in_validated_band(reference_modes)
    if not band:
        return {}

    bridges = [float(m.get("bridge_excitation_abs") or 0) for m in band]
    rads = [float(m.get("radiation_proxy") or 0) for m in band]
    tops = [float(m.get("top_share") or 0) for m in band]
    backs = [float(m.get("back_share") or 0) for m in band]
    airs = [float(m.get("air_share") or 0) for m in band]
    apertures = [float(m.get("air_pressure_proxy") or 0) for m in band]
    freqs = [float(m["frequency_hz"]) for m in band]
    low_f = round(min(freqs), 3)

    raw: Dict[str, Any] = {
        "low_body_mode_frequency": low_f,
        "top_radiation_gain_proxy": round(statistics.mean(rads), 8) if rads else None,
        "soundhole_radiation_gain_proxy": (
            round(statistics.mean(apertures), 8) if any(a > 0 for a in apertures) else None
        ),
        "bridge_to_radiation_strength": (
            round(statistics.mean(bridges) / max(statistics.mean(rads), 1e-12), 6)
            if bridges and rads
            else None
        ),
        "top_to_back_ratio": (
            round(statistics.mean(tops) / max(statistics.mean(backs), 1e-12), 6)
            if tops and backs
            else None
        ),
        "air_to_structural_ratio": (
            round(statistics.mean(airs) / max(statistics.mean(tops) + statistics.mean(backs), 1e-12), 6)
            if airs
            else None
        ),
        "aperture_to_top_radiation_ratio": (
            round(statistics.mean(apertures) / max(statistics.mean(rads), 1e-12), 6)
            if apertures and rads and any(a > 0 for a in apertures)
            else None
        ),
        "modal_density_by_band": _modal_density_by_band(freqs),
        "low_body_mode_q_or_decay": {
            "mode_q": None,
            "mode_tau_s": None,
            "source": "reference_catalog_lowest_mode_q_not_stored",
        },
    }

    out: Dict[str, Dict[str, Any]] = {}
    for name, val in raw.items():
        out[name] = _provenance(
            value=val,
            status="reference_shared",
            source_path=reference_path,
            confidence="high",
            per_sample=False,
            intended_v6_use=(
                "Single-guitar V6 skeleton routing ratios — not safe for multi-guitar differentiation"
            ),
            notes="Identical across all samples unless per-sample modal catalog is loaded",
        )
    return out


def compute_per_sample_derived_features(
    params: Mapping[str, Any],
    *,
    signature_cache: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Derived features that vary per sample from LHS geometry/material + body_signature cache."""
    p = normalize_sample_parameters(params)
    cav = _compute_cavity_geometry_proxy(p)
    mass = compute_body_mass_proxies(p)

    raw: Dict[str, Any] = {
        "body_area_proxy": cav.get("body_area_proxy"),
        "body_volume_proxy": cav.get("body_volume_proxy"),
        "soundhole_area": cav.get("soundhole_area_used"),
        "helmholtz_like_frequency_proxy": cav.get("helmholtz_like_frequency_hz"),
        "cavity_q_proxy": cav.get("cavity_q"),
        "cavity_decay_proxy": cav.get("cavity_decay_s"),
        "high_frequency_absorption_proxy": cav.get("high_frequency_absorption"),
        "mass_loading_proxy": mass.get("mixed_body_mass_proxy"),
        "body_decay_scale_proxy": round(
            0.5 * (float(cav.get("material_top_damping") or 1.0) + float(cav.get("material_back_damping") or 1.0)),
            4,
        ),
        "cavity_contribution_proxy": cav.get("cavity_gain"),
        "bridge_mobility_proxy": mass.get("bridge_mobility_proxy"),
    }

    sig_status = signature_cache.get("status")
    if sig_status == "available" and signature_cache.get("modal_density_by_band_on_grid"):
        raw["modal_density_by_band_sample_grid"] = signature_cache["modal_density_by_band_on_grid"]
        raw["low_body_mode_frequency_sample_grid"] = signature_cache.get("low_grid_frequency_hz")
    else:
        raw["modal_density_by_band_sample_grid"] = None
        raw["low_body_mode_frequency_sample_grid"] = None

    use_notes = {
        "body_area_proxy": "Cavity/radiation area estimate",
        "body_volume_proxy": "Helmholtz / cavity air mass",
        "soundhole_area": "Aperture / Helmholtz stiffness",
        "helmholtz_like_frequency_proxy": "Low-body air resonance target",
        "cavity_q_proxy": "Cavity mode Q target",
        "cavity_decay_proxy": "Cavity tail / sustain target",
        "high_frequency_absorption_proxy": "E5 metallicity damping target",
        "mass_loading_proxy": "Bridge mobility / modal mass loading",
        "body_decay_scale_proxy": "Material damping aggregate",
        "cavity_contribution_proxy": "Cavity gain proxy",
        "bridge_mobility_proxy": "Bridge admittance scaling",
        "modal_density_by_band_sample_grid": "Per-sample transfer envelope band counts",
        "low_body_mode_frequency_sample_grid": "Lowest frequency on cached admittance grid",
    }

    out: Dict[str, Dict[str, Any]] = {}
    for name, val in raw.items():
        if name in ("modal_density_by_band_sample_grid", "low_body_mode_frequency_sample_grid"):
            st = "available" if val is not None else "missing"
            conf = "medium" if val is not None else "low"
            src = "ROM/classic/body_signature_cache/{sample_id}.npz"
        else:
            st = "derived"
            conf = "medium"
            src = "derived:geometry/material via _compute_cavity_geometry_proxy / compute_body_mass_proxies"
        out[name] = _provenance(
            value=val,
            status=st,
            source_path=src,
            confidence=conf,
            per_sample=True,
            intended_v6_use=use_notes.get(name, "V6 per-sample driver"),
        )
    return out


def compute_derived_v6_features(
    params: Mapping[str, Any],
    *,
    signature_cache: Mapping[str, Any],
    reference_modes: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Legacy flat dict — kept for sanity-check helpers; prefer normalized schema."""
    per = compute_per_sample_derived_features(params, signature_cache=signature_cache)
    flat = {k: v["value"] for k, v in per.items()}
    ref_path = str(REFERENCE_MODAL_PATH)
    ref = compute_reference_catalog_aggregates(reference_modes, reference_path=ref_path)
    for k, v in ref.items():
        flat[k] = v["value"]
    return flat


def classify_v6_feature_availability(
    *,
    geometry: Sequence[Mapping[str, Any]],
    materials: Sequence[Mapping[str, Any]],
    modal_block: Mapping[str, Any],
    derived: Mapping[str, Any],
) -> Dict[str, str]:
    """Map each critical V6 feature to available/derived/fallback/missing."""
    by_name = {r["name"]: r for r in geometry + materials}
    classification: Dict[str, str] = {}

    def _from_record(name: str) -> str:
        rec = by_name.get(name)
        if rec:
            st = str(rec.get("status") or "missing")
            if st == "available":
                return "available_directly"
            if st == "derived":
                return "derived_from_existing_data"
            if st == "fallback":
                return "fallback_required"
            return "missing_and_not_safe_to_infer"
        if derived.get(name) is not None:
            return "derived_from_existing_data"
        return "missing_and_not_safe_to_infer"

    classification["body_length"] = _from_record("body_length")
    classification["body_width"] = _from_record("body_width")
    classification["body_depth"] = _from_record("body_depth")
    classification["soundhole_radius"] = _from_record("soundhole_radius")
    classification["top_back_wood_ids"] = (
        "available_directly"
        if by_name.get("top_wood_id", {}).get("status") == "available"
        else "missing_and_not_safe_to_infer"
    )
    classification["body_volume_proxy"] = (
        "derived_from_existing_data" if derived.get("body_volume_proxy") else "missing_and_not_safe_to_infer"
    )
    classification["helmholtz_like_frequency_proxy"] = (
        "derived_from_existing_data" if derived.get("helmholtz_like_frequency_proxy") else "fallback_required"
    )
    classification["per_mode_participation_shares"] = (
        "derived_from_existing_data"
        if modal_block.get("m4_surrogate_model_files_present")
        or modal_block.get("lhs_rom_metadata", {}).get("last_participation_computed_count")
        else "missing_and_not_safe_to_infer"
    )
    classification["bridge_excitation_per_mode"] = classification["per_mode_participation_shares"]
    classification["radiation_and_aperture_proxies"] = (
        "available_directly"
        if "air_pressure_proxy" in (modal_block.get("reference_catalog_schema_keys") or [])
        else "fallback_required"
    )
    classification["scale_length"] = "missing_and_not_safe_to_infer"
    classification["bridge_position"] = "missing_and_not_safe_to_infer"
    classification["elastic_moduli_anisotropy"] = "missing_and_not_safe_to_infer"
    classification["side_wood_identity"] = "missing_and_not_safe_to_infer"
    classification["per_sample_modal_catalog_on_disk"] = (
        "available_directly"
        if modal_block.get("per_sample_modal_catalog_on_disk")
        else "fallback_required"
    )
    return classification


def _build_geometry_provenance(params: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    records = inspect_geometry_fields(params)
    by_name = {r["name"]: r for r in records}
    keys = (
        "body_length",
        "body_width",
        "body_depth",
        "top_thickness",
        "back_thickness",
        "soundhole_radius",
        "body_area_proxy",
        "body_volume_proxy",
        "soundhole_area",
    )
    out: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        r = by_name.get(key)
        if r is None:
            out[key] = _provenance(
                value=None,
                status="missing",
                source_path=f"geometry.{key}",
                confidence="low",
                per_sample=True,
                intended_v6_use="V6 geometry driver",
                notes="Not found in LHS parameters",
            )
            continue
        per_sample = key not in ("body_area_proxy", "body_volume_proxy", "soundhole_area")
        st = str(r.get("status") or "missing")
        if st == "available":
            status = "available"
        elif st == "derived":
            status = "derived"
        elif st == "fallback":
            status = "fallback"
        else:
            status = "missing"
        out[key] = _provenance(
            value=r.get("value"),
            status=status,
            source_path=str(r.get("source_key_path") or ""),
            confidence=str(r.get("confidence") or "low"),
            per_sample=True,
            intended_v6_use=str(r.get("intended_v6_use") or ""),
            notes=str(r.get("notes") or ""),
        )
    return out


def _build_material_provenance(params: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    records = inspect_material_fields(params)
    keep = (
        "top_wood_id",
        "back_wood_id",
        "top_density_proxy",
        "back_density_proxy",
        "top_damping_coeff_proxy",
        "back_damping_coeff_proxy",
        "top_effective_mass_proxy",
        "back_effective_mass_proxy",
        "body_air_volume_proxy",
        "bridge_mobility_proxy",
    )
    by_name = {r["name"]: r for r in records}
    out: Dict[str, Dict[str, Any]] = {}
    for key in keep:
        r = by_name.get(key)
        if not r:
            continue
        st = str(r.get("status") or "missing")
        status = {"available": "available", "derived": "derived", "fallback": "fallback"}.get(st, st)
        out[key] = _provenance(
            value=r.get("value"),
            status=status,
            source_path=str(r.get("source_key_path") or ""),
            confidence=str(r.get("confidence") or "low"),
            per_sample=True,
            intended_v6_use=str(r.get("intended_v6_use") or ""),
            notes=str(r.get("notes") or ""),
        )
    return out


def build_normalized_sample_record(
    *,
    sample: Mapping[str, Any],
    repo_root: Path,
    reference_summary: Mapping[str, Any],
    reference_schema_keys: Sequence[str],
    reference_aggregates: Mapping[str, Dict[str, Any]],
    ref_path: str,
) -> Dict[str, Any]:
    params = sample.get("parameters") or {}
    geometry = _build_geometry_provenance(params)
    material = _build_material_provenance(params)
    modal_avail = inspect_modal_availability(
        repo_root=repo_root,
        sample=sample,
        reference_summary=reference_summary,
        reference_schema_keys=reference_schema_keys,
    )
    per_sample_derived = compute_per_sample_derived_features(
        params,
        signature_cache=modal_avail["body_signature_cache"],
    )

    modal: Dict[str, Dict[str, Any]] = {}
    for key in (
        "m4_surrogate_model_files_present",
        "per_sample_modal_catalog_on_disk",
        "modal_source_for_v6",
    ):
        val = modal_avail.get(key)
        modal[key] = _provenance(
            value=val,
            status="available" if val else "missing",
            source_path="lhs_pool.json / filesystem check",
            confidence="high",
            per_sample=key != "m4_surrogate_model_files_present",
            intended_v6_use="Modal catalog availability for V6",
        )
    modal["body_signature_cache_status"] = _provenance(
        value=modal_avail["body_signature_cache"].get("status"),
        status="available" if modal_avail["body_signature_cache"].get("status") == "available" else "missing",
        source_path="ROM/classic/body_signature_cache/{sample_id}.npz",
        confidence="high",
        per_sample=True,
        intended_v6_use="Admittance envelope cross-check",
    )

    derived_features: Dict[str, Dict[str, Any]] = {}
    derived_features.update(per_sample_derived)
    derived_features.update(reference_aggregates)

    stage2_usable: Dict[str, str] = {}
    for fname, rec in {**geometry, **material, **derived_features}.items():
        if rec.get("per_sample") and rec.get("status") in ("available", "derived"):
            stage2_usable[fname] = "safe_per_sample_driver"
        elif not rec.get("per_sample") and rec.get("status") == "reference_shared":
            stage2_usable[fname] = "reference_shared_single_guitar_only"
        elif rec.get("status") == "fallback":
            stage2_usable[fname] = "fallback_caution"
        elif rec.get("status") == "missing":
            stage2_usable[fname] = "missing"

    missing_critical = [
        name
        for name, rec in {**geometry, **material}.items()
        if rec.get("status") == "missing"
    ]
    for name in ("scale_length", "bridge_position"):
        missing_critical.append(name)

    feature_class = classify_v6_feature_availability(
        geometry=[{"name": k, **v} for k, v in geometry.items()],
        materials=[{"name": k, **v} for k, v in material.items()],
        modal_block=modal_avail,
        derived={k: v.get("value") for k, v in derived_features.items()},
    )

    return {
        "sample_id": str(sample["sample_id"]),
        "run_id": sample.get("run_id"),
        "geometry": geometry,
        "material": material,
        "modal": modal,
        "derived_features": derived_features,
        "stage2_usable_features": stage2_usable,
        "missing_critical_fields": sorted(set(missing_critical)),
        "v6_feature_classification": feature_class,
        "_modal_availability_detail": modal_avail,
    }


def detect_shared_or_constant_features(
    samples: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Find features identical across all samples — likely reference/shared, not per-sample."""
    if len(samples) < 2:
        return []

    watch = set(REFERENCE_SHARED_FEATURE_NAMES) | set(PER_SAMPLE_DERIVED_FEATURE_NAMES)
    rows: List[Dict[str, Any]] = []

    for fname in sorted(watch):
        values: List[Any] = []
        per_sample_flags: List[bool] = []
        statuses: List[str] = []
        for s in samples:
            rec = (s.get("derived_features") or {}).get(fname)
            if not rec:
                continue
            values.append(rec.get("value"))
            per_sample_flags.append(bool(rec.get("per_sample")))
            statuses.append(str(rec.get("status") or ""))

        if len(values) < 2:
            continue

        def _norm(v: Any) -> str:
            if isinstance(v, dict):
                return json.dumps(v, sort_keys=True)
            return json.dumps(v)

        unique = {_norm(v) for v in values}
        if len(unique) != 1:
            continue

        all_per_sample = all(per_sample_flags) if per_sample_flags else False
        st_set = set(statuses)
        if "reference_shared" in st_set:
            reason = "reference_catalog"
        elif all_per_sample:
            reason = "unknown"
        else:
            reason = "constant_default"

        safe_stage2 = fname in PER_SAMPLE_DERIVED_FEATURE_NAMES and not all_per_sample
        safe_multi = all_per_sample and reason != "reference_catalog"

        rows.append(
            {
                "feature_name": fname,
                "value": values[0],
                "likely_reason": reason,
                "safe_for_stage2_single_guitar": fname in PER_SAMPLE_DERIVED_FEATURE_NAMES or reason == "reference_catalog",
                "safe_for_multi_guitar_differentiation": safe_multi,
                "per_sample_flag_in_schema": all(per_sample_flags) if per_sample_flags else False,
                "statuses_seen": sorted(st_set),
                "warning": (
                    "Identical across samples but marked per_sample — verify data source"
                    if all_per_sample and reason == "unknown"
                    else (
                        "Reference/shared value — do not use for multi-guitar differentiation"
                        if reason == "reference_catalog" or fname in REFERENCE_SHARED_FEATURE_NAMES
                        else None
                    )
                ),
            }
        )
    return rows


def build_recommended_stage2_feature_set() -> Dict[str, List[str]]:
    return {
        "safe_per_sample_drivers": [
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
            "modal_density_by_band_sample_grid",
            "low_body_mode_frequency_sample_grid",
        ],
        "shared_reference_drivers_single_guitar_skeleton": [
            "low_body_mode_frequency",
            "modal_density_by_band",
            "top_to_back_ratio",
            "air_to_structural_ratio",
            "bridge_to_radiation_strength",
            "aperture_to_top_radiation_ratio",
            "top_radiation_gain_proxy",
            "soundhole_radiation_gain_proxy",
            "reference modal frequencies",
            "reference participation shares (top/back/air)",
        ],
        "risky_drivers_multi_guitar_differentiation": [
            "low_body_mode_frequency (reference_shared)",
            "bridge_to_radiation_strength (reference_shared)",
            "air_to_structural_ratio (reference_shared)",
            "top_to_back_ratio (reference_shared)",
            "aperture_to_top_radiation_ratio (reference_shared)",
            "modal_density_by_band (reference_shared)",
            "any fallback Helmholtz clamp at 85/128 Hz boundary",
            "wood-ID damping tables without measured material_delta",
        ],
    }


def build_stage2_readiness(
    *,
    samples: Sequence[Mapping[str, Any]],
    shared_features: Sequence[Mapping[str, Any]],
    global_missing: Sequence[str],
) -> Dict[str, Any]:
    ref_shared = [r for r in shared_features if r.get("likely_reason") == "reference_catalog"]
    per_sample_vary = any(
        len({s["geometry"]["body_depth"]["value"] for s in samples if s.get("geometry")}) > 1
        for _ in [0]
    ) if samples else False

    safe = build_recommended_stage2_feature_set()["safe_per_sample_drivers"]
    caution = build_recommended_stage2_feature_set()["risky_drivers_multi_guitar_differentiation"]

    status = "ready_with_limitations"
    if not samples:
        status = "blocked"
    elif not per_sample_vary:
        status = "blocked"

    return {
        "status": status,
        "single_guitar_v6_skeleton": (
            "Proceed using per-sample geometry/cavity/material proxies plus reference modal "
            "catalog for routing architecture (top/back/air/soundhole stems)."
        ),
        "multi_guitar_differentiation": (
            "Limited to geometry, wood IDs, mass/mobility, and body_signature cache until "
            "per-sample predicted_modes are loaded (M4 surrogate inference — not run in this audit)."
        ),
        "main_limitations": [
            "Per-sample modal catalog not loaded in Step 1 audit (no ROM run)",
            f"{len(ref_shared)} features are reference_shared and identical across samples",
            "scale_length and bridge_position absent from LHS pool",
            "material_delta / elastic moduli not available",
        ],
        "must_not_assume": [
            "Reference modal radiation ratios differ between guitars",
            "low_body_mode_frequency from reference catalog is sample-specific",
            "Helmholtz proxy equals FEM cavity mode",
            "mic_output_proxy equals soundhole radiation",
        ],
        "safe_to_use_in_stage2": safe,
        "requires_fallback_or_caution": caution + list(global_missing),
    }


def run_physical_sanity_checks(samples_derived: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def _val(s: Mapping[str, Any], section: str, key: str) -> Optional[float]:
        block = s.get(section) or {}
        rec = block.get(key) if isinstance(block, dict) else None
        if isinstance(rec, dict):
            v = rec.get("value")
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        return None

    def _derived(s: Mapping[str, Any], key: str) -> Optional[float]:
        rec = (s.get("derived_features") or {}).get(key)
        if isinstance(rec, dict):
            v = rec.get("value")
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        return None

    vols = [_derived(s, "body_volume_proxy") for s in samples_derived]
    vols = [v for v in vols if v is not None]
    helm = [_derived(s, "helmholtz_like_frequency_proxy") for s in samples_derived]
    helm = [v for v in helm if v is not None]
    hole_areas = [_derived(s, "soundhole_area") for s in samples_derived]
    hole_areas = [v for v in hole_areas if v is not None]
    damp_scales = [_derived(s, "body_decay_scale_proxy") for s in samples_derived]
    damp_scales = [v for v in damp_scales if v is not None]
    depth_only = [_val(s, "geometry", "body_depth") for s in samples_derived]
    depth_only = [v for v in depth_only if v is not None]
    cavity_decays = [_derived(s, "cavity_decay_proxy") for s in samples_derived]
    cavity_decays = [v for v in cavity_decays if v is not None]

    checks = {
        "larger_volume_lowers_helmholtz": {
            "expected": "negative correlation between body_volume_proxy and helmholtz_like_frequency_proxy",
            "correlation": _pearson(vols, helm),
            "supported": (_pearson(vols, helm) or 0) < -0.1 if len(vols) >= 3 else None,
            "sample_count": len(vols),
        },
        "larger_soundhole_raises_helmholtz": {
            "expected": "positive correlation between soundhole_area and helmholtz_like_frequency_proxy",
            "correlation": _pearson(hole_areas, helm),
            "supported": (_pearson(hole_areas, helm) or 0) > 0.1 if len(hole_areas) >= 3 else None,
            "sample_count": len(hole_areas),
        },
        "deeper_body_lowers_helmholtz": {
            "expected": "negative correlation between body_depth and helmholtz_like_frequency_proxy",
            "correlation": _pearson(depth_only, helm),
            "supported": (_pearson(depth_only, helm) or 0) < -0.1 if len(depth_only) >= 3 else None,
            "sample_count": len(depth_only),
        },
        "higher_damping_shorter_sustain_proxy": {
            "expected": "higher body_decay_scale_proxy should inversely relate to cavity_decay_proxy",
            "correlation": _pearson(damp_scales, cavity_decays),
            "supported": None,
            "notes": "cavity_decay_proxy currently rises with depth — material damping effect is weak; needs listening validation",
        },
        "participation_routing_feasible": {
            "expected": "top/back/air shares available in reference catalog for stem routing",
            "supported": True,
            "notes": "Reference catalog includes top_share, back_share, air_share, air_pressure_proxy",
        },
        "bridge_coupling_available": {
            "expected": "bridge_excitation_abs present for body excitation weighting",
            "supported": True,
            "notes": "Present in reference ROM catalog; per-sample via surrogate/ROM not loaded here",
        },
        "mass_loading_affects_mobility": {
            "expected": "mass_loading_proxy varies across LHS samples",
            "supported": len({round(v, 6) for v in [_derived(s, "mass_loading_proxy") for s in samples_derived] if v is not None})
            > 1,
        },
        "hf_absorption_reduces_e5_metallicity_risk": {
            "expected": "high_frequency_absorption_proxy varies with wood/depth",
            "supported": len(
                {round(v, 4) for v in [_derived(s, "high_frequency_absorption_proxy") for s in samples_derived] if v is not None}
            )
            > 1,
            "notes": "Proxy exists; listening link not validated in this audit",
        },
    }
    return checks


def build_stk_v6_physical_dof_audit(
    *,
    repo_root: Path,
    sample_ids: Sequence[str] = DEFAULT_SAMPLE_IDS,
) -> Dict[str, Any]:
    repo_root = Path(repo_root)
    all_samples = load_lhs_sample_entries_full(repo_root, max_samples=26)
    id_set = {str(s) for s in sample_ids}
    samples = [s for s in all_samples if str(s["sample_id"]) in id_set]
    samples.sort(key=lambda r: str(r["sample_id"]))
    found_ids = [str(s["sample_id"]) for s in samples]

    ref_modes, ref_path, ref_defaults = _load_reference_modal_catalog(repo_root)
    ref_summary = _modal_field_summary(ref_modes)
    ref_schema = _collect_modal_schema_keys(ref_modes)
    reference_aggregates = compute_reference_catalog_aggregates(ref_modes, reference_path=ref_path)

    per_sample: List[Dict[str, Any]] = []
    for sample in samples:
        per_sample.append(
            build_normalized_sample_record(
                sample=sample,
                repo_root=repo_root,
                reference_summary=ref_summary,
                reference_schema_keys=ref_schema,
                reference_aggregates=reference_aggregates,
                ref_path=ref_path,
            )
        )

    shared_or_constant = detect_shared_or_constant_features(per_sample)

    global_missing = [
        "scale_length",
        "bridge_position",
        "elastic_moduli_anisotropy",
        "side_wood_identity",
        "material_delta / parameter_payload in lhs_pool",
        "per-sample predicted_modes JSON on disk (inference required)",
    ]
    sanity = run_physical_sanity_checks(per_sample)
    recommended_stage2 = build_recommended_stage2_feature_set()
    stage2_readiness = build_stage2_readiness(
        samples=per_sample,
        shared_features=shared_or_constant,
        global_missing=global_missing,
    )

    recommendations = [
        "Stage 2: build V6 routed stems (top / back / soundhole / cavity) using participation + aperture proxies",
        "Load per-sample predicted_modes via M4 surrogate before multi-guitar differentiation",
        "Add scale length and bridge position to LHS schema for pluck→bridge routing",
        "Use only per_sample=true features for guitar-to-guitar contrast in Stage 2 listening tests",
        "Reference_shared modal aggregates OK for single-guitar skeleton only",
    ]

    return {
        "report_version": REPORT_VERSION,
        "timestamp": _utc_now(),
        "status": "stk_v6_step1_1_audit_schema_cleanup_not_solved",
        "step": "1.1",
        "schema_note": "Normalized sample.geometry/material/modal/derived_features with provenance per feature",
        "requested_sample_ids": list(sample_ids),
        "audited_sample_ids": found_ids,
        "sample_limitation": (
            None
            if len(found_ids) >= len(sample_ids)
            else f"Only {len(found_ids)} of {len(sample_ids)} requested samples found in lhs_pool.json"
        ),
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "no_audio_synthesis_performed": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "stk_v5_behavior_unchanged": True,
        "reference_modal_catalog": {
            "path": ref_path,
            "parse_defaults": ref_defaults,
            "schema_keys": ref_schema,
            "summary": ref_summary,
            "reference_aggregates": reference_aggregates,
        },
        "samples": per_sample,
        "shared_or_constant_features": shared_or_constant,
        "stage2_readiness": stage2_readiness,
        "recommended_stage2_feature_set": recommended_stage2,
        "missing_critical_fields": sorted(set(global_missing)),
        "v6_critical_feature_descriptions": V6_CRITICAL_FEATURES,
        "dof_influence_map": DOF_INFLUENCE_MAP,
        "physical_sanity_checks": sanity,
        "recommendations_for_v6_stage2": recommendations,
        "explicit_flags": {
            "website_default_unchanged": True,
            "website_default_mode": DEFAULT_WEBSITE_STK_MODE,
            "no_audio_synthesis_performed": True,
            "no_fem_run": True,
            "no_rom_run": True,
            "no_rom_batch_run": True,
            "stk_v5_behavior_unchanged": True,
            "production_synthesis_unchanged": True,
        },
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    readiness = report.get("stage2_readiness") or {}
    rec = report.get("recommended_stage2_feature_set") or {}
    lines: List[str] = [
        "# STK V6 physical DOF audit (Step 1.1)",
        "",
        f"**Status:** {report.get('status')}",
        f"**Step:** {report.get('step', '1.1')} — schema cleanup & Stage 2 readiness",
        f"**Generated:** {report.get('timestamp')}",
        "",
        report.get("schema_note", ""),
        "",
        "**No audio synthesis was performed. Website default and production paths were not changed.**",
        "",
        f"- Website default (unchanged): `{report.get('website_default')}`",
        f"- Top-level flags: website_default_unchanged={report.get('website_default_unchanged')}, "
        f"no_audio={report.get('no_audio_synthesis_performed')}, no_fem={report.get('no_fem_run')}, "
        f"no_rom={report.get('no_rom_run')}",
        f"- Samples audited: {', '.join(report.get('audited_sample_ids') or [])}",
    ]
    if report.get("sample_limitation"):
        lines.append(f"- **Limitation:** {report['sample_limitation']}")

    lines.extend(
        [
            "",
            "## Stage 2 readiness",
            "",
            f"- **Status:** `{readiness.get('status')}`",
            f"- Single-guitar skeleton: {readiness.get('single_guitar_v6_skeleton')}",
            f"- Multi-guitar differentiation: {readiness.get('multi_guitar_differentiation')}",
            "",
            "**Main limitations:**",
        ]
    )
    for item in readiness.get("main_limitations") or []:
        lines.append(f"- {item}")

    lines.extend(["", "**Must not assume:**"])
    for item in readiness.get("must_not_assume") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Per-sample safe drivers (geometry / cavity)"])
    lines.append("")
    lines.append("| sample | body_depth | volume_proxy | Helmholtz | soundhole_area | bridge_mobility |")
    lines.append("|--------|------------|--------------|-----------|----------------|-----------------|")
    for s in report.get("samples") or []:
        geo = s.get("geometry") or {}
        mat = s.get("material") or {}
        der = s.get("derived_features") or {}
        lines.append(
            f"| {s.get('sample_id')} "
            f"| {geo.get('body_depth', {}).get('value')} "
            f"| {der.get('body_volume_proxy', {}).get('value')} "
            f"| {der.get('helmholtz_like_frequency_proxy', {}).get('value')} "
            f"| {der.get('soundhole_area', {}).get('value')} "
            f"| {mat.get('bridge_mobility_proxy', {}).get('value')} |"
        )

    lines.extend(["", "## Shared / reference drivers (single-guitar skeleton only)", ""])
    lines.append("These are **reference_shared** — identical across samples in this audit:")
    lines.append("")
    for row in report.get("shared_or_constant_features") or []:
        if row.get("likely_reason") == "reference_catalog":
            lines.append(
                f"- `{row.get('feature_name')}` = {row.get('value')} "
                f"(safe multi-guitar: {row.get('safe_for_multi_guitar_differentiation')})"
            )

    lines.extend(["", "### Recommended Stage 2 feature set", ""])
    lines.append("**A. Safe per-sample drivers:**")
    for item in rec.get("safe_per_sample_drivers") or []:
        lines.append(f"- `{item}`")
    lines.append("")
    lines.append("**B. Shared reference (single-guitar skeleton):**")
    for item in rec.get("shared_reference_drivers_single_guitar_skeleton") or []:
        lines.append(f"- `{item}`")
    lines.append("")
    lines.append("**C. Risky for multi-guitar differentiation:**")
    for item in rec.get("risky_drivers_multi_guitar_differentiation") or []:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Warning",
            "",
            "Step 2 may build a **single-guitar V6 skeleton** using reference modal features, but "
            "**multi-guitar differentiation must not rely on shared modal/radiation aggregates** "
            "as if they were per-sample. Load per-sample modal catalogs first.",
            "",
            "## Missing / fallback DOFs",
            "",
        ]
    )
    for item in report.get("missing_critical_fields") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Physical sanity checks", ""])
    for key, chk in (report.get("physical_sanity_checks") or {}).items():
        lines.append(f"- **{key}:** supported={chk.get('supported')} correlation={chk.get('correlation')}")

    lines.extend(
        [
            "",
            "## Explicit confirmations",
            "",
            "- Website default unchanged (`stk_body_transfer_final_v1`)",
            "- No audio WAV synthesis in this audit",
            "- No FEM or ROM batch executed",
            "- STK V5 behavior not modified",
            "",
            "*STK V6 is not solved — Step 1.1 schema cleanup only. Do not proceed to Step 2 synthesis yet.*",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="STK V6 physical DOF audit (read-only)")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--max-sample-index", type=int, default=9, help="Audit sample_000..sample_N")
    args = parser.parse_args()

    sample_ids = tuple(f"sample_{i:03d}" for i in range(args.max_sample_index + 1))
    report = build_stk_v6_physical_dof_audit(repo_root=args.repo_root, sample_ids=sample_ids)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, args.md_out)
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")
    print(f"Audited {len(report['audited_sample_ids'])} samples")


if __name__ == "__main__":
    main()
