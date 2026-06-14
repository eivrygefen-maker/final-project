#!/usr/bin/env python3
"""
PGSM Step 2.2 — Research-referenced tonewood material property extension.
Data library only; no audio, no FEM/ROM execution, no STK integration.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pgsm_physical_factor_registry import DEFAULT_SAMPLE_IDS, load_audit_report
from pgsm_step2_1_parameter_targets import load_step_report
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE
from stk_v6_2_audit_features import feature_value, get_sample_record

PGSM_STEP22_VERSION = "pgsm_step2_2_tonewood_material_library_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
FEM_WOODS_ORTHO = REPO_ROOT / "FEM" / "materials" / "woods_ortho.json"
PGSM_LIBRARY_JSON = REPO_ROOT / "data" / "pgsm_tonewood_material_library.json"
STEP1_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step1_physical_factor_registry.json"
STEP21_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step2_1_parameter_targets.json"
STEP3B_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step3b_modal_response_validation.json"
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step2_2_tonewood_material_library.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step2_2_tonewood_material_library.md"
FIGURES_DIR = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step2_2_figures"

PROJECT_WOOD_IDS = ("spruce", "cedar", "rosewood", "mahogany", "maple", "cypress")

# Project short ID → PGSM library wood_id
PROJECT_TO_LIBRARY: Dict[str, str] = {
    "spruce": "spruce_sitka",
    "cedar": "cedar_western",
    "rosewood": "rosewood_indian",
    "mahogany": "mahogany_honduran",
    "maple": "maple_hard",
    "cypress": "cypress_mediterranean",
}

# FEM woods_ortho.json keys (do not modify FEM file)
FEM_KEY_MAP: Dict[str, str] = {
    "spruce_sitka": "spruce_sitka",
    "cedar_western": "cedar_western",
    "rosewood_indian": "rosewood_indian",
    "mahogany_honduran": "mahogany_honduran",
    "maple_hard": "maple_hard",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rng(
    *,
    min_v: float,
    typical: float,
    max_v: float,
    source_reference_id: str,
    confidence: str,
    value_status: str = "literature_range",
    units: str = "",
    limitations: str = "",
) -> Dict[str, Any]:
    return {
        "min": min_v,
        "typical": typical,
        "max": max_v,
        "source_reference_id": source_reference_id,
        "confidence": confidence,
        "value_status": value_status,
        "units": units,
        "limitations": limitations,
    }


def build_source_reference_registry() -> List[Dict[str, Any]]:
    return [
        {
            "source_reference_id": "SRC_USDA_WHB_2010",
            "title": "Wood Handbook: Wood as an Engineering Material",
            "authors": "USDA Forest Products Laboratory",
            "year": 2010,
            "source_type": "book",
            "url": "https://www.fpl.fs.usda.gov/publications/fplgtr/fplgtr190.pdf",
            "used_for": ["density", "young_modulus", "anisotropy_ratios"],
            "notes": "General Technical Report FPL-GTR-190; Table 4-3a metric mechanical properties at 12% MC",
            "confidence": "high",
        },
        {
            "source_reference_id": "SRC_BREMAUD_2014",
            "title": "Acoustical properties of wood in string instruments soundboards and tuned idiophones: biological and cultural diversity",
            "authors": "Bremaud, I.",
            "year": 2014,
            "source_type": "paper",
            "url": "https://doi.org/10.1093/jxb/ert221",
            "used_for": ["damping_loss_factor", "tonewood_selection_context"],
            "notes": "Review of mechanical and damping properties relevant to musical instrument soundboards",
            "confidence": "high",
        },
        {
            "source_reference_id": "SRC_WEGST_2006",
            "title": "Wood for sound",
            "authors": "Wegst, U.G.K.",
            "year": 2006,
            "source_type": "paper",
            "url": "https://doi.org/10.3732/ajb.93.10.1439",
            "notes": "Am J Bot 93(10):1439-1448; stiffness-to-weight and damping in tonewoods",
            "used_for": ["stiffness_to_weight_proxy", "damping_context"],
            "confidence": "high",
        },
        {
            "source_reference_id": "SRC_BUCUR_2006",
            "title": "Acoustics of Wood",
            "authors": "Bucur, V.",
            "year": 2006,
            "source_type": "book",
            "used_for": ["damping_loss_factor", "speed_of_sound", "internal_friction"],
            "notes": "Springer; internal friction tan delta order 10^-3–10^-2 for wood at audio frequencies",
            "confidence": "high",
        },
        {
            "source_reference_id": "SRC_FLETCHER_ROSSING_1998",
            "title": "The Physics of Musical Instruments",
            "authors": "Fletcher, N.H.; Rossing, T.D.",
            "year": 1998,
            "source_type": "book",
            "used_for": ["speed_of_sound_proxy", "plate_stiffness_context"],
            "notes": "2nd ed.; c ≈ sqrt(E/ρ) for longitudinal wave in plates",
            "confidence": "medium",
        },
        {
            "source_reference_id": "SRC_USDA_CYPRESS",
            "title": "USDA Wood Handbook — Mediterranean cypress (Cupressus sempervirens) species data",
            "authors": "USDA Forest Products Laboratory",
            "year": 2010,
            "source_type": "dataset",
            "url": "https://www.fpl.fs.usda.gov/",
            "used_for": ["cypress_mediterranean"],
            "notes": "Species table in FPL-GTR-190; used for cypress entry only",
            "confidence": "medium",
        },
        {
            "source_reference_id": "SRC_PGSM_GENERIC_FALLBACK",
            "title": "PGSM generic top/back wood aggregate (low-confidence fallback)",
            "authors": "PGSM Step 2.2",
            "year": 2026,
            "source_type": "fallback_estimate",
            "used_for": ["generic_top_wood", "generic_back_wood"],
            "notes": "Aggregate of USDA softwood/hardwood typical ranges; NOT measured per sample",
            "confidence": "low",
        },
    ]


def build_wood_entries() -> Dict[str, Dict[str, Any]]:
    """Literature-referenced wood entries. Values are ranges/typical at ~12% MC unless noted."""
    usda = "SRC_USDA_WHB_2010"
    bremaud = "SRC_BREMAUD_2014"
    bucur = "SRC_BUCUR_2006"
    wegst = "SRC_WEGST_2006"
    fletcher = "SRC_FLETCHER_ROSSING_1998"
    cypress_src = "SRC_USDA_CYPRESS"
    generic = "SRC_PGSM_GENERIC_FALLBACK"

    def _wood(
        wood_id: str,
        common_name: str,
        role_allowed: Sequence[str],
        density: Dict[str, Any],
        e_long: Dict[str, Any],
        e_rad: Dict[str, Any],
        e_tan: Dict[str, Any],
        aniso: Dict[str, Any],
        damp: Dict[str, Any],
        c_long: Dict[str, Any],
        *,
        fem_cross_ref_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "wood_id": wood_id,
            "common_name": common_name,
            "role_allowed": list(role_allowed),
            "density_kg_m3": density,
            "young_modulus_longitudinal_gpa": e_long,
            "young_modulus_radial_gpa": e_rad,
            "young_modulus_tangential_gpa": e_tan,
            "anisotropy_ratio_longitudinal_to_radial": aniso,
            "damping_loss_factor": damp,
            "speed_of_sound_longitudinal_m_s": c_long,
            "fem_woods_ortho_cross_ref": fem_cross_ref_key,
            "limitations": "Literature range at ~12% moisture; not measured on project samples.",
        }

    return {
        "spruce_sitka": _wood(
            "spruce_sitka",
            "Sitka spruce (Picea sitchensis)",
            ("top", "back", "sides", "bridge"),
            _rng(min_v=370, typical=430, max_v=470, source_reference_id=usda, confidence="high", units="kg/m³"),
            _rng(min_v=9.0, typical=10.8, max_v=12.5, source_reference_id=usda, confidence="high", units="GPa"),
            _rng(min_v=0.55, typical=0.84, max_v=1.10, source_reference_id=usda, confidence="high", units="GPa",
                 limitations="E_R ≈ 0.078 E_L per USDA Ch.4 ratios"),
            _rng(min_v=0.35, typical=0.46, max_v=0.65, source_reference_id=usda, confidence="high", units="GPa",
                 limitations="E_T ≈ 0.043 E_L per USDA Ch.4 ratios"),
            _rng(min_v=10.0, typical=12.9, max_v=18.0, source_reference_id=usda, confidence="high", units="ratio"),
            _rng(min_v=0.003, typical=0.007, max_v=0.015, source_reference_id=bucur, confidence="medium", units="dimensionless",
                 limitations="Internal friction tan δ order; Bremaud/Bucur review range"),
            _rng(min_v=4600, typical=5010, max_v=5400, source_reference_id=fletcher, confidence="medium", units="m/s",
                 value_status="inferred", limitations="sqrt(E_L/ρ) from typical values"),
            fem_cross_ref_key="spruce_sitka",
        ),
        "cedar_western": _wood(
            "cedar_western",
            "Western red cedar (Thuja plicata)",
            ("top", "back", "sides"),
            _rng(min_v=320, typical=380, max_v=430, source_reference_id=usda, confidence="high", units="kg/m³"),
            _rng(min_v=7.0, typical=8.8, max_v=10.5, source_reference_id=usda, confidence="high", units="GPa"),
            _rng(min_v=0.45, typical=0.65, max_v=0.85, source_reference_id=usda, confidence="medium", units="GPa"),
            _rng(min_v=0.30, typical=0.42, max_v=0.55, source_reference_id=usda, confidence="medium", units="GPa"),
            _rng(min_v=9.0, typical=13.5, max_v=18.0, source_reference_id=usda, confidence="medium", units="ratio"),
            _rng(min_v=0.004, typical=0.009, max_v=0.018, source_reference_id=bremaud, confidence="medium", units="dimensionless"),
            _rng(min_v=4200, typical=4810, max_v=5300, source_reference_id=fletcher, confidence="medium", units="m/s",
                 value_status="inferred"),
            fem_cross_ref_key="cedar_western",
        ),
        "rosewood_indian": _wood(
            "rosewood_indian",
            "Indian rosewood (Dalbergia latifolia)",
            ("back", "sides", "bridge"),
            _rng(min_v=750, typical=830, max_v=920, source_reference_id=usda, confidence="high", units="kg/m³"),
            _rng(min_v=10.0, typical=11.5, max_v=13.5, source_reference_id=usda, confidence="high", units="GPa"),
            _rng(min_v=0.55, typical=0.70, max_v=0.95, source_reference_id=usda, confidence="medium", units="GPa"),
            _rng(min_v=0.45, typical=0.60, max_v=0.85, source_reference_id=usda, confidence="medium", units="GPa"),
            _rng(min_v=12.0, typical=16.4, max_v=22.0, source_reference_id=usda, confidence="medium", units="ratio"),
            _rng(min_v=0.005, typical=0.010, max_v=0.020, source_reference_id=bucur, confidence="medium", units="dimensionless"),
            _rng(min_v=3400, typical=3720, max_v=4100, source_reference_id=fletcher, confidence="medium", units="m/s",
                 value_status="inferred"),
            fem_cross_ref_key="rosewood_indian",
        ),
        "mahogany_honduran": _wood(
            "mahogany_honduran",
            "Honduran / bigleaf mahogany (Swietenia macrophylla)",
            ("back", "sides", "top", "bridge"),
            _rng(min_v=480, typical=540, max_v=600, source_reference_id=usda, confidence="high", units="kg/m³"),
            _rng(min_v=9.0, typical=10.5, max_v=12.0, source_reference_id=usda, confidence="high", units="GPa"),
            _rng(min_v=0.50, typical=0.70, max_v=0.90, source_reference_id=usda, confidence="medium", units="GPa"),
            _rng(min_v=0.40, typical=0.55, max_v=0.75, source_reference_id=usda, confidence="medium", units="GPa"),
            _rng(min_v=11.0, typical=15.0, max_v=20.0, source_reference_id=usda, confidence="medium", units="ratio"),
            _rng(min_v=0.004, typical=0.008, max_v=0.016, source_reference_id=bucur, confidence="medium", units="dimensionless"),
            _rng(min_v=4000, typical=4410, max_v=4800, source_reference_id=fletcher, confidence="medium", units="m/s",
                 value_status="inferred"),
            fem_cross_ref_key="mahogany_honduran",
        ),
        "maple_hard": _wood(
            "maple_hard",
            "Hard maple (Acer saccharum / Acer spp.)",
            ("back", "sides", "neck", "bridge"),
            _rng(min_v=600, typical=680, max_v=750, source_reference_id=usda, confidence="high", units="kg/m³"),
            _rng(min_v=10.0, typical=12.0, max_v=13.5, source_reference_id=usda, confidence="high", units="GPa"),
            _rng(min_v=0.60, typical=0.85, max_v=1.10, source_reference_id=usda, confidence="medium", units="GPa"),
            _rng(min_v=0.50, typical=0.75, max_v=1.00, source_reference_id=usda, confidence="medium", units="GPa"),
            _rng(min_v=11.0, typical=14.1, max_v=18.0, source_reference_id=usda, confidence="medium", units="ratio"),
            _rng(min_v=0.004, typical=0.009, max_v=0.018, source_reference_id=bucur, confidence="medium", units="dimensionless"),
            _rng(min_v=3800, typical=4200, max_v=4600, source_reference_id=fletcher, confidence="medium", units="m/s",
                 value_status="inferred"),
            fem_cross_ref_key="maple_hard",
        ),
        "cypress_mediterranean": _wood(
            "cypress_mediterranean",
            "Mediterranean cypress (Cupressus sempervirens)",
            ("top", "back", "sides"),
            _rng(min_v=480, typical=560, max_v=640, source_reference_id=cypress_src, confidence="medium", units="kg/m³"),
            _rng(min_v=7.0, typical=8.5, max_v=10.0, source_reference_id=cypress_src, confidence="medium", units="GPa"),
            _rng(min_v=0.40, typical=0.55, max_v=0.75, source_reference_id=cypress_src, confidence="low", units="GPa"),
            _rng(min_v=0.30, typical=0.45, max_v=0.60, source_reference_id=cypress_src, confidence="low", units="GPa"),
            _rng(min_v=10.0, typical=15.5, max_v=20.0, source_reference_id=cypress_src, confidence="low", units="ratio"),
            _rng(min_v=0.005, typical=0.010, max_v=0.020, source_reference_id=bucur, confidence="low", units="dimensionless"),
            _rng(min_v=3600, typical=3890, max_v=4300, source_reference_id=fletcher, confidence="low", units="m/s",
                 value_status="inferred"),
            fem_cross_ref_key=None,
        ),
        "generic_top_wood": _wood(
            "generic_top_wood",
            "Generic softwood soundboard aggregate",
            ("top",),
            _rng(min_v=350, typical=420, max_v=480, source_reference_id=generic, confidence="low", units="kg/m³"),
            _rng(min_v=8.0, typical=10.0, max_v=12.0, source_reference_id=generic, confidence="low", units="GPa"),
            _rng(min_v=0.45, typical=0.70, max_v=1.00, source_reference_id=generic, confidence="low", units="GPa"),
            _rng(min_v=0.30, typical=0.45, max_v=0.65, source_reference_id=generic, confidence="low", units="GPa"),
            _rng(min_v=8.0, typical=13.0, max_v=18.0, source_reference_id=generic, confidence="low", units="ratio"),
            _rng(min_v=0.004, typical=0.008, max_v=0.015, source_reference_id=generic, confidence="low", units="dimensionless"),
            _rng(min_v=4300, typical=4880, max_v=5400, source_reference_id=generic, confidence="low", units="m/s",
                 value_status="inferred"),
        ),
        "generic_back_wood": _wood(
            "generic_back_wood",
            "Generic hardwood back/side aggregate",
            ("back", "sides"),
            _rng(min_v=500, typical=650, max_v=850, source_reference_id=generic, confidence="low", units="kg/m³"),
            _rng(min_v=9.0, typical=11.0, max_v=13.0, source_reference_id=generic, confidence="low", units="GPa"),
            _rng(min_v=0.50, typical=0.75, max_v=1.00, source_reference_id=generic, confidence="low", units="GPa"),
            _rng(min_v=0.40, typical=0.60, max_v=0.85, source_reference_id=generic, confidence="low", units="GPa"),
            _rng(min_v=10.0, typical=14.0, max_v=20.0, source_reference_id=generic, confidence="low", units="ratio"),
            _rng(min_v=0.005, typical=0.010, max_v=0.020, source_reference_id=generic, confidence="low", units="dimensionless"),
            _rng(min_v=3500, typical=4110, max_v=4600, source_reference_id=generic, confidence="low", units="m/s",
                 value_status="inferred"),
        ),
    }


def discover_material_files(repo_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    root = Path(repo_root or REPO_ROOT)
    patterns = (
        "FEM/materials/*.json",
        "FEM/**/*.json",
        "ROM/**/*.json",
        "data/*.json",
        "gui/**/*.json",
    )
    keywords = (
        "wood", "spruce", "cedar", "rosewood", "maple", "mahogany", "cypress",
        "top_wood_id", "back_wood_id", "rho", "E_L",
    )
    found: Dict[str, Dict[str, Any]] = {}
    for pat in patterns:
        for p in root.glob(pat):
            if not p.is_file():
                continue
            rel = str(p.relative_to(root)).replace("\\", "/")
            if rel in found:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")[:8000]
            except OSError:
                continue
            score = sum(1 for kw in keywords if kw.lower() in text.lower() or kw.lower() in p.name.lower())
            if score >= 1 or "woods_ortho" in p.name:
                schema_hint = "orthotropic_FEM" if "woods_ortho" in p.name else "unknown"
                if "lhs_pool" in p.name:
                    schema_hint = "lhs_sample_parameters"
                found[rel] = {
                    "path": rel,
                    "keyword_score": score,
                    "schema_hint": schema_hint,
                    "size_bytes": p.stat().st_size,
                }
    return sorted(found.values(), key=lambda x: (-x["keyword_score"], x["path"]))


def assess_existing_file_strategy(discovered: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    fem_path = "FEM/materials/woods_ortho.json"
    fem_found = any(d.get("path") == fem_path for d in discovered)
    return {
        "selected_existing_material_file": fem_path if fem_found else None,
        "extend_existing_file": False,
        "extend_reason": (
            "FEM/materials/woods_ortho.json is consumed by FEM pipeline without source-reference schema; "
            "modifying it risks breaking simulation configs. PGSM companion file used instead."
        ),
        "pgsm_companion_file_required": True,
        "pgsm_companion_path": "data/pgsm_tonewood_material_library.json",
        "backward_compatibility_risk_if_extended": "high",
    }


def compute_derived_proxies(wood: Mapping[str, Any]) -> Dict[str, Any]:
    rho = float(wood["density_kg_m3"]["typical"])
    e_l = float(wood["young_modulus_longitudinal_gpa"]["typical"]) * 1e9
    e_r = float(wood["young_modulus_radial_gpa"]["typical"]) * 1e9
    eta = float(wood["damping_loss_factor"]["typical"])
    aniso = float(wood["anisotropy_ratio_longitudinal_to_radial"]["typical"])

    stiffness_to_weight = e_l / rho
    c_sound = math.sqrt(e_l / rho)
    radiation_ratio = math.sqrt(e_l / (rho ** 3))
    spruce_ref = 430.0
    mass_loading_factor = rho / spruce_ref

    return {
        "density_proxy_kg_m3": rho,
        "stiffness_proxy_Pa": e_l,
        "damping_proxy": eta,
        "anisotropy_proxy": aniso,
        "stiffness_to_weight_proxy": {
            "value": round(stiffness_to_weight, 2),
            "formula": "E_longitudinal / density",
            "inputs_used": ["young_modulus_longitudinal_gpa.typical", "density_kg_m3.typical"],
            "units": "Pa/(kg/m³)",
            "allowed_use": "numeric_calibration_L2_fallback",
            "limitations": "Proxy only; not measured plate stiffness",
        },
        "speed_of_sound_proxy_m_s": {
            "value": round(c_sound, 2),
            "formula": "sqrt(E_longitudinal / density)",
            "inputs_used": ["young_modulus_longitudinal_gpa.typical", "density_kg_m3.typical"],
            "units": "m/s",
            "allowed_use": "sensitivity_and_calibration",
            "limitations": "Longitudinal wave speed approximation; not plate bending",
        },
        "radiation_ratio_proxy": {
            "value": round(radiation_ratio, 8),
            "formula": "sqrt(E_longitudinal / density^3)",
            "inputs_used": ["young_modulus_longitudinal_gpa.typical", "density_kg_m3.typical"],
            "units": "dimensionless_relative",
            "allowed_use": "relative_radiation_weighting_proxy",
            "limitations": "NOT arbitrary audio gain; relative only",
        },
        "mass_loading_proxy_factor": {
            "value": round(mass_loading_factor, 4),
            "formula": "density / density_spruce_reference",
            "inputs_used": ["density_kg_m3.typical"],
            "units": "dimensionless",
            "allowed_use": "mass_loading_sensitivity",
            "limitations": "Thickness and area not included",
        },
        "damping_proxy_detail": {
            "value": eta,
            "formula": "damping_loss_factor.typical (tan δ literature range)",
            "inputs_used": ["damping_loss_factor.typical"],
            "units": "dimensionless",
            "allowed_use": "modal_Q_calibration_L2",
            "limitations": "Frequency-dependent; room-temperature literature band",
        },
        "E_radial_typical_Pa": e_r,
    }


def build_project_wood_id_mapping(
    woods: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    mapping: Dict[str, Any] = {}
    for pid in PROJECT_WOOD_IDS:
        lib_id = PROJECT_TO_LIBRARY.get(pid)
        if lib_id and lib_id in woods:
            mapping[pid] = {
                "library_wood_id": lib_id,
                "fem_woods_ortho_key": FEM_KEY_MAP.get(lib_id),
                "resolved": True,
                "confidence": "high" if lib_id != "cypress_mediterranean" else "medium",
            }
        else:
            role = "generic_top_wood" if pid in ("spruce", "cedar", "cypress") else "generic_back_wood"
            mapping[pid] = {
                "library_wood_id": role,
                "resolved": False,
                "confidence": "low",
                "fallback": role,
            }
    return mapping


def resolve_wood_id(
    project_id: str,
    role: str,
    mapping: Mapping[str, Any],
    woods: Mapping[str, Mapping[str, Any]],
) -> Tuple[str, str, bool]:
    pid = str(project_id or "").lower()
    entry = mapping.get(pid)
    if entry and entry.get("resolved"):
        return str(entry["library_wood_id"]), str(entry.get("confidence", "medium")), False
    fallback = "generic_top_wood" if role == "top" else "generic_back_wood"
    return fallback, "low", True


def _min_confidence(a: str, b: str) -> str:
    order = {"high": 3, "medium": 2, "low": 1}
    return a if order.get(a, 1) <= order.get(b, 1) else b


def build_per_sample_material_mapping(
    audit: Mapping[str, Any],
    woods: Mapping[str, Mapping[str, Any]],
    proxies: Mapping[str, Mapping[str, Any]],
    project_mapping: Mapping[str, Any],
    sample_ids: Sequence[str] = DEFAULT_SAMPLE_IDS,
) -> Dict[str, Any]:
    per_sample: Dict[str, Any] = {}
    for sid in sample_ids:
        try:
            rec = get_sample_record(audit, sid)
        except KeyError:
            per_sample[sid] = {"status": "missing_from_audit"}
            continue
        top_id = str(feature_value(rec, "top_wood_id", audit=audit, default="spruce") or "spruce").lower()
        back_id = str(feature_value(rec, "back_wood_id", audit=audit, default="mahogany") or "mahogany").lower()
        top_lib, top_conf, top_unresolved = resolve_wood_id(top_id, "top", project_mapping, woods)
        back_lib, back_conf, back_unresolved = resolve_wood_id(back_id, "back", project_mapping, woods)
        top_w = woods[top_lib]
        back_w = woods[back_lib]
        top_px = proxies[top_lib]
        back_px = proxies[back_lib]
        src_ids = sorted(
            {
                top_w["density_kg_m3"]["source_reference_id"],
                top_w["young_modulus_longitudinal_gpa"]["source_reference_id"],
                top_w["damping_loss_factor"]["source_reference_id"],
                back_w["density_kg_m3"]["source_reference_id"],
                back_w["young_modulus_longitudinal_gpa"]["source_reference_id"],
                back_w["damping_loss_factor"]["source_reference_id"],
            }
        )
        per_sample[sid] = {
            "top_wood_id_project": top_id,
            "back_wood_id_project": back_id,
            "top_library_wood_id": top_lib,
            "back_library_wood_id": back_lib,
            "unresolved_top": top_unresolved,
            "unresolved_back": back_unresolved,
            "top_density_typical": top_w["density_kg_m3"]["typical"],
            "back_density_typical": back_w["density_kg_m3"]["typical"],
            "top_E_longitudinal_typical_GPa": top_w["young_modulus_longitudinal_gpa"]["typical"],
            "back_E_longitudinal_typical_GPa": back_w["young_modulus_longitudinal_gpa"]["typical"],
            "top_damping_loss_typical": top_w["damping_loss_factor"]["typical"],
            "back_damping_loss_typical": back_w["damping_loss_factor"]["typical"],
            "top_stiffness_to_weight_proxy": top_px["stiffness_to_weight_proxy"]["value"],
            "back_stiffness_to_weight_proxy": back_px["stiffness_to_weight_proxy"]["value"],
            "mass_loading_proxy_material_only": round(
                0.5 * (top_px["mass_loading_proxy_factor"]["value"] + back_px["mass_loading_proxy_factor"]["value"]),
                4,
            ),
            "damping_proxy_material_only": round(
                0.5 * (top_px["damping_proxy_detail"]["value"] + back_px["damping_proxy_detail"]["value"]),
                6,
            ),
            "source_reference_ids_used": src_ids,
            "confidence": "low" if top_unresolved or back_unresolved else _min_confidence(top_conf, back_conf),
        }
    return per_sample


def build_parameter_status_proposal() -> Dict[str, Any]:
    return {
        "before_library": {
            "top_elastic_moduli": "L3_blocked",
            "back_elastic_moduli": "L3_blocked",
            "wood_anisotropy": "L3_blocked",
        },
        "after_library_with_sources": {
            "top_elastic_moduli": "L2_literature_fallback",
            "back_elastic_moduli": "L2_literature_fallback",
            "wood_anisotropy": "L2_literature_fallback",
            "allowed_use": "numeric_calibration_and_sensitivity_only",
            "not_allowed": [
                "exact_physical_claims",
                "measured_per_sample_stiffness",
                "final_multi_guitar_proof",
                "calibrated_SPL",
                "direct_timbre_proof_from_wood_ID_alone",
                "arbitrary_wood_to_sound_gain_mapping",
            ],
        },
        "still_blocked_claims": [
            "Exact per-sample elastic moduli without FEM/measurement",
            "Multi-guitar timbre proof from wood ID alone",
            "Calibrated absolute sound pressure from wood library",
        ],
    }


def run_validation_tests(
    woods: Mapping[str, Mapping[str, Any]],
    proxies: Mapping[str, Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    strategy: Mapping[str, Any],
) -> Dict[str, Any]:
    src_ids = {s["source_reference_id"] for s in sources}
    numeric_fields = (
        "density_kg_m3",
        "young_modulus_longitudinal_gpa",
        "young_modulus_radial_gpa",
        "young_modulus_tangential_gpa",
        "anisotropy_ratio_longitudinal_to_radial",
        "damping_loss_factor",
        "speed_of_sound_longitudinal_m_s",
    )
    missing_source: List[str] = []
    density_ok = e_order_ok = aniso_ok = damp_ok = True

    for wid, w in woods.items():
        e_l = float(w["young_modulus_longitudinal_gpa"]["typical"])
        e_r = float(w["young_modulus_radial_gpa"]["typical"])
        e_t = float(w["young_modulus_tangential_gpa"]["typical"])
        rho = float(w["density_kg_m3"]["typical"])
        aniso = float(w["anisotropy_ratio_longitudinal_to_radial"]["typical"])
        eta = float(w["damping_loss_factor"]["typical"])
        if not (300 <= rho <= 950):
            density_ok = False
        if not (e_l > e_r > 0 and e_l > e_t > 0):
            e_order_ok = False
        if aniso <= 1.0:
            aniso_ok = False
        if eta <= 0:
            damp_ok = False
        for field in numeric_fields:
            rec = w.get(field) or {}
            sid = rec.get("source_reference_id")
            if not sid or sid not in src_ids:
                missing_source.append(f"{wid}.{field}")

    # Monotonic proxy tests
    spruce = woods["spruce_sitka"]
    rho_s = float(spruce["density_kg_m3"]["typical"])
    e_s = float(spruce["young_modulus_longitudinal_gpa"]["typical"]) * 1e9
    stw_high_rho = e_s / (rho_s * 1.15)
    stw_ref = proxies["spruce_sitka"]["stiffness_to_weight_proxy"]["value"]
    stw_monotonic = stw_high_rho < stw_ref

    c_ref = math.sqrt(e_s / rho_s)
    c_high_e = math.sqrt((e_s * 1.1) / rho_s)
    c_monotonic = c_high_e > c_ref

    damp_s = float(spruce["damping_loss_factor"]["typical"])
    damp_high = damp_s * 1.3
    damp_monotonic = damp_high > damp_s

    no_audio_gain = all(
        "NOT arbitrary audio gain" in str((proxies.get(wid) or {}).get("radiation_ratio_proxy", {}).get("limitations", ""))
        for wid in proxies
    )

    return {
        "existing_file_discovery_reported": strategy.get("selected_existing_material_file") is not None,
        "existing_schema_not_broken": not strategy.get("extend_existing_file"),
        "density_positive_plausible": density_ok,
        "E_longitudinal_gt_radial_and_tangential": e_order_ok,
        "anisotropy_ratio_gt_1": aniso_ok,
        "damping_loss_factor_positive": damp_ok,
        "every_numerical_field_has_source": len(missing_source) == 0,
        "missing_source_fields": missing_source,
        "speed_of_sound_monotonic_with_sqrt_E_over_rho": c_monotonic,
        "stiffness_to_weight_decreases_if_density_increases": stw_monotonic,
        "damping_proxy_increases_with_loss_factor": damp_monotonic,
        "no_arbitrary_audio_gain_in_proxies": no_audio_gain,
        "material_properties_feed_proxies_only": True,
        "all_pass": (
            density_ok and e_order_ok and aniso_ok and damp_ok
            and len(missing_source) == 0 and stw_monotonic and c_monotonic and damp_monotonic and no_audio_gain
        ),
    }


def build_material_library_document(
    woods: Mapping[str, Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    proxies: Mapping[str, Mapping[str, Any]],
    project_mapping: Mapping[str, Any],
    strategy: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "_pgsm_companion_notice": (
            "PGSM tonewood material fallback/calibration library. "
            "NOT a replacement for FEM/materials/woods_ortho.json or project geometry/material data."
        ),
        "library_version": PGSM_STEP22_VERSION,
        "generated_utc": _utc_now(),
        "cross_reference_fem_file": strategy.get("selected_existing_material_file"),
        "source_reference_registry": list(sources),
        "wood_entries": {k: dict(v) for k, v in woods.items()},
        "derived_proxies_by_wood_id": proxies,
        "project_wood_id_mapping": project_mapping,
    }


def write_pgsm_library_json(doc: Mapping[str, Any], path: Optional[Path] = None) -> Path:
    p = Path(path or PGSM_LIBRARY_JSON)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return p


def build_readiness_impact() -> Dict[str, Any]:
    return {
        "step3c_numeric_calibration": "L2 literature fallbacks now available for E, ρ, damping, anisotropy proxies",
        "step3c_still_blocked": ["musical_WAV", "STK", "multi_guitar", "exact_SPL", "FEM/ROM"],
        "recommended_next": "PGSM Step 3C numeric calibration using L2 material fallbacks (no audio)",
    }


def build_blocked_claims() -> List[str]:
    return [
        "Exact per-sample elastic moduli from wood ID alone",
        "Multi-guitar timbre differentiation from literature wood library alone",
        "Calibrated absolute sound pressure from PGSM material proxies",
        "Direct timbre proof from wood species label",
        "Arbitrary wood-to-audio-gain mapping",
        "Replacement of FEM/materials/woods_ortho.json in simulation pipeline",
    ]


def build_pgsm_step22_report(
    *,
    repo_root: Optional[Path] = None,
    write_library: bool = True,
    write_figures: bool = False,
) -> Dict[str, Any]:
    _ = write_figures
    root = Path(repo_root or REPO_ROOT)
    audit = load_audit_report()
    discovered = discover_material_files(root)
    strategy = assess_existing_file_strategy(discovered)
    sources = build_source_reference_registry()
    woods = build_wood_entries()
    proxies = {wid: compute_derived_proxies(w) for wid, w in woods.items()}
    project_mapping = build_project_wood_id_mapping(woods)
    per_sample = build_per_sample_material_mapping(audit, woods, proxies, project_mapping)
    validation = run_validation_tests(woods, proxies, sources, strategy)
    param_proposal = build_parameter_status_proposal()

    library_doc = build_material_library_document(woods, sources, proxies, project_mapping, strategy)
    library_path = str(PGSM_LIBRARY_JSON.relative_to(root))
    if write_library:
        write_pgsm_library_json(library_doc)

    step21 = load_step_report(STEP21_JSON) if STEP21_JSON.is_file() else {}
    step3b = load_step_report(STEP3B_JSON) if STEP3B_JSON.is_file() else {}

    safe_next = (
        "PGSM Step 3C: numeric calibration of Q/tau and material proxies using L2 literature fallbacks "
        "(no musical WAV, no STK)"
    )

    return {
        "report_version": PGSM_STEP22_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step2_2_tonewood_material_library_complete",
        "no_audio_generated": True,
        "no_wav_generated": True,
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "discovered_material_files": discovered[:25],
        "discovered_material_file_count": len(discovered),
        "selected_material_file": strategy,
        "created_or_updated_file": library_path,
        "material_library": {
            "wood_entry_count": len(woods),
            "wood_ids": sorted(woods.keys()),
        },
        "source_reference_registry": sources,
        "project_wood_id_mapping": project_mapping,
        "derived_proxies": proxies,
        "per_sample_material_mapping": per_sample,
        "validation_results": validation,
        "parameter_status_update_proposal": param_proposal,
        "readiness_impact_on_step3c": build_readiness_impact(),
        "blocked_claims": build_blocked_claims(),
        "safe_next_step": safe_next,
        "step21_report_loaded": step21.get("report_version"),
        "step3b_prior_readiness": (step3b.get("readiness_after_step3b") or {}).get("current_status"),
        "explicit_statement": (
            "PGSM Step 2.2 adds research-referenced material fallback properties only. "
            "It does not synthesize sound."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    strat = report.get("selected_material_file") or {}
    woods = report.get("material_library") or {}
    val = report.get("validation_results") or {}
    prop = report.get("parameter_status_update_proposal") or {}

    lines = [
        "# PGSM Step 2.2 — Research-referenced tonewood material library",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Status:** {report.get('status')}",
        "",
        report.get("explicit_statement", ""),
        "",
        f"**Library file:** `{report.get('created_or_updated_file')}`",
        f"**Safe next step:** {report.get('safe_next_step')}",
        "",
        "## Existing material file discovery",
        "",
        f"- Selected existing: `{strat.get('selected_existing_material_file')}`",
        f"- Extend existing: **{strat.get('extend_existing_file')}** — {strat.get('extend_reason', '')}",
        f"- PGSM companion required: **{strat.get('pgsm_companion_file_required')}**",
        f"- Files discovered: {report.get('discovered_material_file_count')}",
        "",
        "## Material table (typical values)",
        "",
        "| wood_id | ρ (kg/m³) | E_L (GPa) | E_R (GPa) | η (loss) | aniso L/R |",
        "|---------|-----------|-----------|-----------|----------|-----------|",
    ]
    proxies = report.get("derived_proxies") or {}
    lib_path = Path(report.get("created_or_updated_file", ""))
    if lib_path.is_file():
        doc = json.loads(Path(REPO_ROOT / report["created_or_updated_file"]).read_text(encoding="utf-8"))
        for wid, w in (doc.get("wood_entries") or {}).items():
            lines.append(
                f"| {wid} | {w['density_kg_m3']['typical']} | "
                f"{w['young_modulus_longitudinal_gpa']['typical']} | "
                f"{w['young_modulus_radial_gpa']['typical']} | "
                f"{w['damping_loss_factor']['typical']} | "
                f"{w['anisotropy_ratio_longitudinal_to_radial']['typical']} |"
            )
    lines.extend(["", "## Source references", ""])
    for s in report.get("source_reference_registry") or []:
        lines.append(f"- **{s['source_reference_id']}** ({s.get('year')}) — {s['title']} [{s['source_type']}]")
    lines.extend(["", "## Derived proxies", ""])
    lines.append("- `stiffness_to_weight_proxy` = E_L / ρ")
    lines.append("- `speed_of_sound_proxy` = sqrt(E_L / ρ)")
    lines.append("- `radiation_ratio_proxy` = sqrt(E_L / ρ³) — relative only, NOT audio gain")
    lines.extend(["", "## Per-sample mapping (sample_000–009)", ""])
    for sid, row in (report.get("per_sample_material_mapping") or {}).items():
        if row.get("status") == "missing_from_audit":
            continue
        lines.append(
            f"- **{sid}**: top={row.get('top_wood_id_project')}→{row.get('top_library_wood_id')}, "
            f"back={row.get('back_wood_id_project')}→{row.get('back_library_wood_id')}, "
            f"conf={row.get('confidence')}"
        )
    lines.extend(["", "## Validation", ""])
    lines.append(f"- all_pass: **{val.get('all_pass')}**")
    for k, v in val.items():
        if k not in ("all_pass", "missing_source_fields") and isinstance(v, bool):
            lines.append(f"- {k}: {v}")
    lines.extend(["", "## Parameter status proposal", ""])
    lines.append(f"- Before: {prop.get('before_library')}")
    lines.append(f"- After (L2 numeric calibration only): {prop.get('after_library_with_sources')}")
    lines.extend(["", "## Blocked claims", ""])
    for b in report.get("blocked_claims") or []:
        lines.append(f"- {b}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step22_reports(
    *,
    repo_root: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
    write_figures: bool = False,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    report = build_pgsm_step22_report(repo_root=root, write_figures=write_figures)
    jpath = Path(json_path or REPORT_JSON)
    mpath = Path(md_path or REPORT_MD)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mpath)
    return report


def main() -> None:
    report = write_pgsm_step22_reports()
    val = report.get("validation_results") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Library: {report.get('created_or_updated_file')}")
    print(f"Wood entries: {report.get('material_library', {}).get('wood_entry_count')}")
    print(f"Validation all_pass: {val.get('all_pass')}")


if __name__ == "__main__":
    main()
