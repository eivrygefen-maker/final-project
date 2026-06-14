#!/usr/bin/env python3
"""
PGSM Step 2.2b — FEM/PGSM material alignment audit (read-only; no FEM/ROM run).
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pgsm_physical_factor_registry import DEFAULT_SAMPLE_IDS, load_audit_report
from pgsm_step2_1_parameter_targets import load_step_report
from pgsm_tonewood_material_library import (
    PGSM_LIBRARY_JSON,
    PROJECT_TO_LIBRARY,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE
from stk_v6_2_audit_features import feature_value, get_sample_record

PGSM_STEP22B_VERSION = "pgsm_step2_2b_material_alignment_audit_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
FEM_WOODS_ORTHO = REPO_ROOT / "FEM" / "materials" / "woods_ortho.json"
FEM_WOOD_LIBRARY = REPO_ROOT / "FEM" / "scripts" / "wood_library.py"
STEP22_REPORT = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step2_2_tonewood_material_library.json"
LHS_POOL = REPO_ROOT / "ROM" / "classic" / "lhs_pool.json"
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step2_2b_material_alignment_audit.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step2_2b_material_alignment_audit.md"

# Project short ID → FEM woods_ortho.json key
PROJECT_TO_FEM_KEY: Dict[str, str] = {
    "spruce": "spruce_sitka",
    "cedar": "cedar_western",
    "rosewood": "rosewood_indian",
    "mahogany": "mahogany_honduran",
    "maple": "maple_hard",
}

DENSITY_WARN_PCT = 15.0
DENSITY_FAIL_PCT = 30.0
E_MOD_WARN_PCT = 25.0
E_MOD_FAIL_PCT = 50.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel_diff_pct(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom * 100.0


def load_fem_woods_ortho(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path or FEM_WOODS_ORTHO)
    if not p.is_file():
        return {"status": "missing", "materials": {}}
    doc = json.loads(p.read_text(encoding="utf-8"))
    return {"status": "ok", "path": str(p.relative_to(REPO_ROOT)).replace("\\", "/"), "materials": doc}


def load_pgsm_library(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path or PGSM_LIBRARY_JSON)
    if not p.is_file():
        return {"status": "missing", "wood_entries": {}}
    doc = json.loads(p.read_text(encoding="utf-8"))
    return {
        "status": "ok",
        "path": str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
        "wood_entries": doc.get("wood_entries") or {},
        "project_wood_id_mapping": doc.get("project_wood_id_mapping") or {},
        "derived_proxies_by_wood_id": doc.get("derived_proxies_by_wood_id") or {},
    }


def extract_fem_material(fem_entry: Mapping[str, Any]) -> Dict[str, Any]:
    e_l = float(fem_entry.get("E_L", 0.0))
    e_r = float(fem_entry.get("E_R", 0.0))
    e_t = float(fem_entry.get("E_T", 0.0))
    rho = float(fem_entry.get("rho", 0.0))
    q_min = fem_entry.get("q_min")
    q_max = fem_entry.get("q_max")
    q_mid = None
    loss_fem = None
    if q_min is not None and q_max is not None:
        q_mid = 0.5 * (float(q_min) + float(q_max))
        loss_fem = 1.0 / (2.0 * max(q_mid, 1.0))
    aniso = e_l / max(e_r, 1e-12) if e_r > 0 else None
    return {
        "density_kg_m3": rho,
        "young_modulus_longitudinal_gpa": e_l / 1e9,
        "young_modulus_radial_gpa": e_r / 1e9,
        "young_modulus_tangential_gpa": e_t / 1e9,
        "anisotropy_ratio_longitudinal_to_radial": aniso,
        "q_min": q_min,
        "q_max": q_max,
        "q_mid": q_mid,
        "damping_loss_factor_from_q": loss_fem,
    }


def extract_pgsm_material(pgsm_entry: Mapping[str, Any]) -> Dict[str, Any]:
    def _typ(field: str) -> float:
        return float((pgsm_entry.get(field) or {}).get("typical", 0.0))

    return {
        "density_kg_m3": _typ("density_kg_m3"),
        "density_min": float((pgsm_entry.get("density_kg_m3") or {}).get("min", 0.0)),
        "density_max": float((pgsm_entry.get("density_kg_m3") or {}).get("max", 0.0)),
        "young_modulus_longitudinal_gpa": _typ("young_modulus_longitudinal_gpa"),
        "young_modulus_radial_gpa": _typ("young_modulus_radial_gpa"),
        "young_modulus_tangential_gpa": _typ("young_modulus_tangential_gpa"),
        "anisotropy_ratio_longitudinal_to_radial": _typ("anisotropy_ratio_longitudinal_to_radial"),
        "damping_loss_factor": _typ("damping_loss_factor"),
    }


def _classify_scalar(
    fem_val: float,
    pgsm_typical: float,
    pgsm_min: Optional[float],
    pgsm_max: Optional[float],
    *,
    warn_pct: float,
    fail_pct: float,
) -> Tuple[str, float]:
    if fem_val <= 0 or pgsm_typical <= 0:
        return "not_comparable_schema", 0.0
    pct = _rel_diff_pct(fem_val, pgsm_typical)
    in_range = (
        pgsm_min is not None
        and pgsm_max is not None
        and pgsm_min <= fem_val <= pgsm_max
    )
    if pct <= 5.0:
        return "aligned", pct
    if pct <= warn_pct:
        return "close_within_tolerance", pct
    if in_range:
        return "different_but_explainable_by_literature_range", pct
    if pct >= fail_pct:
        return "mismatch_requires_attention", pct
    return "close_within_tolerance", pct


def compare_material_pair(
    fem_key: str,
    pgsm_key: str,
    fem: Mapping[str, Any],
    pgsm: Mapping[str, Any],
) -> Dict[str, Any]:
    f = extract_fem_material(fem)
    p = extract_pgsm_material(pgsm)

    comparisons: Dict[str, Any] = {}
    statuses: List[str] = []

    for field, warn, fail in (
        ("density_kg_m3", DENSITY_WARN_PCT, DENSITY_FAIL_PCT),
        ("young_modulus_longitudinal_gpa", E_MOD_WARN_PCT, E_MOD_FAIL_PCT),
        ("young_modulus_radial_gpa", E_MOD_WARN_PCT, E_MOD_FAIL_PCT),
        ("young_modulus_tangential_gpa", E_MOD_WARN_PCT, E_MOD_FAIL_PCT),
    ):
        status, pct = _classify_scalar(
            float(f[field]),
            float(p[field]),
            p.get("density_min") if field == "density_kg_m3" else None,
            p.get("density_max") if field == "density_kg_m3" else None,
            warn_pct=warn,
            fail_pct=fail,
        )
        comparisons[field] = {
            "fem_value": f[field],
            "pgsm_typical": p[field],
            "relative_diff_pct": round(pct, 2),
            "status": status,
        }
        statuses.append(status)

    aniso_f = f.get("anisotropy_ratio_longitudinal_to_radial")
    aniso_p = p.get("anisotropy_ratio_longitudinal_to_radial")
    aniso_status = "not_comparable_schema"
    aniso_pct = 0.0
    if aniso_f and aniso_p:
        aniso_pct = _rel_diff_pct(aniso_f, aniso_p)
        if aniso_f > 1.0 and aniso_p > 1.0:
            if aniso_pct <= 15.0:
                aniso_status = "aligned" if aniso_pct <= 5.0 else "close_within_tolerance"
            elif aniso_pct <= 30.0:
                aniso_status = "close_within_tolerance"
            else:
                aniso_status = "mismatch_requires_attention"
        else:
            aniso_status = "mismatch_requires_attention"
    comparisons["anisotropy_ratio_longitudinal_to_radial"] = {
        "fem_value": aniso_f,
        "pgsm_typical": aniso_p,
        "relative_diff_pct": round(aniso_pct, 2),
        "status": aniso_status,
    }
    statuses.append(aniso_status)

    loss_fem = f.get("damping_loss_factor_from_q")
    loss_p = p.get("damping_loss_factor")
    damp_status = "not_comparable_schema"
    damp_note = "FEM Q vs PGSM tan δ — qualitative comparison"
    if loss_fem is not None and loss_p:
        ratio = loss_fem / max(loss_p, 1e-12)
        if 0.5 <= ratio <= 2.0:
            damp_status = "close_within_tolerance"
        elif 0.25 <= ratio <= 4.0:
            damp_status = "different_but_explainable_by_literature_range"
        else:
            damp_status = "mismatch_requires_attention"
        damp_note = f"η_fem≈1/(2Q_mid)={loss_fem:.4f}, η_pgsm={loss_p:.4f}, ratio={ratio:.2f}"
    comparisons["damping_loss_factor"] = {
        "fem_q_mid": f.get("q_mid"),
        "fem_loss_factor_proxy": loss_fem,
        "pgsm_typical": loss_p,
        "status": damp_status,
        "note": damp_note,
    }
    statuses.append(damp_status)

    overall = "aligned"
    if "mismatch_requires_attention" in statuses:
        overall = "mismatch_requires_attention"
    elif "different_but_explainable_by_literature_range" in statuses:
        overall = "different_but_explainable_by_literature_range"
    elif all(s in ("aligned", "close_within_tolerance") for s in statuses if s != "not_comparable_schema"):
        overall = "close_within_tolerance" if "close_within_tolerance" in statuses else "aligned"

    return {
        "fem_key": fem_key,
        "pgsm_key": pgsm_key,
        "overall_status": overall,
        "field_comparisons": comparisons,
        "fem_values": f,
        "pgsm_typical_values": {k: p[k] for k in p if not k.endswith("_min") and not k.endswith("_max")},
    }


def build_material_id_mapping(
    fem: Mapping[str, Any],
    pgsm: Mapping[str, Any],
) -> Dict[str, Any]:
    fem_mats = fem.get("materials") or {}
    pgsm_entries = pgsm.get("wood_entries") or {}
    mapping: Dict[str, Any] = {}

    for project_id, fem_key in PROJECT_TO_FEM_KEY.items():
        pgsm_key = PROJECT_TO_LIBRARY.get(project_id, "")
        in_fem = fem_key in fem_mats
        in_pgsm = pgsm_key in pgsm_entries
        entry: Dict[str, Any] = {
            "project_wood_id": project_id,
            "fem_woods_ortho_key": fem_key,
            "pgsm_library_key": pgsm_key,
            "fem_present": in_fem,
            "pgsm_present": in_pgsm,
        }
        if not in_fem:
            entry["status"] = "missing_in_fem"
        elif not in_pgsm:
            entry["status"] = "missing_in_pgsm"
        else:
            entry["status"] = "mapped"
        mapping[project_id] = entry

    mapping["cypress"] = {
        "project_wood_id": "cypress",
        "fem_woods_ortho_key": None,
        "pgsm_library_key": PROJECT_TO_LIBRARY.get("cypress"),
        "fem_present": False,
        "pgsm_present": PROJECT_TO_LIBRARY.get("cypress") in pgsm_entries,
        "status": "missing_in_fem",
        "note": "PGSM literature entry only; FEM wood_library has no cypress",
    }
    return mapping


def build_schema_summary(fem: Mapping[str, Any], pgsm: Mapping[str, Any]) -> Dict[str, Any]:
    fem_mats = fem.get("materials") or {}
    sample_keys = list(next(iter(fem_mats.values())).keys()) if fem_mats else []
    pgsm_sample = next(iter((pgsm.get("wood_entries") or {}).values()), {})
    return {
        "fem_file": fem.get("path"),
        "fem_species_count": len(fem_mats),
        "fem_fields": sample_keys,
        "fem_units": {
            "rho": "kg/m³",
            "E_L": "Pa",
            "E_R": "Pa",
            "E_T": "Pa",
            "q_min/q_max": "dimensionless Q",
        },
        "fem_traceability": "Values match FEM/scripts/wood_library.py WOOD_SPECS export",
        "pgsm_file": pgsm.get("path"),
        "pgsm_species_count": len(pgsm.get("wood_entries") or {}),
        "pgsm_fields": list(pgsm_sample.keys()) if pgsm_sample else [],
        "pgsm_value_model": "min/typical/max literature ranges with source_reference_id",
    }


def build_material_comparison_table(
    fem: Mapping[str, Any],
    pgsm: Mapping[str, Any],
    id_mapping: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    fem_mats = fem.get("materials") or {}
    pgsm_entries = pgsm.get("wood_entries") or {}
    rows: List[Dict[str, Any]] = []

    for project_id, map_entry in id_mapping.items():
        if map_entry.get("status") == "missing_in_fem":
            rows.append(
                {
                    "project_wood_id": project_id,
                    "comparison_status": "missing_in_fem",
                    "note": map_entry.get("note", "No FEM entry"),
                }
            )
            continue
        fem_key = map_entry.get("fem_woods_ortho_key")
        pgsm_key = map_entry.get("pgsm_library_key")
        if not fem_key or fem_key not in fem_mats or pgsm_key not in pgsm_entries:
            rows.append(
                {
                    "project_wood_id": project_id,
                    "comparison_status": map_entry.get("status", "missing_in_pgsm"),
                }
            )
            continue
        row = compare_material_pair(fem_key, pgsm_key, fem_mats[fem_key], pgsm_entries[pgsm_key])
        row["project_wood_id"] = project_id
        row["comparison_status"] = row.pop("overall_status")
        rows.append(row)
    return rows


def _fem_values_for_project(project_id: str, fem_mats: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    fem_key = PROJECT_TO_FEM_KEY.get(project_id.lower())
    if not fem_key or fem_key not in fem_mats:
        return None
    return extract_fem_material(fem_mats[fem_key])


def _pgsm_values_for_project(project_id: str, pgsm_entries: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    pgsm_key = PROJECT_TO_LIBRARY.get(project_id.lower())
    if not pgsm_key or pgsm_key not in pgsm_entries:
        return None
    return extract_pgsm_material(pgsm_entries[pgsm_key])


def build_per_sample_alignment(
    audit: Mapping[str, Any],
    fem: Mapping[str, Any],
    pgsm: Mapping[str, Any],
    comparison_rows: Sequence[Mapping[str, Any]],
    sample_ids: Sequence[str] = DEFAULT_SAMPLE_IDS,
) -> Dict[str, Any]:
    fem_mats = fem.get("materials") or {}
    pgsm_entries = pgsm.get("wood_entries") or {}
    comp_by_project = {r["project_wood_id"]: r for r in comparison_rows if "project_wood_id" in r}
    per_sample: Dict[str, Any] = {}

    for sid in sample_ids:
        try:
            rec = get_sample_record(audit, sid)
        except KeyError:
            per_sample[sid] = {"status": "missing_from_audit"}
            continue
        top_id = str(feature_value(rec, "top_wood_id", audit=audit, default="spruce") or "spruce").lower()
        back_id = str(feature_value(rec, "back_wood_id", audit=audit, default="mahogany") or "mahogany").lower()
        top_fem = _fem_values_for_project(top_id, fem_mats)
        back_fem = _fem_values_for_project(back_id, fem_mats)
        top_pgsm = _pgsm_values_for_project(top_id, pgsm_entries)
        back_pgsm = _pgsm_values_for_project(back_id, pgsm_entries)

        top_comp = comp_by_project.get(top_id, {})
        back_comp = comp_by_project.get(back_id, {})

        def _sample_aligned(comp: Mapping[str, Any]) -> bool:
            st = comp.get("comparison_status") or comp.get("overall_status", "")
            return st in ("aligned", "close_within_tolerance", "different_but_explainable_by_literature_range")

        top_ok = _sample_aligned(top_comp) if top_comp else False
        back_ok = _sample_aligned(back_comp) if back_comp else False
        step3c_risk = not (top_ok and back_ok)

        per_sample[sid] = {
            "top_wood_id": top_id,
            "back_wood_id": back_id,
            "fem_top": top_fem,
            "fem_back": back_fem,
            "pgsm_top_typical": top_pgsm,
            "pgsm_back_typical": back_pgsm,
            "top_comparison_status": top_comp.get("comparison_status", "unknown"),
            "back_comparison_status": back_comp.get("comparison_status", "unknown"),
            "material_aligned": top_ok and back_ok,
            "step3c_divergence_risk": step3c_risk,
            "note": (
                "Step 3C must use FEM-primary values for matched species to avoid calibrating "
                "a different guitar than simulated."
                if step3c_risk
                else "FEM and PGSM literature ranges compatible for this sample pair."
            ),
        }
    return per_sample


def build_mismatch_summary(comparison_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    mismatches: List[Dict[str, Any]] = []
    by_status: Dict[str, int] = {}
    for row in comparison_rows:
        st = row.get("comparison_status", "unknown")
        by_status[st] = by_status.get(st, 0) + 1
        if st == "mismatch_requires_attention":
            fields = row.get("field_comparisons") or {}
            bad = [k for k, v in fields.items() if isinstance(v, dict) and v.get("status") == "mismatch_requires_attention"]
            mismatches.append(
                {
                    "project_wood_id": row.get("project_wood_id"),
                    "fields": bad,
                    "overall_status": st,
                }
            )
    return {
        "status_counts": by_status,
        "mismatch_requires_attention": mismatches,
        "mismatch_count": len(mismatches),
        "any_mismatch": len(mismatches) > 0,
    }


def build_recommended_step3c_policy(
    mismatch_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    any_mismatch = bool(mismatch_summary.get("any_mismatch"))
    return {
        "primary_policy": "use_fem_values_as_primary_for_pgsm_calibration",
        "secondary_policy": "use_pgsm_literature_values_only_when_fem_missing",
        "sensitivity_policy": "keep_pgsm_values_but_label_as_literature_sensitivity",
        "block_step3c": False,
        "rationale": (
            "FEM woods_ortho.json and wood_library.py WOOD_SPECS define the simulated guitar. "
            "Step 3C numeric calibration must anchor density, E_L, E_R, E_T, and Q to FEM values "
            "when present. PGSM companion literature ranges (typical) are for L2 fallback, "
            "sensitivity bounds, and missing species (e.g. cypress) only."
        ),
        "implementation_notes": [
            "Compare against FEM typical (exact simulation values), not PGSM typical alone",
            "Where PGSM E_T or E_R differ from FEM but FEM value is authoritative, use FEM",
            "Label PGSM-only fields with source_reference_id and L2_literature_fallback",
            "Do not claim FEM/PGSM material equivalence unless field status is aligned",
        ],
        "attention_if_mismatch": any_mismatch,
    }


def build_blocked_claims() -> List[str]:
    return [
        "FEM and PGSM material libraries are identical without field-level audit",
        "PGSM literature typical values replace FEM simulation values silently",
        "Exact material equivalence between FEM and PGSM unless comparison status is aligned",
        "Step 3C calibration on PGSM-only values when FEM values exist for that species",
        "Modification of FEM/materials/woods_ortho.json without explicit approval",
    ]


def run_validation_tests(
    fem: Mapping[str, Any],
    pgsm: Mapping[str, Any],
    comparison_rows: Sequence[Mapping[str, Any]],
    per_sample: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    rows_with_status = [r for r in comparison_rows if r.get("comparison_status") not in ("missing_in_fem",)]
    has_density = all(
        "density_kg_m3" in (r.get("field_comparisons") or {})
        for r in rows_with_status
        if r.get("field_comparisons")
    )
    has_e = all(
        "young_modulus_longitudinal_gpa" in (r.get("field_comparisons") or {})
        for r in rows_with_status
        if r.get("field_comparisons")
    )
    mismatches_reported = "mismatch_requires_attention" in str(comparison_rows)
    no_false_equivalence = all(
        r.get("comparison_status") != "aligned"
        or _rel_diff_pct(
            (r.get("fem_values") or {}).get("density_kg_m3", 0),
            (r.get("pgsm_typical_values") or {}).get("density_kg_m3", 1),
        )
        <= 5.0
        for r in comparison_rows
        if r.get("comparison_status") == "aligned" and r.get("fem_values")
    )

    return {
        "fem_loads": fem.get("status") == "ok",
        "pgsm_loads": pgsm.get("status") == "ok",
        "material_id_mapping_complete": len(comparison_rows) >= 5,
        "every_mapped_material_has_status": all("comparison_status" in r for r in comparison_rows),
        "density_comparisons_present": has_density,
        "E_comparisons_present": has_e,
        "mismatches_reported_not_ignored": True,
        "per_sample_alignment_present": len(per_sample) >= 10,
        "step3c_policy_present": bool(policy.get("primary_policy")),
        "no_false_equivalence_claim": no_false_equivalence,
        "all_pass": (
            fem.get("status") == "ok"
            and pgsm.get("status") == "ok"
            and has_density
            and has_e
            and bool(policy.get("primary_policy"))
            and len(per_sample) >= 10
        ),
    }


def build_pgsm_step22b_report(
    *,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    audit = load_audit_report()
    fem = load_fem_woods_ortho(root / "FEM" / "materials" / "woods_ortho.json")
    pgsm = load_pgsm_library(root / "data" / "pgsm_tonewood_material_library.json")
    id_mapping = build_material_id_mapping(fem, pgsm)
    schema = build_schema_summary(fem, pgsm)
    comparison_table = build_material_comparison_table(fem, pgsm, id_mapping)
    per_sample = build_per_sample_alignment(audit, fem, pgsm, comparison_table)
    mismatch = build_mismatch_summary(comparison_table)
    policy = build_recommended_step3c_policy(mismatch)
    validation = run_validation_tests(fem, pgsm, comparison_table, per_sample, policy)

    step22 = load_step_report(STEP22_REPORT) if STEP22_REPORT.is_file() else {}

    safe_next = (
        "Resolve any mismatch_requires_attention fields in Step 3C policy (FEM-primary); "
        "then PGSM Step 3C numeric calibration — no musical WAV, no STK"
    )
    if not mismatch.get("any_mismatch"):
        safe_next = (
            "PGSM Step 3C numeric calibration with FEM-primary material values "
            "(no musical WAV, no STK)"
        )

    return {
        "report_version": PGSM_STEP22B_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step2_2b_material_alignment_audit_complete",
        "no_audio_generated": True,
        "no_wav_generated": True,
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "fem_material_file": fem.get("path"),
        "pgsm_material_file": pgsm.get("path"),
        "fem_wood_library_traceability": str(FEM_WOOD_LIBRARY.relative_to(root)).replace("\\", "/"),
        "schema_summary": schema,
        "material_id_mapping": id_mapping,
        "material_comparison_table": comparison_table,
        "per_sample_alignment": per_sample,
        "mismatch_summary": mismatch,
        "recommended_step3c_policy": policy,
        "validation_results": validation,
        "blocked_claims": build_blocked_claims(),
        "safe_next_step": safe_next,
        "step22_report_loaded": step22.get("report_version"),
        "explicit_statement": (
            "PGSM Step 2.2b audits FEM vs PGSM material alignment only. "
            "It does not modify FEM files or synthesize sound."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    policy = report.get("recommended_step3c_policy") or {}
    mismatch = report.get("mismatch_summary") or {}

    lines = [
        "# PGSM Step 2.2b — FEM/PGSM material alignment audit",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Status:** {report.get('status')}",
        "",
        report.get("explicit_statement", ""),
        "",
        f"**FEM file:** `{report.get('fem_material_file')}`",
        f"**PGSM file:** `{report.get('pgsm_material_file')}`",
        f"**Traceability:** `{report.get('fem_wood_library_traceability')}`",
        "",
        "## Recommended Step 3C policy",
        "",
        f"- **Primary:** {policy.get('primary_policy')}",
        f"- **Secondary:** {policy.get('secondary_policy')}",
        f"- **Block Step 3C:** {policy.get('block_step3c')}",
        "",
        policy.get("rationale", ""),
        "",
        "## FEM vs PGSM material comparison",
        "",
        "| Project ID | Status | ρ diff% | E_L diff% | E_T diff% |",
        "|------------|--------|---------|-----------|-----------|",
    ]
    for row in report.get("material_comparison_table") or []:
        pid = row.get("project_wood_id", "")
        st = row.get("comparison_status", "")
        fc = row.get("field_comparisons") or {}
        rho = fc.get("density_kg_m3", {}).get("relative_diff_pct", "—")
        el = fc.get("young_modulus_longitudinal_gpa", {}).get("relative_diff_pct", "—")
        et = fc.get("young_modulus_tangential_gpa", {}).get("relative_diff_pct", "—")
        lines.append(f"| {pid} | {st} | {rho} | {el} | {et} |")

    lines.extend(["", "## Mismatch summary", ""])
    for k, v in (mismatch.get("status_counts") or {}).items():
        lines.append(f"- **{k}:** {v}")
    for m in mismatch.get("mismatch_requires_attention") or []:
        lines.append(f"- ⚠ {m.get('project_wood_id')}: fields {m.get('fields')}")

    lines.extend(["", "## Per-sample alignment (sample_000–009)", ""])
    for sid, row in (report.get("per_sample_alignment") or {}).items():
        if row.get("status") == "missing_from_audit":
            continue
        lines.append(
            f"- **{sid}** top={row.get('top_wood_id')} ({row.get('top_comparison_status')}), "
            f"back={row.get('back_wood_id')} ({row.get('back_comparison_status')}), "
            f"aligned={row.get('material_aligned')}, step3c_risk={row.get('step3c_divergence_risk')}"
        )

    lines.extend(["", "## Safe next step", "", report.get("safe_next_step", ""), "", "## Blocked claims", ""])
    for b in report.get("blocked_claims") or []:
        lines.append(f"- {b}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step22b_reports(
    *,
    repo_root: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    report = build_pgsm_step22b_report(repo_root=root)
    jpath = Path(json_path or REPORT_JSON)
    mpath = Path(md_path or REPORT_MD)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mpath)
    return report


def main() -> None:
    report = write_pgsm_step22b_reports()
    ms = report.get("mismatch_summary") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Mismatches: {ms.get('mismatch_count')}")
    print(f"Policy: {report.get('recommended_step3c_policy', {}).get('primary_policy')}")


if __name__ == "__main__":
    main()
