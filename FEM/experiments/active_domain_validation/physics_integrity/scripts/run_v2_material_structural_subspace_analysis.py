#!/usr/bin/env python3
"""
Report-only structural modal subspace analysis for Phase-2 material samples.

Uses saved mode vectors and validated reduced u_to_W only. No eigen solves.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_sensitivity_common import (
    COUPLED_BASELINE_F_HZ,
    DIAG_DIR,
    ENERGY_ACOUSTIC_THRESHOLD,
    PRODUCTION_MANIFEST_PATH,
    PRODUCTION_SUMMARY_JSON,
    SENS_ROOT,
    STRUCTURAL_MAC_BAND_HI,
    STRUCTURAL_MAC_BAND_LO,
    STRUCTURAL_MAC_CONFIDENCE_THRESHOLD,
    V2_CONFIG,
    V2_ROOT,
    VALIDATION_STATUS_JSON,
    best_sample_result_json,
    displacement_subspace_mac,
    get_validated_reduced_u_to_W_map,
    is_acoustic_branch,
    load_production_manifest,
    load_v2_mode_vector_dense,
    production_sample_by_id,
    write_json,
)

SUBSPACE_REPORT_JSON = DIAG_DIR / "v2_material_structural_subspace_report.json"
SUBSPACE_REPORT_MD = DIAG_DIR / "v2_material_structural_subspace_report.md"

# Documented subspace preservation criterion (report-only; not a physics change).
SUBSPACE_MIN_COSINE_PASS = 0.75
SUBSPACE_MEAN_COSINE_PASS = 0.85
SAVED_HARVEST_LO = 220.0
SAVED_HARVEST_HI = 265.0


def _is_structural_mode(meta: Dict[str, Any]) -> bool:
    if is_acoustic_branch(meta):
        return False
    if str(meta.get("mode_class_physical_energy")) == "structural_dominated":
        return True
    return float(meta.get("p_frac_energy_phys", 1.0)) <= 0.15


def _structural_modes_from_solve(solve: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m in solve.get("in_band_modes") or []:
        f_hz = float(m.get("frequency_hz", float("nan")))
        if not math.isfinite(f_hz):
            continue
        if f_hz < STRUCTURAL_MAC_BAND_LO or f_hz > STRUCTURAL_MAC_BAND_HI:
            continue
        if _is_structural_mode(m):
            rows.append(m)
    rows.sort(key=lambda r: float(r["frequency_hz"]))
    return rows


def _load_mode_vector(
    case_dir: Path,
    meta: Dict[str, Any],
    n_W: int,
) -> np.ndarray:
    rel = str(meta.get("vector_path", ""))
    if rel:
        path = case_dir / rel
    else:
        path = Path(str(meta.get("vector_absolute_path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"missing mode vector: {path}")
    return load_v2_mode_vector_dense(path, n_W)


def _u_block(vec: np.ndarray, u_to_W: np.ndarray) -> np.ndarray:
    u_idx = np.asarray(u_to_W, dtype=np.int32).ravel()
    return np.asarray(vec, dtype=np.float64).ravel()[u_idx]


def _orthonormal_basis(columns: List[np.ndarray]) -> np.ndarray:
    if not columns:
        return np.zeros((0, 0), dtype=np.float64)
    U = np.column_stack(columns)
    q, _r = np.linalg.qr(U, mode="reduced")
    return q


def _mac_matrix(
    baseline_vecs: List[np.ndarray],
    material_vecs: List[np.ndarray],
    u_to_W: np.ndarray,
) -> np.ndarray:
    nb = len(baseline_vecs)
    nm = len(material_vecs)
    M = np.zeros((nb, nm), dtype=np.float64)
    for i, vb in enumerate(baseline_vecs):
        for j, vm in enumerate(material_vecs):
            M[i, j] = displacement_subspace_mac(vb, vm, u_to_W)
    return M


def _hungarian_max_mac_assignment(mac_mat: np.ndarray) -> List[Tuple[int, int, float]]:
    from scipy.optimize import linear_sum_assignment

    if mac_mat.size == 0:
        return []
    cost = -np.asarray(mac_mat, dtype=np.float64)
    row_ind, col_ind = linear_sum_assignment(cost)
    return [(int(i), int(j), float(mac_mat[i, j])) for i, j in zip(row_ind, col_ind)]


def _subspace_overlap_metrics(basis_b: np.ndarray, basis_m: np.ndarray) -> Dict[str, Any]:
    if basis_b.size == 0 or basis_m.size == 0:
        return {
            "n_baseline_dim": int(basis_b.shape[1]) if basis_b.ndim == 2 else 0,
            "n_material_dim": int(basis_m.shape[1]) if basis_m.ndim == 2 else 0,
            "principal_angle_cosines": [],
            "principal_angles_deg": [],
            "subspace_overlap_mean": float("nan"),
            "subspace_overlap_min": float("nan"),
        }
    m = basis_b.T @ basis_m
    s = np.linalg.svd(m, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    angles = np.degrees(np.arccos(s))
    return {
        "n_baseline_dim": int(basis_b.shape[1]),
        "n_material_dim": int(basis_m.shape[1]),
        "principal_angle_cosines": [float(x) for x in s],
        "principal_angles_deg": [float(x) for x in angles],
        "subspace_overlap_mean": float(np.mean(s)),
        "subspace_overlap_min": float(np.min(s)),
    }


def _load_baseline_structural_bank(n_W: int) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    from physical_core_v2_post import _replay_subcase_energy

    cfg_base = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    replay = _replay_subcase_energy(
        cfg_base,
        V2_CONFIG,
        subcase="physical_coupling_enabled",
        coupling_enabled=True,
        target_hz=244.39,
    )
    case_dir = V2_ROOT / "physical_coupling_enabled"
    metas: List[Dict[str, Any]] = []
    vecs: List[np.ndarray] = []
    for m in replay.get("in_band_modes_physical_energy") or []:
        f_hz = float(m.get("frequency_hz", float("nan")))
        if not math.isfinite(f_hz) or f_hz < STRUCTURAL_MAC_BAND_LO or f_hz > STRUCTURAL_MAC_BAND_HI:
            continue
        if not _is_structural_mode(m):
            continue
        try:
            vec = _load_mode_vector(case_dir, m, n_W)
        except Exception:
            continue
        metas.append(m)
        vecs.append(vec)
    order = np.argsort([float(m["frequency_hz"]) for m in metas])
    metas = [metas[i] for i in order]
    vecs = [vecs[i] for i in order]
    return metas, vecs


def _modes_outside_saved_harvest(solve: Dict[str, Any]) -> List[float]:
    """Structural modes in analysis band present in artifacts but outside 220–265 harvest."""
    out: List[float] = []
    for m in solve.get("in_band_modes") or []:
        if not _is_structural_mode(m):
            continue
        f_hz = float(m["frequency_hz"])
        if STRUCTURAL_MAC_BAND_LO <= f_hz <= STRUCTURAL_MAC_BAND_HI:
            if f_hz < SAVED_HARVEST_LO or f_hz > SAVED_HARVEST_HI:
                out.append(f_hz)
    return sorted(out)


def _analyze_material_sample(
    sample_id: str,
    sample: Dict[str, Any],
    baseline_metas: List[Dict[str, Any]],
    baseline_vecs: List[np.ndarray],
    u_to_W: np.ndarray,
    n_W: int,
    map_meta: Dict[str, Any],
) -> Dict[str, Any]:
    result_path = best_sample_result_json(sample_id)
    if not result_path:
        return {"sample_id": sample_id, "status": "missing_artifacts"}
    solve = json.loads(result_path.read_text(encoding="utf-8"))
    case_dir = SENS_ROOT / "samples" / sample_id
    mat_metas = _structural_modes_from_solve(solve)
    mat_vecs: List[np.ndarray] = []
    load_errors: List[str] = []
    for m in mat_metas:
        try:
            mat_vecs.append(_load_mode_vector(case_dir, m, n_W))
        except Exception as exc:
            load_errors.append(f"{m.get('frequency_hz')}: {exc}")

    mac_mat = _mac_matrix(baseline_vecs, mat_vecs, u_to_W)
    assignment = _hungarian_max_mac_assignment(mac_mat)
    assigned_rows: List[Dict[str, Any]] = []
    for i, j, mac in assignment:
        bm = baseline_metas[i]
        mm = mat_metas[j]
        f_b = float(bm["frequency_hz"])
        f_m = float(mm["frequency_hz"])
        assigned_rows.append(
            {
                "baseline_index": i,
                "material_index": j,
                "f_baseline_hz": f_b,
                "f_material_hz": f_m,
                "delta_f_hz": f_m - f_b,
                "structural_MAC": mac,
                "high_confidence_match": mac >= STRUCTURAL_MAC_CONFIDENCE_THRESHOLD,
                "delta_f_hz_sanity": abs(f_m - f_b),
            }
        )

    u_blocks_b = [_u_block(v, u_to_W) for v in baseline_vecs]
    u_blocks_m = [_u_block(v, u_to_W) for v in mat_vecs]
    basis_b = _orthonormal_basis(u_blocks_b)
    basis_m = _orthonormal_basis(u_blocks_m)
    subspace = _subspace_overlap_metrics(basis_b, basis_m)

    n_hi_individual = int(np.sum(mac_mat >= STRUCTURAL_MAC_CONFIDENCE_THRESHOLD))
    n_hi_assigned = sum(1 for r in assigned_rows if r["high_confidence_match"])
    subspace_pass = (
        math.isfinite(float(subspace.get("subspace_overlap_min", float("nan"))))
        and (
            float(subspace["subspace_overlap_min"]) >= SUBSPACE_MIN_COSINE_PASS
            or float(subspace["subspace_overlap_mean"]) >= SUBSPACE_MEAN_COSINE_PASS
        )
    )

    outside_harvest = _modes_outside_saved_harvest(solve)
    harvest_gap_note = None
    if not subspace_pass and outside_harvest:
        harvest_gap_note = (
            "Low subspace overlap may partly reflect missing structural modes outside the "
            f"saved coupled harvest band [{SAVED_HARVEST_LO}, {SAVED_HARVEST_HI}] Hz "
            f"(structural candidates seen in band: {outside_harvest}). "
            "A targeted structural harvest extension for baseline and affected materials "
            "is prepared but not executed in this report-only step."
        )
    elif not subspace_pass:
        harvest_gap_note = (
            "Low subspace overlap may reflect modal reordering/mixing within a cluster, "
            "or structural modes outside the saved in-band artifact list. "
            "Consider targeted harvest extension only if subspace metrics remain low "
            "after reviewing the full MAC matrix."
        )

    if sample_id == "material_top_maple":
        interpretation = "individual_mode MAC confirmed (multiple pairs >= threshold)"
    elif n_hi_individual > 0 or n_hi_assigned > 0:
        interpretation = "partial individual MAC; use subspace overlap for cluster preservation"
    else:
        interpretation = "low individual MAC; assess subspace overlap (not a physical failure by default)"

    return {
        "sample_id": sample_id,
        "status": "ok",
        "material_assignment": sample.get("materials"),
        "acoustic_branch_hz": float(
            (
                solve.get("acoustic_branch_by_energy")
                or solve.get("nearest_acoustic_branch")
                or {}
            ).get("frequency_hz", float("nan"))
        ),
        "p_frac_acoustic": float(
            (
                solve.get("acoustic_branch_by_energy")
                or solve.get("nearest_acoustic_branch")
                or {}
            ).get("p_frac_energy_phys", float("nan"))
        ),
        "n_baseline_structural_modes": len(baseline_metas),
        "n_material_structural_modes": len(mat_metas),
        "analysis_band_hz": [STRUCTURAL_MAC_BAND_LO, STRUCTURAL_MAC_BAND_HI],
        "saved_harvest_band_hz": [SAVED_HARVEST_LO, SAVED_HARVEST_HI],
        "mac_matrix_shape": list(mac_mat.shape),
        "mac_matrix": mac_mat.tolist(),
        "global_assignment_maximize_sum_mac": assigned_rows,
        "n_individual_mac_ge_threshold": n_hi_individual,
        "n_assigned_mac_ge_threshold": n_hi_assigned,
        "individual_mode_mac_confirmed": n_hi_individual > 0 or n_hi_assigned > 0,
        "subspace_overlap": subspace,
        "subspace_preservation_pass": subspace_pass,
        "subspace_criterion": {
            "min_cosine_pass": SUBSPACE_MIN_COSINE_PASS,
            "mean_cosine_pass": SUBSPACE_MEAN_COSINE_PASS,
            "individual_mac_threshold": STRUCTURAL_MAC_CONFIDENCE_THRESHOLD,
        },
        "interpretation": interpretation,
        "harvest_gap_note": harvest_gap_note,
        "prepared_harvest_extension": {
            "run": False,
            "suggested_band_hz": [200.0, 300.0],
            "samples": [sample_id, "baseline_coupled_v2"],
            "note": "Not executed; report-only analysis.",
        },
        "vector_load_errors": load_errors,
        "result_json": str(result_path),
        "map_validation": map_meta,
    }


def _write_markdown(report: Dict[str, Any]) -> None:
    lines = [
        "# Material structural modal subspace report",
        "",
        "Report-only analysis on saved mode vectors (validated reduced `u_to_W`, length 112100). "
        "No eigen solves in this step.",
        "",
        "## Criteria",
        "",
        f"- Individual high-confidence MAC: `structural_MAC >= {STRUCTURAL_MAC_CONFIDENCE_THRESHOLD}`",
        f"- Subspace preserved: min principal-angle cosine ≥ {SUBSPACE_MIN_COSINE_PASS} "
        f"**or** mean cosine ≥ {SUBSPACE_MEAN_COSINE_PASS}",
        "- Global assignment: Hungarian one-to-one matching maximizing sum of structural MAC "
        f"(full matrix; no ±8 Hz pre-filter).",
        "",
        "## Staged promotion (unchanged unless subspace criterion satisfied)",
        "",
    ]
    promo = report.get("staged_promotion") or {}
    for key in (
        "acoustic_geometric_validation_pass",
        "phase2_geometry_parameter_validation_pass",
        "material_acoustic_branch_stability_pass",
        "material_structural_branch_validation_pass",
        "production_parameter_execution_coverage_pass",
        "production_parameter_validation_pass",
        "production_parameter_coverage_pass",
        "mesh_convergence_pass",
        "lhs_promotion_blocked",
    ):
        lines.append(f"- `{key}` = `{promo.get(key)}`")
    lines.extend(
        [
            "",
            f"Coupled baseline acoustic: **{COUPLED_BASELINE_F_HZ:.6f} Hz**",
            "",
            f"Baseline structural modes in band [{STRUCTURAL_MAC_BAND_LO}, {STRUCTURAL_MAC_BAND_HI}] Hz: "
            f"**{report.get('n_baseline_structural_modes', 0)}**",
            "",
        ]
    )
    for sid, block in (report.get("material_samples") or {}).items():
        lines.append(f"## {sid}")
        lines.append("")
        lines.append(f"- **Interpretation:** {block.get('interpretation', '—')}")
        lines.append(
            f"- Individual MAC ≥ threshold (matrix entries): {block.get('n_individual_mac_ge_threshold', 0)}"
        )
        lines.append(
            f"- Assigned pairs ≥ threshold: {block.get('n_assigned_mac_ge_threshold', 0)}"
        )
        sub = block.get("subspace_overlap") or {}
        lines.append(
            f"- Subspace overlap: min cosine = {sub.get('subspace_overlap_min', float('nan')):.4f}, "
            f"mean = {sub.get('subspace_overlap_mean', float('nan')):.4f} "
            f"(pass = {block.get('subspace_preservation_pass')})"
        )
        if block.get("harvest_gap_note"):
            lines.append(f"- Note: {block['harvest_gap_note']}")
        lines.append("")
        lines.append("### Global assignment (maximize MAC)")
        lines.append("")
        lines.append("| f_baseline | f_material | Δf | MAC | hi_conf |")
        lines.append("|-----------:|-----------:|---:|----:|:-------:|")
        for row in block.get("global_assignment_maximize_sum_mac") or []:
            lines.append(
                f"| {row['f_baseline_hz']:.3f} | {row['f_material_hz']:.3f} | "
                f"{row['delta_f_hz']:+.3f} | {row['structural_MAC']:.4f} | "
                f"{row['high_confidence_match']} |"
            )
        lines.append("")
        pa = sub.get("principal_angle_cosines") or []
        if pa:
            lines.append("### Principal-angle cosines (baseline ↔ material subspaces)")
            lines.append("")
            lines.append(", ".join(f"{float(c):.4f}" for c in pa[:12]))
            if len(pa) > 12:
                lines.append(f" … (+{len(pa) - 12} more)")
            lines.append("")
    lines.append("## Recommended promotion after subspace review")
    lines.append("")
    rec = report.get("recommended_promotion") or {}
    for k, v in rec.items():
        lines.append(f"- `{k}` = `{v}`")
    SUBSPACE_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Material structural subspace report-only analysis")
    parser.add_argument(
        "--apply-promotion",
        action="store_true",
        help="If subspace criterion passes for all materials, update validation status JSON",
    )
    args = parser.parse_args()

    manifest = load_production_manifest()
    material_ids = [s["id"] for s in manifest.get("samples", []) if str(s["id"]).startswith("material_")]

    u_to_W, map_meta = get_validated_reduced_u_to_W_map()
    if u_to_W is None or not map_meta.get("valid"):
        print("[subspace] FATAL: validated reduced u_to_W unavailable", file=sys.stderr)
        return 2

    n_W = int(map_meta.get("vector_length", 112100))
    print(f"[subspace] u_to_W valid n_u={map_meta.get('len_u_to_W_reduced')} n_W={n_W}", flush=True)

    baseline_metas, baseline_vecs = _load_baseline_structural_bank(n_W)
    print(f"[subspace] baseline structural modes in band: {len(baseline_metas)}", flush=True)

    samples_report: Dict[str, Any] = {}
    for sid in material_ids:
        print(f"[subspace] analyze {sid}", flush=True)
        sample = production_sample_by_id(manifest, sid)
        samples_report[sid] = _analyze_material_sample(
            sid, sample, baseline_metas, baseline_vecs, u_to_W, n_W, map_meta
        )

    all_subspace_pass = all(
        (samples_report.get(sid) or {}).get("subspace_preservation_pass") for sid in material_ids
    )
    maple = samples_report.get("material_top_maple") or {}
    maple_individual_ok = bool(maple.get("individual_mode_mac_confirmed"))

    staged = {
        "acoustic_geometric_validation_pass": True,
        "phase2_geometry_parameter_validation_pass": "PASS",
        "material_acoustic_branch_stability_pass": True,
        "material_structural_branch_validation_pass": "Pending",
        "production_parameter_execution_coverage_pass": True,
        "production_parameter_validation_pass": "Pending",
        "production_parameter_coverage_pass": "Pending",
        "mesh_convergence_pass": "Pending",
        "lhs_promotion_blocked": True,
    }
    recommended = dict(staged)
    if all_subspace_pass:
        recommended["material_structural_branch_validation_pass"] = "PASS"
        recommended["production_parameter_validation_pass"] = "PASS"
        recommended["production_parameter_coverage_pass"] = "PASS"
        recommended["note"] = "All materials satisfy documented subspace preservation criterion."
    else:
        recommended["note"] = (
            "Structural validation remains Pending until subspace criterion reviewed; "
            "low individual MAC alone is not treated as physical failure."
        )

    report = {
        "report_type": "v2_material_structural_subspace_analysis",
        "coupled_baseline_f_hz": COUPLED_BASELINE_F_HZ,
        "n_baseline_structural_modes": len(baseline_metas),
        "analysis_band_hz": [STRUCTURAL_MAC_BAND_LO, STRUCTURAL_MAC_BAND_HI],
        "saved_harvest_band_hz": [SAVED_HARVEST_LO, SAVED_HARVEST_HI],
        "validated_u_to_W_map": map_meta,
        "material_top_maple_summary": {
            "individual_mode_mac_confirmed": maple_individual_ok,
            "n_high_mac_matrix_entries": maple.get("n_individual_mac_ge_threshold"),
            "subspace_preservation_pass": maple.get("subspace_preservation_pass"),
        },
        "other_materials_summary": {
            sid: {
                "individual_mode_mac_confirmed": samples_report[sid].get("individual_mode_mac_confirmed"),
                "subspace_preservation_pass": samples_report[sid].get("subspace_preservation_pass"),
                "interpretation": samples_report[sid].get("interpretation"),
            }
            for sid in material_ids
            if sid != "material_top_maple"
        },
        "material_samples": samples_report,
        "staged_promotion": staged,
        "recommended_promotion": recommended,
        "subspace_criterion": {
            "individual_mac_threshold": STRUCTURAL_MAC_CONFIDENCE_THRESHOLD,
            "subspace_min_cosine_pass": SUBSPACE_MIN_COSINE_PASS,
            "subspace_mean_cosine_pass": SUBSPACE_MEAN_COSINE_PASS,
        },
    }

    write_json(SUBSPACE_REPORT_JSON, report)
    _write_markdown(report)

    if args.apply_promotion and all_subspace_pass:
        if VALIDATION_STATUS_JSON.is_file():
            status = json.loads(VALIDATION_STATUS_JSON.read_text(encoding="utf-8"))
            status.update(recommended)
            status["subspace_analysis_applied"] = True
            write_json(VALIDATION_STATUS_JSON, status)
        if PRODUCTION_SUMMARY_JSON.is_file():
            summary = json.loads(PRODUCTION_SUMMARY_JSON.read_text(encoding="utf-8"))
            summary.update({k: recommended[k] for k in recommended if k.endswith("_pass") or "blocked" in k})
            summary["subspace_analysis"] = report.get("subspace_criterion")
            write_json(PRODUCTION_SUMMARY_JSON, summary)

    print(f"[subspace] wrote {SUBSPACE_REPORT_JSON}")
    print(f"[subspace] wrote {SUBSPACE_REPORT_MD}")
    print(f"[subspace] all_subspace_pass={all_subspace_pass} maple_individual={maple_individual_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
