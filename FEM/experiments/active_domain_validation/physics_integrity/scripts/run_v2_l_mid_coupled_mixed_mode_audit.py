#!/usr/bin/env python3
"""
No-eigensolve L_mid coupled consistency + mixed-mode continuation audit.

Uses existing L_mid meshes, locator JSON, saved coupled mode vectors, and optional seed files.
No new coupled (or acoustic) eigen solves.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mpi4py import MPI

SCRIPT_DIR = Path(__file__).resolve().parent
FEM_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
V2_CONFIG = PHYSICS_ROOT / "configs" / "coupled_physical_core_v2.json"

from mode_diagnostics import pressure_subspace_mac
from physical_fsi_seed_residual_audit import (
    _block_residual_contributions,
    _rayleigh_metrics,
)
from v2_mesh_convergence_common import (
    CONV_DIAG,
    load_manifest,
    mesh_path,
    sample_spec_from_case,
    solve_case_dir,
    write_json,
)

REPORT_JSON = CONV_DIAG / "v2_l_mid_coupled_mixed_mode_audit.json"
REPORT_MD = CONV_DIAG / "v2_l_mid_coupled_mixed_mode_audit.md"
INCREMENTAL_JSON = CONV_DIAG / "v2_l_mid_coupled_mixed_mode_audit.incremental.json"
OVERLAP_NUM_TOL = 1.0e-12
STATUS_JSON = (
    PHYSICS_ROOT / "v2_sensitivity_validation" / "diagnostics" / "v2_validation_status.json"
)

ACOUSTIC_CASES = ("baseline_coupled_v2", "hole_radius_large")
CLUSTER_BAND_HALF_WIDTH_HZ = 40.0
MAC_SINGLE_MIXED = 0.85
MAC_CLUSTER = 0.70
SUBSPACE_OVERLAP_PASS = 0.75


def _write_md(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clamp_unit_interval(value: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    x = float(value)
    if x < 0.0 and x > -OVERLAP_NUM_TOL:
        x = 0.0
    if x > 1.0 and x < 1.0 + OVERLAP_NUM_TOL:
        x = 1.0
    return float(min(1.0, max(0.0, x)))


def _seed_truth_fields(seed_info: Dict[str, Any], seed_source: str) -> Dict[str, Any]:
    is_acoustic_seed = seed_source == "acoustic_coupled_seed.npy" and bool(
        seed_info.get("seed_build_success")
    )
    return {
        "seed_build_success": bool(seed_info.get("seed_build_success")),
        "seed_vector_length": seed_info.get("seed_vector_length"),
        "eps_seed_applied": bool(seed_info.get("eps_seed_applied")),
        "seed_failure_reason": seed_info.get("seed_failure_reason"),
        "locator_pressure_reference_source": seed_source,
        "locator_pressure_reference_is_acoustic_seed": is_acoustic_seed,
        "proxy_reference_diagnostic_only": not is_acoustic_seed,
        "proxy_may_not_justify_mesh_convergence_resume": not is_acoustic_seed,
    }


def _base_report(*, audit_in_progress: bool) -> Dict[str, Any]:
    return {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "audit_in_progress": audit_in_progress,
        "interpretation_note": (
            "Mesh refinement may yield mixed air-structure modes; continuation is judged by "
            "pressure MAC and pressure_reference_to_cluster_projection_overlap to the locator "
            "pressure reference, not by p_frac>=0.85 alone. Proxy pressure references are "
            "diagnostic only and do not justify mesh_convergence_may_resume."
        ),
        "cases": {},
        "mesh_convergence_may_resume": False,
        "staged_status": {
            "mesh_convergence_pass": "Pending",
            "v2_production_promotion_ready": False,
            "lhs_promotion_blocked": True,
        },
        "continuation_criterion_met": False,
    }


def _checkpoint_report(report: Dict[str, Any], cases_out: Dict[str, Any], *, stage: str) -> None:
    report["cases"] = dict(cases_out)
    report["last_checkpoint_stage"] = stage
    report["last_checkpoint_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(INCREMENTAL_JSON, report)
    write_json(REPORT_JSON, report)
    _write_md(REPORT_MD, _report_to_md_lines(report))


def _case_from_manifest(manifest: Dict[str, Any], case_id: str) -> Dict[str, Any]:
    for c in manifest.get("cases") or []:
        if str(c.get("id")) == case_id:
            return c
    raise KeyError(case_id)


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _verify_seed_path(case_dir: Path) -> Dict[str, Any]:
    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"
    seed_meta = case_dir / "diagnostics" / "acoustic_coupled_seed_meta.json"
    build_log = case_dir / "logs" / "build_acoustic_seed.log"
    rescue_logs = list((case_dir / "logs").glob("mesh_convergence*.log"))

    out: Dict[str, Any] = {
        "seed_npy_path": str(seed_npy),
        "seed_meta_path": str(seed_meta),
        "seed_build_success": False,
        "seed_vector_length": None,
        "eps_seed_applied": False,
        "seed_failure_reason": None,
        "seed_build_log_error": None,
    }

    bl = _read_text(build_log)
    if "ValueError: not enough values to unpack" in bl or "expected 6, got 4" in bl:
        out["seed_build_log_error"] = "v2_build_coupled_acoustic_seed API mismatch (solve_evp=False returns 4 values)"
        out["seed_failure_reason"] = out["seed_build_log_error"]

    if seed_npy.is_file():
        try:
            arr = np.load(str(seed_npy))
            out["seed_vector_length"] = int(arr.size)
            out["seed_build_success"] = int(arr.size) > 0 and math.isfinite(float(np.linalg.norm(arr)))
            if not out["seed_build_success"]:
                out["seed_failure_reason"] = "seed file empty or zero norm"
        except Exception as exc:
            out["seed_failure_reason"] = f"failed to load seed npy: {exc}"
    elif not out["seed_failure_reason"]:
        out["seed_failure_reason"] = "acoustic_coupled_seed.npy missing"

    if seed_meta.is_file():
        try:
            meta = json.loads(seed_meta.read_text(encoding="utf-8"))
            out["seed_meta"] = meta
            if int(meta.get("n_reduced_W", 0)) > 0 and out["seed_vector_length"]:
                out["seed_build_success"] = out["seed_build_success"] and (
                    int(meta["n_reduced_W"]) == int(out["seed_vector_length"])
                )
        except Exception as exc:
            out["seed_meta_read_error"] = str(exc)

    for lp in rescue_logs:
        txt = _read_text(lp)
        if "--eps-seed-npy" in txt or "eps-seed-npy" in txt:
            out["eps_seed_applied"] = True
            out["eps_seed_log"] = str(lp)
            break
    if out["eps_seed_applied"] and not out["seed_build_success"]:
        out["seed_application_not_verified"] = True
        out["seed_failure_reason"] = (
            (out.get("seed_failure_reason") or "")
            + "; labeled seeded retry but seed vector not valid"
        ).strip("; ")

    return out


def _assemble_l_mid_v2(
    mesh_file: Path,
    sample: Dict[str, Any],
    *,
    coupling_enabled: bool,
    sorting_tag: str,
) -> Tuple[Any, Any, np.ndarray, np.ndarray, Dict[str, Any]]:
    import copy

    import fem_main_3d as fem3d

    cfg = copy.deepcopy(json.loads(V2_CONFIG.read_text(encoding="utf-8")))
    sc = cfg.setdefault("solver", {})
    sc["mesh_file"] = str(mesh_file.resolve())
    sc["coupled_physical_core_v2_diagnosis"] = True
    sc["coupled_physical_core_v2_coupling_enabled"] = bool(coupling_enabled)
    sc["fsi_coupling_gain"] = 1.0
    sc["fsi_nitsche_enable"] = False
    sc["physics_integrity_capture"] = True
    sc["coupled_air_pressure_restriction_diagnosis"] = True
    sc["coupled_air_pressure_restriction_replay_audit"] = True
    sc["gnhep_block_frobenius_normalize"] = True
    from v2_sensitivity_mesh import sample_geometry

    cfg["geometry"] = sample_geometry(sample)

    sort_dir = solve_case_dir("L_mid", str(sample["id"])) / "sorting_audit" / sorting_tag
    sort_dir.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sort_dir.resolve())

    _msh, _W, A, M = fem3d._solve_coupled_evp(
        mesh_file=mesh_file.resolve(),
        config=cfg,
        num_modes=0,
        solve_evp=False,
    )
    u_to_W = np.asarray(cfg["_coupled_air_u_to_W_map"], dtype=np.int32).ravel()
    p_to_W = np.asarray(cfg["_coupled_air_p_to_W_map"], dtype=np.int32).ravel()
    restr = dict(cfg.get("_coupled_air_pressure_restriction") or {})
    return A, M, u_to_W, p_to_W, restr


def _load_seed_vector(
    case_dir: Path,
    p_to_W: np.ndarray,
    *,
    n_reduced_W: int,
    locator_hz: float,
    saved_modes: List[Dict[str, Any]],
) -> Tuple[np.ndarray, str]:
    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"
    n_W = int(n_reduced_W)
    if seed_npy.is_file():
        seed = np.asarray(np.load(str(seed_npy)), dtype=np.float64).ravel()
        if seed.size == n_W:
            return seed, "acoustic_coupled_seed.npy"
    # Proxy: pressure block of saved coupled mode nearest locator frequency (no acoustic vector stored)
    if saved_modes:
        nearest = min(saved_modes, key=lambda m: abs(float(m["frequency_hz"]) - locator_hz))
        vec = _load_mode_vec(case_dir, nearest)
        seed = np.zeros(n_W, dtype=np.float64)
        seed[p_to_W] = np.asarray(vec[p_to_W], dtype=np.float64).ravel()
        nrm = float(np.linalg.norm(seed))
        if nrm > 0:
            seed /= nrm
        return seed, (
            f"proxy_pressure_from_nearest_saved_coupled_mode_f={nearest['frequency_hz']:.4f} "
            "(acoustic eigenvector not archived; not a true locator mode)"
        )
    seed = np.zeros(n_W, dtype=np.float64)
    if p_to_W.size:
        seed[p_to_W] = 1.0 / math.sqrt(float(p_to_W.size))
    return seed, "unit_pressure_template_on_p_to_W"


def _load_mode_vec(case_dir: Path, meta: Dict[str, Any]) -> np.ndarray:
    from fem_mode_array_utils import load_mode_column_any

    rel = str(meta.get("vector_path", ""))
    path = case_dir / rel if rel else Path(str(meta.get("vector_absolute_path", "")))
    col = load_mode_column_any(path)
    return np.asarray(col.toarray(), dtype=np.float64).ravel()


def _load_saved_modes(case_dir: Path) -> List[Dict[str, Any]]:
    summary_path = case_dir / "diagnostics" / "mode_energy_summary.json"
    if not summary_path.is_file():
        return []
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for m in data.get("modes") or []:
        rel = str(m.get("vector_path", ""))
        if not rel:
            continue
        path = case_dir / rel
        if not path.is_file():
            continue
        rows.append({**m, "vector_path": rel})
    return rows


def _rayleigh_audit(
    mesh_file: Path,
    sample: Dict[str, Any],
    seed: np.ndarray,
    locator_hz: float,
) -> Dict[str, Any]:
    lam0 = (2.0 * math.pi * float(locator_hz)) ** 2
    out: Dict[str, Any] = {}
    for label, coupling in (
        ("coupling_disabled", False),
        ("physical_coupling_enabled", True),
    ):
        A, M, u_to_W, p_to_W, restr = _assemble_l_mid_v2(
            mesh_file, sample, coupling_enabled=coupling, sorting_tag=f"rayleigh_{label}"
        )
        residual = _block_residual_contributions(
            A, M, seed, lam0=lam0, u_idx=u_to_W, p_idx=p_to_W
        )
        rayleigh = _rayleigh_metrics(A, M, seed, seed_f_hz=locator_hz)
        out[label] = {
            "n_reduced_W": int(A.getSize()[0]),
            "n_u_active": int(u_to_W.size),
            "n_p_active": int(p_to_W.size),
            "relative_residual": float(residual["relative_residual"]),
            "rayleigh_f_hz": float(rayleigh["rayleigh_f_hz"]),
            "delta_rayleigh_from_locator_hz": float(rayleigh["delta_rayleigh_from_seed_hz"]),
            "block_residual_contributions": residual.get("block_residual_contributions"),
        }
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass
    dis = out.get("coupling_disabled") or {}
    en = out.get("physical_coupling_enabled") or {}
    out["relative_residual_disabled"] = dis.get("relative_residual")
    out["relative_residual_enabled"] = en.get("relative_residual")
    out["rayleigh_f_disabled_hz"] = dis.get("rayleigh_f_hz")
    out["rayleigh_f_enabled_hz"] = en.get("rayleigh_f_hz")
    out["delta_rayleigh_from_locator_hz"] = en.get("delta_rayleigh_from_locator_hz")
    return out


def _pressure_reference_to_cluster_projection_overlap(
    p_to_W: np.ndarray,
    ref_p: np.ndarray,
    mode_vecs: List[np.ndarray],
) -> Dict[str, Any]:
    """
    One normalized locator pressure reference q_reference vs orthonormal cluster Q_cluster
    on active pressure DOFs: projection_overlap = || Q_cluster^H q_reference ||.
    """
    p_idx = np.asarray(p_to_W, dtype=np.int32).ravel()
    q_reference = np.asarray(ref_p[p_idx], dtype=np.complex128).ravel()
    ref_n = float(np.linalg.norm(q_reference))
    n_in = len(mode_vecs)
    if ref_n <= OVERLAP_NUM_TOL or n_in == 0:
        return {
            "pressure_reference_to_cluster_projection_overlap": float("nan"),
            "n_cluster_vectors": n_in,
            "n_pressure_dofs": int(p_idx.size),
            "metric_type": "single_reference_to_orthonormal_pressure_cluster",
            "note": "||Q_cluster^H q_reference||; not a multi-reference principal-angle SVD.",
        }

    q_reference = q_reference / ref_n
    cols: List[np.ndarray] = []
    for v in mode_vecs:
        p = np.asarray(v[p_idx], dtype=np.complex128).ravel()
        pn = float(np.linalg.norm(p))
        cols.append(p / pn if pn > OVERLAP_NUM_TOL else p)
    P = np.column_stack(cols)
    if P.ndim != 2 or P.shape[1] == 0:
        return {
            "pressure_reference_to_cluster_projection_overlap": float("nan"),
            "n_cluster_vectors": 0,
            "n_pressure_dofs": int(p_idx.size),
            "metric_type": "single_reference_to_orthonormal_pressure_cluster",
            "note": "empty candidate pressure cluster",
        }

    Q_cluster, _ = np.linalg.qr(P, mode="reduced")
    coeff = Q_cluster.conj().T @ q_reference
    coeff = np.atleast_1d(np.asarray(coeff, dtype=np.complex128).ravel())
    projection_overlap = _clamp_unit_interval(float(np.linalg.norm(coeff)))

    return {
        "pressure_reference_to_cluster_projection_overlap": projection_overlap,
        "n_cluster_vectors": int(Q_cluster.shape[1]),
        "n_pressure_dofs": int(p_idx.size),
        "metric_type": "single_reference_to_orthonormal_pressure_cluster",
        "note": (
            "projection_overlap = ||Q_cluster^H q_reference|| with complex-conjugate "
            "transpose; single-reference metric (not multi-angle subspace SVD)."
        ),
    }


def _mixed_mode_analysis(
    case_dir: Path,
    p_to_W: np.ndarray,
    locator_hz: float,
    seed: np.ndarray,
    seed_source: str,
) -> Dict[str, Any]:
    modes = _load_saved_modes(case_dir)
    lo = float(locator_hz) - CLUSTER_BAND_HALF_WIDTH_HZ
    hi = float(locator_hz) + CLUSTER_BAND_HALF_WIDTH_HZ
    ref_p = seed

    candidates: List[Dict[str, Any]] = []
    for m in modes:
        f_hz = float(m.get("frequency_hz", float("nan")))
        if not math.isfinite(f_hz) or f_hz < lo or f_hz > hi:
            continue
        try:
            vec = _load_mode_vec(case_dir, m)
        except Exception as exc:
            continue
        p_mac = pressure_subspace_mac(ref_p, vec, p_to_W)
        candidates.append(
            {
                "frequency_hz": f_hz,
                "p_frac_energy_phys": float(m.get("p_frac_energy_phys", float("nan"))),
                "E_air": m.get("E_air_phys"),
                "E_struct": m.get("E_struct_phys"),
                "mode_class_physical_energy": m.get("mode_class_physical_energy"),
                "pressure_MAC_to_locator_mode": float(p_mac),
                "mode_index": m.get("mode_index"),
                "vector_path": str(m.get("vector_path", "")),
            }
        )

    candidates.sort(key=lambda r: float(r["frequency_hz"]))
    if not candidates:
        return {"candidates": [], "cluster_metrics": {}, "verdict": "L_MID_NO_SAVED_MODES_IN_BAND"}

    macs = [float(c["pressure_MAC_to_locator_mode"]) for c in candidates]
    best = max(candidates, key=lambda c: float(c["pressure_MAC_to_locator_mode"]))
    top3 = sorted(candidates, key=lambda c: -float(c["pressure_MAC_to_locator_mode"]))[:3]
    top3_vecs = [_load_mode_vec(case_dir, c) for c in top3]

    overlap = _pressure_reference_to_cluster_projection_overlap(p_to_W, ref_p, top3_vecs)
    p_fracs = [float(c["p_frac_energy_phys"]) for c in candidates if math.isfinite(float(c["p_frac_energy_phys"]))]

    cluster_metrics = {
        "n_candidates_in_band": len(candidates),
        "band_hz": [lo, hi],
        "max_pressure_MAC": float(max(macs)),
        "best_matching_mode": best,
        "top3_by_MAC": top3,
        "mean_p_frac_in_cluster": float(np.mean(p_fracs)) if p_fracs else float("nan"),
        "max_p_frac_in_cluster": float(max(p_fracs)) if p_fracs else float("nan"),
        "locator_pressure_reference_source": seed_source,
        **overlap,
    }
    return {
        "candidates": candidates,
        "cluster_metrics": cluster_metrics,
        "verdict": None,
    }


def _assign_verdict(
    seed_info: Dict[str, Any],
    rayleigh: Dict[str, Any],
    mixed: Dict[str, Any],
) -> str:
    cm = mixed.get("cluster_metrics") or {}
    max_mac = float(cm.get("max_pressure_MAC", float("nan")))
    proj_overlap = float(
        cm.get("pressure_reference_to_cluster_projection_overlap", float("nan"))
    )
    best = cm.get("best_matching_mode") or {}
    best_p = float(best.get("p_frac_energy_phys", float("nan")))

    rel_dis = float((rayleigh.get("coupling_disabled") or {}).get("relative_residual", float("nan")))
    rel_en = float((rayleigh.get("physical_coupling_enabled") or {}).get("relative_residual", float("nan")))
    d_ray_en = float((rayleigh.get("physical_coupling_enabled") or {}).get("delta_rayleigh_from_locator_hz", float("nan")))

    if seed_info.get("eps_seed_applied") and not seed_info.get("seed_build_success"):
        return "L_MID_SEED_APPLICATION_NOT_VERIFIED"

    if not math.isfinite(max_mac) or cm.get("n_candidates_in_band", 0) == 0:
        return "L_MID_ENERGY_CLASSIFICATION_OR_LAYOUT_SUSPECT"

    if math.isfinite(rel_en) and rel_en > 0.35 and max_mac < 0.45:
        return "L_MID_PHYSICAL_COUPLING_PERTURBS_ACOUSTIC_SEED_STRONGLY"

    if math.isfinite(max_mac) and max_mac >= MAC_SINGLE_MIXED:
        return "L_MID_ACOUSTIC_BRANCH_CONTINUES_AS_SINGLE_MIXED_MODE"

    if math.isfinite(max_mac) and max_mac >= MAC_CLUSTER and (
        (math.isfinite(proj_overlap) and proj_overlap >= SUBSPACE_OVERLAP_PASS)
        or max_mac >= 0.80
    ):
        return "L_MID_ACOUSTIC_BRANCH_CONTINUES_AS_MIXED_CLUSTER"

    if seed_info.get("seed_build_success") and math.isfinite(rel_en) and rel_en < 0.15 and max_mac < 0.55:
        return "L_MID_SEED_REMAINS_GOOD_BUT_RETRIEVAL_FAILED"

    if math.isfinite(rel_dis) and rel_dis < 0.12 and math.isfinite(rel_en) and rel_en > 0.2:
        return "L_MID_PHYSICAL_COUPLING_PERTURBS_ACOUSTIC_SEED_STRONGLY"

    return "L_MID_SEED_REMAINS_GOOD_BUT_RETRIEVAL_FAILED"


def _process_case(
    manifest: Dict[str, Any],
    case_id: str,
    *,
    report: Dict[str, Any],
    cases_out: Dict[str, Any],
) -> Dict[str, Any]:
    case = _case_from_manifest(manifest, case_id)
    case_dir = solve_case_dir("L_mid", case_id)
    mesh_file = mesh_path("L_mid", case_id)
    if not mesh_file.is_file():
        return {"sample_id": case_id, "error": f"missing mesh {mesh_file}"}

    locator_path = case_dir / "diagnostics" / "acoustic_locator_l_mid.json"
    locator = (
        json.loads(locator_path.read_text(encoding="utf-8"))
        if locator_path.is_file()
        else {}
    )
    loc_hz = float(locator.get("locator_frequency_hz", float("nan")))
    l0_ref = float(case.get("reference_f_hz", loc_hz))

    seed_info = _verify_seed_path(case_dir)

    row: Dict[str, Any] = {
        "sample_id": case_id,
        "mesh_file": str(mesh_file),
        "L0_reference_frequency_hz": l0_ref,
        "locator_search_band_hz": locator.get("locator_band_hz"),
        "locator_status": locator.get("locator_status"),
        "locator_acoustic_frequency_hz": loc_hz,
        "locator_shift_from_L0_hz": (
            loc_hz - l0_ref if math.isfinite(loc_hz) else float("nan")
        ),
        "seed_verification": seed_info,
    }
    cases_out[case_id] = row
    _checkpoint_report(report, cases_out, stage=f"{case_id}:locator_and_seed")

    if not math.isfinite(loc_hz):
        row["diagnostic_verdict"] = "L_MID_ENERGY_CLASSIFICATION_OR_LAYOUT_SUSPECT"
        cases_out[case_id] = row
        _checkpoint_report(report, cases_out, stage=f"{case_id}:verdict_no_locator")
        return row

    sample = sample_spec_from_case(case)
    A0, M0, u0, p0, _restr0 = _assemble_l_mid_v2(
        mesh_file, sample, coupling_enabled=True, sorting_tag="mixed_mode"
    )
    n_W = int(A0.getSize()[0])
    try:
        A0.destroy()
        M0.destroy()
    except Exception:
        pass

    saved_modes = _load_saved_modes(case_dir)
    seed, seed_source = _load_seed_vector(
        case_dir,
        p0,
        n_reduced_W=n_W,
        locator_hz=loc_hz,
        saved_modes=saved_modes,
    )
    row.update(_seed_truth_fields(seed_info, seed_source))
    cases_out[case_id] = row
    _checkpoint_report(report, cases_out, stage=f"{case_id}:pressure_reference")

    row["rayleigh_residual_audit"] = _rayleigh_audit(mesh_file, sample, seed, loc_hz)
    cases_out[case_id] = row
    _checkpoint_report(report, cases_out, stage=f"{case_id}:rayleigh_residual")

    mixed = _mixed_mode_analysis(case_dir, p0, loc_hz, seed, seed_source)
    row["mixed_mode_continuation"] = mixed
    row["diagnostic_verdict"] = _assign_verdict(seed_info, row["rayleigh_residual_audit"], mixed)
    cases_out[case_id] = row
    _checkpoint_report(report, cases_out, stage=f"{case_id}:mixed_mode_complete")
    return row


def _report_to_md_lines(report: Dict[str, Any]) -> List[str]:
    lines = [
        "# L_mid coupled mixed-mode continuation audit (no eigensolve)",
        "",
        f"Generated: {report.get('generated_utc')}",
        f"Audit in progress: {report.get('audit_in_progress')}",
        f"Last checkpoint: {report.get('last_checkpoint_stage')} @ {report.get('last_checkpoint_utc')}",
        "",
        str(report.get("interpretation_note", "")),
        "",
        "**mesh_convergence_may_resume:** `False` (until valid acoustic-reference verdict review)",
        "",
    ]
    for cid in ACOUSTIC_CASES:
        row = (report.get("cases") or {}).get(cid)
        if not row:
            lines.append(f"## {cid}")
            lines.append("")
            lines.append("_pending_")
            lines.append("")
            continue
        lines.append(f"## {cid}")
        lines.append("")
        d_l0 = row.get("locator_shift_from_L0_hz")
        d_l0_s = f"{float(d_l0):+.4f}" if d_l0 is not None and math.isfinite(float(d_l0)) else "n/a"
        lines.append(
            f"- Locator f: **{row.get('locator_acoustic_frequency_hz')}** Hz (Δ L0: {d_l0_s})"
        )
        lines.append(
            f"- Seed: build_success={row.get('seed_build_success')} "
            f"length={row.get('seed_vector_length')} "
            f"eps_seed_applied={row.get('eps_seed_applied')} "
            f"reason={row.get('seed_failure_reason')}"
        )
        lines.append(f"- Pressure reference: `{row.get('locator_pressure_reference_source')}`")
        if row.get("proxy_reference_diagnostic_only"):
            lines.append("- **Proxy reference — diagnostic only; does not justify mesh resume**")
        verdict = row.get("diagnostic_verdict")
        if verdict:
            lines.append(f"- **Verdict:** `{verdict}`")
        lines.append("")
        rra = row.get("rayleigh_residual_audit") or {}
        if rra:
            for k in ("coupling_disabled", "physical_coupling_enabled"):
                blk = rra.get(k) or {}
                lines.append(
                    f"### Rayleigh / residual ({k}): "
                    f"rel_res={blk.get('relative_residual')} "
                    f"f_ray={blk.get('rayleigh_f_hz')} "
                    f"Δf_loc={blk.get('delta_rayleigh_from_locator_hz')}"
                )
            lines.append(
                f"- Flattened: rel_dis={rra.get('relative_residual_disabled')} "
                f"rel_en={rra.get('relative_residual_enabled')} "
                f"f_dis={rra.get('rayleigh_f_disabled_hz')} f_en={rra.get('rayleigh_f_enabled_hz')}"
            )
            lines.append("")
        mm = row.get("mixed_mode_continuation") or {}
        cm = mm.get("cluster_metrics") or {}
        if cm:
            lines.append(
                f"### Cluster: n={cm.get('n_candidates_in_band')} "
                f"max_MAC={cm.get('max_pressure_MAC')} "
                f"projection_overlap={cm.get('pressure_reference_to_cluster_projection_overlap')}"
            )
            lines.append("")
            lines.append("| f (Hz) | p_frac | MAC to ref | class |")
            lines.append("|--------|--------|------------|-------|")
            for c in mm.get("candidates") or []:
                pfrac = float(c.get("p_frac_energy_phys", float("nan")))
                lines.append(
                    f"| {float(c['frequency_hz']):.3f} | {pfrac:.4f} | "
                    f"{float(c['pressure_MAC_to_locator_mode']):.4f} | "
                    f"{c.get('mode_class_physical_energy')} |"
                )
            lines.append("")
    return lines


def main() -> int:
    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[l_mid_audit] Requires mpiexec -n 1 for operator assembly", file=sys.stderr)
        return 2

    manifest = load_manifest()
    CONV_DIAG.mkdir(parents=True, exist_ok=True)

    report = _base_report(audit_in_progress=True)
    cases_out: Dict[str, Any] = {}
    _checkpoint_report(report, cases_out, stage="audit_start")

    for cid in ACOUSTIC_CASES:
        print(f"[l_mid_audit] {cid}", flush=True)
        try:
            _process_case(manifest, cid, report=report, cases_out=cases_out)
        except Exception as exc:
            row = cases_out.get(cid) or {"sample_id": cid}
            row["audit_error"] = repr(exc)
            row["diagnostic_verdict"] = None
            cases_out[cid] = row
            _checkpoint_report(report, cases_out, stage=f"{cid}:error")
            raise

    report["audit_in_progress"] = False
    report["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    verdicts = [str((cases_out.get(c) or {}).get("diagnostic_verdict", "")) for c in ACOUSTIC_CASES]
    positive = {
        "L_MID_ACOUSTIC_BRANCH_CONTINUES_AS_SINGLE_MIXED_MODE",
        "L_MID_ACOUSTIC_BRANCH_CONTINUES_AS_MIXED_CLUSTER",
    }
    acoustic_refs_ok = all(
        bool((cases_out.get(c) or {}).get("locator_pressure_reference_is_acoustic_seed"))
        for c in ACOUSTIC_CASES
    )
    continuation_ok = acoustic_refs_ok and all(v in positive for v in verdicts)
    report["continuation_criterion_met"] = continuation_ok
    report["continuation_requires_acoustic_seed_reference"] = True
    _checkpoint_report(report, cases_out, stage="audit_complete")

    if STATUS_JSON.is_file():
        st = json.loads(STATUS_JSON.read_text(encoding="utf-8"))
        st["l_mid_coupled_mixed_mode_audit"] = {
            "report_json": str(REPORT_JSON),
            "incremental_json": str(INCREMENTAL_JSON),
            "continuation_criterion_met": continuation_ok,
            "acoustic_reference_required_for_resume": True,
            "mesh_convergence_may_resume": False,
        }
        write_json(STATUS_JSON, st)

    print(f"[l_mid_audit] wrote {REPORT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
