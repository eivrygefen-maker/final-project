#!/usr/bin/env python3
"""
Pre-resume repair: finite-area L_prod gate revalidation + L_mid acoustic branch rescue.

No L_prod/L_check eigensolves; no v2 physics changes.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_v2_mesh_production_preflight import (
    PREFLIGHT_MESH,
    _case_from_manifest,
    _run_gates,
)
from v2_mesh_convergence_common import (
    CONV_DIAG,
    CONV_MESH,
    CONV_SOLVES,
    _acoustic_branch_ok,
    load_manifest,
    mesh_path,
    run_mpi_case_solve,
    sample_spec_from_case,
    solve_case_dir,
    solve_result_path,
    write_json,
)

REPORT_MD = CONV_DIAG / "v2_mesh_production_preflight_report.md"
REPORT_JSON = CONV_DIAG / "v2_mesh_preresume_repair_report.json"
L_PROD_READY_JSON = CONV_DIAG / "l_prod_repair_ready.json"
AREA_REL_TOL = 0.15


def _aperture_area_ratio(aperture: Dict[str, Any]) -> float:
    for key in ("area_ratio", "area_ratio_vs_pi_r2"):
        if key in aperture:
            return float(aperture[key])
    return float("nan")


def _aperture_pass_fail_closed(aperture: Dict[str, Any]) -> Tuple[bool, str]:
    if not bool(aperture.get("gate_pass")):
        failed = [
            k
            for k, v in (aperture.get("acceptance_checks") or {}).items()
            if not v
        ]
        return False, f"aperture gate_pass=False checks={failed}"
    ratio = _aperture_area_ratio(aperture)
    if not math.isfinite(ratio):
        return False, "area_ratio is NaN or missing (fail-closed)"
    lo = 1.0 - AREA_REL_TOL
    hi = 1.0 + AREA_REL_TOL
    if not (lo <= ratio <= hi):
        return False, f"area_ratio={ratio:.4f} outside [{lo:.2f},{hi:.2f}]"
    return True, f"area_ratio={ratio:.4f} finite and within tolerance"


def revalidate_l_prod_gates(manifest: Dict[str, Any], *, rebuild: bool) -> Dict[str, Any]:
    from run_v2_mesh_production_preflight import _build_l_prod_mesh

    out: Dict[str, Any] = {}
    for case_id in ("baseline_coupled_v2", "hole_radius_large"):
        case = _case_from_manifest(manifest, case_id)
        mesh_file = PREFLIGHT_MESH / f"{case_id}.msh"
        if rebuild or not mesh_file.is_file():
            print(f"[repair] build L_prod mesh {case_id}", flush=True)
            mesh_file = _build_l_prod_mesh(case, manifest)
        elif not mesh_file.is_file():
            raise FileNotFoundError(f"missing preflight mesh: {mesh_file}")
        print(f"[repair] gates-only {case_id}", flush=True)
        row = _run_gates(mesh_file, case, PREFLIGHT_MESH / case_id)
        aperture = row.get("aperture_report") or {}
        area_ok, area_note = _aperture_pass_fail_closed(aperture)
        ratio = _aperture_area_ratio(aperture)
        row["area_ratio_finite"] = math.isfinite(ratio)
        row["area_ratio"] = ratio
        row["area_ratio_pass"] = area_ok
        row["area_ratio_note"] = area_note
        row["all_preflight_pass"] = bool(
            row.get("combined_mesh_gate_pass")
            and row.get("air_adjacency_ok")
            and area_ok
        )
        out[case_id] = row
    return out


def install_repaired_l_prod_meshes(manifest: Dict[str, Any], gate_results: Dict[str, Any]) -> Dict[str, Any]:
    """Copy validated preflight L_prod meshes into convergence tree; drop stale L_prod solves."""
    installed: List[str] = []
    CONV_MESH.joinpath("L_prod").mkdir(parents=True, exist_ok=True)
    for case_id in ("baseline_coupled_v2", "hole_radius_large"):
        if not (gate_results.get(case_id) or {}).get("all_preflight_pass"):
            continue
        src = PREFLIGHT_MESH / f"{case_id}.msh"
        dst = mesh_path("L_prod", case_id)
        if dst.is_file():
            backup = dst.with_suffix(".msh.bad_topology_backup")
            if not backup.is_file():
                shutil.copy2(dst, backup)
        shutil.copy2(src, dst)
        installed.append(case_id)
        solve_dir = solve_case_dir("L_prod", case_id)
        for rp in (solve_dir / "results").glob("result_*.json"):
            rp.unlink()
    ready = {
        "installed_cases": installed,
        "mesh_dir": str(CONV_MESH / "L_prod"),
        "note": "Stale L_prod solve JSON removed; resume will run L_prod eigensolves on repaired meshes.",
    }
    write_json(L_PROD_READY_JSON, ready)
    return ready


def _load_result(level_id: str, case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    p = solve_result_path(level_id, str(case["id"]), float(case["target_hz"]))
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _load_mode_summary(level_id: str, case_id: str) -> List[Dict[str, Any]]:
    p = solve_case_dir(level_id, case_id) / "diagnostics" / "mode_energy_summary.json"
    if not p.is_file():
        return []
    try:
        return list(json.loads(p.read_text(encoding="utf-8")).get("modes") or [])
    except Exception:
        return []


def diagnose_l_mid_acoustic(case: Dict[str, Any]) -> Dict[str, Any]:
    cid = str(case["id"])
    level = "L_mid"
    res = _load_result(level, case)
    log_path = solve_case_dir(level, cid) / "logs" / "mesh_convergence_solve.log"
    out: Dict[str, Any] = {
        "case_id": cid,
        "level_id": level,
        "log_path": str(log_path),
    }
    if not res:
        out["category"] = "missing_result_artifact"
        out["detail"] = "No result JSON on disk."
        return out

    in_band = list(res.get("in_band_modes") or [])
    nconv = int(res.get("nconv_marked", (res.get("eps_batch_diagnostics") or {}).get("nconv_marked", -1)))
    out["v2_converged"] = bool(res.get("v2_converged"))
    out["nconv_marked"] = nconv
    out["num_modes_saved"] = res.get("num_modes_saved")
    out["harvest_band_hz"] = res.get("harvest_band_hz")
    out["acoustic_branch_selection"] = res.get("acoustic_branch_selection")
    out["branch_captured"] = _acoustic_branch_ok(res)
    branch = res.get("acoustic_branch_by_energy") or res.get("nearest_acoustic_branch")
    out["acoustic_f_hz"] = (branch or {}).get("frequency_hz")
    out["p_frac_energy_phys"] = (branch or {}).get("p_frac_energy_phys")

    all_modes = _load_mode_summary(level, cid)
    band = res.get("harvest_band_hz") or [
        float(case["harvest_lo_hz"]),
        float(case["harvest_hi_hz"]),
    ]
    lo, hi = float(band[0]), float(band[1])
    in_band_n = len(in_band)
    acoustic_in_band = [
        m
        for m in in_band
        if str(m.get("mode_class_physical_energy")) == "acoustic_dominated"
        or float(m.get("p_frac_energy_phys", 0.0)) >= 0.85
    ]
    freqs_all = [
        float(m["frequency_hz"])
        for m in all_modes
        if math.isfinite(float(m.get("frequency_hz", float("nan"))))
    ]
    freqs_outside = [f for f in freqs_all if f < lo or f > hi]
    nearest_acoustic_outside = None
    for m in sorted(all_modes, key=lambda r: -float(r.get("p_frac_energy_phys", 0.0))):
        p = float(m.get("p_frac_energy_phys", 0.0))
        if p < 0.35:
            continue
        f_hz = float(m.get("frequency_hz", float("nan")))
        if not math.isfinite(f_hz):
            continue
        if f_hz < lo or f_hz > hi:
            nearest_acoustic_outside = {
                "frequency_hz": f_hz,
                "p_frac_energy_phys": p,
                "mode_class": m.get("mode_class_physical_energy"),
            }
            break

    if nconv <= 0 or not out["v2_converged"]:
        out["category"] = "no_eps_eigenpairs_converged"
        out["detail"] = f"nconv_marked={nconv}; EPS did not mark converged modes."
    elif in_band_n == 0 and freqs_outside:
        out["category"] = "acoustic_branch_outside_harvest_band"
        out["detail"] = (
            f"No modes in harvest [{lo},{hi}] Hz; "
            f"example high-p_frac mode outside band: {nearest_acoustic_outside}"
        )
    elif in_band_n > 0 and not acoustic_in_band:
        out["category"] = "eigenpairs_converged_no_acoustic_dominated_in_band"
        out["detail"] = (
            f"{in_band_n} in-band modes but none acoustic-dominated (p_frac>=0.85)."
        )
    elif in_band_n > 0 and acoustic_in_band and not out["branch_captured"]:
        out["category"] = "post_processing_selection_failure"
        out["detail"] = (
            f"selection={res.get('acoustic_branch_selection')}; "
            "acoustic candidates exist but branch not stored."
        )
    elif not out["branch_captured"]:
        out["category"] = "eigenpairs_converged_no_acoustic_dominated_in_band"
        out["detail"] = "No usable acoustic branch in artifacts."
    else:
        out["category"] = "ok"
        out["detail"] = "Branch already captured."

    if log_path.is_file():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        if "selection=no_acoustic_candidate" in tail or "selection=none" in tail:
            out["log_selection_hint"] = "no_acoustic_candidate_in_band"
    return out


def _rescue_harvest(case: Dict[str, Any], diagnosis: Dict[str, Any]) -> Tuple[float, float, int]:
    lo = float(case["harvest_lo_hz"])
    hi = float(case["harvest_hi_hz"])
    nm = int(case["num_modes"])
    cat = str(diagnosis.get("category", ""))
    if cat == "acoustic_branch_outside_harvest_band":
        margin = 25.0
        lo = max(180.0, lo - margin)
        hi = hi + margin
        nm = max(nm, nm + 4)
    elif cat in (
        "eigenpairs_converged_no_acoustic_dominated_in_band",
        "post_processing_selection_failure",
    ):
        nm = max(nm, nm + 2)
    return lo, hi, nm


def rescue_l_mid_acoustic(
    manifest: Dict[str, Any],
    diagnoses: Dict[str, Dict[str, Any]],
    *,
    force: bool,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for case_id in ("baseline_coupled_v2", "hole_radius_large"):
        case = _case_from_manifest(manifest, case_id)
        diag = diagnoses[case_id]
        if diag.get("category") == "ok" and not force:
            results[case_id] = {"status": "skipped", "reason": "branch already ok"}
            continue

        mesh_file = mesh_path("L_mid", case_id)
        if not mesh_file.is_file():
            results[case_id] = {"status": "failed", "error": f"missing L_mid mesh {mesh_file}"}
            continue

        lo, hi, nm = _rescue_harvest(case, diag)
        case_dir = solve_case_dir("L_mid", case_id)
        rp = solve_result_path("L_mid", case_id, float(case["target_hz"]))
        if rp.is_file():
            backup = rp.with_suffix(".json.pre_rescue_backup")
            if not backup.is_file():
                shutil.copy2(rp, backup)

        print(
            f"[repair] L_mid rescue solve {case_id} band=[{lo},{hi}] nm={nm} "
            f"diagnosis={diag.get('category')}",
            flush=True,
        )
        rc, solve = run_mpi_case_solve(
            sample_spec_from_case(case),
            mesh_file,
            target_hz=float(case["target_hz"]),
            harvest_lo_hz=lo,
            harvest_hi_hz=hi,
            num_modes=nm,
            log_path=case_dir / "logs" / "mesh_convergence_rescue_solve.log",
            case_dir=case_dir,
            select_by_energy=True,
            reference_f_hz=float(case.get("reference_f_hz", case["target_hz"])),
        )
        solve["rescue_diagnosis"] = diag
        solve["rescue_harvest_band_hz"] = [lo, hi]
        write_json(rp, solve)
        ok = rc == 0 and _acoustic_branch_ok(solve)
        branch = solve.get("acoustic_branch_by_energy") or solve.get("nearest_acoustic_branch")
        results[case_id] = {
            "status": "ok" if ok else "solve_failed",
            "solve_exit_code": rc,
            "branch_captured": _acoustic_branch_ok(solve),
            "f_acoustic_hz": (branch or {}).get("frequency_hz"),
            "p_frac_energy_phys": (branch or {}).get("p_frac_energy_phys"),
            "mode_class_physical_energy": (branch or {}).get("mode_class_physical_energy"),
            "rescue_harvest_band_hz": [lo, hi],
            "num_modes": nm,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-resume mesh convergence repair")
    parser.add_argument(
        "--rebuild-l-prod-mesh",
        action="store_true",
        help="Rebuild L_prod preflight meshes before gate revalidation",
    )
    parser.add_argument("--force-l-mid-rescue", action="store_true", help="Rerun L_mid even if branch ok")
    args = parser.parse_args()

    manifest = load_manifest()
    CONV_DIAG.mkdir(parents=True, exist_ok=True)

    gate_results = revalidate_l_prod_gates(manifest, rebuild=bool(args.rebuild_l_prod_mesh))
    l_prod_ok = all((gate_results.get(c) or {}).get("all_preflight_pass") for c in gate_results)

    l_mid_diag = {
        cid: diagnose_l_mid_acoustic(_case_from_manifest(manifest, cid))
        for cid in ("baseline_coupled_v2", "hole_radius_large")
    }
    l_mid_rescue = rescue_l_mid_acoustic(manifest, l_mid_diag, force=bool(args.force_l_mid_rescue))
    l_mid_ok = all((l_mid_rescue.get(c) or {}).get("branch_captured") for c in l_mid_rescue)

    l_prod_install = {}
    if l_prod_ok:
        l_prod_install = install_repaired_l_prod_meshes(manifest, gate_results)

    may_resume = l_prod_ok and l_mid_ok
    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issue_1_area_ratio": {
            "root_cause": "Preflight read aperture['area_ratio'] but audit JSON field is area_ratio_vs_pi_r2",
            "fix": "Fail-closed finite area_ratio; alias added in audit script",
            "area_rel_tol": AREA_REL_TOL,
            "L_prod_gate_revalidation": gate_results,
        },
        "issue_2_l_mid_acoustic": {
            "diagnosis": l_mid_diag,
            "rescue": l_mid_rescue,
            "rescue_strategy": (
                "Reuse L_mid meshes; --select-by-energy; optional widened harvest when "
                "branch outside band; reference_f_hz from manifest."
            ),
        },
        "l_prod_mesh_install": l_prod_install,
        "mesh_convergence_may_resume": may_resume,
        "resume_command": (
            "bash FEM/experiments/active_domain_validation/physics_integrity/scripts/"
            "run_v2_mesh_convergence.sh --resume"
            if may_resume
            else None
        ),
        "do_not_run_yet": ["L_check", "L_prod_eigensolves_until_resume_command"],
    }
    write_json(REPORT_JSON, report)

    lines = [
        "# v2 mesh pre-resume repair report",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        "## Issue 1: L_prod aperture area_ratio (fail-closed)",
        "",
        f"Root cause: {report['issue_1_area_ratio']['root_cause']}",
        "",
    ]
    for cid, row in gate_results.items():
        lines.append(f"### {cid}")
        lines.append(f"- all_preflight_pass: **{row.get('all_preflight_pass')}**")
        lines.append(f"- area_ratio: **{row.get('area_ratio')}** ({row.get('area_ratio_note')})")
        adj = row.get("adjacency_report") or {}
        lines.append(
            f"- adjacency: tag2={adj.get('tag2_total')} air_adj={adj.get('tag2_adjacent_to_air_10')}"
        )
        lines.append("")
    lines.extend(["## Issue 2: L_mid acoustic branch rescue", ""])
    for cid in ("baseline_coupled_v2", "hole_radius_large"):
        d = l_mid_diag[cid]
        r = l_mid_rescue.get(cid) or {}
        lines.append(f"### {cid}")
        lines.append(f"- diagnosis: **{d.get('category')}** — {d.get('detail')}")
        lines.append(f"- rescue status: **{r.get('status')}** branch={r.get('branch_captured')} f={r.get('f_acoustic_hz')}")
        lines.append("")
    lines.extend(
        [
            "## Resume",
            "",
            f"**mesh_convergence_may_resume:** `{may_resume}`",
            "",
        ]
    )
    if may_resume:
        lines.append(f"When ready: `{report['resume_command']}`")
    else:
        lines.append("Do not resume full mesh convergence until both issues pass.")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[repair] wrote {REPORT_JSON}", flush=True)
    print(f"[repair] may_resume={may_resume}", flush=True)
    return 0 if may_resume else 1


if __name__ == "__main__":
    raise SystemExit(main())
