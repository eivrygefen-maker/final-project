#!/usr/bin/env python3
"""
Gates-only L_prod production mesh preflight + diagnosis of existing L_mid solve logs.

Does not run eigen solves, L_check, or resume full mesh convergence.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_mesh_convergence_common import CONV_MESH, CONV_ROOT, CONV_SOLVES, load_manifest, mesh_path, write_json
from v2_mesh_convergence_mesh import build_level_mesh
from v2_sensitivity_gates import run_mesh_gates

PREFLIGHT_ROOT = CONV_ROOT / "preflight"
PREFLIGHT_MESH = PREFLIGHT_ROOT / "L_prod"
PREFLIGHT_DIAG = CONV_ROOT / "diagnostics"
REPORT_MD = PREFLIGHT_DIAG / "v2_mesh_production_preflight_report.md"
REPORT_JSON = PREFLIGHT_DIAG / "v2_mesh_production_preflight_report.json"
CONFIG_DIR = PREFLIGHT_ROOT / "configs"

ORIGINAL_L_PROD_FAILURE = {
    "mesh_profile": "FEM_ALLOW_FOM=1",
    "n_nodes": 369209,
    "n_tetrahedra": 2146052,
    "aperture": {
        "area_ratio": 1.1701,
        "z_span_m": 0.004,
        "horizontal_area_fraction": 0.854539,
        "gate_pass": False,
        "failed_checks": ["area_within_15pct", "horizontal_fraction_ok"],
    },
    "air_adjacency": {
        "tag2_total": 12767,
        "air_adjacent": 0,
        "wood_only": 12767,
        "soundhole_p_in_air_subgraph": 0,
        "hypothesis": "Tag-2 facets on wood rim, not air aperture connected to cavity",
    },
}


def _case_from_manifest(manifest: Dict[str, Any], case_id: str) -> Dict[str, Any]:
    for c in manifest.get("cases") or []:
        if str(c.get("id")) == case_id:
            return c
    raise KeyError(case_id)


def _run_gates(mesh_path: Path, case: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    import subprocess

    py = sys.executable
    hole_r = float((case.get("geometry") or {}).get("hole_radius", 0.047))
    gates_dir = out_dir / "gates"
    gates = run_mesh_gates(mesh_path, hole_radius_m=hole_r, gates_dir=gates_dir)

    aperture_dir = gates_dir / "soundhole_aperture_audit"
    aperture_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            py,
            str(SCRIPT_DIR / "audit_soundhole_aperture_geometry.py"),
            "--mesh",
            str(mesh_path.resolve()),
            "--hole-radius",
            str(hole_r),
            "--out-dir",
            str(aperture_dir),
        ],
        check=False,
    )
    aperture_json = aperture_dir / "soundhole_aperture_geometry_report.json"
    aperture = (
        json.loads(aperture_json.read_text(encoding="utf-8"))
        if aperture_json.is_file()
        else {"gate_pass": False, "error": "missing aperture report"}
    )

    adjacency_dir = gates_dir / "soundhole_air_audit"
    adjacency_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "mpiexec",
            "-n",
            "1",
            py,
            str(SCRIPT_DIR / "audit_soundhole_air_adjacency.py"),
            "--mesh",
            str(mesh_path.resolve()),
            "--out-dir",
            str(adjacency_dir),
            "--skip-xdmf",
        ],
        check=False,
    )
    adjacency_json = adjacency_dir / "soundhole_air_adjacency_report.json"
    adjacency = (
        json.loads(adjacency_json.read_text(encoding="utf-8"))
        if adjacency_json.is_file()
        else {"error": "missing adjacency report"}
    )

    fc = adjacency.get("facet_counts") or {}
    dof = adjacency.get("pressure_dof_audit") or {}
    tag2 = int(fc.get("tag2_total", 0))
    air_adj = int(fc.get("tag2_adjacent_to_air_10", 0))
    wood_only = int(fc.get("tag2_adjacent_wood_only", 0))
    p_air = int(dof.get("n_p_soundhole_in_air_subgraph", 0))

    aperture_ok = bool(aperture.get("gate_pass"))
    area_ratio = float(
        aperture.get("area_ratio", aperture.get("area_ratio_vs_pi_r2", float("nan")))
    )
    z_span = float(aperture.get("z_span_m", float("inf")))
    horiz_frac = float(aperture.get("horizontal_area_fraction", 0.0))
    area_ratio_ok = math.isfinite(area_ratio) and (1.0 - 0.15) <= area_ratio <= (1.0 + 0.15)
    adjacency_ok = (
        tag2 > 0
        and air_adj == tag2
        and wood_only == 0
        and p_air > 0
    )
    aperture_pass = aperture_ok and area_ratio_ok

    return {
        "combined_mesh_gate_pass": bool(gates.get("combined_mesh_gate_pass")),
        "aperture_gate_pass": aperture_pass,
        "area_ratio": area_ratio,
        "area_ratio_finite": math.isfinite(area_ratio),
        "area_ratio_pass": area_ratio_ok,
        "air_adjacency_ok": adjacency_ok,
        "all_preflight_pass": bool(
            gates.get("combined_mesh_gate_pass") and aperture_pass and adjacency_ok
        ),
        "aperture_report": aperture,
        "adjacency_report": {
            "tag2_total": tag2,
            "tag2_adjacent_to_air_10": air_adj,
            "tag2_adjacent_wood_only": wood_only,
            "n_p_soundhole_in_air_subgraph": p_air,
            "area_ratio": area_ratio,
            "z_span_m": z_span,
            "horizontal_area_fraction": horiz_frac,
        },
        "mesh_gates": gates,
    }


def _build_l_prod_mesh(case: Dict[str, Any], manifest: Dict[str, Any]) -> Path:
    level_def = (manifest.get("mesh_levels") or {}).get("L_prod") or {}
    cid = str(case["id"])
    PREFLIGHT_MESH.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out_msh = PREFLIGHT_MESH / f"{cid}.msh"
    built_path = mesh_path("L_prod", cid)
    if built_path.is_file():
        built_path.unlink()
    audit = build_level_mesh(case, "L_prod", level_def, config_dir=CONFIG_DIR)
    built = Path(audit.get("mesh_file", built_path))
    import shutil

    PREFLIGHT_MESH.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, out_msh)
    write_json(PREFLIGHT_MESH / f"{cid}_mesh_audit.json", audit)
    return out_msh.resolve()


def _diagnose_l_mid_log(case_id: str) -> Dict[str, Any]:
    log_path = CONV_SOLVES / "L_mid" / case_id / "logs" / "mesh_convergence_solve.log"
    result_glob = list((CONV_SOLVES / "L_mid" / case_id / "results").glob("result_*.json"))
    out: Dict[str, Any] = {
        "case_id": case_id,
        "log_path": str(log_path),
        "log_found": log_path.is_file(),
        "result_json_found": bool(result_glob),
    }
    if result_glob:
        try:
            res = json.loads(result_glob[0].read_text(encoding="utf-8"))
            out["v2_converged"] = res.get("v2_converged")
            out["nconv_marked"] = (res.get("eps_batch_diagnostics") or {}).get("nconv_marked")
            out["num_modes_saved"] = res.get("num_modes_saved")
            branch = res.get("acoustic_branch_by_energy") or res.get("nearest_acoustic_branch")
            out["branch_captured"] = branch is not None
            out["acoustic_f_hz"] = (branch or {}).get("frequency_hz")
        except Exception as exc:
            out["result_parse_error"] = str(exc)

    if not log_path.is_file():
        out["diagnosis"] = "log_missing"
        out["category"] = "unknown"
        return out

    text = log_path.read_text(encoding="utf-8", errors="replace")
    tail = text[-12000:] if len(text) > 12000 else text
    out["log_tail_chars"] = len(tail)

    categories: List[str] = []
    if re.search(r"timeout|timed out|Time limit", tail, re.I):
        categories.append("timeout")
    if re.search(r"MemoryError|out of memory|OOM|Cannot allocate", tail, re.I):
        categories.append("resource_limit")
    if re.search(r"KSP|SNES|EPS|diverged|not converged|failed to converge", tail, re.I):
        categories.append("eps_nonconvergence")
    if re.search(r"v2_converged=False|returncode=3|solve_failed", tail, re.I):
        categories.append("solver_or_gate_failure")
    if re.search(r"no_acoustic_candidate|branch_captured=False|selection=none", tail, re.I):
        categories.append("branch_capture_failure")
    if re.search(r"Traceback|ERROR|FATAL", tail):
        categories.append("exception")

    m_rc = re.search(r"\[v2_sensitivity_solve\].*v2_converged=(\w+)", tail)
    if m_rc:
        out["log_v2_converged"] = m_rc.group(1)
    m_sel = re.search(r"selection=(\S+)", tail)
    if m_sel:
        out["log_selection"] = m_sel.group(1)

    if "eps_nonconvergence" in categories:
        primary = "eps_nonconvergence"
    elif "branch_capture_failure" in categories:
        primary = "branch_capture_failure"
    elif "resource_limit" in categories or "timeout" in categories:
        primary = "resource_limit"
    elif "exception" in categories:
        primary = "exception"
    elif out.get("v2_converged") is False and out.get("branch_captured") is False:
        primary = "branch_capture_post_processing"
    elif out.get("v2_converged") is False:
        primary = "eps_nonconvergence"
    else:
        primary = "solver_or_gate_failure" if categories else "unknown"

    out["categories"] = categories
    out["category"] = primary
    out["diagnosis"] = (
        f"L_mid {case_id}: primary={primary}; tags={categories or ['none']}. "
        "No rerun in this repair stage."
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="L_prod mesh preflight (gates only)")
    parser.add_argument(
        "--skip-hole-radius",
        action="store_true",
        help="Only run baseline_coupled_v2 preflight",
    )
    parser.add_argument(
        "--gates-only-revalidate",
        action="store_true",
        help="Re-run gates on existing preflight/L_prod/*.msh (no remesh)",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    PREFLIGHT_DIAG.mkdir(parents=True, exist_ok=True)

    if args.gates_only_revalidate:
        from run_v2_mesh_preresume_repair import revalidate_l_prod_gates

        gate_results = revalidate_l_prod_gates(manifest, rebuild=False)
        all_pass = bool(
            all((gate_results.get(c) or {}).get("all_preflight_pass") for c in gate_results)
        )
        write_json(REPORT_JSON, {"L_prod_gate_revalidation": gate_results})
        print(f"[preflight] gates-only all_preflight_pass={all_pass}", flush=True)
        return 0 if all_pass else 1

    repair_note = (
        "Production FOM now uses the same air-opening geometry + aperture tagging as "
        "FEM_VALIDATION_MESH: cavity fused with through-plate air channel, conformal "
        "wood/air re-fragment, air-side aperture as facet tag 2 (not legacy wood-rim picker)."
    )

    preflight_results: Dict[str, Any] = {}
    baseline_case = _case_from_manifest(manifest, "baseline_coupled_v2")
    print("[preflight] build + gate L_prod baseline_coupled_v2", flush=True)
    mesh_b = _build_l_prod_mesh(baseline_case, manifest)
    preflight_results["baseline_coupled_v2"] = {
        "mesh_file": str(mesh_b),
        **_run_gates(mesh_b, baseline_case, PREFLIGHT_MESH / "baseline_coupled_v2"),
    }

    hole_result: Optional[Dict[str, Any]] = None
    if (
        not args.skip_hole_radius
        and preflight_results["baseline_coupled_v2"].get("all_preflight_pass")
    ):
        hole_case = _case_from_manifest(manifest, "hole_radius_large")
        print("[preflight] build + gate L_prod hole_radius_large", flush=True)
        mesh_h = _build_l_prod_mesh(hole_case, manifest)
        hole_result = {
            "mesh_file": str(mesh_h),
            **_run_gates(mesh_h, hole_case, PREFLIGHT_MESH / "hole_radius_large"),
        }
        preflight_results["hole_radius_large"] = hole_result
    elif not args.skip_hole_radius:
        preflight_results["hole_radius_large"] = {
            "skipped": True,
            "reason": "baseline L_prod preflight did not pass",
        }

    l_mid_diag = [
        _diagnose_l_mid_log("baseline_coupled_v2"),
        _diagnose_l_mid_log("hole_radius_large"),
    ]

    baseline_pass = bool(preflight_results["baseline_coupled_v2"].get("all_preflight_pass"))
    hole_pass = bool((hole_result or {}).get("all_preflight_pass"))
    may_resume = baseline_pass and (args.skip_hole_radius or hole_pass or hole_result is None)

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repair_summary": repair_note,
        "original_L_prod_failure": ORIGINAL_L_PROD_FAILURE,
        "L_prod_preflight": preflight_results,
        "L_mid_solve_failure_diagnosis": l_mid_diag,
        "mesh_convergence_may_resume": may_resume,
        "resume_conditions": (
            "Rebuild L_prod meshes under v2_mesh_convergence/mesh/L_prod/ from repaired "
            "preflight artifacts, then resume run_v2_mesh_convergence.sh (not automatic). "
            "Do not start L_check until L_prod gates pass on production meshes."
        ),
        "staged_status": {
            "mesh_convergence_pass": "Pending",
            "v2_production_promotion_ready": False,
            "lhs_promotion_blocked": True,
        },
    }
    write_json(REPORT_JSON, report)

    lines = [
        "# v2 mesh production preflight report",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        "## Root cause (original L_prod failure)",
        "",
        "Production `FEM_ALLOW_FOM=1` used **legacy** soundhole geometry (short top cylinder + ",
        "exterior-shell facet picker for tag 2). That tagged **wood rim** facets with ",
        "`z_span≈4 mm`, `horizontal_area_fraction≈0.85`, and **zero** tag-2 adjacency to air tag 10.",
        "",
        "Validation / L_mid used **air-opening** geometry (cavity + through-plate channel) and ",
        "air-side aperture selection — gates passed.",
        "",
        "## Repair applied (`build_3d_guitar.py`)",
        "",
        repair_note,
        "",
        "## Repaired L_prod gate results",
        "",
    ]
    for cid, row in preflight_results.items():
        lines.append(f"### {cid}")
        if row.get("skipped"):
            lines.append(f"- skipped: {row.get('reason')}")
            continue
        lines.append(f"- mesh: `{row.get('mesh_file')}`")
        lines.append(f"- all_preflight_pass: **{row.get('all_preflight_pass')}**")
        adj = row.get("adjacency_report") or {}
        lines.append(
            f"- aperture: pass={row.get('aperture_gate_pass')} "
            f"area_ratio={adj.get('area_ratio')} z_span={adj.get('z_span_m')} "
            f"horiz_frac={adj.get('horizontal_area_fraction')}"
        )
        lines.append(
            f"- adjacency: tag2={adj.get('tag2_total')} air_adj={adj.get('tag2_adjacent_to_air_10')} "
            f"wood_only={adj.get('tag2_adjacent_wood_only')} p_in_air={adj.get('n_p_soundhole_in_air_subgraph')}"
        )
        lines.append("")

    lines.extend(["## L_mid existing solve failures (diagnosis only)", ""])
    for d in l_mid_diag:
        lines.append(f"### {d.get('case_id')}")
        lines.append(f"- category: **{d.get('category')}**")
        lines.append(f"- {d.get('diagnosis')}")
        if d.get("v2_converged") is not None:
            lines.append(
                f"- artifact: v2_converged={d.get('v2_converged')} branch={d.get('branch_captured')} "
                f"f={d.get('acoustic_f_hz')}"
            )
        lines.append("")

    lines.extend(
        [
            "## Mesh convergence resume",
            "",
            f"**May resume:** `{may_resume}`",
            "",
            report["resume_conditions"],
            "",
            "**Do not** auto-resume or start L_check from this script.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[preflight] wrote {REPORT_MD}", flush=True)
    print(f"[preflight] baseline_pass={baseline_pass} may_resume={may_resume}", flush=True)
    return 0 if baseline_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
