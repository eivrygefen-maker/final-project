#!/usr/bin/env python3
"""
L_mid acoustic locator + targeted coupled capture (existing L_mid meshes only).

Does not remesh, alter v2 physics, or run L_prod/L_check solves.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_mesh_convergence_common import (
    CONV_DIAG,
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
from v2_sensitivity_common import (
    COUPLED_BASELINE_F_HZ,
    REPO_ROOT,
    targeted_harvest_from_locator,
)

REPORT_JSON = CONV_DIAG / "v2_l_mid_acoustic_locator_rescue_report.json"
REPORT_MD = CONV_DIAG / "v2_l_mid_acoustic_locator_rescue_report.md"

LOCATOR_POLICY = {
    "locator_harvest_lo_hz": 150.0,
    "locator_harvest_hi_hz": 350.0,
    "coupled_harvest_half_width_hz": 22.0,
    "coupled_harvest_half_width_min_hz": 18.0,
    "coupled_harvest_half_width_max_hz": 28.0,
}

ACOUSTIC_CASES = ("baseline_coupled_v2", "hole_radius_large")


def _case_from_manifest(manifest: Dict[str, Any], case_id: str) -> Dict[str, Any]:
    for c in manifest.get("cases") or []:
        if str(c.get("id")) == case_id:
            return c
    raise KeyError(case_id)


def _l0_reference_hz(case: Dict[str, Any]) -> float:
    return float(case.get("reference_f_hz", COUPLED_BASELINE_F_HZ))


def _diagnose_prior_result(case: Dict[str, Any]) -> Dict[str, Any]:
    p = solve_result_path("L_mid", str(case["id"]), float(case["target_hz"]))
    if not p.is_file():
        return {"prior_result": "missing"}
    res = json.loads(p.read_text(encoding="utf-8"))
    in_band = list(res.get("in_band_modes") or [])
    n_acoustic = sum(
        1
        for m in in_band
        if str(m.get("mode_class_physical_energy")) == "acoustic_dominated"
        or float(m.get("p_frac_energy_phys", 0.0)) >= 0.85
    )
    out = {
        "v2_converged": bool(res.get("v2_converged")),
        "nconv_marked": int(res.get("nconv_marked", -1)),
        "num_modes_saved": res.get("num_modes_saved"),
        "harvest_band_hz": res.get("harvest_band_hz"),
        "acoustic_branch_selection": res.get("acoustic_branch_selection"),
        "branch_captured": _acoustic_branch_ok(res),
        "n_in_band": len(in_band),
        "n_acoustic_dominated_in_band": n_acoustic,
    }
    if not out["v2_converged"] or int(out.get("nconv_marked", 0)) <= 0:
        out["failure_class"] = "no_eps_eigenpairs_converged"
    elif len(in_band) == 0:
        out["failure_class"] = "acoustic_branch_outside_harvest_band"
    elif n_acoustic == 0:
        out["failure_class"] = "eigenpairs_converged_no_acoustic_dominated_in_band"
    elif not out["branch_captured"]:
        out["failure_class"] = "post_processing_selection_failure"
    else:
        out["failure_class"] = "ok"
    return out


def _run_locator(
    case: Dict[str, Any],
    mesh_file: Path,
    case_dir: Path,
    *,
    policy: Dict[str, Any],
) -> Tuple[int, Dict[str, Any]]:
    case_dir.mkdir(parents=True, exist_ok=True)
    sample_json = case_dir / "sample_spec.json"
    sample_json.write_text(json.dumps(sample_spec_from_case(case), indent=2), encoding="utf-8")
    out_json = case_dir / "diagnostics" / "acoustic_locator_l_mid.json"
    log_path = case_dir / "logs" / "acoustic_locator_l_mid.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    l0_ref = _l0_reference_hz(case)
    cmd = [
        "mpiexec",
        "-n",
        "1",
        sys.executable,
        str(SCRIPT_DIR / "v2_sensitivity_locator.py"),
        "--mesh",
        str(mesh_file.resolve()),
        "--sample-json",
        str(sample_json.resolve()),
        "--locator-lo-hz",
        str(float(policy["locator_harvest_lo_hz"])),
        "--locator-hi-hz",
        str(float(policy["locator_harvest_hi_hz"])),
        "--reference-hz",
        str(l0_ref),
        "--num-modes",
        "24",
        "--out-json",
        str(out_json.resolve()),
    ]
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT, check=False
        )
    if not out_json.is_file():
        return proc.returncode, {"locator_status": "failed", "error": "no locator output"}
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    payload["L0_reference_frequency_hz"] = l0_ref
    payload["locator_shift_from_L0_hz"] = (
        float(payload.get("locator_frequency_hz", float("nan"))) - l0_ref
        if math.isfinite(float(payload.get("locator_frequency_hz", float("nan"))))
        else float("nan")
    )
    payload["locator_exit_code"] = int(proc.returncode)
    write_json(out_json, payload)
    return int(proc.returncode), payload


def _coupled_attempt(
    case: Dict[str, Any],
    mesh_file: Path,
    case_dir: Path,
    *,
    label: str,
    target_hz: float,
    harvest_lo: float,
    harvest_hi: float,
    num_modes: int,
    eps_seed_npy: Optional[Path] = None,
) -> Dict[str, Any]:
    log_path = case_dir / "logs" / f"mesh_convergence_{label}.log"
    rc, solve = run_mpi_case_solve(
        sample_spec_from_case(case),
        mesh_file,
        target_hz=float(target_hz),
        harvest_lo_hz=float(harvest_lo),
        harvest_hi_hz=float(harvest_hi),
        num_modes=int(num_modes),
        log_path=log_path,
        case_dir=case_dir,
        select_by_energy=True,
        reference_f_hz=_l0_reference_hz(case),
        eps_seed_npy=eps_seed_npy,
    )
    branch = solve.get("acoustic_branch_by_energy") or solve.get("nearest_acoustic_branch")
    row = {
        "attempt_label": label,
        "coupled_target_hz": float(target_hz),
        "coupled_harvest_band_hz": [float(harvest_lo), float(harvest_hi)],
        "requested_modes": int(num_modes),
        "solve_exit_code": int(rc),
        "nconv_marked": int(solve.get("nconv_marked", -1)),
        "num_modes_saved": int(solve.get("num_modes_saved", -1)),
        "branch_captured": _acoustic_branch_ok(solve),
        "f_acoustic_hz": (branch or {}).get("frequency_hz"),
        "p_frac_energy_phys": (branch or {}).get("p_frac_energy_phys"),
        "mode_class_physical_energy": (branch or {}).get("mode_class_physical_energy"),
        "acoustic_branch_selection": solve.get("acoustic_branch_selection"),
        "eps_seed_used": bool(eps_seed_npy),
        "spectral_crowding_note": None,
    }
    if not row["branch_captured"] and row["nconv_marked"] > 0:
        row["spectral_crowding_note"] = (
            "EPS converged modes but no acoustic-dominated branch in harvest band; "
            "may indicate spectral crowding/retrieval failure."
        )
    row["_solve_payload"] = solve
    return row


def _build_and_run_seed(
    case: Dict[str, Any],
    mesh_file: Path,
    case_dir: Path,
    locator_hz: float,
    policy: Dict[str, Any],
) -> Tuple[Path, Dict[str, Any]]:
    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"
    seed_meta = case_dir / "diagnostics" / "acoustic_coupled_seed_meta.json"
    sample_json = case_dir / "sample_spec.json"
    log_path = case_dir / "logs" / "build_acoustic_seed.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mpiexec",
        "-n",
        "1",
        sys.executable,
        str(SCRIPT_DIR / "v2_build_coupled_acoustic_seed.py"),
        "--mesh",
        str(mesh_file.resolve()),
        "--sample-json",
        str(sample_json.resolve()),
        "--locator-hz",
        str(float(locator_hz)),
        "--locator-lo-hz",
        str(float(policy["locator_harvest_lo_hz"])),
        "--locator-hi-hz",
        str(float(policy["locator_harvest_hi_hz"])),
        "--reference-hz",
        str(_l0_reference_hz(case)),
        "--num-modes",
        "24",
        "--out-npy",
        str(seed_npy.resolve()),
        "--out-meta-json",
        str(seed_meta.resolve()),
    ]
    with open(log_path, "w", encoding="utf-8") as logf:
        subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT, check=False)
    meta = json.loads(seed_meta.read_text(encoding="utf-8")) if seed_meta.is_file() else {}
    return seed_npy, meta


def process_case(manifest: Dict[str, Any], case_id: str) -> Dict[str, Any]:
    case = _case_from_manifest(manifest, case_id)
    mesh_file = mesh_path("L_mid", case_id)
    if not mesh_file.is_file():
        return {"sample_id": case_id, "status": "failed", "error": f"missing mesh {mesh_file}"}

    case_dir = solve_case_dir("L_mid", case_id)
    prior = _diagnose_prior_result(case)

    loc_rc, locator = _run_locator(case, mesh_file, case_dir, policy=LOCATOR_POLICY)
    loc_hz = float(locator.get("locator_frequency_hz", float("nan")))
    loc_status = str(locator.get("locator_status", "failed"))

    row: Dict[str, Any] = {
        "sample_id": case_id,
        "mesh_file": str(mesh_file),
        "L0_reference_frequency_hz": _l0_reference_hz(case),
        "locator_search_band_hz": [
            LOCATOR_POLICY["locator_harvest_lo_hz"],
            LOCATOR_POLICY["locator_harvest_hi_hz"],
        ],
        "locator_status": loc_status,
        "locator_acoustic_frequency_hz": loc_hz,
        "locator_shift_from_L0_hz": locator.get("locator_shift_from_L0_hz"),
        "locator_selection_method": locator.get("locator_selection_method"),
        "prior_failure_diagnosis": prior,
        "coupled_attempts": [],
    }

    if loc_status != "ok" or not math.isfinite(loc_hz):
        row["status"] = "locator_failed"
        return row

    targeted = targeted_harvest_from_locator(loc_hz, LOCATOR_POLICY)
    att1 = _coupled_attempt(
        case,
        mesh_file,
        case_dir,
        label="locator_targeted",
        target_hz=targeted["target_hz"],
        harvest_lo=targeted["harvest_lo_hz"],
        harvest_hi=targeted["harvest_hi_hz"],
        num_modes=24,
    )
    row["coupled_attempts"].append(att1)

    final_ok = bool(att1.get("branch_captured"))
    if not final_ok:
        seed_npy, seed_meta = _build_and_run_seed(case, mesh_file, case_dir, loc_hz, LOCATOR_POLICY)
        half = 28.0
        att2 = _coupled_attempt(
            case,
            mesh_file,
            case_dir,
            label="locator_targeted_seeded",
            target_hz=loc_hz,
            harvest_lo=loc_hz - half,
            harvest_hi=loc_hz + half,
            num_modes=32,
            eps_seed_npy=seed_npy if seed_npy.is_file() else None,
        )
        att2["acoustic_seed_meta"] = seed_meta
        row["coupled_attempts"].append(att2)
        final_ok = bool(att2.get("branch_captured"))
        if not final_ok and att2.get("nconv_marked", 0) > 0:
            row["interpretation"] = "spectral_crowding_retrieval_failure"

    if final_ok:
        win = row["coupled_attempts"][-1]
        solve = win.get("_solve_payload") or {}
        solve["locator_rescue"] = {
            k: v
            for k, v in row.items()
            if k not in ("_solve_payload",)
        }
        solve["branch_captured"] = True
        write_json(solve_result_path("L_mid", case_id, float(case["target_hz"])), solve)

    row["branch_captured"] = final_ok
    row["status"] = "ok" if final_ok else "coupled_capture_failed"
    if final_ok:
        last = row["coupled_attempts"][-1]
        row.update(
            {
                "coupled_target_hz": last.get("coupled_target_hz"),
                "coupled_harvest_band_hz": last.get("coupled_harvest_band_hz"),
                "f_acoustic_hz": last.get("f_acoustic_hz"),
                "p_frac_energy_phys": last.get("p_frac_energy_phys"),
                "mode_class_physical_energy": last.get("mode_class_physical_energy"),
            }
        )
    return row


def _write_md(report: Dict[str, Any]) -> None:
    lines = [
        "# L_mid acoustic locator rescue report",
        "",
        f"Generated: {report.get('generated_utc')}",
        "",
        "## Policy",
        "",
        f"- Locator search: `{report.get('locator_search_band_hz')}` Hz (does not assume L0 f is in prior coupled harvest)",
        f"- Targeted coupled half-width: up to `{LOCATOR_POLICY['coupled_harvest_half_width_max_hz']}` Hz",
        "- Seeded retry: acoustic cavity mode embedded in coupled W (`v2_build_coupled_acoustic_seed.py`)",
        "",
    ]
    for cid, row in (report.get("cases") or {}).items():
        lines.append(f"## {cid}")
        lines.append("")
        lines.append(f"- L0 reference f: **{row.get('L0_reference_frequency_hz')}** Hz")
        lines.append(f"- Locator status: **{row.get('locator_status')}**")
        lines.append(
            f"- Locator f: **{row.get('locator_acoustic_frequency_hz')}** Hz "
            f"(Δ from L0: {row.get('locator_shift_from_L0_hz'):+.4f} Hz)"
        )
        prior = row.get("prior_failure_diagnosis") or {}
        lines.append(f"- Prior failure class: `{prior.get('failure_class')}`")
        lines.append(f"- Final status: **{row.get('status')}** branch_captured={row.get('branch_captured')}")
        if row.get("interpretation"):
            lines.append(f"- Note: {row['interpretation']}")
        lines.append("")
        for att in row.get("coupled_attempts") or []:
            lines.append(
                f"### {att.get('attempt_label')}: target={att.get('coupled_target_hz')} "
                f"band={att.get('coupled_harvest_band_hz')} modes={att.get('requested_modes')}"
            )
            lines.append(
                f"- branch={att.get('branch_captured')} f={att.get('f_acoustic_hz')} "
                f"p_frac={att.get('p_frac_energy_phys')} selection={att.get('acoustic_branch_selection')}"
            )
            if att.get("spectral_crowding_note"):
                lines.append(f"- {att['spectral_crowding_note']}")
            lines.append("")
    lines.extend(
        [
            "## Resume",
            "",
            f"**mesh_convergence_may_resume:** `{report.get('mesh_convergence_may_resume')}`",
            "",
        ]
    )
    if report.get("resume_command"):
        lines.append(f"```bash\n{report['resume_command']}\n```")
    lines.append("")
    lines.append("Do not auto-start L_prod or L_check from this script.")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="L_mid acoustic locator rescue")
    args = parser.parse_args()
    manifest = load_manifest()
    CONV_DIAG.mkdir(parents=True, exist_ok=True)

    cases_report: Dict[str, Any] = {}
    for cid in ACOUSTIC_CASES:
        print(f"[l_mid_rescue] process {cid}", flush=True)
        cases_report[cid] = process_case(manifest, cid)

    all_ok = all(bool((cases_report.get(c) or {}).get("branch_captured")) for c in ACOUSTIC_CASES)
    may_resume = all_ok

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "locator_search_band_hz": [
            LOCATOR_POLICY["locator_harvest_lo_hz"],
            LOCATOR_POLICY["locator_harvest_hi_hz"],
        ],
        "cases": cases_report,
        "mesh_convergence_may_resume": may_resume,
        "resume_command": (
            "bash FEM/experiments/active_domain_validation/physics_integrity/scripts/"
            "run_v2_mesh_convergence.sh --resume"
            if may_resume
            else None
        ),
        "staged_status": {
            "mesh_convergence_pass": "Pending",
            "v2_production_promotion_ready": False,
            "lhs_promotion_blocked": True,
        },
    }
    write_json(REPORT_JSON, report)
    _write_md(report)

    status_path = (
        Path(__file__).resolve().parents[1]
        / "v2_sensitivity_validation"
        / "diagnostics"
        / "v2_validation_status.json"
    )
    if status_path.is_file():
        st = json.loads(status_path.read_text(encoding="utf-8"))
        st["l_mid_acoustic_locator_rescue"] = {
            "may_resume": may_resume,
            "report_json": str(REPORT_JSON),
        }
        write_json(status_path, st)

    print(f"[l_mid_rescue] wrote {REPORT_MD}", flush=True)
    print(f"[l_mid_rescue] may_resume={may_resume}", flush=True)
    return 0 if may_resume else 1


if __name__ == "__main__":
    raise SystemExit(main())
