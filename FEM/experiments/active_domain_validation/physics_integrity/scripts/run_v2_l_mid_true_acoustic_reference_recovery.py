#!/usr/bin/env python3
"""
Recover true acoustic-only locator pressure eigenvectors on L_mid meshes,
build coupled-W seeds (no coupled EPS), then rerun the no-eigensolve mixed-mode audit.
"""
from __future__ import annotations

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
    load_manifest,
    mesh_path,
    sample_spec_from_case,
    solve_case_dir,
    write_json,
)
from v2_sensitivity_common import COUPLED_BASELINE_F_HZ, REPO_ROOT

RECOVERY_JSON = CONV_DIAG / "v2_l_mid_true_acoustic_reference_recovery_report.json"
RECOVERY_MD = CONV_DIAG / "v2_l_mid_true_acoustic_reference_recovery_report.md"

ACOUSTIC_CASES = ("baseline_coupled_v2", "hole_radius_large")
ACOUSTIC_REFERENCE_SOURCE = "acoustic_only_locator_eigenvector"

LOCATOR_POLICY = {
    "locator_harvest_lo_hz": 150.0,
    "locator_harvest_hi_hz": 350.0,
}


def _true_ref_paths(case_dir: Path) -> Dict[str, Path]:
    d = case_dir / "diagnostics" / "l_mid_true_ref"
    return {
        "dir": d,
        "pressure_npy": d / "acoustic_locator_pressure.npy",
        "pressure_meta": d / "acoustic_locator_pressure_meta.json",
        "locator_json": case_dir / "diagnostics" / "acoustic_locator_l_mid.json",
        "seed_npy": case_dir / "diagnostics" / "acoustic_coupled_seed.npy",
        "seed_meta": case_dir / "diagnostics" / "acoustic_coupled_seed_meta.json",
        "build_log": case_dir / "logs" / "build_acoustic_seed.log",
        "locator_log": case_dir / "logs" / "acoustic_locator_true_ref.log",
    }


def _inspect_archives(paths: Dict[str, Path]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "pressure_npy_exists": paths["pressure_npy"].is_file(),
        "pressure_meta_exists": paths["pressure_meta"].is_file(),
        "seed_npy_exists": paths["seed_npy"].is_file(),
        "seed_meta_exists": paths["seed_meta"].is_file(),
        "locator_json_exists": paths["locator_json"].is_file(),
        "archived_pressure_vector_valid": False,
        "archived_seed_valid": False,
    }
    if paths["pressure_meta"].is_file():
        try:
            pm = json.loads(paths["pressure_meta"].read_text(encoding="utf-8"))
            out["pressure_meta"] = pm
            out["archived_pressure_vector_valid"] = (
                paths["pressure_npy"].is_file()
                and pm.get("locator_pressure_reference_source") == ACOUSTIC_REFERENCE_SOURCE
                and int(pm.get("n_p_active", 0)) > 0
            )
        except Exception as exc:
            out["pressure_meta_error"] = str(exc)
    if paths["seed_meta"].is_file():
        try:
            sm = json.loads(paths["seed_meta"].read_text(encoding="utf-8"))
            out["seed_meta"] = sm
            out["archived_seed_valid"] = (
                paths["seed_npy"].is_file()
                and bool(sm.get("seed_layout_valid"))
                and sm.get("locator_pressure_reference_source") == ACOUSTIC_REFERENCE_SOURCE
                and bool(sm.get("acoustic_locator_vector_saved"))
            )
        except Exception as exc:
            out["seed_meta_error"] = str(exc)
    if paths["locator_json"].is_file():
        try:
            out["locator_json"] = json.loads(paths["locator_json"].read_text(encoding="utf-8"))
        except Exception as exc:
            out["locator_json_error"] = str(exc)
    return out


def _l0_reference_hz(case: Dict[str, Any]) -> float:
    return float(case.get("reference_f_hz", COUPLED_BASELINE_F_HZ))


def _run_locator_with_vector(
    mesh_file: Path,
    sample_json: Path,
    paths: Dict[str, Path],
    *,
    reference_hz: float,
) -> Tuple[int, Dict[str, Any]]:
    paths["dir"].mkdir(parents=True, exist_ok=True)
    out_json = paths["locator_json"]
    log_path = paths["locator_log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
        str(LOCATOR_POLICY["locator_harvest_lo_hz"]),
        "--locator-hi-hz",
        str(LOCATOR_POLICY["locator_harvest_hi_hz"]),
        "--reference-hz",
        str(reference_hz),
        "--num-modes",
        "24",
        "--out-json",
        str(out_json.resolve()),
        "--out-pressure-mode-npy",
        str(paths["pressure_npy"].resolve()),
        "--out-pressure-mode-meta-json",
        str(paths["pressure_meta"].resolve()),
    ]
    with open(log_path, "w", encoding="utf-8") as logf:
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT).returncode
    if not out_json.is_file():
        return rc, {"locator_status": "failed", "error": "no locator json"}
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    payload["locator_exit_code"] = int(rc)
    write_json(out_json, payload)
    return int(rc), payload


def _run_seed_build(
    mesh_file: Path,
    sample_json: Path,
    paths: Dict[str, Path],
    *,
    locator_hz: float,
    reference_hz: float,
) -> Tuple[int, Dict[str, Any]]:
    log_path = paths["build_log"]
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
        str(LOCATOR_POLICY["locator_harvest_lo_hz"]),
        "--locator-hi-hz",
        str(LOCATOR_POLICY["locator_harvest_hi_hz"]),
        "--reference-hz",
        str(reference_hz),
        "--num-modes",
        "24",
        "--out-npy",
        str(paths["seed_npy"].resolve()),
        "--out-meta-json",
        str(paths["seed_meta"].resolve()),
        "--out-pressure-npy",
        str(paths["pressure_npy"].resolve()),
    ]
    with open(log_path, "w", encoding="utf-8") as logf:
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT).returncode
    meta = (
        json.loads(paths["seed_meta"].read_text(encoding="utf-8"))
        if paths["seed_meta"].is_file()
        else {}
    )
    return int(rc), meta


def _validate_recovery(paths: Dict[str, Path]) -> Dict[str, Any]:
    import numpy as np

    out: Dict[str, Any] = {
        "acoustic_locator_vector_saved": False,
        "locator_pressure_reference_source": None,
        "seed_build_success": False,
        "seed_vector_length": None,
        "seed_layout_valid": False,
        "eps_seed_applied": False,
        "eps_seed_available_for_later_retry": False,
        "seed_failure_reason": None,
    }
    if paths["pressure_meta"].is_file():
        pm = json.loads(paths["pressure_meta"].read_text(encoding="utf-8"))
        out["acoustic_locator_vector_saved"] = bool(
            paths["pressure_npy"].is_file()
            and pm.get("locator_pressure_reference_source") == ACOUSTIC_REFERENCE_SOURCE
        )
        out["locator_pressure_n_p"] = int(pm.get("n_p_active", 0))
    if paths["seed_meta"].is_file():
        sm = json.loads(paths["seed_meta"].read_text(encoding="utf-8"))
        out["locator_pressure_reference_source"] = sm.get("locator_pressure_reference_source")
        out["seed_layout_valid"] = bool(sm.get("seed_layout_valid"))
        out["seed_vector_length"] = int(sm.get("seed_vector_length", sm.get("n_reduced_W", 0)))
        if paths["seed_npy"].is_file():
            arr = np.load(str(paths["seed_npy"]))
            out["seed_vector_length"] = int(arr.size)
            out["seed_build_success"] = (
                int(arr.size) > 0
                and math.isfinite(float(np.linalg.norm(arr)))
                and out["seed_layout_valid"]
                and sm.get("locator_pressure_reference_source") == ACOUSTIC_REFERENCE_SOURCE
            )
        if not out["seed_build_success"]:
            out["seed_failure_reason"] = "seed/meta validation failed after recovery"
    out["eps_seed_available_for_later_retry"] = bool(out["seed_build_success"])
    return out


def _process_case(manifest: Dict[str, Any], case_id: str) -> Dict[str, Any]:
    case = next(c for c in manifest["cases"] if str(c["id"]) == case_id)
    mesh_file = mesh_path("L_mid", case_id)
    case_dir = solve_case_dir("L_mid", case_id)
    paths = _true_ref_paths(case_dir)
    sample_json = case_dir / "sample_spec.json"
    sample_json.write_text(json.dumps(sample_spec_from_case(case), indent=2), encoding="utf-8")

    row: Dict[str, Any] = {
        "sample_id": case_id,
        "mesh_file": str(mesh_file),
        "inspect": _inspect_archives(paths),
        "actions": [],
    }
    if not mesh_file.is_file():
        row["status"] = "failed"
        row["error"] = f"missing mesh {mesh_file}"
        return row

    insp = row["inspect"]
    loc_hz = float("nan")
    if insp.get("locator_json"):
        loc_hz = float(insp["locator_json"].get("locator_frequency_hz", float("nan")))

    need_locator = not bool(insp.get("archived_pressure_vector_valid"))
    if need_locator:
        row["actions"].append("run_acoustic_cavity_locator_with_vector_archive")
        rc, loc = _run_locator_with_vector(
            mesh_file, sample_json, paths, reference_hz=_l0_reference_hz(case)
        )
        row["locator_run"] = {"exit_code": rc, "payload": loc}
        loc_hz = float(loc.get("locator_frequency_hz", float("nan")))
        if rc != 0 or not bool(loc.get("acoustic_locator_vector_saved")):
            row["status"] = "locator_vector_recovery_failed"
            return row
    else:
        row["actions"].append("reuse_archived_pressure_vector")
        if paths["pressure_meta"].is_file():
            pm = json.loads(paths["pressure_meta"].read_text(encoding="utf-8"))
            loc_hz = float(pm.get("locator_frequency_hz", loc_hz))

    need_seed = not bool(insp.get("archived_seed_valid")) or need_locator
    if need_seed:
        row["actions"].append("build_coupled_W_acoustic_seed")
        if not math.isfinite(loc_hz):
            row["status"] = "failed"
            row["error"] = "no locator frequency for seed build"
            return row
        src, sm = _run_seed_build(
            mesh_file,
            sample_json,
            paths,
            locator_hz=loc_hz,
            reference_hz=_l0_reference_hz(case),
        )
        row["seed_build"] = {"exit_code": src, "meta": sm}
    else:
        row["actions"].append("reuse_valid_coupled_W_seed")

    row["validation"] = _validate_recovery(paths)
    ok = bool(row["validation"].get("seed_build_success")) and bool(
        row["validation"].get("acoustic_locator_vector_saved")
    )
    row["status"] = "ok" if ok else "validation_failed"
    row["locator_acoustic_frequency_hz"] = loc_hz
    return row


def _write_recovery_md(report: Dict[str, Any]) -> None:
    lines = [
        "# L_mid true acoustic reference recovery",
        "",
        f"Generated: {report.get('generated_utc')}",
        "",
        "Acoustic-cavity-only locator eigenvectors archived under "
        "`diagnostics/l_mid_true_ref/`. Coupled-W seeds rebuilt with fixed seed builder. "
        "No coupled EPS solves in this step.",
        "",
    ]
    for cid, row in (report.get("cases") or {}).items():
        lines.append(f"## {cid}")
        lines.append("")
        lines.append(f"- Status: **{row.get('status')}**")
        lines.append(f"- Actions: {', '.join(row.get('actions') or [])}")
        val = row.get("validation") or {}
        lines.append(
            f"- Validation: vector_saved={val.get('acoustic_locator_vector_saved')} "
            f"seed_ok={val.get('seed_build_success')} "
            f"n_W={val.get('seed_vector_length')} "
            f"source={val.get('locator_pressure_reference_source')}"
        )
        lines.append("")
    audit = report.get("mixed_mode_audit") or {}
    lines.append("## Mixed-mode audit rerun")
    lines.append("")
    lines.append(f"- Exit code: {audit.get('exit_code')}")
    lines.append(f"- Report: `{audit.get('report_json')}`")
    lines.append("")
    RECOVERY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    manifest = load_manifest()
    CONV_DIAG.mkdir(parents=True, exist_ok=True)

    cases: Dict[str, Any] = {}
    for cid in ACOUSTIC_CASES:
        print(f"[true_ref_recovery] {cid}", flush=True)
        cases[cid] = _process_case(manifest, cid)

    all_ok = all(str((cases.get(c) or {}).get("status")) == "ok" for c in ACOUSTIC_CASES)
    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "acoustic_reference_source_required": ACOUSTIC_REFERENCE_SOURCE,
        "cases": cases,
        "all_cases_recovered": all_ok,
        "mesh_convergence_may_resume": False,
        "staged_status": {
            "mesh_convergence_pass": "Pending",
            "v2_production_promotion_ready": False,
            "lhs_promotion_blocked": True,
        },
    }

    audit_rc = 1
    if all_ok:
        print("[true_ref_recovery] rerunning no-eigensolve mixed-mode audit", flush=True)
        audit_cmd = [
            "mpiexec",
            "-n",
            "1",
            sys.executable,
            str(SCRIPT_DIR / "run_v2_l_mid_coupled_mixed_mode_audit.py"),
            "--require-true-acoustic-reference",
        ]
        proc = subprocess.run(audit_cmd, cwd=str(REPO_ROOT), check=False)
        audit_rc = int(proc.returncode)
    report["mixed_mode_audit"] = {
        "exit_code": audit_rc,
        "report_json": str(CONV_DIAG / "v2_l_mid_coupled_mixed_mode_audit.json"),
        "report_md": str(CONV_DIAG / "v2_l_mid_coupled_mixed_mode_audit.md"),
        "skipped": not all_ok,
    }
    write_json(RECOVERY_JSON, report)
    _write_recovery_md(report)
    print(f"[true_ref_recovery] wrote {RECOVERY_JSON}", flush=True)
    return 0 if all_ok and audit_rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
