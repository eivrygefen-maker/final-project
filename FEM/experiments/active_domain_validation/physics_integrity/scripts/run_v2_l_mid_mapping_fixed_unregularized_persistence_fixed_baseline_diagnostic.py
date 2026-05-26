#!/usr/bin/env python3
"""
Replacement mapping-corrected baseline after persistence self-test passes.

Output tree: seed_branch_recovery_diagnostic_mapping_fixed_unregularized_persistence_fixed/
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FEM_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_mesh_convergence_common import (
    CONV_DIAG,
    load_manifest,
    mesh_path,
    sample_spec_from_case,
    solve_case_dir,
    write_json,
)
from v2_mapping_fixed_baseline_evaluator import (
    OUT_SUBDIR_PERSISTENCE_FIXED,
    VERDICT_BRANCH_RECOVERED,
    VERDICT_INCONSISTENT,
    VERDICT_PERSISTENCE_FAILURE,
    evaluate_mapping_fixed_baseline_artifacts,
)
from v2_sensitivity_common import REPO_ROOT, hz_result_tag

CASE_ID = "baseline_coupled_v2"
NUM_MODES = 64
SOLVE_SCRIPT = SCRIPT_DIR / "v2_sensitivity_solve.py"
SIGMA_OFFSETS_HZ = (0.5, -0.5, 1.0, -1.0, 2.0, -2.0)
SEED_F_HZ = 243.0754171175576
SELF_TEST_JSON = CONV_DIAG / "v2_mapping_fixed_candidate_persistence_self_test.json"

REPORT_JSON = (
    CONV_DIAG / "v2_l_mid_mapping_fixed_unregularized_persistence_fixed_baseline_diagnostic.json"
)
REPORT_MD = (
    CONV_DIAG / "v2_l_mid_mapping_fixed_unregularized_persistence_fixed_baseline_diagnostic.md"
)
VM_SHELL = (
    "bash FEM/experiments/active_domain_validation/physics_integrity/scripts/"
    "run_v2_mapping_fixed_persistence_fixed_baseline_vm.sh"
)


def _out_dir(case_dir: Path) -> Path:
    return case_dir / OUT_SUBDIR_PERSISTENCE_FIXED


def _self_test_passed() -> Tuple[bool, Dict[str, Any]]:
    if not SELF_TEST_JSON.is_file():
        return False, {"reason": "self_test_json_missing", "path": str(SELF_TEST_JSON)}
    try:
        data = json.loads(SELF_TEST_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, {"reason": f"self_test_json_load_failed:{exc}"}
    return bool(data.get("self_test_pass")), data


def _artifact_status(out_dir: Path, seed_npy: Path) -> Dict[str, Any]:
    summary = out_dir / "diagnostics" / "mode_energy_summary.json"
    bank = out_dir / "diagnostics" / "eps_candidate_bank.json"
    modes_dir = out_dir / "modes"
    n_slot = len(list(modes_dir.glob("candidate_eps_slot_*.smx.npz"))) if modes_dir.is_dir() else 0
    result_glob = list((out_dir / "results").glob("result_*.json"))
    return {
        "output_dir": str(out_dir),
        "mode_energy_summary_exists": summary.is_file(),
        "eps_candidate_bank_exists": bank.is_file(),
        "candidate_vector_files": n_slot,
        "result_json_count": len(result_glob),
        "seed_npy_exists": seed_npy.is_file(),
        "artifacts_ok": bool(
            out_dir.is_dir()
            and summary.is_file()
            and bank.is_file()
            and seed_npy.is_file()
            and n_slot > 0
        ),
    }


def _load_solve_result(out_dir: Path, target_hz: float) -> Dict[str, Any]:
    result_path = out_dir / "results" / f"result_{hz_result_tag(target_hz)}.json"
    if not result_path.is_file():
        results = sorted((out_dir / "results").glob("result_*.json"))
        if not results:
            return {}
        result_path = results[-1]
    return json.loads(result_path.read_text(encoding="utf-8"))


def _run_solve(
    sample: Dict[str, Any],
    mesh_file: Path,
    *,
    target_hz: float,
    seed_npy: Path,
    out_dir: Path,
) -> Tuple[int, Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_json = out_dir / "sample_spec.json"
    sample_json.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    log_path = out_dir / "logs" / "mapping_fixed_persistence_fixed_baseline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mpiexec",
        "-n",
        "1",
        sys.executable,
        "-u",
        str(SOLVE_SCRIPT.resolve()),
        "--sample-id",
        str(sample["id"]),
        "--mesh",
        str(mesh_file.resolve()),
        "--sample-json",
        str(sample_json.resolve()),
        "--case-dir",
        str(out_dir.resolve()),
        "--target-hz",
        str(float(target_hz)),
        "--reference-f-hz",
        str(float(target_hz)),
        "--num-modes",
        str(int(NUM_MODES)),
        "--eps-seed-npy",
        str(seed_npy.resolve()),
        "--seed-branch-recovery-diagnostic",
        "--seed-branch-mapping-fixed-unregularized-diagnostic",
    ]
    launch_record = {
        "exact_command_argv": cmd,
        "shell_command": shlex.join(cmd),
        "output_case_dir": str(out_dir),
        "persistence_self_test_json": str(SELF_TEST_JSON),
    }
    write_json(out_dir / "launch_record.json", launch_record)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env, check=False
    )
    log_path.write_text(
        "\n".join(
            [
                f"return_code={completed.returncode}",
                f"command={shlex.join(cmd)}",
                "",
                completed.stdout or "",
                "",
                completed.stderr or "",
            ]
        ),
        encoding="utf-8",
    )
    result = _load_solve_result(out_dir, target_hz)
    return int(completed.returncode), result


def _write_md(report: Dict[str, Any]) -> None:
    ev = report.get("evaluation") or {}
    lines = [
        "# L_mid mapping-corrected baseline (persistence-fixed replacement)",
        "",
        f"Generated: {report.get('generated_utc')}",
        f"**Verdict:** `{ev.get('diagnostic_verdict')}`",
        "",
        f"**self_test_pass:** `{report.get('persistence_self_test_pass')}`",
        "",
    ]
    pf = ev.get("persistence_failure")
    if pf:
        lines.append(f"**persistence_failure:** `{pf}`")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _refresh_static_reports() -> None:
    from run_v2_solver_root_cause_and_forward_risk_audit import main as audit_main
    from write_v2_st_singular_mass_rehabilitation_plan import main as rehab_main

    rehab_main()
    audit_main()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    case_dir = solve_case_dir("L_mid", CASE_ID)
    mesh_file = mesh_path("L_mid", CASE_ID)
    out_dir = _out_dir(case_dir)
    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"
    seed_meta_path = case_dir / "diagnostics" / "acoustic_coupled_seed_meta.json"
    seed_meta = (
        json.loads(seed_meta_path.read_text(encoding="utf-8")) if seed_meta_path.is_file() else {}
    )
    target_hz = float(seed_meta.get("locator_frequency_hz", SEED_F_HZ))
    fallback_sample = sample_spec_from_case(case)
    st_pass, st_data = _self_test_passed()

    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_id": CASE_ID,
        "output_subdir": OUT_SUBDIR_PERSISTENCE_FIXED,
        "recommended_vm_command": VM_SHELL,
        "persistence_self_test_pass": st_pass,
        "persistence_self_test": st_data,
        "prior_run_inconclusive": {
            "output_subdir": "seed_branch_recovery_diagnostic_mapping_fixed_unregularized",
            "verdict": VERDICT_PERSISTENCE_FAILURE,
            "nconv_marked": 56,
            "num_vectors_saved": 0,
            "not_evidence_for_st_failure": True,
        },
        "mesh_convergence_may_resume": False,
    }

    if args.prepare_only:
        report["evaluation"] = {
            "diagnostic_verdict": "PENDING_SELF_TEST_AND_REPLACEMENT_BASELINE",
            "note": f"Run {VM_SHELL} on VM.",
        }
        write_json(REPORT_JSON, report)
        _write_md(report)
        _refresh_static_reports()
        return 0

    if not args.evaluate_only and not st_pass:
        report["evaluation"] = {
            "diagnostic_verdict": "PERSISTENCE_SELF_TEST_FAILED_NO_EIGENSOLVE",
            "self_test": st_data,
        }
        write_json(REPORT_JSON, report)
        _write_md(report)
        print("[persistence_fixed_baseline] ABORT: persistence self-test not passed", file=sys.stderr)
        return 2

    if args.evaluate_only:
        solve_result = _load_solve_result(out_dir, target_hz)
        status = _artifact_status(out_dir, seed_npy)
        if not solve_result:
            report["evaluation"] = {
                "diagnostic_verdict": VERDICT_INCONSISTENT,
                "artifact_status": status,
            }
        else:
            report["evaluation"] = evaluate_mapping_fixed_baseline_artifacts(
                out_dir=out_dir,
                case_dir=case_dir,
                mesh_file=mesh_file,
                seed_npy=seed_npy,
                seed_meta=seed_meta,
                fallback_sample=fallback_sample,
                solve_result=solve_result,
                target_hz=target_hz,
            )
            report["evaluation"]["artifact_status"] = status
        write_json(REPORT_JSON, report)
        _write_md(report)
        _refresh_static_reports()
        verdict = (report.get("evaluation") or {}).get("diagnostic_verdict", VERDICT_INCONSISTENT)
        if verdict == VERDICT_PERSISTENCE_FAILURE:
            return 2
        if verdict == VERDICT_BRANCH_RECOVERED:
            return 0
        if verdict == VERDICT_INCONSISTENT:
            return 2
        return 1

    rc, solve_result = _run_solve(
        fallback_sample, mesh_file, target_hz=target_hz, seed_npy=seed_npy, out_dir=out_dir
    )
    report["solve_return_code"] = rc
    status = _artifact_status(out_dir, seed_npy)
    if rc != 0 or not solve_result:
        report["evaluation"] = {
            "diagnostic_verdict": VERDICT_INCONSISTENT,
            "artifact_status": status,
        }
    else:
        report["evaluation"] = evaluate_mapping_fixed_baseline_artifacts(
            out_dir=out_dir,
            case_dir=case_dir,
            mesh_file=mesh_file,
            seed_npy=seed_npy,
            seed_meta=seed_meta,
            fallback_sample=fallback_sample,
            solve_result=solve_result,
            target_hz=target_hz,
        )
        report["evaluation"]["artifact_status"] = status
    write_json(REPORT_JSON, report)
    _write_md(report)
    _refresh_static_reports()
    verdict = (report.get("evaluation") or {}).get("diagnostic_verdict", VERDICT_INCONSISTENT)
    print(f"[persistence_fixed_baseline] rc={rc} verdict={verdict}", flush=True)
    if verdict == VERDICT_PERSISTENCE_FAILURE:
        return 2
    if verdict == VERDICT_BRANCH_RECOVERED:
        return 0
    if verdict == VERDICT_INCONSISTENT:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
