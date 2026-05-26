#!/usr/bin/env python3
"""
L_mid baseline: unregularized-offset seed-branch recovery diagnostic (experiment-only).

Prepared for exactly one permitted baseline EPS rerun on the VM. Does not run unless
invoked explicitly (no automatic execution in CI/local dev).
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
from v2_sensitivity_common import REPO_ROOT, hz_result_tag

CASE_ID = "baseline_coupled_v2"
NUM_MODES = 48
SOLVE_SCRIPT = SCRIPT_DIR / "v2_sensitivity_solve.py"
SIGMA_OFFSETS_HZ = (0.5, -0.5, 1.0, -1.0, 2.0, -2.0)

REPORT_JSON = CONV_DIAG / "v2_l_mid_seed_branch_unregularized_offset_diagnostic.json"
REPORT_MD = CONV_DIAG / "v2_l_mid_seed_branch_unregularized_offset_diagnostic.md"
OUT_SUBDIR = "seed_branch_recovery_diagnostic_unregularized_offset"


def _out_dir(case_dir: Path) -> Path:
    return case_dir / OUT_SUBDIR


def _run_solve(
    sample: Dict[str, Any],
    mesh_file: Path,
    *,
    target_hz: float,
    seed_npy: Path,
    out_dir: Path,
) -> Tuple[int, Dict[str, Any], str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_json = out_dir / "sample_spec.json"
    sample_json.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    log_path = out_dir / "logs" / "seed_branch_recovery_diagnostic.log"
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
        "--seed-branch-unregularized-offset-diagnostic",
    ]
    launch_record = {
        "exact_command_argv": cmd,
        "shell_command": shlex.join(cmd),
        "working_directory": str(REPO_ROOT.resolve()),
        "output_case_dir": str(out_dir),
        "log_path": str(log_path),
        "diagnostic_requires_unregularized_ST": True,
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
                "# unregularized-offset seed_branch_recovery_diagnostic",
                f"return_code={completed.returncode}",
                f"command={shlex.join(cmd)}",
                "",
                "=== stdout ===",
                completed.stdout or "",
                "",
                "=== stderr ===",
                completed.stderr or "",
            ]
        ),
        encoding="utf-8",
    )
    launch_record["return_code"] = int(completed.returncode)
    write_json(out_dir / "launch_record.json", launch_record)

    result_path = out_dir / "results" / f"result_{hz_result_tag(target_hz)}.json"
    result: Dict[str, Any] = {}
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    return int(completed.returncode), result, str(log_path)


def _filter_imports():
    from v2_seed_branch_candidate_filter import (
        FILTER_POLICY,
        VERDICT_ST_REGULARIZATION_REQUIRED,
        assess_physical_eligibility,
        branch_recovery_from_row,
        extract_st_operator_fields,
        replay_candidate_metrics,
    )
    from run_v2_l_mid_seed_branch_recovery_diagnostic import (
        _assemble_operators,
        _load_mode_vec,
        _validate_prelaunch,
    )

    return {
        "FILTER_POLICY": FILTER_POLICY,
        "VERDICT_ST_REGULARIZATION_REQUIRED": VERDICT_ST_REGULARIZATION_REQUIRED,
        "assess_physical_eligibility": assess_physical_eligibility,
        "branch_recovery_from_row": branch_recovery_from_row,
        "extract_st_operator_fields": extract_st_operator_fields,
        "replay_candidate_metrics": replay_candidate_metrics,
        "_assemble_operators": _assemble_operators,
        "_load_mode_vec": _load_mode_vec,
        "_validate_prelaunch": _validate_prelaunch,
    }


def _evaluate(
    out_dir: Path,
    mesh_file: Path,
    sample: Dict[str, Any],
    seed: np.ndarray,
    seed_meta: Dict[str, Any],
    solve_result: Dict[str, Any],
) -> Dict[str, Any]:
    flt = _filter_imports()
    extract_st_operator_fields = flt["extract_st_operator_fields"]
    assess_physical_eligibility = flt["assess_physical_eligibility"]
    replay_candidate_metrics = flt["replay_candidate_metrics"]
    branch_recovery_from_row = flt["branch_recovery_from_row"]
    VERDICT_ST_REGULARIZATION_REQUIRED = flt["VERDICT_ST_REGULARIZATION_REQUIRED"]
    FILTER_POLICY = flt["FILTER_POLICY"]
    _assemble_operators = flt["_assemble_operators"]
    _load_mode_vec = flt["_load_mode_vec"]

    seed_f = float(seed_meta.get("locator_frequency_hz", solve_result.get("target_hz", float("nan"))))
    st_fields = extract_st_operator_fields(solve_result)
    op_ok = bool(st_fields["diagnostic_operator_consistent_with_replay"])

    summary_path = out_dir / "diagnostics" / "mode_energy_summary.json"
    modes: List[Dict[str, Any]] = []
    if summary_path.is_file():
        modes = json.loads(summary_path.read_text(encoding="utf-8")).get("modes") or []

    u_to_W = np.asarray(solve_result.get("u_to_W") or [], dtype=np.int32).ravel()
    p_to_W = np.asarray(solve_result.get("p_to_W") or [], dtype=np.int32).ravel()
    if u_to_W.size == 0:
        A0, M0, u_to_W, p_to_W = _assemble_operators(mesh_file, sample, "maps")
        try:
            A0.destroy()
            M0.destroy()
        except Exception:
            pass

    p_seed = np.asarray(seed[p_to_W], dtype=np.float64).ravel()
    A, M, u_idx, p_idx = _assemble_operators(mesh_file, sample, "evaluate_all")
    candidates: List[Dict[str, Any]] = []
    try:
        for m in modes:
            try:
                vec = _load_mode_vec(out_dir, str(m["vector_path"]))
            except Exception:
                continue
            f_hz = float(m["frequency_hz"])
            p_blk = np.asarray(vec[p_to_W], dtype=np.float64).ravel()
            mac = float(
                abs(np.vdot(p_seed, p_blk))
                / (float(np.linalg.norm(p_seed)) * float(np.linalg.norm(p_blk)))
                if float(np.linalg.norm(p_seed)) > 0 and float(np.linalg.norm(p_blk)) > 0
                else float("nan")
            )
            replay = replay_candidate_metrics(
                A, M, vec, u_to_W=u_idx, p_to_W=p_idx, reported_f_hz=f_hz
            )
            elig = assess_physical_eligibility(
                reported_f_hz=f_hz,
                replay_metrics=replay,
                pressure_mac_to_true_seed=mac,
                seed_f_hz=seed_f,
                require_mac=True,
                require_seed_frequency_match=True,
            )
            row = {
                "continuation_seed_applied": continuation,
                "candidate_index": m.get("mode_index"),
                "vector_file": str(out_dir / str(m["vector_path"])),
                "reported_frequency_hz": f_hz,
                "reported_p_frac_energy_phys": m.get("p_frac_energy_phys"),
                "reported_mode_class_physical_energy": m.get("mode_class_physical_energy"),
                "pressure_MAC_to_true_acoustic_seed": mac,
                "replay_rayleigh_lambda": replay["replay_rayleigh_lambda"],
                "replay_rayleigh_frequency_hz": replay["replay_rayleigh_frequency_hz"],
                "replay_relative_residual": replay["replay_relative_residual"],
                "algebraic_lambda_one_suspect": replay["algebraic_lambda_one_suspect"],
                "reported_vs_replay_frequency_consistent": replay[
                    "reported_vs_replay_frequency_consistent"
                ],
                "physically_eligible_after_filter": elig["physically_eligible_after_filter"],
                "branch_recovery_pass": bool(
                    elig["branch_recovery_pass"] and op_ok
                ),
                "rejection_reasons": list(elig["rejection_reasons"]),
                **st_fields,
            }
            if not op_ok:
                row["rejection_reasons"].append("st_regularization_used_eps_not_replay_consistent")
            candidates.append(row)
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass

    branch_pool = [c for c in candidates if branch_recovery_from_row(c)]
    best = (
        max(branch_pool, key=lambda r: float(r["pressure_MAC_to_true_acoustic_seed"]))
        if branch_pool
        else {}
    )

    continuation = bool(
        solve_result.get("continuation_seed_applied")
        or (solve_result.get("eps_seed") or {}).get("eps_initial_space_set")
        or (solve_result.get("seed_branch_recovery_diagnostic") or {}).get(
            "continuation_seed_applied"
        )
    )

    if not continuation:
        verdict = "DIAGNOSTIC_SOLVER_NOT_APPLIED"
    elif not op_ok:
        verdict = VERDICT_ST_REGULARIZATION_REQUIRED
    elif best:
        verdict = "SEED_BRANCH_RECOVERED_IN_DIAGNOSTIC_MODE"
    elif candidates:
        verdict = "FILTERED_DIAGNOSTIC_NO_PHYSICAL_BRANCH_RECOVERED"
    else:
        verdict = "DIAGNOSTIC_OUTPUT_OR_REPLAY_INCONSISTENT"

    return {
        "seed_rayleigh_f_hz": seed_f,
        "filter_policy": FILTER_POLICY,
        "st_operator_fields": st_fields,
        "candidates": candidates,
        "recovered_mode": best,
        "any_branch_recovery_pass": bool(branch_pool),
        "diagnostic_verdict": verdict,
        "continuation_seed_applied": continuation,
    }


def _write_md(report: Dict[str, Any]) -> None:
    ev = report.get("evaluation") or {}
    st = ev.get("st_operator_fields") or {}
    lines = [
        "# L_mid unregularized-offset seed-branch recovery diagnostic",
        "",
        f"Generated: {report.get('generated_utc')}",
        "",
        f"**Verdict:** `{ev.get('diagnostic_verdict')}`",
        "",
        f"- diagnostic_operator_consistent_with_replay: {st.get('diagnostic_operator_consistent_with_replay')}",
        f"- actual_sigma_hz: {st.get('actual_sigma_hz')}",
        f"- actual_st_a_shift_frac: {st.get('actual_st_a_shift_frac')}",
        f"- actual_st_mass_reg_frac: {st.get('actual_st_mass_reg_frac')}",
        "",
        f"**mesh_convergence_may_resume:** `{report.get('mesh_convergence_may_resume')}`",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write launch_record and report stub without running mpiexec.",
    )
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
    target_hz = float(seed_meta.get("locator_frequency_hz", 243.0754171175576))

    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_id": CASE_ID,
        "output_subdir": OUT_SUBDIR,
        "mesh_convergence_may_resume": False,
        "staged_status": {
            "mesh_convergence_pass": "Pending",
            "v2_production_promotion_ready": False,
            "lhs_promotion_blocked": True,
        },
        "maximum_additional_baseline_solves_before_escalation": 1,
        "maximum_additional_code_fix_cycles_before_reconsidering_solver_architecture": 1,
    }

    if args.prepare_only:
        local_half = max(0.5, target_hz * 0.01)
        ladder = [target_hz + float(o) for o in SIGMA_OFFSETS_HZ]
        report["prepare_only"] = True
        report["recommended_vm_command"] = (
            "bash FEM/experiments/active_domain_validation/physics_integrity/scripts/"
            "run_v2_l_mid_seed_branch_unregularized_offset_diagnostic.sh"
        )
        report["sigma_ladder_policy"] = {
            "seed_rayleigh_f_hz": target_hz,
            "offsets_hz_from_seed": list(SIGMA_OFFSETS_HZ),
            "never_sigma_at_seed_rayleigh_frequency": True,
            "diagnostic_sigma_hz_ladder": ladder,
            "diagnostic_sigma_offset_from_seed_hz": [float(o) for o in SIGMA_OFFSETS_HZ],
            "diagnostic_local_band_hz": [target_hz - local_half, target_hz + local_half],
            "st_a_shift_frac_must_be_zero_for_verdict": True,
            "st_mass_reg_frac_must_be_zero_for_verdict": True,
        }
        report["evaluation"] = {
            "diagnostic_verdict": "PENDING_VM_RUN",
            "diagnostic_requires_unregularized_ST": True,
            "note": "Run recommended_vm_command on VM after code sync (one permitted baseline solve).",
        }
        write_json(REPORT_JSON, report)
        _write_md(report)
        return 0

    flt = _filter_imports()
    _validate_prelaunch = flt["_validate_prelaunch"]
    pre = _validate_prelaunch(mesh_file, seed_npy, seed_meta)
    if not pre["valid"]:
        report["evaluation"] = {"diagnostic_verdict": "DIAGNOSTIC_SOLVER_NOT_APPLIED", "prelaunch": pre}
        write_json(REPORT_JSON, report)
        _write_md(report)
        return 1

    seed = np.load(str(seed_npy))
    sample = sample_spec_from_case(case)
    rc, solve_result, log_path = _run_solve(
        sample, mesh_file, target_hz=target_hz, seed_npy=seed_npy, out_dir=out_dir
    )
    evaluation = _evaluate(out_dir, mesh_file, sample, seed, seed_meta, solve_result)
    report["evaluation"] = evaluation
    report["solve_exit_code"] = rc
    report["log_path"] = log_path
    report["output_dir"] = str(out_dir)
    write_json(REPORT_JSON, report)
    _write_md(report)
    print(f"[unreg_offset_diag] verdict={evaluation['diagnostic_verdict']}", flush=True)
    return 0 if evaluation["diagnostic_verdict"] == "SEED_BRANCH_RECOVERED_IN_DIAGNOSTIC_MODE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
