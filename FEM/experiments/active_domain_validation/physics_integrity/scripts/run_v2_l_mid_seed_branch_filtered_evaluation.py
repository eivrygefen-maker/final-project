#!/usr/bin/env python3
"""
VM-only report-only evaluation of filtered seed-branch recovery diagnostic candidates.

Reads artifacts under seed_branch_recovery_diagnostic_filtered/ on the VM.
Does not run an eigensolve. Safe to run locally only when those artifacts exist; otherwise
writes FILTERED_DIAGNOSTIC_OUTPUT_OR_REPLAY_INCONSISTENT with a pending-runtime note.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FEM_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fem_main_3d as fem3d
from v2_mesh_convergence_common import (
    CONV_DIAG,
    load_manifest,
    mesh_path,
    sample_spec_from_case,
    solve_case_dir,
    write_json,
)
from v2_seed_branch_candidate_filter import (
    FILTER_POLICY,
    VERDICT_FILTERED_BRANCH_RECOVERED,
    VERDICT_FILTERED_INCONSISTENT,
    VERDICT_FILTERED_NO_BRANCH,
    assess_physical_eligibility,
    assign_filtered_evaluation_verdict,
    branch_recovery_from_row,
    replay_candidate_metrics,
)

OUT_JSON = CONV_DIAG / "v2_l_mid_seed_branch_recovery_filtered_evaluation.json"
OUT_MD = CONV_DIAG / "v2_l_mid_seed_branch_recovery_filtered_evaluation.md"

CASE_ID = "baseline_coupled_v2"
SEED_F_HZ = 243.0754171175576
FILTERED_SUBDIR = "seed_branch_recovery_diagnostic_filtered"
VM_OPERATOR_EXPECTED_SAVED_MODES = 7


def _load_mode_vec(path: Path) -> np.ndarray:
    from fem_mode_array_utils import load_mode_column_any

    return np.asarray(load_mode_column_any(path).toarray(), dtype=np.float64).ravel()


def _pressure_mac(seed_block: np.ndarray, p_block: np.ndarray) -> float:
    a = np.asarray(seed_block, dtype=np.complex128).ravel()
    b = np.asarray(p_block, dtype=np.complex128).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(abs(np.vdot(a, b)) / (na * nb))


def _assemble_replay(mesh_file: Path, sample: Dict[str, Any], filtered_dir: Path):
    from v2_build_coupled_acoustic_seed import _assemble_reduced_coupled_replay

    sort_dir = filtered_dir / "sorting_filtered_evaluation"
    sort_dir.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sort_dir.resolve())
    A, M, cfg = _assemble_reduced_coupled_replay(mesh_file, sample, coupling_enabled=True)
    u_to_W = np.asarray(cfg["_coupled_air_u_to_W_map"], dtype=np.int32).ravel()
    p_to_W = np.asarray(cfg["_coupled_air_p_to_W_map"], dtype=np.int32).ravel()
    return A, M, u_to_W, p_to_W


def _artifact_status(filtered_dir: Path, seed_npy: Path) -> Dict[str, Any]:
    summary = filtered_dir / "diagnostics" / "mode_energy_summary.json"
    modes_dir = filtered_dir / "modes"
    result_glob = list((filtered_dir / "results").glob("result_*.json"))
    return {
        "filtered_dir": str(filtered_dir),
        "filtered_dir_exists": filtered_dir.is_dir(),
        "mode_energy_summary_exists": summary.is_file(),
        "modes_dir_exists": modes_dir.is_dir(),
        "result_json_count": len(result_glob),
        "seed_npy_exists": seed_npy.is_file(),
        "artifacts_ok": bool(
            filtered_dir.is_dir()
            and summary.is_file()
            and seed_npy.is_file()
            and modes_dir.is_dir()
        ),
    }


def main() -> int:
    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    case_dir = solve_case_dir("L_mid", CASE_ID)
    filtered_dir = case_dir / FILTERED_SUBDIR
    mesh_file = mesh_path("L_mid", CASE_ID)
    sample = sample_spec_from_case(case)
    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"

    status = _artifact_status(filtered_dir, seed_npy)
    generated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    report: Dict[str, Any] = {
        "generated_utc": generated,
        "evidence_scope": "requires_VM_runtime_artifact_evaluation",
        "case_id": CASE_ID,
        "seed_frequency_hz": SEED_F_HZ,
        "seed_file": str(seed_npy),
        "filter_policy": FILTER_POLICY,
        "artifact_status": status,
        "vm_operator_evidence": {
            "note": "Operator-reported VM facts are not read from this script; see merged root-cause audit.",
            "expected_saved_modes": VM_OPERATOR_EXPECTED_SAVED_MODES,
            "prior_unfiltered_false_recovery": {
                "file": "seed_branch_recovery_diagnostic/modes/mode_243075_004.smx.npz",
                "interpretation": "lambda≈1 sigma/mapping spurious; origin not localized to Dirichlet rows",
                "source": "reported_from_VM_operator_evidence",
            },
        },
        "candidates": [],
        "verdict": None,
        "verdict_pending_until_vm_run": True,
        "mesh_convergence_may_resume": False,
        "staged_status": {
            "mesh_convergence_pass": "Pending",
            "v2_production_promotion_ready": False,
            "lhs_promotion_blocked": True,
        },
    }

    if not status["artifacts_ok"]:
        report["verdict"] = VERDICT_FILTERED_INCONSISTENT
        report["verdict_reason"] = "Filtered diagnostic artifacts or seed not present at expected paths."
        report["runtime_evaluation_completed"] = False
        write_json(OUT_JSON, report)
        _write_md(report, pending=True)
        print(f"[filtered_eval] artifacts missing -> {report['verdict']}", flush=True)
        return 0

    summary = json.loads(
        (filtered_dir / "diagnostics" / "mode_energy_summary.json").read_text(encoding="utf-8")
    )
    modes = summary.get("modes") or []
    seed_w = _load_mode_vec(seed_npy)
    candidates: List[Dict[str, Any]] = []

    A, M, u_to_W, p_to_W = _assemble_replay(mesh_file, sample, filtered_dir)
    p_seed = np.asarray(seed_w[p_to_W], dtype=np.float64).ravel()
    try:
        for m in modes:
            rel = str(m.get("vector_path", ""))
            path = filtered_dir / rel
            if not path.is_file():
                path = case_dir / rel
            if not path.is_file():
                continue
            vec = _load_mode_vec(path)
            f_hz = float(m["frequency_hz"])
            mac = _pressure_mac(p_seed, np.asarray(vec[p_to_W], dtype=np.float64).ravel())
            replay = replay_candidate_metrics(
                A, M, vec, u_to_W=u_to_W, p_to_W=p_to_W, reported_f_hz=f_hz
            )
            elig = assess_physical_eligibility(
                reported_f_hz=f_hz,
                replay_metrics=replay,
                pressure_mac_to_true_seed=mac,
                seed_f_hz=SEED_F_HZ,
                require_mac=True,
                require_seed_frequency_match=True,
            )
            candidates.append(
                {
                    "candidate_index": m.get("mode_index"),
                    "vector_file": str(path),
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
                    "branch_recovery_pass": elig["branch_recovery_pass"],
                    "rejection_reasons": elig["rejection_reasons"],
                }
            )
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass

    candidates.sort(key=lambda r: int(r.get("candidate_index") or 0))
    artifacts_ok = len(candidates) > 0
    verdict = assign_filtered_evaluation_verdict(
        candidates,
        artifacts_ok=artifacts_ok,
        expected_mode_count=VM_OPERATOR_EXPECTED_SAVED_MODES,
    )

    report["candidates"] = candidates
    report["verdict"] = verdict
    report["verdict_pending_until_vm_run"] = False
    report["runtime_evaluation_completed"] = True
    report["evidence_scope"] = "VM_runtime_artifact_evaluation"
    report["summary"] = {
        "num_candidates_evaluated": len(candidates),
        "num_branch_recovery_pass": sum(1 for c in candidates if c.get("branch_recovery_pass")),
        "num_physically_eligible": sum(
            1 for c in candidates if c.get("physically_eligible_after_filter")
        ),
        "any_branch_recovery_pass": any(branch_recovery_from_row(c) for c in candidates),
    }
    report["verdict_reason"] = {
        VERDICT_FILTERED_BRANCH_RECOVERED: "At least one candidate passes branch_recovery_pass gates.",
        VERDICT_FILTERED_NO_BRANCH: "Candidates present but none pass branch_recovery_pass.",
        VERDICT_FILTERED_INCONSISTENT: "Missing/corrupt artifacts or candidate count mismatch.",
    }.get(verdict, "")

    write_json(OUT_JSON, report)
    _write_md(report, pending=False)
    print(f"[filtered_eval] verdict={verdict} n={len(candidates)}", flush=True)
    return 0


def _write_md(report: Dict[str, Any], *, pending: bool) -> None:
    lines = [
        "# L_mid filtered seed-branch recovery evaluation",
        "",
        f"Generated: {report.get('generated_utc')}",
        "",
    ]
    if pending:
        lines.extend(
            [
                "**Status:** VM runtime evaluation **pending** (artifacts not found at expected paths).",
                "",
                f"**Provisional verdict:** `{report.get('verdict')}`",
                "",
                f"Reason: {report.get('verdict_reason')}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "**Status:** VM runtime evaluation **completed**.",
                "",
                f"**Verdict:** `{report.get('verdict')}`",
                "",
                f"- Candidates evaluated: {report.get('summary', {}).get('num_candidates_evaluated')}",
                f"- branch_recovery_pass count: {report.get('summary', {}).get('num_branch_recovery_pass')}",
                "",
            ]
        )
        cands = report.get("candidates") or []
        if cands:
            lines.extend(
                [
                    "## Candidate details (7 saved modes)",
                    "",
                    "| candidate_index | vector_file | reported_f_hz | replay_lambda | replay_f_hz | residual | algebraic_lambda_one_suspect | physically_eligible | branch_recovery_pass | rejection_reasons |",
                    "|---|---|---:|---:|---:|---:|---|---|---|---|",
                ]
            )
            for c in cands:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(c.get("candidate_index")),
                            str(c.get("vector_file")).replace("\n", " "),
                            str(c.get("reported_frequency_hz")),
                            str(c.get("replay_rayleigh_lambda")),
                            str(c.get("replay_rayleigh_frequency_hz")),
                            str(c.get("replay_relative_residual")),
                            str(c.get("algebraic_lambda_one_suspect")),
                            str(c.get("physically_eligible_after_filter")),
                            str(c.get("branch_recovery_pass")),
                            ",".join(c.get("rejection_reasons") or []),
                        ]
                    )
                    + " |"
                )
    lines.append("**mesh_convergence_may_resume:** `False`\n")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
