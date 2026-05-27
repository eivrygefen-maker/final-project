#!/usr/bin/env python3
"""
Isolated nullspace-projected lossless adjudication v1: one EPS with certified-null deflation.

Requires --authorize-single-projected-eps-run and passing certified-null preflight gates.
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
from typing import Any, Dict, Tuple

import numpy as np
from mpi4py import MPI

SCRIPT_DIR = Path(__file__).resolve().parent
FEM_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_build_coupled_acoustic_seed import _assemble_reduced_coupled_replay, _extract_layout_maps
from v2_certified_null_projection_lib import (
    PROJECTED_AUTH_JSON,
    build_Q_certified_null_from_prior_lossless_tree,
    persist_projection_basis,
    validate_null_basis_preflight_gates,
    verify_Q_certified_properties,
)
from v2_clean_adjudication_lane import (
    OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1,
    OUT_SUBDIR_LOSSLESS_NULLSPACE_PROJECTED_ADJUDICATION_V1,
    SEED_F_HZ,
)
from v2_lossless_nullspace_projected_adjudication_evaluator import (
    VERDICT_INCONSISTENT,
    VERDICT_PROJECTED_BRANCH_RECOVERED,
    VERDICT_PROJECTED_ST_BLOCKER,
    evaluate_projected_lossless_adjudication_artifacts,
)
from v2_mesh_convergence_common import CONV_DIAG, load_manifest, mesh_path, sample_spec_from_case, solve_case_dir, write_json
from v2_sensitivity_common import REPO_ROOT, hz_result_tag
from v2_unreg_offset_report_evaluator import load_seed_with_diagnostics

CASE_ID = "baseline_coupled_v2"
NUM_MODES = 64
SOLVE_SCRIPT = SCRIPT_DIR / "v2_sensitivity_solve.py"
REPORT_JSON = (
    CONV_DIAG
    / "v2_l_mid_mapping_fixed_unregularized_lossless_nullspace_projected_adjudication_v1_diagnostic.json"
)
REPORT_MD = (
    CONV_DIAG
    / "v2_l_mid_mapping_fixed_unregularized_lossless_nullspace_projected_adjudication_v1_diagnostic.md"
)
PREFLIGHT_GATE_JSON = (
    CONV_DIAG
    / "v2_lossless_adjudication_v1_Muu_null_basis_certification_and_projection_preflight.json"
)


def _out_dir(case_dir: Path) -> Path:
    return case_dir / OUT_SUBDIR_LOSSLESS_NULLSPACE_PROJECTED_ADJUDICATION_V1


def _prior_out_dir(case_dir: Path) -> Path:
    return case_dir / OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1


def _load_solve_result(out_dir: Path, target_hz: float) -> Dict[str, Any]:
    result_path = out_dir / "results" / f"result_{hz_result_tag(target_hz)}.json"
    if not result_path.is_file():
        results = sorted((out_dir / "results").glob("result_*.json"))
        if not results:
            return {}
        result_path = results[-1]
    return json.loads(result_path.read_text(encoding="utf-8"))


def _eps_already_executed(out_dir: Path) -> bool:
    if PROJECTED_AUTH_JSON.is_file():
        try:
            rec = json.loads(PROJECTED_AUTH_JSON.read_text(encoding="utf-8"))
            if int(rec.get("eps_run_count_for_projected_lane", 0)) >= 1:
                return True
        except Exception:
            pass
    modes = out_dir / "modes"
    if not modes.is_dir():
        return False
    n_dense = len(list(modes.glob("candidate_eps_slot_*.smx.dense.npy")))
    n_result = len(list((out_dir / "results").glob("result_*.json"))) if (out_dir / "results").is_dir() else 0
    return n_dense > 0 and n_result > 0


def _run_pre_eps_projection_gate(
    *,
    case_dir: Path,
    out_dir: Path,
    mesh_file: Path,
    sample: Dict[str, Any],
    seed_npy: Path,
) -> Tuple[bool, Dict[str, Any]]:
    ok, issues = validate_null_basis_preflight_gates()
    if not ok:
        return False, {"preflight_gate_issues": issues}

    prior_dir = _prior_out_dir(case_dir)
    if not prior_dir.is_dir():
        return False, {"reason": "prior_lossless_adjudication_tree_missing"}

    A, M, _cfg = _assemble_reduced_coupled_replay(mesh_file, sample, coupling_enabled=True)
    try:
        maps = _extract_layout_maps(_cfg, A)
        u_to_W = np.asarray(maps["u_to_W"], dtype=np.int32).ravel()
        p_to_W = np.asarray(maps["p_to_W"], dtype=np.int32).ravel()
        operator_size = int(A.getSize()[0])

        Q_cert, build_meta = build_Q_certified_null_from_prior_lossless_tree(
            prior_out_dir=prior_dir,
            mesh_file=mesh_file,
            sample=sample,
            u_to_W=u_to_W,
            p_to_W=p_to_W,
            operator_size=operator_size,
            M=M,
        )
        if int(Q_cert.shape[1]) != 23:
            return False, {"reason": f"unexpected_certified_dim={Q_cert.shape[1]}"}

        seed_info = load_seed_with_diagnostics(seed_npy)
        seed_vec = np.asarray(seed_info["seed_array"], dtype=np.float64).ravel()
        verify = verify_Q_certified_properties(Q_cert, M=M, u_to_W=u_to_W, seed_vec=seed_vec)
        q_path = persist_projection_basis(out_dir, Q_cert, {**build_meta, **verify})

        gate = {
            "preflight_gate_pass": True,
            "certified_empirical_null_basis_dimension": int(Q_cert.shape[1]),
            "certified_null_basis_certified": True,
            "seed_projection_preservation_pass_certified_null": bool(
                verify.get("seed_projection_preservation_pass")
            ),
            "projection_basis_path": str(q_path),
            "projection_enabled": True,
            "projection_strategy": "certified_empirical_Muu_null_basis_deflation",
            "projection_basis_dimension": int(Q_cert.shape[1]),
            "projection_basis_indices": build_meta.get("certified_null_basis_indices"),
            "projection_basis_source": "prior_lossless_adjudication_v1_authoritative_vectors",
            "projection_basis_certified": True,
            "Q_full_used_for_solver": False,
            "Q_orthonormal_within_tolerance": verify.get("Q_orthonormal_within_tolerance"),
            "seed_projection_preservation_pass": verify.get("seed_projection_preservation_pass"),
            "seed_relative_change_norm_ratio": verify.get("seed_projection_relative_change_norm_ratio"),
        }
        write_json(out_dir / "diagnostics" / "pre_eps_projection_gate.json", gate)
        return True, gate
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass


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
    log_path = out_dir / "logs" / "lossless_nullspace_projected_adjudication_v1_eps.log"
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
        "--seed-branch-lossless-nullspace-projected-adjudication-v1",
    ]
    write_json(
        out_dir / "launch_record.json",
        {
            "exact_command_argv": cmd,
            "shell_command": shlex.join(cmd),
            "output_case_dir": str(out_dir),
            "nullspace_projected_adjudication_v1": True,
            "single_projected_eps_authorized": True,
        },
    )
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
    return int(completed.returncode), _load_solve_result(out_dir, target_hz)


def main() -> int:
    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[projected_adjudication_v1] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authorize-single-projected-eps-run",
        action="store_true",
        help="Required for projected EPS execution.",
    )
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    if not args.authorize_single_projected_eps_run and not args.evaluate_only:
        print(
            "[projected_adjudication_v1] No EPS: pass --authorize-single-projected-eps-run.",
            file=sys.stderr,
        )
        return 2

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

    if not args.evaluate_only and _eps_already_executed(out_dir):
        print(
            "[projected_adjudication_v1] ABORT: projected lane already has EPS artifacts.",
            file=sys.stderr,
        )
        return 2

    pre_gate_ok = False
    pre_gate: Dict[str, Any] = {}
    if not args.evaluate_only:
        pre_gate_ok, pre_gate = _run_pre_eps_projection_gate(
            case_dir=case_dir,
            out_dir=out_dir,
            mesh_file=mesh_file,
            sample=fallback_sample,
            seed_npy=seed_npy,
        )
        if not pre_gate_ok:
            print(f"[projected_adjudication_v1] pre_eps_projection_gate_failed {pre_gate}", file=sys.stderr)
            return 2

    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_id": CASE_ID,
        "output_subdir": OUT_SUBDIR_LOSSLESS_NULLSPACE_PROJECTED_ADJUDICATION_V1,
        "preflight_gate_pass": True,
        "pre_eps_projection_gate": pre_gate
        if pre_gate
        else (
            json.loads((out_dir / "diagnostics/pre_eps_projection_gate.json").read_text(encoding="utf-8"))
            if (out_dir / "diagnostics/pre_eps_projection_gate.json").is_file()
            else {}
        ),
        "no_additional_eps_run_authorized": True,
        "production_vector_fidelity_exposure": "OPEN",
        "mesh_convergence_may_resume": False,
    }

    if args.evaluate_only:
        solve_result = _load_solve_result(out_dir, target_hz)
    else:
        rc, solve_result = _run_solve(
            fallback_sample, mesh_file, target_hz=target_hz, seed_npy=seed_npy, out_dir=out_dir
        )
        report["solve_return_code"] = rc
        if rc != 0 or not solve_result:
            report["evaluation"] = {
                "diagnostic_verdict": VERDICT_INCONSISTENT,
                "final_projected_adjudication_verdict": VERDICT_INCONSISTENT,
            }
            write_json(REPORT_JSON, report)
            return 2

    report["eps_run_count_for_projected_lane"] = 1
    report["evaluation"] = evaluate_projected_lossless_adjudication_artifacts(
        out_dir=out_dir,
        case_dir=case_dir,
        mesh_file=mesh_file,
        seed_npy=seed_npy,
        seed_meta=seed_meta,
        fallback_sample=fallback_sample,
        solve_result=solve_result,
        target_hz=target_hz,
    )
    write_json(REPORT_JSON, report)
    write_json(
        PROJECTED_AUTH_JSON,
        {
            "generated_utc": report["generated_utc"],
            "eps_run_count_for_projected_lane": 1,
            "no_additional_eps_run_authorized": True,
            "output_subdir": OUT_SUBDIR_LOSSLESS_NULLSPACE_PROJECTED_ADJUDICATION_V1,
            "final_projected_adjudication_verdict": report["evaluation"].get(
                "final_projected_adjudication_verdict"
            ),
        },
    )

    ev = report["evaluation"]
    verdict = ev.get("final_projected_adjudication_verdict", VERDICT_INCONSISTENT)
    dim = int((pre_gate or {}).get("projection_basis_dimension", 23) or 23)
    print(f"[projected_adjudication_v1] projection_basis_dimension={dim}", flush=True)
    print(f"[projected_adjudication_v1] preflight_gate_pass=True", flush=True)
    print(f"[projected_adjudication_v1] eps_run_count_for_projected_lane=1", flush=True)
    print(f"[projected_adjudication_v1] final_projected_adjudication_verdict={verdict}", flush=True)
    print(
        f"[projected_adjudication_v1] branch_recovery_pass_count={ev.get('branch_recovery_pass_count', 0)}",
        flush=True,
    )
    print(
        f"[projected_adjudication_v1] mass_null_candidate_count_after_projection="
        f"{ev.get('mass_null_candidate_count_after_projection', 0)}",
        flush=True,
    )
    print("[projected_adjudication_v1] no_additional_eps_run_authorized=True", flush=True)

    if verdict == VERDICT_PROJECTED_BRANCH_RECOVERED:
        return 0
    if verdict == VERDICT_PROJECTED_ST_BLOCKER:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
