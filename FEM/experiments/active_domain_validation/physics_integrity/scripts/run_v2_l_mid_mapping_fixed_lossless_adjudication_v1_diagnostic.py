#!/usr/bin/env python3
"""
Isolated lossless adjudication v1: one EPS solve + lossless-authoritative evaluation.

Requires --authorize-single-eps-run and passing preflight gate contract.
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

SCRIPT_DIR = Path(__file__).resolve().parent
FEM_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_clean_adjudication_lane import OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1
from v2_clean_lane_preflight_gate import validate_gate_contract_for_eps_authorization
from v2_lossless_adjudication_evaluator import (
    VERDICT_BRANCH_RECOVERED,
    VERDICT_INCONSISTENT,
    VERDICT_LOSSLESS_ROUNDTRIP_FAILURE,
    VERDICT_PERSISTENCE_FAILURE,
    evaluate_lossless_adjudication_artifacts,
)
from v2_mesh_convergence_common import CONV_DIAG, load_manifest, mesh_path, sample_spec_from_case, solve_case_dir, write_json
from v2_sensitivity_common import REPO_ROOT, hz_result_tag

CASE_ID = "baseline_coupled_v2"
NUM_MODES = 64
SOLVE_SCRIPT = SCRIPT_DIR / "v2_sensitivity_solve.py"
SEED_F_HZ = 243.0754171175576
PREFLIGHT_JSON = CONV_DIAG / "v2_lossless_adjudication_v1_policy_equivalence_preflight.json"
REPORT_JSON = CONV_DIAG / "v2_l_mid_mapping_fixed_unregularized_lossless_adjudication_v1_diagnostic.json"
REPORT_MD = CONV_DIAG / "v2_l_mid_mapping_fixed_unregularized_lossless_adjudication_v1_diagnostic.md"
AUTH_RECORD_JSON = CONV_DIAG / "v2_lossless_adjudication_v1_eps_authorization_record.json"


def _out_dir(case_dir: Path) -> Path:
    return case_dir / OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1


def _load_gate() -> Dict[str, Any]:
    if not PREFLIGHT_JSON.is_file():
        return {}
    return json.loads(PREFLIGHT_JSON.read_text(encoding="utf-8"))


def _load_solve_result(out_dir: Path, target_hz: float) -> Dict[str, Any]:
    result_path = out_dir / "results" / f"result_{hz_result_tag(target_hz)}.json"
    if not result_path.is_file():
        results = sorted((out_dir / "results").glob("result_*.json"))
        if not results:
            return {}
        result_path = results[-1]
    return json.loads(result_path.read_text(encoding="utf-8"))


def _eps_already_executed(out_dir: Path) -> bool:
    if AUTH_RECORD_JSON.is_file():
        try:
            rec = json.loads(AUTH_RECORD_JSON.read_text(encoding="utf-8"))
            if int(rec.get("eps_run_count_for_this_lane", 0)) >= 1:
                return True
        except Exception:
            pass
    modes = out_dir / "modes"
    if not modes.is_dir():
        return False
    n_dense = len(list(modes.glob("candidate_eps_slot_*.smx.dense.npy")))
    n_result = len(list((out_dir / "results").glob("result_*.json"))) if (out_dir / "results").is_dir() else 0
    return n_dense > 0 and n_result > 0


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
    log_path = out_dir / "logs" / "lossless_adjudication_v1_eps.log"
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
        "--seed-branch-lossless-adjudication-v1",
    ]
    write_json(
        out_dir / "launch_record.json",
        {
            "exact_command_argv": cmd,
            "shell_command": shlex.join(cmd),
            "output_case_dir": str(out_dir),
            "lossless_adjudication_v1": True,
            "single_eps_authorized": True,
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


def _write_md(report: Dict[str, Any]) -> None:
    ev = report.get("evaluation") or {}
    lines = [
        "# L_mid lossless adjudication v1 (isolated EPS)",
        "",
        f"Generated: {report.get('generated_utc')}",
        f"**Verdict:** `{ev.get('diagnostic_verdict')}`",
        "",
        f"**preflight_gate_pass:** `{report.get('preflight_gate_pass')}`",
        f"**eps_run_count_for_this_lane:** `{report.get('eps_run_count_for_this_lane')}`",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authorize-single-eps-run",
        action="store_true",
        help="Required for EPS execution.",
    )
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    gate = _load_gate()
    ok, issues = validate_gate_contract_for_eps_authorization(gate)
    if not ok:
        print(f"[lossless_adjudication_v1] pre_eps_gate_failed issues={issues}", file=sys.stderr)
        return 2

    if not args.authorize_single_eps_run and not args.evaluate_only:
        print(
            "[lossless_adjudication_v1] No EPS: pass --authorize-single-eps-run after gate confirmation.",
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

    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_id": CASE_ID,
        "output_subdir": OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1,
        "preflight_gate_pass": True,
        "single_lossless_adjudication_run_authorized": bool(args.authorize_single_eps_run),
        "no_additional_eps_run_authorized": True,
        "production_vector_fidelity_exposure": "OPEN",
        "mesh_convergence_may_resume": False,
    }

    if not args.evaluate_only and _eps_already_executed(out_dir):
        print(
            "[lossless_adjudication_v1] ABORT: isolated tree already has EPS artifacts; "
            "no_additional_eps_run_authorized=True",
            file=sys.stderr,
        )
        return 2

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
                "verdict_reason": "eps_solve_failed_or_empty_result",
            }
            write_json(REPORT_JSON, report)
            _write_md(report)
            return 2

    report["eps_run_count_for_this_lane"] = 1
    report["evaluation"] = evaluate_lossless_adjudication_artifacts(
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
        AUTH_RECORD_JSON,
        {
            "generated_utc": report["generated_utc"],
            "eps_run_count_for_this_lane": 1,
            "no_additional_eps_run_authorized": True,
            "output_subdir": OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1,
            "diagnostic_verdict": report["evaluation"].get("diagnostic_verdict"),
        },
    )
    _write_md(report)

    verdict = report["evaluation"].get("diagnostic_verdict", VERDICT_INCONSISTENT)
    print(f"[lossless_adjudication_v1] verdict={verdict}", flush=True)
    if verdict in (VERDICT_PERSISTENCE_FAILURE, VERDICT_LOSSLESS_ROUNDTRIP_FAILURE, VERDICT_INCONSISTENT):
        return 2
    if verdict == VERDICT_BRANCH_RECOVERED:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
