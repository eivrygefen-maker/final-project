#!/usr/bin/env python3
"""
L_mid baseline-only seeded-branch recovery diagnostic (experiment-only solver mode).

Uses sigma centered on validated seed Rayleigh frequency — not production harvest policy.
Does not rerun standard seeded_retrieval or acoustic locator.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FEM_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_seed_branch_candidate_filter import (
    FILTER_POLICY,
    VERDICT_SPURIOUS_SELECTED,
    assess_physical_eligibility,
    replay_candidate_metrics,
)
from v2_mesh_convergence_common import (
    CONV_DIAG,
    load_manifest,
    mesh_path,
    sample_spec_from_case,
    solve_case_dir,
    write_json,
)
from v2_sensitivity_common import REPO_ROOT, hz_result_tag

REPORT_JSON = CONV_DIAG / "v2_l_mid_seed_branch_recovery_diagnostic.json"
REPORT_MD = CONV_DIAG / "v2_l_mid_seed_branch_recovery_diagnostic.md"
SOLVE_SCRIPT = SCRIPT_DIR / "v2_sensitivity_solve.py"

CASE_ID = "baseline_coupled_v2"
FREQ_TOL_FRAC = float(FILTER_POLICY["seed_frequency_match_max_relative"])
MAC_TOL = float(FILTER_POLICY["pressure_mac_to_true_seed_min"])
REPLAY_RESIDUAL_OK = float(FILTER_POLICY["replay_relative_residual_max"])
NUM_MODES = 48


def _diag_output_dir(case_dir: Path, *, filtered: bool = False) -> Path:
    if filtered:
        return case_dir / "seed_branch_recovery_diagnostic_filtered"
    return case_dir / "seed_branch_recovery_diagnostic"


def _load_prior_standard_run(case_dir: Path) -> Dict[str, Any]:
    """Summarize completed standard seeded_retrieval (no re-run)."""
    retrieval = case_dir / "seeded_retrieval"
    out: Dict[str, Any] = {
        "standard_seeded_retrieval_dir": str(retrieval),
        "continuation_seed_applied": None,
        "normal_harvest_policy_did_not_recover_seed_branch": None,
        "nconv": None,
        "standard_st_sigma_hz": None,
        "standard_harvest_band_hz": None,
    }
    result_paths = list((retrieval / "results").glob("result_*.json")) if retrieval.is_dir() else []
    if not result_paths:
        return out
    res = json.loads(result_paths[0].read_text(encoding="utf-8"))
    eps = res.get("eps_batch_diagnostics") or {}
    out["continuation_seed_applied"] = bool(
        eps.get("continuation_seed_applied") or (res.get("eps_seed") or {}).get("eps_initial_space_set")
    )
    out["nconv"] = int(res.get("nconv", eps.get("nconv_marked", -1)))
    out["standard_harvest_band_hz"] = res.get("harvest_band_hz")
    out["standard_st_sigma_hz"] = float(res.get("st_sigma_hz_used", float("nan")))
    seed_f = float(res.get("target_hz", float("nan")))
    modes = []
    summary = retrieval / "diagnostics" / "mode_energy_summary.json"
    if summary.is_file():
        modes = json.loads(summary.read_text(encoding="utf-8")).get("modes") or []
    near = [
        m
        for m in modes
        if math.isfinite(seed_f)
        and abs(float(m.get("frequency_hz", 0)) - seed_f) / seed_f <= FREQ_TOL_FRAC
    ]
    out["normal_harvest_policy_did_not_recover_seed_branch"] = len(near) == 0
    out["modes_near_seed_in_standard_harvest_hz"] = len(near)
    return out


def _validate_prelaunch(
    mesh_file: Path,
    seed_npy: Path,
    seed_meta: Dict[str, Any],
) -> Dict[str, Any]:
    errors: List[str] = []
    n_w = int(seed_meta.get("n_reduced_W", 0))
    seed_len = None
    if seed_npy.is_file():
        seed_len = int(np.load(str(seed_npy)).size)
        if n_w > 0 and seed_len != n_w:
            errors.append(f"seed length {seed_len} != meta n_reduced_W {n_w}")
    else:
        errors.append(f"missing seed: {seed_npy}")
    if not mesh_file.is_file():
        errors.append(f"missing mesh: {mesh_file}")
    cli = subprocess.run(
        [sys.executable, str(SOLVE_SCRIPT), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    help_text = (cli.stdout or "") + (cli.stderr or "")
    if "--seed-branch-recovery-diagnostic" not in help_text:
        errors.append("v2_sensitivity_solve missing --seed-branch-recovery-diagnostic")
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "seed_vector_length": seed_len,
        "seed_layout_valid": bool(seed_meta.get("seed_layout_valid")),
    }


def _run_diagnostic_solve(
    sample: Dict[str, Any],
    mesh_file: Path,
    *,
    target_hz: float,
    seed_npy: Path,
    out_dir: Path,
    filtered_harvest: bool = False,
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
    ]
    if filtered_harvest:
        cmd.append("--seed-branch-filtered-diagnostic")
    launch_record = {
        "exact_command_argv": cmd,
        "shell_command": shlex.join(cmd),
        "working_directory": str(REPO_ROOT.resolve()),
        "mesh_path": str(mesh_file),
        "seed_file_path": str(seed_npy),
        "output_case_dir": str(out_dir),
        "log_path": str(log_path),
    }
    write_json(out_dir / "launch_record.json", launch_record)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env, check=False
    )
    log_body = "\n".join(
        [
            f"# seed_branch_recovery_diagnostic worker log",
            f"return_code={completed.returncode}",
            f"command={shlex.join(cmd)}",
            "",
            "=== stdout ===",
            completed.stdout or "",
            "",
            "=== stderr ===",
            completed.stderr or "",
        ]
    )
    log_path.write_text(log_body, encoding="utf-8")
    launch_record["return_code"] = int(completed.returncode)
    launch_record["log_bytes"] = log_path.stat().st_size if log_path.is_file() else 0
    write_json(out_dir / "launch_record.json", launch_record)

    result_path = out_dir / "results" / f"result_{hz_result_tag(target_hz)}.json"
    result: Dict[str, Any] = {}
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    return int(completed.returncode), result, log_body


def _assemble_operators(mesh_file: Path, sample: Dict[str, Any], tag: str):
    import fem_main_3d as fem3d
    from v2_build_coupled_acoustic_seed import _assemble_reduced_coupled_replay

    sort_dir = _diag_output_dir(solve_case_dir("L_mid", CASE_ID)) / "sorting_replay" / tag
    sort_dir.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sort_dir.resolve())
    A, M, cfg = _assemble_reduced_coupled_replay(mesh_file, sample, coupling_enabled=True)
    u_to_W = np.asarray(cfg["_coupled_air_u_to_W_map"], dtype=np.int32).ravel()
    p_to_W = np.asarray(cfg["_coupled_air_p_to_W_map"], dtype=np.int32).ravel()
    return A, M, u_to_W, p_to_W


def _load_mode_vec(out_dir: Path, rel: str) -> np.ndarray:
    from fem_mode_array_utils import load_mode_column_any

    return np.asarray(load_mode_column_any(out_dir / rel).toarray(), dtype=np.float64).ravel()


def _evaluate(
    out_dir: Path,
    mesh_file: Path,
    sample: Dict[str, Any],
    seed: np.ndarray,
    seed_meta: Dict[str, Any],
    solve_result: Dict[str, Any],
) -> Dict[str, Any]:
    seed_f = float(seed_meta.get("locator_frequency_hz", solve_result.get("target_hz", float("nan"))))
    diag_block = solve_result.get("seed_branch_recovery_diagnostic") or {}
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

    p_seed_block = np.asarray(seed[p_to_W], dtype=np.float64).ravel()
    A, M, u_idx, p_idx = _assemble_operators(mesh_file, sample, "evaluate_all")
    ranked: List[Dict[str, Any]] = []
    try:
        for m in modes:
            try:
                vec = _load_mode_vec(out_dir, str(m["vector_path"]))
            except Exception:
                continue
            f_hz = float(m["frequency_hz"])
            p_mode_block = np.asarray(vec[p_to_W], dtype=np.float64).ravel()
            mac_block = float(
                abs(np.vdot(p_seed_block, p_mode_block))
                / (float(np.linalg.norm(p_seed_block)) * float(np.linalg.norm(p_mode_block)))
                if float(np.linalg.norm(p_seed_block)) > 0 and float(np.linalg.norm(p_mode_block)) > 0
                else float("nan")
            )
            d_frac = abs(f_hz - seed_f) / seed_f if seed_f > 0 else float("inf")
            replay = replay_candidate_metrics(
                A, M, vec, u_to_W=u_idx, p_to_W=p_idx, reported_f_hz=f_hz
            )
            elig = assess_physical_eligibility(
                reported_f_hz=f_hz,
                replay_metrics=replay,
                pressure_mac_to_true_seed=mac_block,
                seed_f_hz=seed_f,
                require_mac=True,
                require_seed_frequency_match=True,
            )
            ranked.append(
                {
                    **m,
                    "pressure_MAC_to_seed_p_block": mac_block,
                    "pressure_MAC_to_true_acoustic_reference": mac_block,
                    "frequency_delta_from_seed_rayleigh_hz": f_hz - seed_f,
                    "frequency_delta_fraction": float(d_frac),
                    "replay_rayleigh_eigenvalue": replay["replay_rayleigh_lambda"],
                    "replay_rayleigh_f_hz": replay["replay_rayleigh_frequency_hz"],
                    "replay_relative_residual_of_recovered_mode": replay["replay_relative_residual"],
                    "algebraic_lambda_one_suspect": replay["algebraic_lambda_one_suspect"],
                    "reported_vs_replay_frequency_consistent": replay[
                        "reported_vs_replay_frequency_consistent"
                    ],
                    "physically_eligible_after_filter": elig["physically_eligible_after_filter"],
                    "rejection_reasons": elig["rejection_reasons"],
                    "recovers_true_seed_branch": elig["recovers_true_seed_branch"],
                }
            )
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass

    eligible = [r for r in ranked if r.get("physically_eligible_after_filter")]
    branch_pool = [r for r in eligible if r.get("recovers_true_seed_branch")]
    best = (
        max(branch_pool, key=lambda r: float(r["pressure_MAC_to_seed_p_block"]))
        if branch_pool
        else (
            max(eligible, key=lambda r: float(r["pressure_MAC_to_seed_p_block"]))
            if eligible
            else {}
        )
    )

    prior_false = [
        r
        for r in ranked
        if float(r.get("frequency_delta_fraction", float("inf"))) <= FREQ_TOL_FRAC
        and not r.get("physically_eligible_after_filter")
    ]

    mac = float(best.get("pressure_MAC_to_seed_p_block", float("nan")))
    d_frac = float(best.get("frequency_delta_fraction", float("inf")))
    rel_res = float(best.get("replay_relative_residual_of_recovered_mode", float("nan")))
    lam_one = bool(best.get("algebraic_lambda_one_suspect", False))
    continuation = bool(
        solve_result.get("continuation_seed_applied")
        or (solve_result.get("eps_seed") or {}).get("eps_initial_space_set")
        or diag_block.get("continuation_seed_applied")
    )
    recovery_ok = bool(best and best.get("recovers_true_seed_branch") and continuation)

    p_frac = float(best.get("p_frac_energy_phys", float("nan")))
    energy_note = None
    if recovery_ok and p_frac < 0.05:
        energy_note = "branch recovered by true-reference metrics; p_frac classification reported separately"

    any_ineligible_near_seed = len(prior_false) > 0
    if not continuation:
        verdict = "DIAGNOSTIC_SOLVER_NOT_APPLIED"
    elif recovery_ok:
        verdict = "SEED_BRANCH_RECOVERED_IN_DIAGNOSTIC_MODE"
    elif any_ineligible_near_seed or (ranked and not eligible):
        verdict = VERDICT_SPURIOUS_SELECTED
    else:
        verdict = "SEED_BRANCH_NOT_RECOVERED_EVEN_IN_DIAGNOSTIC_MODE"

    return {
        "seed_rayleigh_f_hz": seed_f,
        "physical_filter_policy": FILTER_POLICY,
        "solver_mode": diag_block.get("solver_mode", "seeded_branch_recovery_diagnostic"),
        "standard_harvest_sigma_policy_unchanged": True,
        "standard_policy_not_used_for_this_diagnostic": True,
        "seed_file_used": str(solve_result.get("eps_seed", {}).get("seed_file_used")),
        "seed_layout_valid": solve_result.get("eps_seed", {}).get("seed_layout_valid"),
        "continuation_seed_applied": continuation,
        "diagnostic_sigma_hz": diag_block.get("diagnostic_sigma_hz"),
        "diagnostic_sigma_retry_ladder_hz": diag_block.get("diagnostic_sigma_retry_ladder_hz"),
        "diagnostic_local_band_hz": diag_block.get("diagnostic_local_band_hz"),
        "diagnostic_harvest_window_hz": diag_block.get("diagnostic_harvest_window_hz"),
        "st_sigma_hz_used": solve_result.get("st_sigma_hz_used"),
        "nconv": solve_result.get("nconv"),
        "num_modes_saved": solve_result.get("num_modes_saved"),
        "recovered_mode": best,
        "all_candidates_with_physical_filter": ranked,
        "physically_eligible_candidates": eligible,
        "prior_reported_near_seed_but_ineligible": prior_false,
        "any_existing_saved_candidate_recovers_true_seed_branch": bool(branch_pool),
        "recovery_success": recovery_ok,
        "energy_classification_note": energy_note,
        "diagnostic_verdict": verdict,
    }


def _write_md(report: Dict[str, Any]) -> None:
    row = report.get("baseline_coupled_v2") or {}
    prior = report.get("prior_standard_seeded_retrieval") or {}
    ev = row.get("evaluation") or {}
    rec = ev.get("recovered_mode") or {}
    lines = [
        "# L_mid seed-branch recovery diagnostic (baseline)",
        "",
        f"Generated: {report.get('generated_utc')}",
        "",
        "## Prior standard seeded retrieval (already executed — not re-run)",
        "",
        f"- continuation_seed_applied: {prior.get('continuation_seed_applied')}",
        f"- EPS nconv: {prior.get('nconv')}",
        f"- standard ST sigma: {prior.get('standard_st_sigma_hz')} Hz",
        f"- standard harvest band: {prior.get('standard_harvest_band_hz')}",
        f"- normal_harvest_policy_did_not_recover_seed_branch: "
        f"{prior.get('normal_harvest_policy_did_not_recover_seed_branch')}",
        "",
        "## Diagnostic mode run",
        "",
        f"- **Verdict:** `{ev.get('diagnostic_verdict')}`",
        f"- solver_mode: {ev.get('solver_mode')}",
        f"- diagnostic_sigma_hz: {ev.get('diagnostic_sigma_hz')}",
        f"- continuation_seed_applied: {ev.get('continuation_seed_applied')}",
        f"- recovered f: {rec.get('frequency_hz')} Hz, MAC={rec.get('pressure_MAC_to_true_acoustic_reference')}",
        f"- p_frac: {rec.get('p_frac_energy_phys')} ({rec.get('mode_class_physical_energy')})",
        f"- replay residual: {rec.get('replay_relative_residual_of_recovered_mode')}",
        "",
        f"Log: `{row.get('log_path')}`",
        "",
        f"**mesh_convergence_may_resume:** `{report.get('mesh_convergence_may_resume')}`",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true", default=True)
    parser.add_argument(
        "--filtered-harvest",
        action="store_true",
        help="Run filtered diagnostic EPS (sigma-spurious reject on); separate output dir.",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    case_dir = solve_case_dir("L_mid", CASE_ID)
    mesh_file = mesh_path("L_mid", CASE_ID)
    out_dir = _diag_output_dir(case_dir, filtered=bool(args.filtered_harvest))
    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"
    seed_meta_path = case_dir / "diagnostics" / "acoustic_coupled_seed_meta.json"
    seed_meta = json.loads(seed_meta_path.read_text(encoding="utf-8")) if seed_meta_path.is_file() else {}
    target_hz = float(seed_meta.get("locator_frequency_hz", 243.0754171175576))

    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_id": CASE_ID,
        "prior_standard_seeded_retrieval": _load_prior_standard_run(case_dir),
        "mesh_convergence_may_resume": False,
        "staged_status": {
            "mesh_convergence_pass": "Pending",
            "v2_production_promotion_ready": False,
            "lhs_promotion_blocked": True,
        },
    }

    pre = _validate_prelaunch(mesh_file, seed_npy, seed_meta)
    if not pre["valid"]:
        report["baseline_coupled_v2"] = {
            "diagnostic_verdict": "DIAGNOSTIC_SOLVER_NOT_APPLIED",
            "prelaunch": pre,
        }
        write_json(REPORT_JSON, report)
        _write_md(report)
        return 1

    seed = np.load(str(seed_npy))
    sample = sample_spec_from_case(case)
    rc, solve_result, _log = _run_diagnostic_solve(
        sample,
        mesh_file,
        target_hz=target_hz,
        seed_npy=seed_npy,
        out_dir=out_dir,
        filtered_harvest=bool(args.filtered_harvest),
    )
    evaluation = _evaluate(out_dir, mesh_file, sample, seed, seed_meta, solve_result)
    report["baseline_coupled_v2"] = {
        "solve_exit_code": rc,
        "output_dir": str(out_dir),
        "log_path": str(out_dir / "logs" / "seed_branch_recovery_diagnostic.log"),
        "launch_record": str(out_dir / "launch_record.json"),
        "evaluation": evaluation,
        "diagnostic_verdict": evaluation["diagnostic_verdict"],
    }
    write_json(REPORT_JSON, report)
    _write_md(report)
    print(f"[seed_branch_diag] wrote {REPORT_JSON}", flush=True)
    return 0 if evaluation["diagnostic_verdict"] == "SEED_BRANCH_RECOVERED_IN_DIAGNOSTIC_MODE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
