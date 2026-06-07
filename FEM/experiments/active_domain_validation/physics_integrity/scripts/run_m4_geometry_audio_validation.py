#!/usr/bin/env python3
"""
Isolated M4 geometry/audio validation (experimental — does not touch production).

Test A: provenance audit on production runs.
Test B: sample-mesh Stage A + aperture proxy + narrow-band solve on validation run trees.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_DIR = SCRIPT_DIR.parent / "docs"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import DEFAULT_RUN_ID_SUFFIX, load_lhs_pool  # noqa: E402
from v2_b3_m4_validation_lib import (  # noqa: E402
    BAND_281,
    BAND_390,
    _production_legacy_peak,
    attach_aperture_mask,
    build_narrow_band_chunk_targets,
    collect_checkpoint_report,
    collect_solve_band_results,
    evaluate_validation_gates,
    prepare_validation_run_tree,
    validation_run_id,
    validation_tree_ready,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, rel  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

DEFAULT_LHS = "ROM/classic/lhs_pool.json"
DEFAULT_SAMPLES = ("sample_001", "sample_034")


def _parse_samples(arg: str) -> List[str]:
    return [p.strip() for p in arg.split(",") if p.strip()]


def _repo_rel(path: Path, repo_root: Path) -> str:
    return rel(path, repo_root=repo_root)


def run_test_a(repo_root: Path, samples: Sequence[str]) -> Dict[str, Any]:
    json_out = DOCS_DIR / "M4_OPERATOR_PROVENANCE_AUDIT.json"
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "audit_m4_operator_provenance.py"),
        "--samples",
        ",".join(samples),
        "--dolfinx",
        "--json-out",
        str(json_out),
    ]
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    result: Dict[str, Any] = {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr[-8000:] if proc.stderr else "",
        "json_out": _repo_rel(json_out, repo_root),
    }
    if json_out.is_file():
        try:
            result["audit"] = json.loads(json_out.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return result


def _stage_a_command(
    *,
    repo_root: Path,
    val_root: Path,
    generated_mesh: Path,
    core_config: Path,
) -> List[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "v2_b3_m4_validation_checkpoint_export.py"),
        "--mesh-level",
        "L_prod",
        "--output-dir",
        str(val_root / "lprod" / "checkpoint"),
        "--core-config",
        str(core_config),
        "--use-sample-operator-mesh",
        str(generated_mesh),
        "--B3-synthesis-region-dofs",
        "best_effort",
    ]


def _narrow_band_solve_command(
    *,
    val_root: Path,
    targets_json: Path,
) -> List[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "v2_b3_checkpoint_solve_target_list.py"),
        "--checkpoint-dir",
        str(val_root / "lprod" / "checkpoint"),
        "--targets-json",
        str(targets_json),
        "--output-dir",
        str(val_root / "validation" / "narrow_band_solve"),
        "--factor-solver",
        "mkl_pardiso",
        "--nev",
        "8",
        "--ncv",
        "16",
    ]


def run_test_b_sample(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    sample_id: str,
    run_id_suffix: str,
    execute: bool,
    load_dolfinx: bool,
    reuse_validation_checkpoint: bool = False,
    mask_and_solve_only: bool = False,
) -> Dict[str, Any]:
    run_id = validation_run_id(sample_id)
    out: Dict[str, Any] = {
        "sample_id": sample_id,
        "validation_run_id": run_id,
        "status": "planned",
    }
    try:
        prep = prepare_validation_run_tree(
            repo_root=repo_root,
            pool=pool,
            sample_id=sample_id,
            run_id_suffix=run_id_suffix,
            reuse_existing=reuse_validation_checkpoint,
        )
    except FileNotFoundError as exc:
        out["status"] = "FAIL"
        out["error"] = str(exc)
        return out

    val_root: Path = prep["validation_run_root"]
    prod_root: Path = prep["production_run_root"]
    generated_mesh: Path = prep["generated_mesh_path"]
    core_config: Path = prep["resolved_core_config"]

    out["validation_run_root"] = _repo_rel(val_root, repo_root)
    out["production_run_root"] = _repo_rel(prod_root, repo_root)
    out["generated_mesh_path"] = _repo_rel(generated_mesh, repo_root)
    out["operator_mesh_path"] = out["generated_mesh_path"]
    from v2_b3_m4_validation_lib import _sha256_file  # noqa: WPS433

    out["generated_mesh_sha256"] = _sha256_file(generated_mesh)

    stage_a_cmd = _stage_a_command(
        repo_root=repo_root,
        val_root=val_root,
        generated_mesh=generated_mesh,
        core_config=core_config,
    )
    out["stage_a_command"] = " ".join(stage_a_cmd)

    targets_doc = build_narrow_band_chunk_targets(sample_id=sample_id, run_id=run_id)
    targets_path = val_root / "validation" / "narrow_band_targets.json"
    targets_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(targets_path, targets_doc)
    solve_cmd = _narrow_band_solve_command(val_root=val_root, targets_json=targets_path)
    out["narrow_band_solve_command"] = " ".join(solve_cmd)
    out["narrow_band_solve_env"] = {
        "B3_MIC_PROXY_MODE": "aperture_pressure_rms_v1",
        "B3_EXPERIMENTAL_APERTURE_MASK_NPZ": str(val_root / "validation" / "aperture_pressure_mask.npz"),
    }

    if not execute:
        out["status"] = "commands_ready"
        return out

    skip_stage_a = bool(
        reuse_validation_checkpoint
        and validation_tree_ready(val_root)
        and (val_root / "lprod" / "checkpoint" / "built_metadata.json").is_file()
    )
    out["reuse_validation_checkpoint"] = skip_stage_a
    if mask_and_solve_only:
        skip_stage_a = True

    if not skip_stage_a:
        proc_a = subprocess.run(stage_a_cmd, cwd=str(repo_root), capture_output=True, text=True)
        out["stage_a_returncode"] = proc_a.returncode
        out["stage_a_stdout"] = proc_a.stdout[-4000:]
        out["stage_a_stderr"] = proc_a.stderr[-4000:]
        if proc_a.returncode != 0:
            out["status"] = "FAIL"
            out["error"] = "validation_stage_a_failed"
            return out
    else:
        out["stage_a_skipped"] = True

    ckpt_report = collect_checkpoint_report(
        repo_root=repo_root,
        val_root=val_root,
        generated_mesh=generated_mesh,
        load_dolfinx=load_dolfinx,
    )
    out.update(ckpt_report)

    if not out.get("operator_mesh_matches_generated"):
        out["status"] = "FAIL"
        out["error"] = "operator_mesh_matches_generated=false"
        return out

    try:
        mask_summary = attach_aperture_mask(
            val_root=val_root,
            generated_mesh=generated_mesh,
            pool=pool,
            sample_id=sample_id,
            core_config_path=core_config,
            write_diagnostics=True,
        )
    except Exception as exc:  # noqa: BLE001
        out["status"] = "FAIL"
        out["error"] = f"aperture_mask_failed:{type(exc).__name__}:{exc}"
        return out

    out["p_idx_aperture_count"] = int(mask_summary.get("n_p_aperture_dofs") or 0)
    out["aperture_coordinate_bbox_min"] = mask_summary.get("coordinate_bbox_min")
    out["aperture_coordinate_bbox_max"] = mask_summary.get("coordinate_bbox_max")
    out["aperture_mask_method"] = mask_summary.get("mask_method")
    out["mic_output_method"] = mask_summary.get("mic_output_method")
    out["aperture_mask_npz"] = mask_summary.get("mask_npz_path")
    out["narrow_band_solve_env"]["B3_EXPERIMENTAL_APERTURE_MASK_NPZ"] = str(
        mask_summary.get("mask_npz_path") or out["narrow_band_solve_env"]["B3_EXPERIMENTAL_APERTURE_MASK_NPZ"]
    )

    legacy_281 = _production_legacy_peak(prod_root, BAND_281)
    legacy_390 = _production_legacy_peak(prod_root, BAND_390)
    out["old_mic_proxy_281"] = (legacy_281 or {}).get("mic_output_proxy")
    out["old_mic_proxy_390"] = (legacy_390 or {}).get("mic_output_proxy")
    out["old_mic_method_281"] = (legacy_281 or {}).get("mic_output_method")
    out["production_peak_281_hz"] = (legacy_281 or {}).get("frequency_hz")
    out["production_peak_390_hz"] = (legacy_390 or {}).get("frequency_hz")

    env = os.environ.copy()
    env.update(out["narrow_band_solve_env"])
    proc_s = subprocess.run(solve_cmd, cwd=str(repo_root), capture_output=True, text=True, env=env)
    out["narrow_band_solve_returncode"] = proc_s.returncode
    out["narrow_band_solve_stdout"] = proc_s.stdout[-4000:]
    out["narrow_band_solve_stderr"] = proc_s.stderr[-4000:]
    if proc_s.returncode != 0:
        out["status"] = "FAIL"
        out["error"] = "narrow_band_solve_failed"
        return out

    solve_results = collect_solve_band_results(val_root / "validation" / "narrow_band_solve" / "solver_result.json")
    out.update(solve_results)
    out["deduped_modes_270_290_hz"] = solve_results.get("deduped_modes_270_290_hz") or []
    out["deduped_modes_380_400_hz"] = solve_results.get("deduped_modes_380_400_hz") or []

    if out["deduped_modes_270_290_hz"]:
        peak = max(
            out["deduped_modes_270_290_hz"],
            key=lambda m: float(m.get("mic_output_proxy") or 0.0),
        )
        out["new_aperture_mic_proxy_281"] = peak.get("mic_output_proxy")
        out["new_mic_method_281"] = peak.get("mic_output_method")
        out["validation_peak_281_hz"] = peak.get("frequency_hz")
    if out["deduped_modes_380_400_hz"]:
        peak390 = max(
            out["deduped_modes_380_400_hz"],
            key=lambda m: float(m.get("mic_output_proxy") or 0.0),
        )
        out["new_aperture_mic_proxy_390"] = peak390.get("mic_output_proxy")
        out["new_mic_method_390"] = peak390.get("mic_output_method")
        out["validation_peak_390_hz"] = peak390.get("frequency_hz")

    out["status"] = "PASS" if out.get("solver_status") == "PASS" else "FAIL"
    return out


def _production_recommendation(validation_pass: bool) -> Dict[str, Any]:
    if not validation_pass:
        return {
            "decision": "VALIDATION_INCOMPLETE",
            "action": "Run Test B with --execute on VM; do not change production until validation_pass=true.",
        }
    return {
        "decision": "RERUN_ALL_35_SAMPLES",
        "required_production_changes": [
            "Stage A: resolve operator mesh from run-tree lprod/mesh/L_prod/sample_XXX.msh (not baseline_coupled_v2)",
            "Region DOF export: add p_idx_aperture to region_dof_indices.npz at checkpoint export",
            "Audio coupling: default mic_output_method=aperture_pressure_rms_proxy_v1",
            "Aggregation: keep 0.05 Hz dedupe; audit raw duplicate accepts per chunk",
        ],
        "migration_commands": [
            "# 1) Mark production 000-035 pending re-run in tracking (do not auto-delete catalogs)",
            "# 2) After code fix merged, rerun batch:",
            "python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py \\",
            "  --lhs-json ROM/classic/lhs_pool.json --samples 0-35 --workers 3 --execute \\",
            "  --continue-on-fail --run-rom-prepredict --run-rom-compare --rom-nonblocking",
            "# 3) Do NOT resume sample_036 until validation_pass on new pipeline",
            "# 4) Discard ROM artifacts trained on pre-fix catalogs; rebuild after new aggregation",
            "rm -rf FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/*/runs/*/rom/",
        ],
        "estimated_rerun_hours_sequential": 26.25,
        "invalidate_fields": ["frequency_hz", "mic_output_proxy", "ROM intensity targets"],
        "retain_for_audit": ["existing aggregation/", "M4_OPERATOR_PROVENANCE_AUDIT.json"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M4 geometry/audio validation (isolated).")
    parser.add_argument("--lhs-json", type=Path, default=Path(DEFAULT_LHS))
    parser.add_argument("--samples", default=",".join(DEFAULT_SAMPLES))
    parser.add_argument("--run-id-suffix", default=DEFAULT_RUN_ID_SUFFIX)
    parser.add_argument("--test-a-only", action="store_true")
    parser.add_argument("--test-b-only", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run validation Stage A + narrow-band solve (requires DOLFINx/PETSc on VM).",
    )
    parser.add_argument(
        "--reuse-validation-checkpoint",
        action="store_true",
        help="Skip Stage A when sample_XXX_geometryfix_validation checkpoint already exists.",
    )
    parser.add_argument(
        "--mask-and-solve-only",
        action="store_true",
        help="With --execute: only rebuild aperture mask + narrow-band solve (no Stage A).",
    )
    parser.add_argument("--dolfinx", action="store_true", default=True)
    parser.add_argument("--no-dolfinx", action="store_true")
    parser.add_argument("--json-out", type=Path, default=DOCS_DIR / "M4_GEOMETRY_AUDIO_VALIDATION.json")
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    pool = load_lhs_pool(args.lhs_json if args.lhs_json.is_absolute() else repo_root / args.lhs_json)
    samples = _parse_samples(str(args.samples))
    load_dolfinx = bool(args.dolfinx) and not bool(args.no_dolfinx)

    report: Dict[str, Any] = {
        "schema": "m4_geometry_audio_validation_v2",
        "samples": samples,
        "execute": bool(args.execute),
        "test_a_results": None,
        "test_b_results": [],
        "validation_pass": None,
        "decision": None,
        "gate_failures": [],
    }

    if not args.test_b_only:
        report["test_a_results"] = run_test_a(repo_root, samples)
        print(report["test_a_results"].get("stdout", ""))

    if not args.test_a_only:
        report["test_b_results"] = [
            run_test_b_sample(
                repo_root=repo_root,
                pool=pool,
                sample_id=sid,
                run_id_suffix=str(args.run_id_suffix),
                execute=bool(args.execute),
                load_dolfinx=load_dolfinx,
                reuse_validation_checkpoint=bool(args.reuse_validation_checkpoint or args.mask_and_solve_only),
                mask_and_solve_only=bool(args.mask_and_solve_only),
            )
            for sid in samples
        ]
        for row in report["test_b_results"]:
            print(
                f"test_b {row.get('sample_id')}: status={row.get('status')} "
                f"mesh_match={row.get('operator_mesh_matches_generated')} "
                f"aperture_dofs={row.get('p_idx_aperture_count')}"
            )
            if row.get("narrow_band_solve_command"):
                print(f"  solve_cmd: {row['narrow_band_solve_command']}")
                if row.get("narrow_band_solve_env"):
                    for k, v in row["narrow_band_solve_env"].items():
                        print(f"  export {k}={v}")

    if report["test_b_results"]:
        if args.execute:
            passed, failures = evaluate_validation_gates(report["test_b_results"])
            report["gate_failures"] = failures
            report["validation_pass"] = bool(passed)
            rec = _production_recommendation(bool(passed))
            report["decision"] = rec.get("decision")
            report["production_recommendation"] = rec
        else:
            report["validation_pass"] = None
            report["gate_failures"] = []
            report["decision"] = "COMMANDS_READY_RUN_WITH_EXECUTE"
            report["production_recommendation"] = _production_recommendation(False)
    else:
        report["decision"] = "TEST_A_ONLY"

    out = args.json_out if args.json_out.is_absolute() else repo_root / args.json_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"validation_report={_repo_rel(out, repo_root)}")
    print(f"validation_pass={report.get('validation_pass')}")
    print(f"decision={report.get('decision')}")
    if report.get("gate_failures"):
        print(f"gate_failures={report['gate_failures']}")
    return 0 if report.get("validation_pass") is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
