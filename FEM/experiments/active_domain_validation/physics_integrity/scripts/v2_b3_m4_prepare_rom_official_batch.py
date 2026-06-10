#!/usr/bin/env python3
"""Prepare the first official small ROM production batch (indexes 0-4 inclusive)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_bounded_lhs_reset import (  # noqa: E402
    apply_bounded_lhs_reset,
    plan_bounded_lhs_reset,
    sample_id_for_index,
)
from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    build_batch_sample_entry,
    build_lhs_batch_spec,
    load_lhs_pool,
    specs_generated_dir,
)
from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
    DATASET_VERSION_REFERENCE,
    DATASET_VERSION_ROM,
    LEVEL_PROD_REFERENCE,
    LEVEL_ROM_PROD,
    MESH_PROFILE_REFERENCE,
    MESH_PROFILE_ROM,
    checkpoint_export_mesh_level,
    resolve_mesh_profile,
)
from v2_b3_m4_shared_export import (  # noqa: E402
    export_graphs_fixture,
    graphs_destination_dir,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402
from v2_mesh_convergence_common import mesh_path  # noqa: E402

DEFAULT_LHS_REL = "ROM/classic/lhs_pool.json"
OFFICIAL_START_INDEX = 0
OFFICIAL_END_INDEX = 4
OFFICIAL_RUN_ID_SUFFIX = "rom_official_v1"
OFFICIAL_BATCH_ID = "lhs_rom_official_v1_20260610"
PRESERVED_VALIDATION_RUN_IDS = ("sample_002_rom_prod_004",)
PREP_SCHEMA = "m4_rom_official_batch_prepare_v1"


def git_head_sha(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip()


def guitars_run_root(repo_root: Path, sample_id: str, run_id: str) -> Path:
    return (
        repo_root
        / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
        / sample_id
        / "runs"
        / run_id
    )


def build_official_sample_rows(
    *,
    pool: Mapping[str, Any],
    batch_id: str,
    lhs_source_path: str,
    start_index: int,
    end_index: int,
    run_id_suffix: str,
) -> List[Dict[str, Any]]:
    entries = list(pool.get("entries") or [])
    rows: List[Dict[str, Any]] = []
    for i in range(start_index, end_index + 1):
        if i >= len(entries):
            raise ValueError(f"lhs index {i} out of range for pool entries (len={len(entries)})")
        entry = entries[i]
        sid = str(entry.get("id") or sample_id_for_index(i))
        run_id = f"{sid}_{run_id_suffix}"
        rows.append(
            build_batch_sample_entry(
                pool=pool,
                entry=entry,
                lhs_row_index=i,
                run_id_suffix=run_id_suffix,
                batch_id=batch_id,
                lhs_source_path=lhs_source_path,
                mesh_profile=MESH_PROFILE_ROM,
                dataset_version=DATASET_VERSION_ROM,
            )
        )
    return rows


def verify_unique_run_roots(
    *,
    repo_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        sid = str(row["sample_id"])
        rid = str(row["run_id"])
        if rid in seen:
            checks.append(
                {
                    "sample_id": sid,
                    "run_id": rid,
                    "unique_run_id": False,
                    "run_root_exists": None,
                    "ok": False,
                    "error": "duplicate_run_id_in_batch",
                }
            )
            continue
        seen.add(rid)
        root = guitars_run_root(repo_root, sid, rid)
        exists = root.exists()
        checks.append(
            {
                "sample_id": sid,
                "run_id": rid,
                "lhs_index": row.get("lhs_row_index"),
                "run_root": rel(root, repo_root=repo_root),
                "run_root_abs": str(root),
                "unique_run_id": True,
                "run_root_exists": exists,
                "ok": not exists,
                "error": "run_root_already_exists" if exists else None,
            }
        )
    return checks


def build_prepare_report(
    *,
    repo_root: Path,
    lhs_path: Path,
    batch_id: str,
    run_id_suffix: str,
    start_index: int,
    end_index: int,
    execute_reset: bool,
) -> Dict[str, Any]:
    lhs_rel = rel(lhs_path, repo_root=repo_root)
    pool = load_lhs_pool(lhs_path)
    mesh_default = resolve_mesh_profile()
    mesh_reference = resolve_mesh_profile(
        mesh_profile=MESH_PROFILE_REFERENCE,
        dataset_version=DATASET_VERSION_REFERENCE,
    )
    reset_plan = plan_bounded_lhs_reset(
        repo_root=repo_root,
        lhs_path=lhs_path,
        start_index=start_index,
        end_index=end_index,
        preserved_run_ids=PRESERVED_VALIDATION_RUN_IDS,
        run_id_suffix=run_id_suffix,
    )
    if execute_reset:
        reset_result = apply_bounded_lhs_reset(
            repo_root=repo_root,
            lhs_path=lhs_path,
            start_index=start_index,
            end_index=end_index,
            preserved_run_ids=PRESERVED_VALIDATION_RUN_IDS,
            run_id_suffix=run_id_suffix,
        )
        pool = load_lhs_pool(lhs_path)
    else:
        reset_result = None

    sample_rows = build_official_sample_rows(
        pool=pool,
        batch_id=batch_id,
        lhs_source_path=lhs_rel,
        start_index=start_index,
        end_index=end_index,
        run_id_suffix=run_id_suffix,
    )
    run_checks = verify_unique_run_roots(repo_root=repo_root, rows=sample_rows)
    batch_spec = build_lhs_batch_spec(
        pool=pool,
        samples=sample_rows,
        batch_id=batch_id,
        lhs_source_path=lhs_rel,
        run_id_suffix=run_id_suffix,
        mesh_profile=MESH_PROFILE_ROM,
        dataset_version=DATASET_VERSION_ROM,
    )
    gen_dir = specs_generated_dir(repo_root)
    gen_dir.mkdir(parents=True, exist_ok=True)
    spec_path = gen_dir / f"{batch_id}.json"
    write_json_atomic(spec_path, batch_spec)

    example_sid = str(sample_rows[0]["sample_id"])
    rom_mesh_out = mesh_path(LEVEL_ROM_PROD, example_sid)
    ref_mesh_out = mesh_path(LEVEL_PROD_REFERENCE, example_sid)

    mapping = [
        {
            "lhs_index": row["lhs_row_index"],
            "sample_id": row["sample_id"],
            "run_id": row["run_id"],
            "run_root": rel(
                guitars_run_root(repo_root, str(row["sample_id"]), str(row["run_id"])),
                repo_root=repo_root,
            ),
        }
        for row in sample_rows
    ]

    return {
        "schema": PREP_SCHEMA,
        "generated_utc": utc_now(),
        "git_commit_sha": git_head_sha(repo_root),
        "batch_id": batch_id,
        "run_id_suffix": run_id_suffix,
        "lhs_json": lhs_rel,
        "spec_path": rel(spec_path, repo_root=repo_root),
        "start_index": start_index,
        "end_index": end_index,
        "index_5_plus_excluded": True,
        "workers": 3,
        "mesh_profile_default": mesh_default.provenance_fields(),
        "mesh_profile_explicit_reference": mesh_reference.provenance_fields(),
        "stage_a_checkpoint_export_mesh_level": checkpoint_export_mesh_level(),
        "preserved_validation_runs": list(PRESERVED_VALIDATION_RUN_IDS),
        "reset_plan": reset_plan,
        "reset_executed": bool(execute_reset),
        "reset_result": reset_result,
        "sample_mapping": mapping,
        "unique_run_root_checks": run_checks,
        "all_run_roots_unique_and_absent": all(bool(c.get("ok")) for c in run_checks),
        "shared_graph_structure_example": rel(
            graphs_destination_dir(
                shared_root=Path("/media/sf_gmar"),
                shape_name="classic",
                sample_id=str(sample_rows[0]["sample_id"]),
                run_id=str(sample_rows[0]["run_id"]),
            ),
            repo_root=repo_root,
        ),
        "expected_mesh_output_path": rel(rom_mesh_out, repo_root=repo_root),
        "reference_mesh_output_path": rel(ref_mesh_out, repo_root=repo_root),
        "cleanup_then_compaction_order": [
            "graph_export_before_cleanup",
            "preserve_target_plan_before_cleanup",
            "preserve_comparison_provenance_before_cleanup",
            "minimal_rom_compaction",
            "delete_shared_sample_artifacts",
            "verify_cleanup_barrier",
        ],
        "graph_export_blocks_next_sample": True,
        "fem_launched": False,
    }


def render_launch_block(*, batch_id: str, run_id_suffix: str) -> str:
    return "\n".join(
        [
            "cd ~/final-project",
            "git pull --ff-only",
            "source .venv/bin/activate",
            "",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export NUMEXPR_NUM_THREADS=1",
            "",
            f"GIT_SHA=$(git rev-parse HEAD)",
            f"echo git_commit_sha=$GIT_SHA",
            "",
            "python FEM/experiments/active_domain_validation/physics_integrity/scripts/"
            "v2_b3_m4_bounded_lhs_reset.py \\",
            "  --lhs-json ROM/classic/lhs_pool.json \\",
            f"  --start-index {OFFICIAL_START_INDEX} \\",
            f"  --end-index {OFFICIAL_END_INDEX} \\",
            "  --preserve-run-id sample_002_rom_prod_004 \\",
            f"  --run-id-suffix {run_id_suffix} \\",
            "  --execute",
            "",
            "python FEM/experiments/active_domain_validation/physics_integrity/scripts/"
            "run_m4_production_pipeline.py \\",
            "  --lhs-json ROM/classic/lhs_pool.json \\",
            f"  --batch-id {batch_id} \\",
            f"  --run-id-suffix {run_id_suffix} \\",
            f"  --start-index {OFFICIAL_START_INDEX} \\",
            f"  --end-index {OFFICIAL_END_INDEX} \\",
            "  --max-samples 5 \\",
            "  --workers 3 \\",
            "  --mesh-profile rom \\",
            "  --dataset-version m4_geometry_corrected_rommesh_v1 \\",
            "  --strict-production \\",
            "  --compact-after-sample \\",
            "  --compact-blocking \\",
            "  --isolated-subprocess \\",
            "  --no-skip-completed \\",
            "  --execute",
        ]
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare official ROM production batch for LHS indexes 0-4 (no FEM launch)."
    )
    parser.add_argument("--lhs-json", type=Path, default=Path(DEFAULT_LHS_REL))
    parser.add_argument("--batch-id", default=OFFICIAL_BATCH_ID)
    parser.add_argument("--run-id-suffix", default=OFFICIAL_RUN_ID_SUFFIX)
    parser.add_argument("--start-index", type=int, default=OFFICIAL_START_INDEX)
    parser.add_argument("--end-index", type=int, default=OFFICIAL_END_INDEX)
    parser.add_argument(
        "--execute-reset",
        action="store_true",
        help="Apply bounded LHS reset for the selected index range.",
    )
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    lhs_path = args.lhs_json if args.lhs_json.is_absolute() else repo_root / args.lhs_json
    if not lhs_path.is_file():
        print(f"error: missing --lhs-json: {lhs_path}", file=sys.stderr)
        return 2

    report = build_prepare_report(
        repo_root=repo_root,
        lhs_path=lhs_path,
        batch_id=str(args.batch_id),
        run_id_suffix=str(args.run_id_suffix),
        start_index=int(args.start_index),
        end_index=int(args.end_index),
        execute_reset=bool(args.execute_reset),
    )
    report_path = args.report_path
    if report_path is None:
        report_path = specs_generated_dir(repo_root) / f"{args.batch_id}_prepare.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_path, report)
    launch_path = report_path.with_name(f"{args.batch_id}_launch.sh")
    launch_path.write_text(render_launch_block(batch_id=str(args.batch_id), run_id_suffix=str(args.run_id_suffix)), encoding="utf-8")

    print(f"batch_id={report['batch_id']}")
    print(f"git_commit_sha={report['git_commit_sha']}")
    print("sample_mapping:")
    for row in report["sample_mapping"]:
        print(
            f"  lhs_index={row['lhs_index']} sample_id={row['sample_id']} "
            f"run_id={row['run_id']} run_root={row['run_root']}"
        )
    print(f"index_5_plus_excluded={str(report['index_5_plus_excluded']).lower()}")
    print(f"all_run_roots_unique_and_absent={report['all_run_roots_unique_and_absent']}")
    print(f"reset_executed={str(report['reset_executed']).lower()}")
    print(f"report={rel(report_path, repo_root=repo_root)}")
    print(f"launch_script={rel(launch_path, repo_root=repo_root)}")
    print("fem_launched=false")
    return 0 if report["all_run_roots_unique_and_absent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
