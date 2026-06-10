#!/usr/bin/env python3
"""Prepare the first official ROM production batch via full-pool reset + normal LHS selection."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_full_lhs_pool_reset import (  # noqa: E402
    apply_full_lhs_pool_reset,
    plan_full_lhs_pool_reset,
    reset_pool_entries,
    verify_all_entries_pending,
)
from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    build_batch_sample_entry,
    load_lhs_pool,
    select_lhs_samples,
    specs_generated_dir,
)
from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
    DATASET_VERSION_REFERENCE,
    DATASET_VERSION_ROM,
    MESH_PROFILE_REFERENCE,
    MESH_PROFILE_ROM,
    checkpoint_export_mesh_level,
    resolve_mesh_profile,
)
from v2_b3_m4_shared_export import (  # noqa: E402
    sample_plots_destination_dir,
    summaries_destination_dir,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

DEFAULT_LHS_REL = "ROM/classic/lhs_pool.json"
OFFICIAL_MAX_SAMPLES = 5
OFFICIAL_RUN_ID_SUFFIX = "rom_official_v1"
OFFICIAL_BATCH_ID = "lhs_rom_official_v1_20260610"
PREP_SCHEMA = "m4_rom_official_batch_prepare_v2"


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


def simulate_post_reset_selection(
    *,
    pool: Mapping[str, Any],
    max_samples: int,
    run_id_suffix: str,
) -> List[Dict[str, Any]]:
    """Use the same selection logic as run_m4_production_pipeline after a full reset."""
    reset_pool = reset_pool_entries(dict(pool))
    status_doc = {"samples": {}}
    selection, _skipped = select_lhs_samples(
        reset_pool,
        status_doc,
        max_samples=max_samples,
        skip_completed=True,
        include_only_pending=True,
        run_id_suffix=run_id_suffix,
    )
    return selection


def verify_unique_run_roots(
    *,
    repo_root: Path,
    selection: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in selection:
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
                "lhs_index": row.get("lhs_row_index"),
                "sample_id": sid,
                "run_id": rid,
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
    max_samples: int,
    execute_reset: bool,
) -> Dict[str, Any]:
    lhs_rel = rel(lhs_path, repo_root=repo_root)
    pool = load_lhs_pool(lhs_path)
    mesh_default = resolve_mesh_profile()
    mesh_reference = resolve_mesh_profile(
        mesh_profile=MESH_PROFILE_REFERENCE,
        dataset_version=DATASET_VERSION_REFERENCE,
    )
    reset_plan = plan_full_lhs_pool_reset(
        repo_root=repo_root,
        lhs_path=lhs_path,
        run_id_suffix=run_id_suffix,
    )
    if execute_reset:
        reset_result = apply_full_lhs_pool_reset(
            repo_root=repo_root,
            lhs_path=lhs_path,
            run_id_suffix=run_id_suffix,
        )
        pool = load_lhs_pool(lhs_path)
    else:
        reset_result = None
        pool = reset_pool_entries(dict(pool))

    all_pending = verify_all_entries_pending(pool)
    selection = simulate_post_reset_selection(
        pool=pool,
        max_samples=max_samples,
        run_id_suffix=run_id_suffix,
    )
    run_checks = verify_unique_run_roots(repo_root=repo_root, selection=selection)

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
        for row in selection
    ]

    first_sample = str(selection[0]["sample_id"]) if selection else "sample_000"
    first_run = str(selection[0]["run_id"]) if selection else f"sample_000_{run_id_suffix}"

    return {
        "schema": PREP_SCHEMA,
        "generated_utc": utc_now(),
        "git_commit_sha": git_head_sha(repo_root),
        "batch_id": batch_id,
        "run_id_suffix": run_id_suffix,
        "max_samples": max_samples,
        "lhs_json": lhs_rel,
        "selection_mode": "normal_lhs_pipeline_order",
        "bounded_index_selection": False,
        "workers": 3,
        "mesh_profile_default": mesh_default.provenance_fields(),
        "mesh_profile_explicit_reference": mesh_reference.provenance_fields(),
        "stage_a_checkpoint_export_mesh_level": checkpoint_export_mesh_level(),
        "historical_run_trees_preserved": [
            "sample_002_rom_prod_004",
            "sample_002_m4prod2_strict_clean5",
        ],
        "reset_plan": reset_plan,
        "reset_executed": bool(execute_reset),
        "reset_result": reset_result,
        "all_entries_pending_after_reset": all_pending,
        "post_reset_pipeline_selection": mapping,
        "post_reset_selection_count": len(selection),
        "unique_run_root_checks": run_checks,
        "all_run_roots_unique_and_absent": all(bool(c.get("ok")) for c in run_checks),
        "shared_graph_structure_example": rel(
            sample_plots_destination_dir(
                shared_root=Path("/media/sf_gmar"),
                shape_name="classic",
                sample_id=first_sample,
            ),
            repo_root=repo_root,
        ),
        "shared_summaries_structure_example": rel(
            summaries_destination_dir(shared_root=Path("/media/sf_gmar"), shape_name="classic"),
            repo_root=repo_root,
        ),
        "graph_export_blocks_next_sample": True,
        "fem_launched": False,
    }


def render_launch_block(*, batch_id: str, run_id_suffix: str, max_samples: int) -> str:
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
            "GIT_SHA=$(git rev-parse HEAD)",
            "echo git_commit_sha=$GIT_SHA",
            "",
            "python FEM/experiments/active_domain_validation/physics_integrity/scripts/"
            "v2_b3_m4_full_lhs_pool_reset.py \\",
            "  --lhs-json ROM/classic/lhs_pool.json \\",
            f"  --run-id-suffix {run_id_suffix}",
            "",
            "python FEM/experiments/active_domain_validation/physics_integrity/scripts/"
            "v2_b3_m4_full_lhs_pool_reset.py \\",
            "  --lhs-json ROM/classic/lhs_pool.json \\",
            f"  --run-id-suffix {run_id_suffix} \\",
            "  --execute",
            "",
            "python FEM/experiments/active_domain_validation/physics_integrity/scripts/"
            "run_m4_production_pipeline.py \\",
            "  --lhs-json ROM/classic/lhs_pool.json \\",
            f"  --batch-id {batch_id} \\",
            f"  --run-id-suffix {run_id_suffix} \\",
            f"  --max-samples {max_samples} \\",
            "  --workers 3 \\",
            "  --mesh-profile rom \\",
            "  --dataset-version m4_geometry_corrected_rommesh_v1 \\",
            "  --strict-production \\",
            "  --compact-after-sample \\",
            "  --compact-blocking \\",
            "  --isolated-subprocess \\",
            "  --execute",
        ]
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare official ROM batch: full-pool reset + normal LHS selection (no FEM)."
    )
    parser.add_argument("--lhs-json", type=Path, default=Path(DEFAULT_LHS_REL))
    parser.add_argument("--batch-id", default=OFFICIAL_BATCH_ID)
    parser.add_argument("--run-id-suffix", default=OFFICIAL_RUN_ID_SUFFIX)
    parser.add_argument("--max-samples", type=int, default=OFFICIAL_MAX_SAMPLES)
    parser.add_argument(
        "--execute-reset",
        action="store_true",
        help="Apply full LHS pool reset on this machine (default: simulate only).",
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
        max_samples=int(args.max_samples),
        execute_reset=bool(args.execute_reset),
    )
    report_path = args.report_path
    if report_path is None:
        report_path = specs_generated_dir(repo_root) / f"{args.batch_id}_prepare.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_path, report)
    launch_path = report_path.with_name(f"{args.batch_id}_launch.sh")
    launch_path.write_text(
        render_launch_block(
            batch_id=str(args.batch_id),
            run_id_suffix=str(args.run_id_suffix),
            max_samples=int(args.max_samples),
        ),
        encoding="utf-8",
    )

    print(f"batch_id={report['batch_id']}")
    print(f"git_commit_sha={report['git_commit_sha']}")
    print(f"selection_mode={report['selection_mode']}")
    print("post_reset_pipeline_selection:")
    for row in report["post_reset_pipeline_selection"]:
        print(
            f"  lhs_index={row['lhs_index']} sample_id={row['sample_id']} "
            f"run_id={row['run_id']} run_root={row['run_root']}"
        )
    print(f"all_entries_pending_after_reset={report['all_entries_pending_after_reset']}")
    print(f"all_run_roots_unique_and_absent={report['all_run_roots_unique_and_absent']}")
    print(f"reset_executed={str(report['reset_executed']).lower()}")
    print(f"report={rel(report_path, repo_root=repo_root)}")
    print(f"launch_script={rel(launch_path, repo_root=repo_root)}")
    print("fem_launched=false")
    ok = (
        report["all_entries_pending_after_reset"]
        and report["all_run_roots_unique_and_absent"]
        and report["post_reset_selection_count"] == int(args.max_samples)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
