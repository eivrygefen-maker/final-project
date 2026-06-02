#!/usr/bin/env python3
"""Create a canonical B3 pipeline run manifest (M1.5, manifest-only helper)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_petsc_util import write_json_atomic  # noqa: E402

SCHEMA = "b3_pipeline_run_manifest_v1"
ALLOWED_MODES = ("timing", "rich", "synthesis")
ALLOWED_STATUSES = {"PENDING", "PASS", "FAIL", "SKIPPED"}

REPO_ROOT = SCRIPT_DIR.parents[4]
PHYSICS_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = (
    "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs"
)
MESH_CONVERGENCE_MANIFEST = (
    "FEM/experiments/active_domain_validation/physics_integrity/configs/v2_mesh_convergence_manifest.json"
)
CORE_CONFIG = (
    "FEM/experiments/active_domain_validation/physics_integrity/configs/coupled_physical_core_v2.json"
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _utc_run_id_prefix() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _resolve_root(path_arg: str, *, is_default: bool = False) -> Path:
    p = Path(path_arg).expanduser()
    resolved = p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()
    if is_default and not _is_within(resolved, REPO_ROOT):
        raise ValueError(
            "default --output-root resolved outside repository root; "
            f"resolved={resolved} repo_root={REPO_ROOT}"
        )
    return resolved


def _mk_run_id(*, run_id: Optional[str], tag: Optional[str]) -> str:
    if run_id:
        return str(run_id).strip()
    suffix = str(tag).strip() if tag else "manual"
    suffix = suffix.replace(" ", "_")
    return f"{_utc_run_id_prefix()}_{suffix}"


def _validate_status(value: str, *, field: str) -> str:
    v = str(value).strip().upper()
    if v not in ALLOWED_STATUSES:
        raise ValueError(f"invalid {field}={value!r}; allowed: {sorted(ALLOWED_STATUSES)}")
    return v


def _stage_defaults(mode: str) -> Dict[str, Dict[str, Any]]:
    rich = mode in ("rich", "synthesis")
    c_status = "SKIPPED" if mode != "synthesis" else "PENDING"
    return {
        "A": {
            "status": "PENDING",
            "script": "scripts/v2_b3_checkpoint_export.py",
            "command": None,
            "checkpoint_dir": None,
            "export_manifest": None,
        },
        "B": {
            "status": "PENDING",
            "script": "scripts/v2_b3_checkpoint_solve.py",
            "command": None,
            "solve_dir": None,
            "result_json": None,
            "rich_modal_requested": bool(rich),
            "rich_modal_dir": None,
        },
        "C": {
            "status": c_status,
            "script": "scripts/v2_b3_rich_modal_post.py",
            "command": None,
            "synthesis_dir": None,
            "modes_synthesis_json": None,
        },
    }


def _manifest_for_args(args: argparse.Namespace) -> Dict[str, Any]:
    run_id = _mk_run_id(run_id=args.run_id, tag=args.tag)
    mode = str(args.mode).strip().lower()
    stages = _stage_defaults(mode)
    rich = mode in ("rich", "synthesis")
    c_requested = mode == "synthesis"

    if args.checkpoint_dir:
        ckpt = str(Path(args.checkpoint_dir).expanduser().resolve())
        stages["A"]["checkpoint_dir"] = ckpt
        stages["A"]["export_manifest"] = str(
            (Path(ckpt) / "checkpoint_export_manifest.json").resolve()
        )
        stages["A"]["status"] = "PASS"

    if args.solve_dir:
        solve = str(Path(args.solve_dir).expanduser().resolve())
        stages["B"]["solve_dir"] = solve
        stages["B"]["result_json"] = str((Path(solve) / "result.json").resolve())
        if rich:
            stages["B"]["rich_modal_dir"] = str((Path(solve) / "rich_modal").resolve())
        stages["B"]["status"] = "PASS"

    if args.synthesis_dir:
        syn = str(Path(args.synthesis_dir).expanduser().resolve())
        stages["C"]["synthesis_dir"] = syn
        stages["C"]["modes_synthesis_json"] = str(
            (Path(syn) / "modes_synthesis.json").resolve()
        )
        stages["C"]["status"] = "PASS"
    elif c_requested:
        stages["C"]["status"] = "PENDING"

    if args.stage_a_status:
        stages["A"]["status"] = _validate_status(args.stage_a_status, field="stage_a_status")
    if args.stage_b_status:
        stages["B"]["status"] = _validate_status(args.stage_b_status, field="stage_b_status")
    if args.stage_c_status:
        stages["C"]["status"] = _validate_status(args.stage_c_status, field="stage_c_status")

    # Enforce linking rules.
    if stages["B"]["status"] == "PASS" and not stages["A"]["checkpoint_dir"]:
        raise ValueError("Stage B PASS requires --checkpoint-dir (Stage A reference)")
    if c_requested and not rich:
        raise ValueError("synthesis mode requires rich export policy")
    if stages["C"]["status"] in ("PENDING", "PASS") and not stages["B"]["rich_modal_requested"]:
        raise ValueError("Stage C requires rich modal requested in Stage B")
    if stages["C"]["status"] in ("PENDING", "PASS") and not stages["B"]["solve_dir"]:
        raise ValueError("Stage C requires --solve-dir (Stage B reference)")
    if stages["C"]["status"] == "PASS" and not stages["C"]["synthesis_dir"]:
        raise ValueError("Stage C PASS requires --synthesis-dir")

    return {
        "schema": SCHEMA,
        "run_id": run_id,
        "created_utc": _utc_now(),
        "source": {
            "mesh_level": str(args.mesh_level),
            "mesh_convergence_manifest": MESH_CONVERGENCE_MANIFEST,
            "core_config": CORE_CONFIG,
        },
        "policy": {
            "mode": mode,
            "rich_export": bool(rich),
            "stage_c_requested": bool(c_requested),
            "selection_reason": str(args.selection_reason or "manual"),
        },
        "stages": stages,
        "environment": {
            "stage_a_env": "production_venv",
            "stage_b_env": "solver_mkl",
            "stage_c_env": "production_venv",
        },
    }


def _append_index(index_path: Path, payload: Dict[str, Any]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": payload["run_id"],
        "created_utc": payload["created_utc"],
        "mode": payload["policy"]["mode"],
        "selection_reason": payload["policy"]["selection_reason"],
        "stage_a_status": payload["stages"]["A"]["status"],
        "stage_b_status": payload["stages"]["B"]["status"],
        "stage_c_status": payload["stages"]["C"]["status"],
        "checkpoint_dir": payload["stages"]["A"]["checkpoint_dir"],
        "solve_dir": payload["stages"]["B"]["solve_dir"],
        "synthesis_dir": payload["stages"]["C"]["synthesis_dir"],
        "rich_modal_requested": payload["stages"]["B"]["rich_modal_requested"],
    }
    with index_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def run_manifest_cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create B3 pipeline run manifest only (no stage execution)."
    )
    parser.add_argument("--run-id", help="Explicit run id. Default: <utc>_<tag>")
    parser.add_argument("--tag", help="Tag used when --run-id is omitted.")
    parser.add_argument("--mode", choices=ALLOWED_MODES, required=True)
    parser.add_argument("--mesh-level", default="L_prod")
    parser.add_argument("--selection-reason", default="manual")
    parser.add_argument("--checkpoint-dir", help="Optional existing Stage A checkpoint directory")
    parser.add_argument("--solve-dir", help="Optional existing Stage B solve directory")
    parser.add_argument("--synthesis-dir", help="Optional existing Stage C synthesis directory")
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Root for manifest/log/index folders. "
            "Default: FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs"
        ),
    )
    parser.add_argument(
        "--append-index",
        action="store_true",
        help="Append one-line summary to output-root/index/runs_index.jsonl",
    )
    parser.add_argument("--force", action="store_true", help="Allow overwriting existing manifest path")
    parser.add_argument("--stage-a-status", help="Optional explicit status override for Stage A")
    parser.add_argument("--stage-b-status", help="Optional explicit status override for Stage B")
    parser.add_argument("--stage-c-status", help="Optional explicit status override for Stage C")
    args = parser.parse_args(argv)

    payload = _manifest_for_args(args)
    out_root = _resolve_root(
        args.output_root,
        is_default=(str(args.output_root) == DEFAULT_OUTPUT_ROOT),
    )
    manifest_path = out_root / "manifests" / f"run_{payload['run_id']}.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(
            f"[B3_pipeline_manifest] manifest exists: {manifest_path} (use --force to overwrite)"
        )
    write_json_atomic(manifest_path, payload)

    if args.append_index:
        index_path = out_root / "index" / "runs_index.jsonl"
        _append_index(index_path, payload)

    print(f"[B3_pipeline_manifest] created {manifest_path}", flush=True)
    if args.append_index:
        print(f"[B3_pipeline_manifest] appended {(out_root / 'index' / 'runs_index.jsonl')}", flush=True)
    return 0


def main() -> int:
    return run_manifest_cli()


if __name__ == "__main__":
    raise SystemExit(main())
