#!/usr/bin/env python3
"""M4.5.1 — small multi-guitar M4 batch dry-run planner (no execution)."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_pipeline_dry_run import (  # noqa: E402
    GUITARS_ROOT,
    PIPELINE_RUNS,
    build_dry_run_plan,
    _write_tree,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

SCRIPTS_REL = Path("FEM/experiments/active_domain_validation/physics_integrity/scripts")
FREEZE_SCRIPT = f"{SCRIPTS_REL.as_posix()}/v2_b3_m4_freeze_first_e2e_run.py"
AGG_PASS = "AGGREGATION_PASS"
TERMINAL_E2E = "LPROD_WORKERS_AND_AGGREGATION_PASS"
CHUNK_POLICY_V1_1 = "v1_1"

STAGE_LABELS = (
    ("stage0_resolve", "Stage 0 — sample/config"),
    ("stage1_scout_mesh", "Stage 1 — scout mesh"),
    ("stage1_scout_export", "Stage 1 — scout checkpoint"),
    ("stage2_scout_discovery", "Stage 2 — scout discovery"),
    ("stage3_zones_plan", "Stage 3 — zones + L_prod plan"),
    ("stage4_lprod_mesh", "Stage 4 — L_prod mesh"),
    ("stage4_lprod_export", "Stage 4 — L_prod checkpoint"),
    ("stage5_workers", "Stage 5 — L_prod workers"),
    ("stage6_aggregate", "Stage 6 — aggregation"),
    ("stage6_freeze", "Stage 6 — freeze milestone"),
)


def _load_batch_spec(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    if not isinstance(data.get("samples"), list) or not data["samples"]:
        raise ValueError(f"{path}: samples[] required")
    return data


def _classify_run_status(run_root: Path, *, production_mode: bool = False) -> str:
    if not run_root.is_dir():
        return "planned_new_run"

    agg_path = run_root / "aggregation" / "aggregation_result.json"
    if agg_path.is_file():
        try:
            agg = load_json(agg_path)
            if str(agg.get("status")) == AGG_PASS and agg.get("final_aggregation_ready"):
                if production_mode:
                    from v2_b3_m4_production_freeze import production_freeze_complete  # noqa: WPS433

                    if production_freeze_complete(run_root):
                        return "already_complete_reuse"
                    return "resume_possible"
                return "already_complete_reuse"
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    manifest_path = run_root / "pipeline_run_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = load_json(manifest_path)
            term = str(manifest.get("terminal_status") or "")
            if term == TERMINAL_E2E:
                return "already_complete_reuse"
            stages = manifest.get("stages") or {}
            if stages.get("stage5_workers", {}).get("status") == "FAIL":
                return "requires_review"
            if stages.get("stage6_aggregate", {}).get("status") == "FAIL":
                return "requires_review"
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    worker_root = run_root / "worker_results"
    if worker_root.is_dir():
        for chunk_dir in worker_root.iterdir():
            if not chunk_dir.is_dir():
                continue
            wr = chunk_dir / "worker_result.json"
            if wr.is_file():
                try:
                    w = load_json(wr)
                    if str(w.get("status")) == "FAIL":
                        return "requires_review"
                except (OSError, ValueError, json.JSONDecodeError):
                    continue

    if any(run_root.joinpath(p).exists() for p in ("scout", "lprod", "sample", "worker_results")):
        return "resume_possible"

    return "planned_new_run"


def _should_write_run_tree(*, reuse_status: str, force: bool) -> bool:
    if reuse_status == "planned_new_run":
        return True
    if reuse_status == "already_complete_reuse":
        return False
    return bool(force)


def _patch_plan_for_m45(plan: Dict[str, Any], *, batch_id: str) -> None:
    """Align dry-run stubs with M4.4+ naming (chunk v1_1, preview paths)."""
    run_root: Path = plan["run_root"]
    rel_run = plan["rel_run"]
    renamed: Dict[Path, Any] = {}
    manifest: Optional[Dict[str, Any]] = None

    for path, payload in plan["files"].items():
        if path.name == "worker_chunk_plan.placeholder.json":
            path = run_root / "lprod" / "worker_chunk_plan.preview.json"
            payload["chunk_policy_version"] = CHUNK_POLICY_V1_1
            payload["lprod_target_plan_path"] = f"{rel_run}/lprod/lprod_target_plan.json"
            payload["status"] = "PLANNED_NOT_EXECUTED"
        elif path.name == "lprod_target_plan.placeholder.json":
            path = run_root / "lprod" / "lprod_target_plan.json"
            payload["target_generation_policy"] = "gapless_grid_v2_segment_endpoint_plus_coverage_repair"
            payload["status"] = "PLANNED_NOT_EXECUTED"
        elif path.name == "density_zones.placeholder.json":
            path = run_root / "scout" / "density_zones.json"
            payload["status"] = "PLANNED_NOT_EXECUTED"
        elif path.name == "pipeline_run_manifest.json":
            manifest = payload
        renamed[path] = payload

    plan["files"] = renamed

    if manifest is None:
        return
    if manifest:
        manifest["policy_versions"]["chunk_policy_version"] = CHUNK_POLICY_V1_1
        manifest["batch_id"] = batch_id
        manifest["mode"] = "m4_5_1_batch_dry_run"
        st3 = manifest.setdefault("stages", {}).setdefault("stage3_zones_plan", {})
        st3["artifact_paths"] = [
            f"{rel_run}/scout/density_zones.json",
            f"{rel_run}/lprod/lprod_target_plan.json",
            f"{rel_run}/lprod/worker_chunk_plan.preview.json",
        ]
        st3["status"] = "PLANNED"
        manifest["stages"]["stage6_aggregate"]["artifact_paths"] = [
            f"{rel_run}/aggregation/aggregation_result.json",
            f"{rel_run}/aggregation/modes_summary.json",
            f"{rel_run}/aggregation/runtime_summary.json",
        ]
        manifest["stages"]["stage6_freeze"] = {
            "status": "PLANNED",
            "artifact_paths": [f"{rel_run}/freeze/first_end_to_end_run_manifest.json"],
            "command_preview": (
                f"python {FREEZE_SCRIPT} --run-dir {rel_run} --force  # after AGGREGATION_PASS"
            ),
        }


def _per_sample_stage_plan(
    *,
    plan: Dict[str, Any],
    reuse_status: str,
    spec_entry: Dict[str, Any],
) -> Dict[str, Any]:
    manifest = plan["files"][plan["run_root"] / "pipeline_run_manifest.json"]
    stages_out: List[Dict[str, Any]] = []
    for key, label in STAGE_LABELS:
        st = manifest.get("stages", {}).get(key, {})
        stages_out.append(
            {
                "stage_key": key,
                "stage": label,
                "status": st.get("status", "PLANNED"),
                "command_preview": st.get("command_preview"),
                "artifact_paths": st.get("artifact_paths") or [],
            }
        )
    return {
        "sample_id": spec_entry["sample_id"],
        "run_id": spec_entry["run_id"],
        "lhs_source_id": spec_entry.get("lhs_source_id"),
        "reuse_status": reuse_status,
        "run_root": plan["rel_run"],
        "will_execute": False,
        "stages": stages_out,
        "validation": {
            "sample_input_ok": bool(spec_entry.get("sample_input")),
            "overlay_dir": f"pipeline_runs/config_overlays/{spec_entry['sample_id']}",
        },
    }


def _build_sample_plan(
    *,
    repo_root: Path,
    spec: Dict[str, Any],
    entry: Dict[str, Any],
    batch_id: str,
    force: bool,
) -> Dict[str, Any]:
    fp = spec.get("frequency_policy") or {}
    sample = copy.deepcopy(entry["sample_input"])
    run_id = str(entry["run_id"])
    run_root = GUITARS_ROOT / entry["sample_id"] / "runs" / run_id
    reuse_status = _classify_run_status(run_root)

    if entry["sample_id"] in (spec.get("exclude_from_batch") or []):
        raise ValueError(f"sample {entry['sample_id']} is excluded from batch execution")

    plan = build_dry_run_plan(
        repo_root=repo_root,
        sample=sample,
        run_id=run_id,
        freq_min=float(fp.get("band_hz", [60, 550])[0]),
        freq_max=float(fp.get("band_hz", [60, 550])[1]),
        scout_spacing_hz=float(fp.get("scout_spacing_hz", 7.5)),
        scout_half_width_hz=float(fp.get("scout_half_width_hz", 3.75)),
        zone_spacing_dense=float(fp.get("zone_spacing_hz", {}).get("ZONE_1_dense", 6.0)),
        zone_spacing_medium=float(fp.get("zone_spacing_hz", {}).get("ZONE_2_medium", 9.0)),
        zone_spacing_sparse=float(fp.get("zone_spacing_hz", {}).get("ZONE_3_sparse", 12.5)),
        workers=int(fp.get("workers", 3)),
        prod_python="/home/vboxuser/final-project/.venv/bin/python",
        solver_python="/home/vboxuser/solver-mkl/venv/bin/python",
    )
    _patch_plan_for_m45(plan, batch_id=batch_id)

    wrote_tree = False
    write_error: Optional[str] = None
    if _should_write_run_tree(reuse_status=reuse_status, force=force):
        try:
            _write_tree(plan, force=force or reuse_status == "planned_new_run")
            wrote_tree = True
            batch_plan_path = run_root / "m4_5_batch_dry_run_plan.json"
            per_sample = _per_sample_stage_plan(
                plan=plan, reuse_status=reuse_status, spec_entry=entry
            )
            per_sample["batch_id"] = batch_id
            per_sample["generated_utc"] = utc_now()
            write_json_atomic(batch_plan_path, per_sample)
            summary = run_root / "dry_run_summary.md"
            summary.write_text(
                _render_per_sample_summary(per_sample, batch_id=batch_id),
                encoding="utf-8",
            )
        except FileExistsError as exc:
            write_error = str(exc)

    per_sample_plan = _per_sample_stage_plan(
        plan=plan, reuse_status=reuse_status, spec_entry=entry
    )
    per_sample_plan["wrote_run_tree"] = wrote_tree
    per_sample_plan["write_error"] = write_error
    return per_sample_plan


def _render_per_sample_summary(per_sample: Dict[str, Any], *, batch_id: str) -> str:
    lines = [
        f"# M4.5.1 batch dry-run — {per_sample.get('sample_id')}",
        "",
        f"- batch_id: `{batch_id}`",
        f"- run_id: `{per_sample.get('run_id')}`",
        f"- reuse_status: **{per_sample.get('reuse_status')}**",
        f"- will_execute: **false**",
        "",
        "## Stages (planned)",
        "",
    ]
    for st in per_sample.get("stages") or []:
        cmd = str(st.get("command_preview") or "(planner only)")
        short = f"{cmd[:77]}..." if len(cmd) > 80 else cmd
        lines.append(f"- **{st.get('stage')}**: {st.get('status')} — `{short}`")
    lines.extend(["", "No mesh build, scout solve, workers, aggregation, or freeze executed.", ""])
    return "\n".join(lines)


def _render_batch_plan_md(batch: Dict[str, Any]) -> str:
    lines = [
        f"# M4.5.1 batch dry-run — {batch.get('batch_id')}",
        "",
        "**will_execute=false** — planning only; no solvers.",
        "",
        f"- Reference (frozen, not in batch): `{batch.get('reference_sample_id')}` / `{batch.get('reference_run_id')}`",
        f"- Samples: {', '.join(batch.get('sample_ids') or [])}",
        "",
        "## Reuse policy",
        "",
        "| Status | Meaning | Write run tree on dry-run |",
        "|--------|---------|---------------------------|",
        "| `planned_new_run` | No run dir | yes |",
        "| `already_complete_reuse` | E2E PASS | no |",
        "| `resume_possible` | Partial tree | only with `--force` |",
        "| `requires_review` | FAIL artifacts | only with `--force` |",
        "",
        "## Per-sample",
        "",
    ]
    for row in batch.get("samples") or []:
        lines.append(
            f"- **{row['sample_id']}** (`{row['run_id']}`): {row['reuse_status']}"
            + (" — tree written" if row.get("wrote_run_tree") else "")
        )
        if row.get("write_error"):
            lines.append(f"  - write skipped: {row['write_error']}")
    lines.extend(
        [
            "",
            "## Non-goals",
            "",
            "- No batch execution (M4.5)",
            "- No Stage C / rich modal / cleanup / promotion",
            "",
            f"Generated: {batch.get('generated_utc')}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_per_sample_commands(batch: Dict[str, Any]) -> str:
    lines = [
        f"# Per-sample command previews — {batch.get('batch_id')}",
        "",
        "Planned execution order per guitar (not run by this dry-run).",
        "",
    ]
    for row in batch.get("samples") or []:
        lines.extend([f"## {row['sample_id']} — {row['run_id']}", ""])
        for st in row.get("stages") or []:
            if st.get("command_preview"):
                lines.append(f"### {st.get('stage')}")
                lines.append(f"```bash")
                lines.append(str(st["command_preview"]))
                lines.append("```")
                lines.append("")
    return "\n".join(lines)


def run_batch_dry_run(
    *,
    repo_root: Path,
    spec_path: Path,
    batch_id: Optional[str],
    force: bool,
) -> Dict[str, Any]:
    spec = _load_batch_spec(spec_path)
    bid = batch_id or str(spec.get("batch_id") or "m4_5_batch")
    if spec.get("batch_id") and batch_id and spec["batch_id"] != batch_id:
        raise ValueError(f"--batch-id {batch_id!r} does not match spec {spec['batch_id']!r}")

    batch_dir = PIPELINE_RUNS / "batches" / bid
    batch_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: List[Dict[str, Any]] = []
    for entry in spec["samples"]:
        sid = str(entry.get("sample_id") or "")
        if sid in (spec.get("exclude_from_batch") or []):
            continue
        sample_rows.append(
            _build_sample_plan(
                repo_root=repo_root,
                spec=spec,
                entry=entry,
                batch_id=bid,
                force=force,
            )
        )

    batch_plan = {
        "schema": "m4_5_small_lhs_batch_plan_v1",
        "will_execute": False,
        "generated_utc": utc_now(),
        "batch_id": bid,
        "spec_path": rel(spec_path, repo_root=repo_root),
        "reference_sample_id": spec.get("reference_sample_id"),
        "reference_run_id": spec.get("reference_run_id"),
        "sample_ids": [r["sample_id"] for r in sample_rows],
        "sample_count": len(sample_rows),
        "frequency_policy": spec.get("frequency_policy"),
        "samples": sample_rows,
        "no_solver_executed": True,
        "safety": {
            "no_mesh_build": True,
            "no_stage_a_b": True,
            "no_workers": True,
            "no_aggregation": True,
            "no_freeze": True,
            "no_cleanup": True,
        },
    }

    manifest = {
        "schema": "m4_5_small_lhs_batch_manifest_v1",
        "will_execute": False,
        "generated_utc": batch_plan["generated_utc"],
        "batch_id": bid,
        "batch_dir": rel(batch_dir, repo_root=repo_root),
        "status": "BATCH_DRY_RUN_PLANNED",
        "sample_count": len(sample_rows),
        "samples": [
            {
                "sample_id": r["sample_id"],
                "run_id": r["run_id"],
                "reuse_status": r["reuse_status"],
                "run_root": r["run_root"],
                "wrote_run_tree": r.get("wrote_run_tree"),
            }
            for r in sample_rows
        ],
    }

    existing = [
        batch_dir / name
        for name in ("batch_plan.json", "batch_manifest.json", "batch_plan.md", "per_sample_commands.md")
        if (batch_dir / name).is_file()
    ]
    if existing and not force:
        raise FileExistsError(
            f"batch outputs exist in {batch_dir} ({len(existing)} files); use --force"
        )

    write_json_atomic(batch_dir / "batch_plan.json", batch_plan)
    write_json_atomic(batch_dir / "batch_manifest.json", manifest)
    (batch_dir / "batch_plan.md").write_text(_render_batch_plan_md(batch_plan), encoding="utf-8")
    (batch_dir / "per_sample_commands.md").write_text(
        _render_per_sample_commands(batch_plan), encoding="utf-8"
    )

    return batch_plan


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="M4.5.1: small multi-guitar M4 batch dry-run planner (no execution)."
    )
    parser.add_argument("--samples-json", type=Path, required=True)
    parser.add_argument("--batch-id", help="Override spec batch_id.")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Refused: dry-run only.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite batch outputs / eligible run trees.")
    args = parser.parse_args(argv)

    if not args.dry_run:
        print("error: this script is dry-run only", file=sys.stderr)
        return 2

    repo_root = detect_repo_root(SCRIPT_DIR)
    spec_path = args.samples_json if args.samples_json.is_absolute() else repo_root / args.samples_json
    if not spec_path.is_file():
        print(f"error: missing --samples-json: {spec_path}", file=sys.stderr)
        return 2

    try:
        batch = run_batch_dry_run(
            repo_root=repo_root,
            spec_path=spec_path,
            batch_id=args.batch_id,
            force=bool(args.force),
        )
    except (ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("will_execute=false")
    print(f"batch_id={batch['batch_id']}")
    print(f"sample_count={batch['sample_count']}")
    print(f"samples={','.join(batch['sample_ids'])}")
    print("wrote batch_plan.json/md")
    print("wrote batch_manifest.json")
    print("wrote per_sample_commands.md")
    print("wrote per-sample dry-run plans")
    print("no solver executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
