#!/usr/bin/env python3
"""M4.5-pre — freeze and summarize first successful M4 end-to-end run (read-only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_worker_run_lib import (  # noqa: E402
    chunk_ids_from_worker_plan,
    chunk_worker_pass_status,
    detect_repo_root,
    existing_real_worker_result,
    load_json,
    rel,
    utc_now,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

FREEZE_DIR_NAME = "freeze"
REFERENCE_SAMPLE_ID = "sample_001"
REPORT_DOC_REL = (
    "FEM/experiments/active_domain_validation/physics_integrity/docs/"
    "B3_M4_5_PRE_FIRST_END_TO_END_RUN_REPORT.md"
)
TERMINAL_E2E = "LPROD_WORKERS_AND_AGGREGATION_PASS"
AGG_STATUS_PASS = "AGGREGATION_PASS"
SCOUT_TERMINAL_READY = "SCOUT_PASS_TARGET_PLAN_READY"
CHECKPOINT_TERMINAL_READY = "LPROD_CHECKPOINT_READY"

NON_GOALS = [
    "Stage C / rich modal export not run",
    "No audio / STK export",
    "No production promotion",
    "No cleanup or archival of legacy runs",
    "Multi-guitar LHS batch not yet run",
    "Only sample_001 validated end-to-end on VM",
]

NEXT_STEPS = [
    "M4.5 — small multi-guitar batch (2–3 real LHS samples)",
    "M4.6 — validation/comparison against reference/legacy expectations",
    "M4.7 — promote new pipeline as main path",
    "Cleanup/archive only after small batch passes",
]

ARTIFACT_INDEX_FILES = ("artifact_index.json", "artifact_index.md")


def resolve_freeze_config(
    sample_id: str,
    *,
    freeze_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Return manifest/summary basename prefix and schema for freeze outputs."""
    if freeze_prefix:
        prefix = freeze_prefix.strip().removesuffix("_")
    elif sample_id == REFERENCE_SAMPLE_ID:
        prefix = "first_end_to_end"
    else:
        prefix = "sample_e2e"
    schema = (
        "m4_first_end_to_end_freeze_v1"
        if prefix == "first_end_to_end"
        else "m4_sample_e2e_freeze_v1"
    )
    return {
        "prefix": prefix,
        "schema": schema,
        "manifest_name": f"{prefix}_run_manifest.json",
        "summary_name": f"{prefix}_run_summary.md",
        "write_reference_report": prefix == "first_end_to_end",
    }


def _safe_load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _artifact_catalog(run_root: Path, repo_root: Path) -> List[Dict[str, Any]]:
    ckpt = run_root / "lprod" / "checkpoint"
    entries: List[Tuple[str, Path, bool]] = [
        ("sample_input", run_root / "sample" / "sample_input.json", False),
        ("resolved_core_config", run_root / "sample" / "resolved_core_config.json", False),
        ("sample_resolved_config_manifest", run_root / "sample" / "sample_resolved_config_manifest.json", False),
        ("scout_result", run_root / "scout" / "scout_result.json", False),
        ("density_zones", run_root / "scout" / "density_zones.json", False),
        ("scout_plan", run_root / "scout" / "scout_plan.json", False),
        ("lprod_target_plan", run_root / "lprod" / "lprod_target_plan.json", True),
        ("worker_chunk_plan", run_root / "lprod" / "worker_chunk_plan.preview.json", False),
        ("lprod_checkpoint_manifest", run_root / "lprod" / "checkpoint" / "checkpoint_export_manifest.json", True),
        ("lprod_checkpoint_A", ckpt / "A.npz", False),
        ("lprod_checkpoint_M", ckpt / "M.npz", False),
        ("lprod_checkpoint_K", ckpt / "K.npz", False),
        ("lprod_checkpoint_built_metadata", ckpt / "built_metadata.json", False),
        ("remaining_workers_manifest", run_root / "worker_results" / "remaining_workers_m4_4_1b_4_manifest.json", False),
        ("aggregation_result", run_root / "aggregation" / "aggregation_result.json", True),
        ("aggregation_modes_summary", run_root / "aggregation" / "modes_summary.json", False),
        ("aggregation_runtime_summary", run_root / "aggregation" / "runtime_summary.json", False),
        ("aggregation_modes_catalog", run_root / "aggregation" / "modes_catalog.jsonl", False),
        ("aggregation_mode_plot", run_root / "aggregation" / "mode_frequency_plot.png", False),
        ("partial_aggregation_result", run_root / "aggregation" / "partial_aggregation_result.json", False),
        ("pipeline_run_manifest", run_root / "pipeline_run_manifest.json", False),
        ("full_aggregation_preview_manifest", run_root / "pipeline_run_manifest.m4_4_full_aggregation_preview.json", False),
        ("workers_complete_preview_manifest", run_root / "pipeline_run_manifest.m4_4_workers_complete_preview.json", False),
    ]
    out: List[Dict[str, Any]] = []
    for artifact_id, path, essential in entries:
        out.append(
            {
                "id": artifact_id,
                "path": rel(path, repo_root=repo_root),
                "exists": path.is_file(),
                "essential": essential,
            }
        )
    for chunk_id in chunk_ids_from_worker_plan(run_root):
        wr = run_root / "worker_results" / chunk_id / "worker_result.json"
        out.append(
            {
                "id": f"worker_result_{chunk_id}",
                "path": rel(wr, repo_root=repo_root),
                "exists": wr.is_file(),
                "essential": True,
                "chunk_id": chunk_id,
            }
        )
    return out


def _collect_worker_rows(run_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for chunk_id in chunk_ids_from_worker_plan(run_root):
        path = run_root / "worker_results" / chunk_id / "worker_result.json"
        doc = _safe_load(path) or {}
        unique = doc.get("unique_modes") or []
        unique_count = len(unique) if isinstance(unique, list) else 0
        rows.append(
            {
                "chunk_id": chunk_id,
                "status": doc.get("status"),
                "targets_attempted": doc.get("targets_attempted"),
                "targets_passed": doc.get("targets_passed"),
                "unique_mode_count": unique_count,
                "warnings": list(doc.get("warnings") or [])[:5],
                "errors": list(doc.get("errors") or [])[:5],
                "real_result": existing_real_worker_result(path),
            }
        )
    return rows


def _validate_milestone(*, run_root: Path) -> List[str]:
    errors: List[str] = []
    agg_path = run_root / "aggregation" / "aggregation_result.json"
    agg = _safe_load(agg_path)
    if not agg:
        errors.append(f"missing essential: {agg_path}")
    else:
        if str(agg.get("status")) != AGG_STATUS_PASS:
            errors.append(f"aggregation status={agg.get('status')!r} expected {AGG_STATUS_PASS!r}")
        if not agg.get("final_aggregation_ready"):
            errors.append("aggregation final_aggregation_ready is not true")

    if not (run_root / "lprod" / "lprod_target_plan.json").is_file():
        errors.append("missing essential: lprod/lprod_target_plan.json")
    ckpt_manifest = run_root / "lprod" / "checkpoint" / "checkpoint_export_manifest.json"
    if not ckpt_manifest.is_file():
        errors.append("missing essential: lprod/checkpoint/checkpoint_export_manifest.json")
    else:
        cm = _safe_load(ckpt_manifest) or {}
        if str(cm.get("status")) not in ("PASS",) and not cm.get("export_pass"):
            errors.append(f"lprod checkpoint manifest not PASS: {cm.get('status')}")

    planned = chunk_ids_from_worker_plan(run_root)
    if not planned:
        errors.append("no chunks in lprod/worker_chunk_plan.preview.json")
    for chunk_id in planned:
        wr = run_root / "worker_results" / chunk_id / "worker_result.json"
        if not wr.is_file():
            errors.append(f"missing worker_result for {chunk_id}")
            continue
        st = chunk_worker_pass_status(run_root, chunk_id)
        if not st:
            errors.append(f"chunk {chunk_id} is not PASS/PASS_WITH_WARNING (real worker)")
    return errors


def _run_rel(path: Path, *, run_root: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(run_root)).replace("\\", "/")
    except ValueError:
        return rel(path, repo_root=repo_root)


def _stage_table(
    *,
    run_root: Path,
    repo_root: Path,
    manifest: Dict[str, Any],
    agg: Optional[Dict[str, Any]],
    target_plan: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    stages = manifest.get("stages") or {}
    freq = manifest.get("frequency_policy") or {}

    def _paths(*rels: str) -> List[str]:
        return [_run_rel(run_root / r, run_root=run_root, repo_root=repo_root) for r in rels]

    scout_ckpt = run_root / "scout" / "checkpoint" / "checkpoint_export_manifest.json"
    scout_res = run_root / "scout" / "scout_result.json"
    scout_zones = run_root / "scout" / "density_zones.json"

    s0_ok = (run_root / "sample" / "sample_input.json").is_file() and (
        run_root / "sample" / "resolved_core_config.json"
    ).is_file()
    s1_ok = scout_ckpt.is_file() or (run_root / "lprod" / "checkpoint" / "checkpoint_export_manifest.json").is_file()
    s2_ok = scout_res.is_file() or (run_root / "scout" / "discovery").is_dir()
    s3_ok = bool(target_plan) and str(stages.get("stage3_zones_plan", {}).get("status")) in (
        "PASS",
        "COMPLETE",
    )
    s4_ok = str(stages.get("stage4_lprod_export", {}).get("status")) == "PASS" or (
        run_root / "lprod" / "checkpoint" / "checkpoint_export_manifest.json"
    ).is_file()
    s5_ok = agg and int(agg.get("completed_chunk_count") or 0) == int(agg.get("planned_chunk_count") or 0)
    s6_ok = agg and str(agg.get("status")) == AGG_STATUS_PASS

    return [
        {
            "stage": "Stage 0 — sample/config",
            "status": "PASS" if s0_ok else "INCOMPLETE",
            "artifact_paths": _paths("sample/sample_input.json", "sample/resolved_core_config.json"),
        },
        {
            "stage": "Stage 1 — scout mesh/checkpoint",
            "status": "PASS" if s1_ok else "INCOMPLETE",
            "artifact_paths": _paths("scout/scout_plan.json", "lprod/checkpoint/checkpoint_export_manifest.json"),
        },
        {
            "stage": "Stage 2 — scout discovery",
            "status": "PASS" if s2_ok else "OPTIONAL/MISSING",
            "artifact_paths": _paths("scout/scout_result.json", "scout/discovery"),
        },
        {
            "stage": "Stage 3 — zones + adaptive L_prod target plan",
            "status": "PASS" if s3_ok else "INCOMPLETE",
            "artifact_paths": _paths("lprod/lprod_target_plan.json", "scout/density_zones.json"),
        },
        {
            "stage": "Stage 4 — L_prod mesh/checkpoint",
            "status": "PASS" if s4_ok else "INCOMPLETE",
            "artifact_paths": _paths("lprod/checkpoint/checkpoint_export_manifest.json", "lprod/mesh"),
        },
        {
            "stage": "Stage 5 — L_prod workers",
            "status": "PASS" if s5_ok else "INCOMPLETE",
            "artifact_paths": _paths("worker_results"),
        },
        {
            "stage": "Stage 6 — aggregation",
            "status": "PASS" if s6_ok else "INCOMPLETE",
            "artifact_paths": _paths("aggregation/aggregation_result.json", "aggregation/modes_summary.json"),
        },
        {
            "stage_note": "scout_spacing_hz",
            "value": freq.get("scout_spacing_hz"),
        },
        {
            "stage_note": "lprod_zone_spacing_hz",
            "value": freq.get("zone_spacing_hz"),
        },
    ]


def _key_metrics(
    *,
    manifest: Dict[str, Any],
    agg: Dict[str, Any],
    target_plan: Optional[Dict[str, Any]],
    preview: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    coverage = (target_plan or {}).get("coverage_check") or {}
    return {
        "sample_id": agg.get("sample_id") or manifest.get("sample_id"),
        "run_id": agg.get("run_id") or manifest.get("run_id"),
        "target_count": coverage.get("target_count") or len((target_plan or {}).get("targets_hz") or []),
        "worker_chunk_count": agg.get("planned_chunk_count"),
        "completed_chunk_count": agg.get("completed_chunk_count"),
        "missing_chunk_count": agg.get("missing_chunk_count"),
        "failed_chunk_count": agg.get("failed_chunk_count"),
        "raw_mode_count": agg.get("raw_mode_count"),
        "deduped_mode_count": agg.get("deduped_mode_count"),
        "dedupe_tolerance_hz": agg.get("dedupe_tolerance_hz"),
        "final_aggregation_ready": agg.get("final_aggregation_ready"),
        "coverage_pass": coverage.get("pass"),
        "coverage_max_gap_hz": coverage.get("max_gap_hz"),
        "terminal_status": (preview or {}).get("terminal_status") or TERMINAL_E2E,
        "aggregation_status": agg.get("status"),
    }


def _planning_summary(manifest: Dict[str, Any], target_plan: Optional[Dict[str, Any]], chunk_plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    freq = manifest.get("frequency_policy") or {}
    coverage = (target_plan or {}).get("coverage_check") or {}
    return {
        "scout_spacing_hz": freq.get("scout_spacing_hz"),
        "lprod_zone_spacing_hz": freq.get("zone_spacing_hz"),
        "target_generation_policy": (target_plan or {}).get("target_generation_policy"),
        "chunk_policy_version": (chunk_plan or {}).get("chunk_policy_version"),
        "coverage_pass": coverage.get("pass"),
        "coverage_max_gap_hz": coverage.get("max_gap_hz"),
        "target_count": coverage.get("target_count"),
    }


def _render_report_md(payload: Dict[str, Any]) -> str:
    m = payload["key_metrics"]
    plan = payload["planning_summary"]
    lines = [
        "# B3 M4.5-pre — First successful M4 end-to-end run",
        "",
        "## 1. Milestone statement",
        "",
        "**This is the first successful M4 end-to-end run for one guitar sample.**",
        "",
        f"Frozen run: `{m.get('run_id')}` (`{m.get('sample_id')}`) on VM. "
        "Documentation and metadata only; no solver re-execution.",
        "",
        "## 2. Pipeline stages and status",
        "",
        "| Stage | Status | Primary artifacts |",
        "|-------|--------|-------------------|",
    ]
    for row in payload.get("stage_statuses") or []:
        if "stage" not in row:
            continue
        paths = ", ".join(f"`{p}`" for p in (row.get("artifact_paths") or [])[:2])
        if len(row.get("artifact_paths") or []) > 2:
            paths += ", …"
        lines.append(f"| {row['stage']} | **{row['status']}** | {paths} |")

    lines.extend(
        [
            "",
            "## 3. Key run metrics",
            "",
            "| Field | Value |",
            "|-------|-------|",
        ]
    )
    for key in (
        "sample_id",
        "run_id",
        "target_count",
        "worker_chunk_count",
        "completed_chunk_count",
        "failed_chunk_count",
        "raw_mode_count",
        "deduped_mode_count",
        "dedupe_tolerance_hz",
        "final_aggregation_ready",
        "terminal_status",
        "aggregation_status",
    ):
        lines.append(f"| {key} | **{m.get(key)}** |")

    lines.extend(["", "## 4. Worker summary", "", "| chunk_id | status | targets | unique_modes |", "|----------|--------|---------|--------------|"])
    for w in payload["worker_summary"]:
        lines.append(
            f"| {w.get('chunk_id')} | {w.get('status')} | "
            f"{w.get('targets_passed')}/{w.get('targets_attempted')} | {w.get('unique_mode_count')} |"
        )

    zone_sp = plan.get("lprod_zone_spacing_hz")
    if isinstance(zone_sp, dict):
        zone_line = (
            f"ZONE_1_dense={zone_sp.get('ZONE_1_dense')} Hz, "
            f"ZONE_2_medium={zone_sp.get('ZONE_2_medium')} Hz, "
            f"ZONE_3_sparse={zone_sp.get('ZONE_3_sparse')} Hz"
        )
    else:
        zone_line = str(zone_sp)

    lines.extend(
        [
            "",
            "## 5. Adaptive planning summary",
            "",
            f"- Scout spacing: **{plan.get('scout_spacing_hz')} Hz**",
            f"- L_prod zone policy: **{zone_line}**",
            f"- Target generation policy: `{plan.get('target_generation_policy')}`",
            f"- Chunk policy version: `{plan.get('chunk_policy_version')}`",
            f"- Target coverage pass: **{plan.get('coverage_pass')}**",
            f"- Coverage max gap: **{plan.get('coverage_max_gap_hz')} Hz**",
            f"- Target count: **{plan.get('target_count')}**",
            "",
            "## 6. Artifact index",
            "",
            "See `freeze/artifact_index.md` in the run tree for the full exists/missing table.",
            "",
            "| Artifact | Exists | Essential |",
            "|----------|--------|-----------|",
        ]
    )
    for a in payload["artifact_index"]:
        if a.get("chunk_id"):
            continue
        lines.append(
            f"| {a.get('id')} | {'yes' if a.get('exists') else 'no'} | "
            f"{'yes' if a.get('essential') else 'no'} |"
        )

    lines.extend(["", "## 7. Explicit non-goals / not yet done", ""])
    for item in payload["non_goals"]:
        lines.append(f"- {item}")

    lines.extend(["", "## 8. Next steps", ""])
    for item in payload["next_steps"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "---",
            f"*Generated by `v2_b3_m4_freeze_first_e2e_run.py` at {payload.get('generated_utc')}.*",
            "",
        ]
    )
    return "\n".join(lines)


def _render_summary_md(payload: Dict[str, Any]) -> str:
    m = payload["key_metrics"]
    cfg = payload.get("freeze_cfg") or {}
    title = (
        "First M4 end-to-end freeze"
        if cfg.get("prefix") == "first_end_to_end"
        else "M4 sample E2E freeze"
    )
    return "\n".join(
        [
            f"# {title} — {m.get('run_id')}",
            "",
            f"- **status:** {m.get('aggregation_status')}",
            f"- **terminal_status:** {m.get('terminal_status')}",
            f"- **chunks:** {m.get('completed_chunk_count')}/{m.get('worker_chunk_count')} PASS",
            f"- **modes:** raw={m.get('raw_mode_count')} deduped={m.get('deduped_mode_count')}",
            f"- **targets:** {m.get('target_count')} (coverage max gap {payload['planning_summary'].get('coverage_max_gap_hz')} Hz)",
            "",
            "Full report: `docs/B3_M4_5_PRE_FIRST_END_TO_END_RUN_REPORT.md`",
            "",
        ]
    )


def _render_artifact_index_md(artifacts: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# Artifact index — first M4 end-to-end freeze",
        "",
        "| ID | Exists | Essential | Path |",
        "|----|--------|-----------|------|",
    ]
    for a in artifacts:
        lines.append(
            f"| {a.get('id')} | {'yes' if a.get('exists') else 'no'} | "
            f"{'yes' if a.get('essential') else 'no'} | `{a.get('path')}` |"
        )
    lines.append("")
    return "\n".join(lines)


def build_freeze_payload(
    *,
    repo_root: Path,
    run_root: Path,
    freeze_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = _safe_load(run_root / "pipeline_run_manifest.json") or {}
    agg = _safe_load(run_root / "aggregation" / "aggregation_result.json") or {}
    target_plan = _safe_load(run_root / "lprod" / "lprod_target_plan.json")
    chunk_plan = _safe_load(run_root / "lprod" / "worker_chunk_plan.preview.json")
    preview = _safe_load(run_root / "pipeline_run_manifest.m4_4_full_aggregation_preview.json")
    artifacts = _artifact_catalog(run_root, repo_root)
    missing_optional = [a["id"] for a in artifacts if not a.get("exists") and not a.get("essential")]
    sample_id = str(agg.get("sample_id") or manifest.get("sample_id") or "")
    cfg = freeze_cfg or resolve_freeze_config(sample_id)

    payload: Dict[str, Any] = {
        "schema": cfg["schema"],
        "freeze_cfg": cfg,
        "generated_utc": utc_now(),
        "sample_id": agg.get("sample_id") or manifest.get("sample_id"),
        "run_id": agg.get("run_id") or manifest.get("run_id"),
        "terminal_status": (preview or {}).get("terminal_status") or TERMINAL_E2E,
        "status": agg.get("status") or AGG_STATUS_PASS,
        "stage_statuses": _stage_table(
            run_root=run_root,
            repo_root=repo_root,
            manifest=manifest,
            agg=agg or None,
            target_plan=target_plan,
        ),
        "key_metrics": _key_metrics(
            manifest=manifest, agg=agg, target_plan=target_plan, preview=preview
        ),
        "planning_summary": _planning_summary(manifest, target_plan, chunk_plan),
        "worker_summary": _collect_worker_rows(run_root),
        "artifact_index": artifacts,
        "artifact_paths": {a["id"]: a["path"] for a in artifacts if a.get("exists")},
        "missing_optional_artifacts": missing_optional,
        "non_goals": NON_GOALS,
        "next_steps": NEXT_STEPS,
        "milestone_validation_errors": [],
        "no_solver_executed": True,
    }
    return payload


def write_freeze_outputs(
    *,
    repo_root: Path,
    run_root: Path,
    payload: Dict[str, Any],
    force: bool,
) -> Path:
    freeze_dir = run_root / FREEZE_DIR_NAME
    freeze_dir.mkdir(parents=True, exist_ok=True)
    cfg = payload.get("freeze_cfg") or resolve_freeze_config(str(payload.get("sample_id") or ""))
    manifest_name = cfg["manifest_name"]
    summary_name = cfg["summary_name"]
    output_names = (manifest_name, summary_name) + ARTIFACT_INDEX_FILES

    existing = [freeze_dir / name for name in output_names if (freeze_dir / name).is_file()]
    if existing and not force:
        raise FileExistsError(
            f"freeze outputs exist ({len(existing)} files); use --force to overwrite: {freeze_dir}"
        )

    manifest_body = {
        "schema": cfg["schema"],
        "generated_utc": payload["generated_utc"],
        "sample_id": payload["sample_id"],
        "run_id": payload["run_id"],
        "terminal_status": payload["terminal_status"],
        "status": payload["status"],
        "stage_statuses": payload["stage_statuses"],
        "key_metrics": payload["key_metrics"],
        "planning_summary": payload["planning_summary"],
        "artifact_paths": payload["artifact_paths"],
        "missing_optional_artifacts": payload["missing_optional_artifacts"],
        "non_goals": payload["non_goals"],
        "next_steps": payload["next_steps"],
        "freeze_dir": rel(freeze_dir, repo_root=repo_root),
        "freeze_prefix": cfg["prefix"],
        "report_doc": REPORT_DOC_REL if cfg.get("write_reference_report") else None,
    }
    write_json_atomic(freeze_dir / manifest_name, manifest_body)
    (freeze_dir / summary_name).write_text(_render_summary_md(payload), encoding="utf-8")
    write_json_atomic(
        freeze_dir / "artifact_index.json",
        {
            "schema": "m4_freeze_artifact_index_v1",
            "generated_utc": payload["generated_utc"],
            "run_id": payload["run_id"],
            "artifacts": payload["artifact_index"],
            "missing_optional_artifacts": payload["missing_optional_artifacts"],
        },
    )
    (freeze_dir / "artifact_index.md").write_text(
        _render_artifact_index_md(payload["artifact_index"]), encoding="utf-8"
    )

    if cfg.get("write_reference_report"):
        report_path = repo_root / REPORT_DOC_REL
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.is_file() and not force:
            raise FileExistsError(f"report exists (use --force): {report_path}")
        report_path.write_text(_render_report_md(payload), encoding="utf-8")
    return freeze_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="M4.5-pre: freeze first successful M4 end-to-end run (read-only)."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--freeze-prefix",
        choices=("auto", "first_end_to_end", "sample_e2e"),
        default="auto",
        help="Freeze manifest basename prefix (auto: sample_001→first_end_to_end, else sample_e2e).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing freeze/ and report.")
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    run_root = args.run_dir if args.run_dir.is_absolute() else repo_root / args.run_dir
    run_root = run_root.resolve()

    errors = _validate_milestone(run_root=run_root)
    if errors:
        print("error: milestone preconditions not met:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    manifest = _safe_load(run_root / "pipeline_run_manifest.json") or {}
    sample_id = str(manifest.get("sample_id") or "")
    freeze_cfg = resolve_freeze_config(
        sample_id,
        freeze_prefix=None if args.freeze_prefix == "auto" else args.freeze_prefix,
    )
    payload = build_freeze_payload(
        repo_root=repo_root, run_root=run_root, freeze_cfg=freeze_cfg
    )
    payload["milestone_validation_errors"] = errors

    try:
        freeze_dir = write_freeze_outputs(
            repo_root=repo_root,
            run_root=run_root,
            payload=payload,
            force=bool(args.force),
        )
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    m = payload["key_metrics"]
    prefix = (payload.get("freeze_cfg") or {}).get("prefix", "freeze")
    print(f"M4 freeze written ({prefix})")
    print(f"sample_id={m.get('sample_id')}")
    print(f"run_id={m.get('run_id')}")
    print(f"status={m.get('aggregation_status')}")
    print(
        f"completed_chunks={m.get('completed_chunk_count')}/{m.get('worker_chunk_count')}"
    )
    print(f"deduped_modes={m.get('deduped_mode_count')}")
    print(f"freeze_dir={rel(freeze_dir, repo_root=repo_root)}")
    print("no solver executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
