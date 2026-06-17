#!/usr/bin/env python3
"""Shape-agnostic modal discovery audit (advisory; does not block production)."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from v2_b3_m4_target_candidate_audit_lib import (  # noqa: E402
    TARGET_CANDIDATE_AUDIT_FILENAME,
    load_target_candidate_audit_rows,
)

AUDIT_SCHEMA = "m4_modal_discovery_audit_v1"
AUDIT_JSON_REL = "validation/modal_discovery_audit.json"
AUDIT_MD_REL = "validation/modal_discovery_audit.md"

CLASSIFICATIONS = (
    "TARGET_PLAN_TOO_SPARSE",
    "SOLVER_RETURNS_TOO_FEW_CANDIDATES",
    "CANDIDATE_FILTER_TOO_STRICT",
    "DEDUP_TOO_AGGRESSIVE",
    "FREQUENCY_RANGE_TOO_NARROW",
    "WORKER_DIAGNOSTICS_MISSING",
    "BOUNDARY_OR_MESH_SUPPRESSION_SUSPECTED",
    "UNKNOWN",
)

TARGETS_PASSED_NOTE = (
    "targets_passed on worker_result means numerical target solve PASS "
    "(setup+solve succeeded for each target), not guaranteed mode discovery."
)

WORKER_DIAGNOSTIC_FIELDS = (
    "target_hz",
    "solve_status",
    "solver_factor",
    "requested_eigenpairs",
    "candidate_count_raw",
    "candidate_count_after_residual",
    "candidate_count_after_physical_filters",
    "accepted_mode_count",
    "rejected_candidate_count",
    "rejection_reasons",
    "min_residual",
    "acceptance_policy",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def _chunk_ids_from_plan(chunk_plan: Mapping[str, Any]) -> List[str]:
    return [str(c.get("chunk_id")) for c in (chunk_plan.get("chunks") or []) if c.get("chunk_id")]


def _target_count_from_plan(
    *,
    target_plan: Mapping[str, Any],
    chunk_plan: Mapping[str, Any],
    agg: Mapping[str, Any],
) -> int:
    if target_plan.get("target_count") is not None:
        return int(target_plan["target_count"])
    if target_plan.get("targets_hz"):
        return len(target_plan["targets_hz"])
    total = agg.get("total_targets_attempted")
    if total is not None:
        return int(total)
    return sum(len(c.get("targets_hz") or []) for c in (chunk_plan.get("chunks") or []))


def _zone_contribution(target_plan: Mapping[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for meta in target_plan.get("target_metadata") or []:
        zone = str(meta.get("zone_id") or "unknown")
        counts[zone] = counts.get(zone, 0) + 1
    return counts


def _compaction_state(run_root: Path) -> Dict[str, Any]:
    worker_root = run_root / "worker_results"
    compaction_manifest = run_root / "compaction" / "compaction_manifest.json"
    minimal_manifest = run_root / "compaction" / "m4_minimal_rom_durable_compaction_manifest_v1.json"
    compacted = compaction_manifest.is_file() or minimal_manifest.is_file()
    worker_dir_present = worker_root.is_dir()
    worker_dirs = sorted(p.name for p in worker_root.iterdir() if p.is_dir()) if worker_dir_present else []
    heavy_present = False
    for cid in worker_dirs[:3]:
        chunk_dir = worker_root / cid
        if (chunk_dir / "solver_result.json").is_file():
            heavy_present = True
            break
    return {
        "compaction_manifest_present": compaction_manifest.is_file() or minimal_manifest.is_file(),
        "compaction_already_applied": compacted,
        "worker_results_dir_present": worker_dir_present,
        "worker_chunk_dir_count": len(worker_dirs),
        "heavy_worker_artifacts_present": heavy_present,
    }


def _target_rows_from_solver(solver: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for trow in solver.get("targets") or []:
        if not isinstance(trow, dict):
            continue
        rejection = dict(trow.get("candidate_rejection_tally") or {})
        converged = trow.get("converged_mode_count")
        accepted_n = trow.get("accepted_mode_count_in_interval")
        rejected = sum(int(v) for v in rejection.values()) if rejection else None
        if rejected is None and converged is not None and accepted_n is not None:
            try:
                rejected = max(0, int(converged) - int(accepted_n))
            except (TypeError, ValueError):
                rejected = None
        rows.append(
            {
                "target_hz": _finite_or_none(trow.get("target_frequency_hz")),
                "solve_status": trow.get("status"),
                "solver_factor": trow.get("factor_solver") or trow.get("factor_solver_effective"),
                "requested_eigenpairs": trow.get("nev"),
                "requested_ncv": trow.get("ncv"),
                "candidate_count_raw": converged,
                "candidate_count_after_residual": converged,
                "candidate_count_after_physical_filters": accepted_n,
                "accepted_mode_count": accepted_n,
                "rejected_candidate_count": rejected,
                "rejection_reasons": rejection,
                "min_residual": trow.get("min_eps_compute_error_relative"),
                "acceptance_policy": trow.get("acceptance_policy"),
                "acceptance_freq_lo_hz": trow.get("acceptance_freq_lo_hz"),
                "acceptance_freq_hi_hz": trow.get("acceptance_freq_hi_hz"),
                "per_target_acceptance_window_hz": trow.get("per_target_acceptance_window_hz"),
                "no_candidate_produced": bool(
                    trow.get("status") == "PASS"
                    and (converged is None or int(converged) == 0)
                    and (accepted_n is None or int(accepted_n) == 0)
                ),
            }
        )
    return rows


def _audit_chunk(
    *,
    run_root: Path,
    chunk_id: str,
    chunk_plan_entry: Optional[Mapping[str, Any]],
    agg_detail: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    chunk_dir = run_root / "worker_results" / chunk_id
    worker_path = chunk_dir / "worker_result.json"
    solver_path = chunk_dir / "solver_result.json"
    audit_jsonl_path = chunk_dir / TARGET_CANDIDATE_AUDIT_FILENAME
    chunk_targets_path = chunk_dir / "chunk_targets.json"

    worker = _load_json(worker_path) if worker_path.is_file() else {}
    solver = _load_json(solver_path) if solver_path.is_file() else {}
    chunk_targets = _load_json(chunk_targets_path) if chunk_targets_path.is_file() else {}
    candidate_rows = load_target_candidate_audit_rows(chunk_dir)

    targets_attempted = worker.get("targets_attempted")
    if targets_attempted is None and agg_detail:
        targets_attempted = agg_detail.get("targets_attempted")
    if targets_attempted is None and chunk_plan_entry:
        targets_attempted = len(chunk_plan_entry.get("targets_hz") or [])
    if targets_attempted is None and chunk_targets.get("targets"):
        targets_attempted = len(chunk_targets["targets"])

    targets_passed = worker.get("targets_passed")
    if targets_passed is None and agg_detail:
        targets_passed = agg_detail.get("targets_passed")

    mode_count = agg_detail.get("mode_count") if agg_detail else None
    if mode_count is None and worker:
        mode_count = len(worker.get("accepted_mode_records") or worker.get("unique_modes") or [])

    target_hz_plan = list(chunk_plan_entry.get("targets_hz") or []) if chunk_plan_entry else []
    if not target_hz_plan and chunk_targets.get("targets"):
        target_hz_plan = [t.get("target_hz") for t in chunk_targets["targets"]]

    target_windows = list(chunk_plan_entry.get("target_windows_hz") or []) if chunk_plan_entry else []

    per_target: List[Dict[str, Any]] = []
    if candidate_rows:
        per_target = [dict(r) for r in candidate_rows]
    elif solver:
        per_target = _target_rows_from_solver(solver)

    return {
        "chunk_id": chunk_id,
        "worker_result_present": worker_path.is_file(),
        "solver_result_present": solver_path.is_file(),
        "target_candidate_audit_present": audit_jsonl_path.is_file(),
        "chunk_targets_present": chunk_targets_path.is_file(),
        "classification": (agg_detail or {}).get("classification"),
        "worker_status": worker.get("status") or (agg_detail or {}).get("worker_status"),
        "targets_attempted": targets_attempted,
        "targets_passed": targets_passed,
        "targets_passed_means_solve_success_not_mode_discovery": True,
        "mode_count": mode_count,
        "empty_chunk": bool((mode_count or 0) == 0),
        "all_targets_passed_but_zero_modes": bool(
            targets_attempted is not None
            and targets_passed is not None
            and int(targets_attempted) > 0
            and int(targets_passed) == int(targets_attempted)
            and int(mode_count or 0) == 0
        ),
        "target_hz_plan": target_hz_plan,
        "target_windows_hz": target_windows,
        "freq_range_hz": chunk_plan_entry.get("freq_range_hz") if chunk_plan_entry else chunk_targets.get("freq_range_hz"),
        "per_target_diagnostics": per_target,
    }


def _missing_worker_diagnostic_fields(per_target_rows: Sequence[Mapping[str, Any]]) -> List[str]:
    missing: List[str] = []
    if not per_target_rows:
        return list(WORKER_DIAGNOSTIC_FIELDS)
    sample = per_target_rows[0]
    for key in WORKER_DIAGNOSTIC_FIELDS:
        if key not in sample:
            missing.append(key)
    return missing


def classify_modal_discovery_issue(
    *,
    target_count: int,
    raw_mode_count: int,
    deduped_mode_count: int,
    dedup_removed: int,
    modes_per_target: Optional[float],
    empty_chunk_count: int,
    chunk_count: int,
    all_targets_passed_zero_mode_chunks: int,
    candidate_level_diagnostics_available: bool,
    frequency_range_hz: Optional[Sequence[float]],
    zone_contribution: Mapping[str, int],
    aggregate_rejection_tally: Mapping[str, int],
    avg_converged_per_target: Optional[float],
) -> str:
    freq_hi = None
    if frequency_range_hz and len(frequency_range_hz) >= 2:
        freq_hi = _finite_or_none(frequency_range_hz[1])

    if not candidate_level_diagnostics_available:
        if dedup_removed <= max(2, int(0.15 * max(raw_mode_count, 1))):
            if raw_mode_count < max(15, int(0.35 * target_count)):
                return "WORKER_DIAGNOSTICS_MISSING"

    if freq_hi is not None and freq_hi < 450.0 and raw_mode_count < max(15, int(0.35 * target_count)):
        return "FREQUENCY_RANGE_TOO_NARROW"

    if target_count < 20:
        return "TARGET_PLAN_TOO_SPARSE"

    if raw_mode_count > 0 and dedup_removed > max(3, int(0.25 * raw_mode_count)):
        return "DEDUP_TOO_AGGRESSIVE"

    if candidate_level_diagnostics_available:
        if aggregate_rejection_tally and sum(aggregate_rejection_tally.values()) > raw_mode_count:
            dominant = max(aggregate_rejection_tally, key=lambda k: aggregate_rejection_tally[k])
            if aggregate_rejection_tally[dominant] >= 3:
                if dominant in (
                    "support_participation_fail",
                    "inactive_dof_violation",
                    "boundary_dof_violation",
                ):
                    return "BOUNDARY_OR_MESH_SUPPRESSION_SUSPECTED"
                return "CANDIDATE_FILTER_TOO_STRICT"
        if avg_converged_per_target is not None and avg_converged_per_target < 0.5:
            return "SOLVER_RETURNS_TOO_FEW_CANDIDATES"

    if all_targets_passed_zero_mode_chunks >= max(1, chunk_count // 4):
        if candidate_level_diagnostics_available:
            return "CANDIDATE_FILTER_TOO_STRICT"
        return "WORKER_DIAGNOSTICS_MISSING"

    if empty_chunk_count > 0 and modes_per_target is not None and modes_per_target < 0.25:
        return "WORKER_DIAGNOSTICS_MISSING"

    if zone_contribution and not candidate_level_diagnostics_available:
        return "WORKER_DIAGNOSTICS_MISSING"

    return "UNKNOWN"


def build_modal_discovery_audit(
    *,
    run_root: Path,
    shape_name: Optional[str] = None,
) -> Dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    agg = _load_json(run_root / "aggregation" / "aggregation_result.json")
    chunk_plan = _load_json(run_root / "lprod" / "worker_chunk_plan.preview.json")
    target_plan = _load_json(run_root / "lprod" / "lprod_target_plan.json")
    manifest = _load_json(run_root / "pipeline_run_manifest.json")
    sample_input = _load_json(run_root / "sample" / "sample_input.json")

    sample_id = agg.get("sample_id") or manifest.get("sample_id") or chunk_plan.get("sample_id")
    run_id = agg.get("run_id") or manifest.get("run_id") or chunk_plan.get("run_id")
    if shape_name is None:
        shape_name = (
            sample_input.get("shape_name")
            or (sample_input.get("shape_context") or {}).get("shape_name")
            or sample_input.get("geometry_shape_type")
        )

    chunk_ids = _chunk_ids_from_plan(chunk_plan)
    agg_details_by_id = {
        str(d.get("chunk_id")): d for d in (agg.get("chunk_details") or []) if d.get("chunk_id")
    }
    chunk_plan_by_id = {str(c.get("chunk_id")): c for c in (chunk_plan.get("chunks") or []) if c.get("chunk_id")}

    chunk_audits = [
        _audit_chunk(
            run_root=run_root,
            chunk_id=cid,
            chunk_plan_entry=chunk_plan_by_id.get(cid),
            agg_detail=agg_details_by_id.get(cid),
        )
        for cid in chunk_ids
    ]

    target_count = _target_count_from_plan(target_plan=target_plan, chunk_plan=chunk_plan, agg=agg)
    raw_mode_count = int(agg.get("raw_mode_count") or 0)
    deduped_mode_count = int(agg.get("deduped_mode_count") or 0)
    dedup_removed = max(0, raw_mode_count - deduped_mode_count)
    chunk_count = len(chunk_ids) or int(agg.get("planned_chunk_count") or 0)
    empty_chunks = [c["chunk_id"] for c in chunk_audits if c.get("empty_chunk")]
    empty_chunk_count = len(empty_chunks)
    sum_chunk_modes = sum(int(c.get("mode_count") or 0) for c in chunk_audits)
    aggregation_loss = max(0, sum_chunk_modes - raw_mode_count)

    modes_per_target = (float(raw_mode_count) / float(target_count)) if target_count > 0 else None
    modes_per_chunk = (float(raw_mode_count) / float(chunk_count)) if chunk_count > 0 else None

    compaction = _compaction_state(run_root)
    worker_results_present = compaction["worker_results_dir_present"] and compaction["worker_chunk_dir_count"] > 0

    all_per_target: List[Dict[str, Any]] = []
    for ca in chunk_audits:
        all_per_target.extend(list(ca.get("per_target_diagnostics") or []))

    candidate_audit_files = sum(1 for ca in chunk_audits if ca.get("target_candidate_audit_present"))
    solver_files = sum(1 for ca in chunk_audits if ca.get("solver_result_present"))

    has_target_level = bool(all_per_target)
    candidate_level_diagnostics_available = bool(
        has_target_level
        and (
            candidate_audit_files > 0
            or solver_files > 0
            or any(
                r.get("candidate_count_raw") is not None or r.get("accepted_mode_count") is not None
                for r in all_per_target
            )
        )
    )

    missing_diagnostics: List[str] = []
    if not candidate_level_diagnostics_available:
        if compaction["compaction_already_applied"] and not compaction["heavy_worker_artifacts_present"]:
            missing_diagnostics.append("worker heavy artifacts removed by compaction")
        if candidate_audit_files == 0:
            missing_diagnostics.append(f"worker_results/*/target_candidate_audit.jsonl")
        if solver_files == 0:
            missing_diagnostics.append("worker_results/*/solver_result.json targets[]")
        missing_diagnostics.extend(_missing_worker_diagnostic_fields(all_per_target))

    aggregate_rejection: Dict[str, int] = {}
    converged_samples: List[float] = []
    for row in all_per_target:
        for reason, count in (row.get("rejection_reasons") or {}).items():
            aggregate_rejection[str(reason)] = aggregate_rejection.get(str(reason), 0) + int(count)
        raw_c = row.get("candidate_count_raw")
        if raw_c is not None:
            try:
                converged_samples.append(float(raw_c))
            except (TypeError, ValueError):
                pass

    avg_converged = (
        sum(converged_samples) / len(converged_samples) if converged_samples else None
    )

    all_targets_passed_zero_mode_chunks = sum(
        1 for c in chunk_audits if c.get("all_targets_passed_but_zero_modes")
    )

    zone_contribution = _zone_contribution(target_plan)
    frequency_range_hz = target_plan.get("frequency_range_hz") or agg.get("frequency_range_hz")

    classification = classify_modal_discovery_issue(
        target_count=target_count,
        raw_mode_count=raw_mode_count,
        deduped_mode_count=deduped_mode_count,
        dedup_removed=dedup_removed,
        modes_per_target=modes_per_target,
        empty_chunk_count=empty_chunk_count,
        chunk_count=chunk_count,
        all_targets_passed_zero_mode_chunks=all_targets_passed_zero_mode_chunks,
        candidate_level_diagnostics_available=candidate_level_diagnostics_available,
        frequency_range_hz=frequency_range_hz,
        zone_contribution=zone_contribution,
        aggregate_rejection_tally=aggregate_rejection,
        avg_converged_per_target=avg_converged,
    )

    recommendations: List[str] = []
    if not candidate_level_diagnostics_available:
        recommendations.append(
            "Re-run BOX workers with target_candidate_audit.jsonl enabled (post-solve hook) "
            "before compaction to capture per-target candidate/rejection counts."
        )
        recommendations.append(
            "Preserve worker_results/<chunk_id>/solver_result.json targets[] until audit completes, "
            "or rely on target_candidate_audit.jsonl as the durable lightweight diagnostic."
        )
    if classification == "WORKER_DIAGNOSTICS_MISSING":
        recommendations.append(
            "Current run shows mode loss before aggregation; inspect per-target rejection_reasons "
            "after the next worker pass."
        )
    if empty_chunk_count > 0:
        recommendations.append(
            f"Investigate {empty_chunk_count} empty chunk(s): {', '.join(empty_chunks[:6])}"
            + (" ..." if len(empty_chunks) > 6 else "")
        )

    return {
        "schema": AUDIT_SCHEMA,
        "generated_utc": utc_now(),
        "advisory_only": True,
        "sample_id": sample_id,
        "run_id": run_id,
        "shape_name": shape_name,
        "target_count": target_count,
        "chunk_count": chunk_count,
        "raw_mode_count": raw_mode_count,
        "deduped_mode_count": deduped_mode_count,
        "dedup_removed_count": dedup_removed,
        "aggregation_loss_count": aggregation_loss,
        "modes_per_target": modes_per_target,
        "modes_per_chunk": modes_per_chunk,
        "empty_chunk_count": empty_chunk_count,
        "empty_chunks": empty_chunks,
        "completed_chunk_count": agg.get("completed_chunk_count"),
        "failed_chunk_count": agg.get("failed_chunk_count"),
        "missing_chunk_count": agg.get("missing_chunk_count"),
        "targets_passed_semantics": TARGETS_PASSED_NOTE,
        "frequency_range_hz": frequency_range_hz,
        "zone_contribution": zone_contribution,
        "dense_sparse_zone_contribution": {
            "ZONE_1_dense": zone_contribution.get("ZONE_1_dense", 0),
            "ZONE_2_medium": zone_contribution.get("ZONE_2_medium", 0),
            "ZONE_3_sparse": zone_contribution.get("ZONE_3_sparse", 0),
        },
        "worker_results_present_at_audit_time": worker_results_present,
        "compaction_state": compaction,
        "candidate_level_diagnostics_available": candidate_level_diagnostics_available,
        "missing_diagnostics": sorted(set(missing_diagnostics)),
        "classification": classification,
        "classification_candidates": list(CLASSIFICATIONS),
        "aggregate_rejection_tally": aggregate_rejection,
        "avg_converged_candidates_per_target": avg_converged,
        "chunks": chunk_audits,
        "recommendations": recommendations,
        "sources": {
            "aggregation_result": str(run_root / "aggregation" / "aggregation_result.json"),
            "worker_chunk_plan_preview": str(run_root / "lprod" / "worker_chunk_plan.preview.json"),
            "lprod_target_plan": str(run_root / "lprod" / "lprod_target_plan.json"),
        },
    }


def render_modal_discovery_audit_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Modal discovery audit",
        "",
        f"- **Schema**: `{report.get('schema')}`",
        f"- **Generated**: {report.get('generated_utc')}",
        f"- **Advisory only**: {report.get('advisory_only')}",
        f"- **Sample**: `{report.get('sample_id')}`",
        f"- **Run**: `{report.get('run_id')}`",
        f"- **Shape**: `{report.get('shape_name')}`",
        f"- **Classification**: `{report.get('classification')}`",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"| --- | --- |",
        f"| target_count | {report.get('target_count')} |",
        f"| chunk_count | {report.get('chunk_count')} |",
        f"| raw_mode_count | {report.get('raw_mode_count')} |",
        f"| deduped_mode_count | {report.get('deduped_mode_count')} |",
        f"| dedup_removed_count | {report.get('dedup_removed_count')} |",
        f"| aggregation_loss_count | {report.get('aggregation_loss_count')} |",
        f"| modes_per_target | {report.get('modes_per_target')} |",
        f"| modes_per_chunk | {report.get('modes_per_chunk')} |",
        f"| empty_chunk_count | {report.get('empty_chunk_count')} |",
        f"| candidate_level_diagnostics_available | {report.get('candidate_level_diagnostics_available')} |",
        "",
        "## targets_passed semantics",
        "",
        str(report.get("targets_passed_semantics") or TARGETS_PASSED_NOTE),
        "",
        "## Frequency coverage",
        "",
        f"- frequency_range_hz: `{report.get('frequency_range_hz')}`",
        f"- zone_contribution: `{json.dumps(report.get('zone_contribution') or {}, sort_keys=True)}`",
        "",
        "## Worker / compaction",
        "",
        f"- worker_results_present_at_audit_time: `{report.get('worker_results_present_at_audit_time')}`",
    ]
    comp = report.get("compaction_state") or {}
    lines.append(f"- compaction_already_applied: `{comp.get('compaction_already_applied')}`")
    lines.append(f"- heavy_worker_artifacts_present: `{comp.get('heavy_worker_artifacts_present')}`")
    lines.append("")

    if not report.get("candidate_level_diagnostics_available"):
        lines.extend(
            [
                "## Missing diagnostics",
                "",
                "```text",
                "candidate_level_diagnostics_available=false",
                f"missing_diagnostics={json.dumps(report.get('missing_diagnostics') or [], sort_keys=True)}",
                "```",
                "",
            ]
        )

    if report.get("empty_chunks"):
        lines.extend(["## Empty chunks", ""])
        for cid in report.get("empty_chunks") or []:
            lines.append(f"- `{cid}`")
        lines.append("")

    lines.extend(["## Chunk summary", ""])
    lines.append("| chunk_id | targets_attempted | targets_passed | mode_count | empty | all_passed_zero_modes |")
    lines.append("| --- | ---: | ---: | ---: | --- | --- |")
    for ch in report.get("chunks") or []:
        lines.append(
            f"| `{ch.get('chunk_id')}` | {ch.get('targets_attempted')} | {ch.get('targets_passed')} | "
            f"{ch.get('mode_count')} | {ch.get('empty_chunk')} | {ch.get('all_targets_passed_but_zero_modes')} |"
        )
    lines.append("")

    if report.get("recommendations"):
        lines.extend(["## Recommendations", ""])
        for rec in report.get("recommendations") or []:
            lines.append(f"- {rec}")
        lines.append("")

    return "\n".join(lines)


def write_modal_discovery_audit(
    *,
    run_root: Path,
    shape_name: Optional[str] = None,
) -> Tuple[Path, Path, Dict[str, Any]]:
    report = build_modal_discovery_audit(run_root=run_root, shape_name=shape_name)
    json_path = run_root / AUDIT_JSON_REL
    md_path = run_root / AUDIT_MD_REL
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_modal_discovery_audit_markdown(report), encoding="utf-8")
    return json_path, md_path, report
