#!/usr/bin/env python3
"""M4.4.1b-3 — aggregate L_prod worker results (partial or full; no solver execution)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_mode_audio_coupling import (  # noqa: E402
    STK_ROM_GUIDANCE,
    merge_audio_coupling_into_catalog_record,
    summarize_audio_coupling,
)
from v2_b3_m4_mode_provenance import (  # noqa: E402
    PROVENANCE_FIELD_KEYS,
    merge_provenance_into_catalog_record,
)
from v2_b3_mode_region_participation import (  # noqa: E402
    STK_DAMPING_GUIDANCE,
    enrich_participation_catalog_metadata,
    merge_participation_into_catalog_record,
    summarize_participation_shares,
)
from v2_b3_m4_worker_run_lib import (  # noqa: E402
    PASS_LIKE,
    detect_repo_root,
    existing_real_worker_result,
    load_json,
    rel,
    utc_now,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

DEDUPE_TOLERANCE_HZ = 0.05
PARTIAL_STATUS = "PARTIAL_AGGREGATION_PASS_WITH_MISSING_CHUNKS"
FULL_STATUS = "AGGREGATION_PASS"
TERMINAL_PARTIAL = "PARTIAL_AGGREGATION_READY"
TERMINAL_FULL = "LPROD_WORKERS_AND_AGGREGATION_PASS"

PARTIAL_OUTPUTS = (
    "partial_aggregation_result.json",
    "partial_aggregation_result.md",
    "partial_modes_catalog.jsonl",
    "partial_modes_summary.json",
    "partial_runtime_summary.json",
    "partial_warnings_and_failures.json",
)

FINAL_OUTPUTS = (
    "aggregation_result.json",
    "aggregation_result.md",
    "modes_catalog.jsonl",
    "modes_summary.json",
    "runtime_summary.json",
    "warnings_and_failures.json",
)


def _chunk_ids_from_plan(chunk_plan: Dict[str, Any]) -> List[str]:
    return [str(c.get("chunk_id")) for c in (chunk_plan.get("chunks") or []) if c.get("chunk_id")]


def _load_chunk_targets(run_root: Path, chunk_id: str) -> Dict[float, Dict[str, Any]]:
    path = run_root / "worker_results" / chunk_id / "chunk_targets.json"
    if not path.is_file():
        return {}
    try:
        doc = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    out: Dict[float, Dict[str, Any]] = {}
    for t in doc.get("targets") or []:
        if t.get("target_hz") is None:
            continue
        hz = float(t["target_hz"])
        out[hz] = dict(t)
    return out


def _classify_chunk(
    *,
    run_root: Path,
    chunk_id: str,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return (classification, worker_result, solver_result)."""
    chunk_dir = run_root / "worker_results" / chunk_id
    worker_path = chunk_dir / "worker_result.json"
    solver_path = chunk_dir / "solver_result.json"

    if not worker_path.is_file():
        return "missing", None, None

    try:
        worker = load_json(worker_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return "missing", None, None

    if worker.get("status") in ("DRY_RUN_PLANNED",) or worker.get("mode") in (
        "m4_4_1a_dry_run",
        "m4_4_1b_1_smoke_dry_run",
    ):
        if not existing_real_worker_result(worker_path):
            return "missing", worker, None

    status = str(worker.get("status") or "FAIL")
    if status == "FAIL":
        solver = load_json(solver_path) if solver_path.is_file() else None
        return "failed", worker, solver

    if status in PASS_LIKE or existing_real_worker_result(worker_path):
        solver = None
        if solver_path.is_file():
            try:
                solver = load_json(solver_path)
            except (OSError, ValueError, json.JSONDecodeError):
                solver = None
        return "completed", worker, solver

    return "missing", worker, None


def _collect_mode_records(
    *,
    repo_root: Path,
    run_root: Path,
    chunk_id: str,
    worker: Dict[str, Any],
    solver: Optional[Dict[str, Any]],
    target_meta: Dict[float, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    worker_rel = rel(run_root / "worker_results" / chunk_id / "worker_result.json", repo_root=repo_root)
    solver_rel = rel(run_root / "worker_results" / chunk_id / "solver_result.json", repo_root=repo_root)

    if solver and isinstance(solver.get("targets"), list):
        for row in solver["targets"]:
            target_hz = row.get("target_frequency_hz") or row.get("target_hz")
            if target_hz is None:
                continue
            t_hz = float(target_hz)
            meta = target_meta.get(t_hz) or {}
            mode_list = row.get("accepted_modes") or []
            if not mode_list:
                mode_list = row.get("accepted_frequencies_hz") or []
            for mi, am in enumerate(mode_list):
                if isinstance(am, dict) and am.get("frequency_hz") is not None:
                    f_hz = float(am["frequency_hz"])
                elif isinstance(am, (int, float)):
                    f_hz = float(am)
                else:
                    continue
                rec = {
                    "frequency_hz": round(f_hz, 6),
                    "chunk_id": chunk_id,
                    "target_hz": t_hz,
                    "zone_id": meta.get("zone_id") or row.get("zone_id"),
                    "spacing_hz": meta.get("spacing_hz"),
                    "window_hz": meta.get("window_hz"),
                    "source": "solver_result.targets.accepted_modes",
                    "source_worker_result": worker_rel,
                    "source_solver_result": solver_rel,
                    "mode_index": mi,
                    "target_index": row.get("target_index"),
                }
                if isinstance(am, dict):
                    merge_participation_into_catalog_record(rec, am)
                    merge_audio_coupling_into_catalog_record(rec, am)
                    merge_provenance_into_catalog_record(rec, am)
                records.append(rec)
        if records:
            return records

    freqs = worker.get("unique_modes") or worker.get("accepted_modes") or []
    if isinstance(freqs, list) and freqs and isinstance(freqs[0], dict):
        for mi, am in enumerate(freqs):
            f_hz = float(am.get("frequency_hz", 0))
            rec = {
                "frequency_hz": round(f_hz, 6),
                "chunk_id": chunk_id,
                "target_hz": am.get("target_hz"),
                "zone_id": am.get("zone_id"),
                "source": "worker_result.unique_modes",
                "source_worker_result": worker_rel,
                "source_solver_result": solver_rel,
                "mode_index": mi,
            }
            merge_participation_into_catalog_record(rec, am)
            merge_audio_coupling_into_catalog_record(rec, am)
            merge_provenance_into_catalog_record(rec, am)
            records.append(rec)
        return records

    for mi, f_hz in enumerate(freqs if isinstance(freqs, list) else []):
        f = float(f_hz)
        meta = target_meta.get(f) or {}
        nearest_t = None
        if target_meta:
            nearest_t = min(target_meta.keys(), key=lambda t: abs(t - f))
        records.append(
            {
                "frequency_hz": round(f, 6),
                "chunk_id": chunk_id,
                "target_hz": nearest_t,
                "zone_id": meta.get("zone_id"),
                "spacing_hz": meta.get("spacing_hz"),
                "window_hz": meta.get("window_hz"),
                "source": "worker_result.unique_modes",
                "source_worker_result": worker_rel,
                "source_solver_result": solver_rel,
                "mode_index": mi,
            }
        )
    return records


def _dedupe_catalog(
    records: Sequence[Dict[str, Any]],
    *,
    tol_hz: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (deduped_catalog, merge_groups)."""
    if not records:
        return [], []
    sorted_recs = sorted(records, key=lambda r: float(r["frequency_hz"]))
    groups: List[List[Dict[str, Any]]] = [[sorted_recs[0]]]
    for rec in sorted_recs[1:]:
        if abs(float(rec["frequency_hz"]) - float(groups[-1][-1]["frequency_hz"])) <= tol_hz:
            groups[-1].append(rec)
        else:
            groups.append([rec])

    deduped: List[Dict[str, Any]] = []
    merge_meta: List[Dict[str, Any]] = []
    for group in groups:
        rep = dict(group[0])
        rep["frequency_hz"] = round(
            sum(float(g["frequency_hz"]) for g in group) / len(group), 6
        )
        rep["provenance_count"] = len(group)
        rep["provenance_chunk_ids"] = sorted({str(g["chunk_id"]) for g in group})
        rep["provenance_sources"] = [g.get("source") for g in group]
        for g in group:
            if g.get("participation_status") in ("computed", "fallback"):
                merge_participation_into_catalog_record(rep, g)
                merge_audio_coupling_into_catalog_record(rep, g)
                break
        else:
            for g in group:
                if g.get("dominant_region"):
                    merge_participation_into_catalog_record(rep, g)
                    merge_audio_coupling_into_catalog_record(rep, g)
                    break
            else:
                for g in group:
                    if g.get("audio_coupling_status") in ("computed", "partial", "proxy"):
                        merge_audio_coupling_into_catalog_record(rep, g)
                        break
        if len(group) > 1:
            merge_meta.append(
                {
                    "representative_frequency_hz": rep["frequency_hz"],
                    "merged_count": len(group),
                    "chunk_ids": rep["provenance_chunk_ids"],
                }
            )
        deduped.append(rep)
    return deduped, merge_meta


def _render_result_md(report: Dict[str, Any], *, partial: bool) -> str:
    title = "Partial aggregation" if partial else "Full aggregation"
    lines = [
        f"# {title} — {report.get('sample_id')}",
        "",
        f"- status: **{report.get('status')}**",
    ]
    if partial:
        lines.append(f"- partial_ok: **{report.get('partial_ok')}**")
    lines.extend(
        [
        f"- final_aggregation_ready: **{report.get('final_aggregation_ready')}**",
        f"- planned chunks: **{report.get('planned_chunk_count')}**",
        f"- completed: **{report.get('completed_chunk_count')}**",
        f"- missing: **{report.get('missing_chunk_count')}**",
        f"- failed: **{report.get('failed_chunk_count')}**",
        f"- raw modes: **{report.get('raw_mode_count')}**",
        f"- deduped modes: **{report.get('deduped_mode_count')}**",
        f"- dedupe tolerance: **{report.get('dedupe_tolerance_hz')} Hz**",
        "",
        "## Completed chunks",
        "",
        ]
    )
    for cid in report.get("completed_chunks") or []:
        lines.append(f"- `{cid}`")
    if partial:
        lines.extend(["", "## Missing chunks", ""])
        for cid in report.get("missing_chunks") or []:
            lines.append(f"- `{cid}`")
    if report.get("failed_chunks"):
        lines.extend(["", "## Failed chunks", ""])
        for cid in report.get("failed_chunks") or []:
            lines.append(f"- `{cid}`")
    lines.append("")
    return "\n".join(lines)


def build_aggregation_report(
    *,
    repo_root: Path,
    run_root: Path,
    partial_ok: bool,
) -> Dict[str, Any]:
    chunk_plan_path = run_root / "lprod" / "worker_chunk_plan.preview.json"
    target_plan_path = run_root / "lprod" / "lprod_target_plan.json"
    manifest_path = run_root / "pipeline_run_manifest.json"

    errors: List[str] = []
    if not chunk_plan_path.is_file():
        errors.append("missing lprod/worker_chunk_plan.preview.json")

    chunk_plan = load_json(chunk_plan_path) if chunk_plan_path.is_file() else {"chunks": []}
    target_plan = load_json(target_plan_path) if target_plan_path.is_file() else {}
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}

    planned_ids = _chunk_ids_from_plan(chunk_plan)
    completed: List[str] = []
    missing: List[str] = []
    failed: List[str] = []
    chunk_details: List[Dict[str, Any]] = []
    all_records: List[Dict[str, Any]] = []
    total_attempted = 0
    total_passed = 0
    warnings: List[str] = []
    failures: List[str] = []

    for chunk_id in planned_ids:
        classification, worker, solver = _classify_chunk(run_root=run_root, chunk_id=chunk_id)
        detail: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "classification": classification,
        }
        if worker:
            detail["worker_status"] = worker.get("status")
            detail["targets_attempted"] = worker.get("targets_attempted")
            detail["targets_passed"] = worker.get("targets_passed")
            total_attempted += int(worker.get("targets_attempted") or 0)
            total_passed += int(worker.get("targets_passed") or 0)
        if classification == "completed" and worker:
            completed.append(chunk_id)
            target_meta = _load_chunk_targets(run_root, chunk_id)
            recs = _collect_mode_records(
                repo_root=repo_root,
                run_root=run_root,
                chunk_id=chunk_id,
                worker=worker,
                solver=solver,
                target_meta=target_meta,
            )
            detail["mode_count"] = len(recs)
            all_records.extend(recs)
        elif classification == "failed":
            failed.append(chunk_id)
            failures.append(f"{chunk_id}: worker_status={worker.get('status') if worker else 'unknown'}")
        else:
            missing.append(chunk_id)
        chunk_details.append(detail)

    if failed and not partial_ok:
        errors.append(f"failed chunks present ({len(failed)}); use --partial-ok or fix workers")

    if missing and not partial_ok:
        errors.append(
            f"missing {len(missing)} of {len(planned_ids)} chunks; pass --partial-ok for partial aggregation"
        )

    deduped, merge_groups = _dedupe_catalog(all_records, tol_hz=DEDUPE_TOLERANCE_HZ)
    if merge_groups:
        warnings.append(
            f"frequency dedupe merged {len(merge_groups)} groups within {DEDUPE_TOLERANCE_HZ} Hz"
        )

    final_ready = not missing and not failed and len(completed) == len(planned_ids)
    if final_ready:
        status = FULL_STATUS
    elif errors and not partial_ok:
        status = "FAIL"
    elif partial_ok and completed and not failed:
        status = PARTIAL_STATUS
    elif partial_ok and completed:
        status = PARTIAL_STATUS
        warnings.append("some chunks failed but partial_ok set")
    else:
        status = "FAIL"

    schema = "m4_aggregation_result_v1" if final_ready else "m4_partial_aggregation_result_v1"

    return {
        "schema": schema,
        "will_execute": False,
        "generated_utc": utc_now(),
        "sample_id": manifest.get("sample_id") or chunk_plan.get("sample_id"),
        "run_id": manifest.get("run_id") or chunk_plan.get("run_id"),
        "status": status,
        "partial_ok": bool(partial_ok),
        "final_aggregation_ready": final_ready,
        "planned_chunk_count": len(planned_ids),
        "completed_chunk_count": len(completed),
        "missing_chunk_count": len(missing),
        "failed_chunk_count": len(failed),
        "completed_chunks": completed,
        "missing_chunks": missing,
        "failed_chunks": failed,
        "chunk_details": chunk_details,
        "total_targets_attempted": total_attempted,
        "total_targets_passed": total_passed,
        "raw_mode_count": len(all_records),
        "deduped_mode_count": len(deduped),
        "dedupe_tolerance_hz": DEDUPE_TOLERANCE_HZ,
        "dedupe_merge_groups": merge_groups,
        "frequency_range_hz": target_plan.get("frequency_range_hz"),
        "unique_modes_hz": [r["frequency_hz"] for r in deduped],
        "all_mode_records": all_records,
        "deduped_catalog": deduped,
        "errors": errors,
        "warnings": warnings,
        "failures": failures,
    }


def _enrich_catalog_participation(records: Sequence[Dict[str, Any]]) -> None:
    for rec in records:
        if rec.get("participation_status") in ("computed", "fallback"):
            enrich_participation_catalog_metadata(rec)


def _write_common_artifacts(
    *,
    repo_root: Path,
    paths: Dict[str, Path],
    catalog_path: Path,
    modes_summary_path: Path,
    runtime_path: Path,
    warn_fail_path: Path,
    plot_path: Path,
    report: Dict[str, Any],
    result_body: Dict[str, Any],
    deduped_catalog: List[Dict[str, Any]],
    all_records: List[Dict[str, Any]],
    partial: bool,
) -> None:
    _enrich_catalog_participation(all_records)
    _enrich_catalog_participation(deduped_catalog)

    with catalog_path.open("w", encoding="utf-8") as fh:
        for rec in sorted(all_records, key=lambda r: float(r["frequency_hz"])):
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    prov_path = catalog_path.parent / "mode_provenance.jsonl"
    deduped_path = catalog_path.parent / "modes_catalog_deduped.jsonl"
    with prov_path.open("w", encoding="utf-8") as fh:
        for rec in sorted(all_records, key=lambda r: float(r["frequency_hz"])):
            prov = {k: rec.get(k) for k in PROVENANCE_FIELD_KEYS if k in rec}
            prov.update(
                {
                    "sample_id": report.get("sample_id"),
                    "run_id": report.get("run_id"),
                    "frequency_hz": rec.get("frequency_hz"),
                    "chunk_id": rec.get("chunk_id"),
                    "target_hz": rec.get("target_hz"),
                    "cavity_air_share": rec.get("cavity_air_share"),
                    "exterior_air_share": rec.get("exterior_air_share"),
                }
            )
            fh.write(json.dumps(prov, sort_keys=True) + "\n")
    with deduped_path.open("w", encoding="utf-8") as fh:
        for rec in sorted(deduped_catalog, key=lambda r: float(r["frequency_hz"])):
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    region_counts: Dict[str, int] = {}
    coupling_counts: Dict[str, int] = {}
    participation_computed = 0
    for rec in deduped_catalog:
        dom = str(rec.get("dominant_region") or "unknown")
        region_counts[dom] = region_counts.get(dom, 0) + 1
        coupling = str(rec.get("coupling_class") or "weak_or_unknown")
        coupling_counts[coupling] = coupling_counts.get(coupling, 0) + 1
        if rec.get("participation_status") in ("computed", "fallback"):
            participation_computed += 1

    audio_summary = summarize_audio_coupling(deduped_catalog)
    modes_summary = {
        "schema": "m4_partial_modes_summary_v1" if partial else "m4_modes_summary_v1",
        "generated_utc": report.get("generated_utc"),
        "sample_id": report.get("sample_id"),
        "run_id": report.get("run_id"),
        "dedupe_tolerance_hz": DEDUPE_TOLERANCE_HZ,
        "raw_mode_count": report.get("raw_mode_count"),
        "deduped_mode_count": report.get("deduped_mode_count"),
        "unique_modes_hz": report.get("unique_modes_hz"),
        "frequency_range_hz": report.get("frequency_range_hz"),
        "dominant_region_counts": region_counts,
        "normalized_dominant_region_counts": region_counts,
        "coupling_class_counts": coupling_counts,
        "share_summary": summarize_participation_shares(deduped_catalog),
        "stk_damping_guidance": STK_DAMPING_GUIDANCE,
        "audio_coupling_computed_count": audio_summary.get("audio_coupling_computed_count"),
        "bridge_coupling_available_count": audio_summary.get("bridge_coupling_available_count"),
        "mic_proxy_available_count": audio_summary.get("mic_proxy_available_count"),
        "radiation_proxy_summary": audio_summary.get("radiation_proxy_summary"),
        "modal_norm_summary": audio_summary.get("modal_norm_summary"),
        "audio_coupling_summary": audio_summary,
        "stk_rom_guidance": STK_ROM_GUIDANCE,
        "participation_computed_count": participation_computed,
        "by_chunk": [
            {
                "chunk_id": d["chunk_id"],
                "mode_count": d.get("mode_count"),
                "targets_passed": d.get("targets_passed"),
            }
            for d in report.get("chunk_details") or []
            if d.get("classification") == "completed"
        ],
    }
    write_json_atomic(modes_summary_path, modes_summary)

    runtime_summary = {
        "schema": "m4_partial_runtime_summary_v1" if partial else "m4_runtime_summary_v1",
        "generated_utc": report.get("generated_utc"),
        "aggregation_only": True,
        "no_solver_executed": True,
        "planned_chunk_count": report.get("planned_chunk_count"),
        "completed_chunk_count": report.get("completed_chunk_count"),
        "total_targets_attempted": report.get("total_targets_attempted"),
        "total_targets_passed": report.get("total_targets_passed"),
        "raw_mode_count": report.get("raw_mode_count"),
        "deduped_mode_count": report.get("deduped_mode_count"),
        "participation_computed_count": modes_summary.get("participation_computed_count"),
        "dominant_region_counts": modes_summary.get("dominant_region_counts"),
        "normalized_dominant_region_counts": modes_summary.get("normalized_dominant_region_counts"),
        "coupling_class_counts": modes_summary.get("coupling_class_counts"),
        "share_summary": modes_summary.get("share_summary"),
        "stk_damping_guidance": modes_summary.get("stk_damping_guidance"),
        "audio_coupling_computed_count": modes_summary.get("audio_coupling_computed_count"),
        "bridge_coupling_available_count": modes_summary.get("bridge_coupling_available_count"),
        "mic_proxy_available_count": modes_summary.get("mic_proxy_available_count"),
        "radiation_proxy_summary": modes_summary.get("radiation_proxy_summary"),
        "modal_norm_summary": modes_summary.get("modal_norm_summary"),
        "audio_coupling_summary": modes_summary.get("audio_coupling_summary"),
        "stk_rom_guidance": modes_summary.get("stk_rom_guidance"),
    }
    try:
        from v2_b3_m4_runtime_provenance import (  # noqa: E402
            collect_m4_runtime_provenance,
            merge_runtime_summary,
        )

        run_root = runtime_path.parent.parent
        prov = collect_m4_runtime_provenance(
            run_root=run_root,
            workers_requested=int(
                (load_json(run_root / "m4_run_one_sample_plan.json") or {}).get("workers") or 1
            ),
        )
        runtime_summary = merge_runtime_summary(runtime_summary, prov)
    except Exception:
        pass
    write_json_atomic(runtime_path, runtime_summary)

    warn_fail = {
        "schema": "m4_partial_warnings_and_failures_v1" if partial else "m4_warnings_and_failures_v1",
        "generated_utc": report.get("generated_utc"),
        "warnings": report.get("warnings") or [],
        "failures": report.get("failures") or [],
        "missing_chunks": report.get("missing_chunks") or [],
        "failed_chunks": report.get("failed_chunks") or [],
    }
    write_json_atomic(warn_fail_path, warn_fail)

    extra_plots = _try_mode_plots(
        agg_dir=plot_path.parent,
        deduped=deduped_catalog,
        report=result_body,
        partial=partial,
        frequency_plot_path=plot_path,
    )

    result_body["output_paths"] = {k: rel(v, repo_root=repo_root) for k, v in paths.items()}
    result_body["output_paths"][plot_path.name] = rel(plot_path, repo_root=repo_root)
    for name, p in extra_plots.items():
        result_body["output_paths"][name] = rel(p, repo_root=repo_root)


def _write_outputs(
    *,
    repo_root: Path,
    run_root: Path,
    report: Dict[str, Any],
    force: bool,
) -> None:
    agg_dir = run_root / "aggregation"
    agg_dir.mkdir(parents=True, exist_ok=True)
    all_records = list(report.get("all_mode_records") or [])
    deduped_catalog = list(report.get("deduped_catalog") or [])
    result_body = {k: v for k, v in report.items() if k not in ("all_mode_records", "deduped_catalog")}
    final_ready = bool(report.get("final_aggregation_ready"))

    if final_ready:
        paths = {name: agg_dir / name for name in FINAL_OUTPUTS}
        plot_path = agg_dir / "mode_frequency_plot.png"
        for p in list(paths.values()) + [plot_path]:
            if p.is_file() and not force:
                raise FileExistsError(f"aggregation output exists (use --force): {p}")
        result_path = paths["aggregation_result.json"]
        write_json_atomic(result_path, result_body)
        paths["aggregation_result.md"].write_text(
            _render_result_md(result_body, partial=False), encoding="utf-8"
        )
        _write_common_artifacts(
            repo_root=repo_root,
            paths=paths,
            catalog_path=paths["modes_catalog.jsonl"],
            modes_summary_path=paths["modes_summary.json"],
            runtime_path=paths["runtime_summary.json"],
            warn_fail_path=paths["warnings_and_failures.json"],
            plot_path=plot_path,
            report=report,
            result_body=result_body,
            deduped_catalog=deduped_catalog,
            all_records=all_records,
            partial=False,
        )
        write_json_atomic(result_path, result_body)
        report["output_paths"] = result_body["output_paths"]
        return

    paths = {name: agg_dir / name for name in PARTIAL_OUTPUTS}
    plot_path = agg_dir / "partial_mode_frequency_plot.png"
    for p in list(paths.values()) + [plot_path]:
        if p.is_file() and not force:
            raise FileExistsError(f"aggregation output exists (use --force): {p}")
    result_path = paths["partial_aggregation_result.json"]
    write_json_atomic(result_path, result_body)
    paths["partial_aggregation_result.md"].write_text(
        _render_result_md(result_body, partial=True), encoding="utf-8"
    )
    _write_common_artifacts(
        repo_root=repo_root,
        paths=paths,
        catalog_path=paths["partial_modes_catalog.jsonl"],
        modes_summary_path=paths["partial_modes_summary.json"],
        runtime_path=paths["partial_runtime_summary.json"],
        warn_fail_path=paths["partial_warnings_and_failures.json"],
        plot_path=plot_path,
        report=report,
        result_body=result_body,
        deduped_catalog=deduped_catalog,
        all_records=all_records,
        partial=True,
    )
    write_json_atomic(result_path, result_body)
    report["output_paths"] = result_body["output_paths"]


COUPLING_CLASS_COLORS = {
    "top_back_mixed": "#e67e22",
    "back_dominant": "#3498db",
    "top_dominant": "#2ecc71",
    "air_dominant": "#9b59b6",
    "weak_or_unknown": "#95a5a6",
}

DOMINANT_REGION_MARKERS = {
    "top": "^",
    "back": "s",
    "air": "o",
    "unknown": "x",
}


def _mode_scalar(rec: Mapping[str, Any], key: str, *, fallbacks: Sequence[str] = ()) -> Optional[float]:
    for name in (key,) + tuple(fallbacks):
        val = rec.get(name)
        if val is None:
            continue
        try:
            out = float(val)
        except (TypeError, ValueError):
            continue
        if out == out:  # finite
            return out
    return None


def _apply_freq_band(ax: Any, report: Mapping[str, Any]) -> None:
    band = report.get("frequency_range_hz") or [60, 550]
    if len(band) == 2:
        ax.set_xlim(float(band[0]), float(band[1]))


def _try_mode_plots(
    *,
    agg_dir: Path,
    deduped: Sequence[Dict[str, Any]],
    report: Dict[str, Any],
    partial: bool,
    frequency_plot_path: Path,
) -> Dict[str, Path]:
    """Write frequency-only and audio-relevant mode plots from catalog metadata."""
    out: Dict[str, Path] = {}
    if not deduped:
        return out
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
    except ImportError:
        report.setdefault("warnings", []).append("matplotlib unavailable; skipped mode plots")
        return out

    agg_dir.mkdir(parents=True, exist_ok=True)
    chunk_note = f"{report.get('completed_chunk_count')}/{report.get('planned_chunk_count')} chunks"
    label = "Partial modes" if partial else "Aggregated modes"

    # Legacy frequency-only plot (y=1)
    freqs = [float(r["frequency_hz"]) for r in deduped]
    if freqs:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(freqs, [1.0] * len(freqs), s=12, alpha=0.7, c="#7f8c8d")
        ax.set_xlabel("frequency_hz")
        ax.set_ylabel("mode index (unit)")
        ax.set_title(f"{label} — frequency only ({chunk_note})")
        _apply_freq_band(ax, report)
        fig.tight_layout()
        frequency_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(frequency_plot_path, dpi=120)
        plt.close(fig)

    def _scatter_by_coupling(
        *,
        y_key: str,
        y_label: str,
        filename: str,
        fallbacks: Sequence[str] = (),
        size_scale: float = 120.0,
    ) -> None:
        groups: Dict[str, List[Tuple[float, float]]] = {}
        for rec in deduped:
            yv = _mode_scalar(rec, y_key, fallbacks=fallbacks)
            if yv is None:
                continue
            cc = str(rec.get("coupling_class") or "weak_or_unknown")
            groups.setdefault(cc, []).append((float(rec["frequency_hz"]), yv))
        if not groups:
            report.setdefault("warnings", []).append(f"no data for plot {filename}")
            return
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for cc, pts in sorted(groups.items()):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            sizes = [max(8.0, min(80.0, size_scale * abs(y))) for y in ys]
            ax.scatter(
                xs,
                ys,
                s=sizes,
                alpha=0.75,
                label=cc,
                c=COUPLING_CLASS_COLORS.get(cc, "#95a5a6"),
                edgecolors="white",
                linewidths=0.3,
            )
        ax.set_xlabel("frequency_hz")
        ax.set_ylabel(y_label)
        ax.set_title(f"{label} — {y_label} by coupling_class ({chunk_note})")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        _apply_freq_band(ax, report)
        fig.tight_layout()
        path = agg_dir / filename
        fig.savefig(path, dpi=120)
        plt.close(fig)
        out[filename] = path

    _scatter_by_coupling(
        y_key="radiation_proxy",
        y_label="radiation_proxy",
        filename="mode_frequency_vs_radiation_proxy.png",
        fallbacks=("mic_output_proxy",),
        size_scale=200.0,
    )
    _scatter_by_coupling(
        y_key="mic_output_proxy",
        y_label="mic_output_proxy",
        filename="mode_frequency_vs_mic_output_proxy.png",
        size_scale=200.0,
    )
    _scatter_by_coupling(
        y_key="bridge_excitation_coupling",
        y_label="bridge_excitation_coupling",
        filename="mode_frequency_vs_bridge_excitation.png",
        fallbacks=("bridge_excitation_abs",),
        size_scale=200.0,
    )

    # Top/back/air shares — three series
    share_groups: Dict[str, List[Tuple[float, float]]] = {
        "top_share": [],
        "back_share": [],
        "air_share": [],
    }
    for rec in deduped:
        fh = float(rec["frequency_hz"])
        for sk in share_groups:
            sv = _mode_scalar(rec, sk)
            if sv is not None:
                share_groups[sk].append((fh, sv))
    if any(share_groups.values()):
        fig, ax = plt.subplots(figsize=(11, 4.5))
        share_colors = {"top_share": "#2ecc71", "back_share": "#3498db", "air_share": "#9b59b6"}
        for sk, pts in share_groups.items():
            if not pts:
                continue
            ax.scatter(
                [p[0] for p in pts],
                [p[1] for p in pts],
                s=22,
                alpha=0.65,
                label=sk,
                c=share_colors.get(sk, "#7f8c8d"),
            )
        ax.set_xlabel("frequency_hz")
        ax.set_ylabel("normalized share")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{label} — top/back/air shares ({chunk_note})")
        ax.legend(loc="upper right", fontsize=8)
        _apply_freq_band(ax, report)
        fig.tight_layout()
        share_path = agg_dir / "mode_frequency_vs_top_back_air_share.png"
        fig.savefig(share_path, dpi=120)
        plt.close(fig)
        out[share_path.name] = share_path

    return out


def _write_manifest_preview(
    *,
    run_root: Path,
    manifest: Dict[str, Any],
    report: Dict[str, Any],
) -> None:
    preview = json.loads(json.dumps(manifest))
    preview["updated_utc"] = utc_now()
    preview["will_execute"] = False
    preview["mode"] = "m4_4_1b_3_partial_aggregation"
    preview["terminal_status"] = TERMINAL_PARTIAL
    preview["pipeline_terminal_unchanged_note"] = (
        "main pipeline_run_manifest.json not modified by partial aggregation"
    )
    preview["partial_aggregation"] = {
        "status": report.get("status"),
        "final_aggregation_ready": report.get("final_aggregation_ready"),
        "completed_chunks": report.get("completed_chunks"),
        "missing_chunks": report.get("missing_chunks"),
        "output_paths": report.get("output_paths"),
    }
    stages = preview.setdefault("stages", {})
    st5 = stages.setdefault("stage5_workers", {})
    if st5.get("status") not in ("PASS",):
        st5["status"] = "PARTIAL_PASS"
    st6 = stages.setdefault("stage6_aggregate", {})
    st6["status"] = "PARTIAL_READY"
    st6["partial_aggregation_status"] = report.get("status")
    st6["updated_utc"] = utc_now()
    write_json_atomic(run_root / "pipeline_run_manifest.m4_4_partial_aggregation_preview.json", preview)


def _write_full_manifest_preview(
    *,
    run_root: Path,
    manifest: Dict[str, Any],
    report: Dict[str, Any],
) -> None:
    preview = json.loads(json.dumps(manifest))
    preview["updated_utc"] = utc_now()
    preview["will_execute"] = False
    preview["mode"] = "m4_4_full_aggregation"
    preview["terminal_status"] = TERMINAL_FULL
    preview["pipeline_terminal_unchanged_note"] = (
        "main pipeline_run_manifest.json not modified by full aggregation preview"
    )
    preview["full_aggregation"] = {
        "status": report.get("status"),
        "final_aggregation_ready": report.get("final_aggregation_ready"),
        "completed_chunks": report.get("completed_chunks"),
        "deduped_mode_count": report.get("deduped_mode_count"),
        "output_paths": report.get("output_paths"),
    }
    stages = preview.setdefault("stages", {})
    st5 = stages.setdefault("stage5_workers", {})
    st5["status"] = "PASS"
    st5["updated_utc"] = utc_now()
    st6 = stages.setdefault("stage6_aggregate", {})
    st6["status"] = "PASS"
    st6["aggregation_status"] = report.get("status")
    st6["updated_utc"] = utc_now()
    write_json_atomic(run_root / "pipeline_run_manifest.m4_4_full_aggregation_preview.json", preview)


def run_dry_run(*, repo_root: Path, run_root: Path, partial_ok: bool) -> int:
    report = build_aggregation_report(repo_root=repo_root, run_root=run_root, partial_ok=partial_ok)
    if report.get("errors") and not partial_ok:
        print("error: aggregation precheck failed:", file=sys.stderr)
        for e in report["errors"]:
            print(f"  - {e}", file=sys.stderr)
        return 2

    print("will_execute=false")
    print(f"status={report.get('status')}")
    print(f"planned_chunks={report.get('planned_chunk_count')}")
    print(f"completed_chunks={report.get('completed_chunk_count')}")
    print(f"missing_chunks={report.get('missing_chunk_count')}")
    print(f"failed_chunks={report.get('failed_chunk_count')}")
    print(f"completed={report.get('completed_chunks')}")
    print(f"missing={report.get('missing_chunks')}")
    print(f"raw_mode_count={report.get('raw_mode_count')}")
    print(f"deduped_mode_count={report.get('deduped_mode_count')}")
    print(f"final_aggregation_ready={report.get('final_aggregation_ready')}")
    print("no solver executed")
    return 0


def run_execute(*, repo_root: Path, run_root: Path, partial_ok: bool, force: bool) -> int:
    report = build_aggregation_report(repo_root=repo_root, run_root=run_root, partial_ok=partial_ok)
    if report.get("errors"):
        print("error: aggregation failed:", file=sys.stderr)
        for e in report["errors"]:
            print(f"  - {e}", file=sys.stderr)
        return 2

    if report.get("status") == "FAIL":
        print("error: aggregation status FAIL", file=sys.stderr)
        return 2

    report["will_execute"] = True
    try:
        _write_outputs(
            repo_root=repo_root,
            run_root=run_root,
            report=report,
            force=force,
        )
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    manifest = load_json(run_root / "pipeline_run_manifest.json")
    if report.get("final_aggregation_ready"):
        _write_full_manifest_preview(run_root=run_root, manifest=manifest, report=report)
        terminal = TERMINAL_FULL
        from v2_b3_m4_terminal_status_lib import promote_after_aggregation_pass  # noqa: WPS433

        promote_after_aggregation_pass(run_root)
    else:
        _write_manifest_preview(run_root=run_root, manifest=manifest, report=report)
        terminal = TERMINAL_PARTIAL

    print(f"status={report.get('status')}")
    print(f"planned_chunks={report.get('planned_chunk_count')}")
    print(f"completed_chunks={report.get('completed_chunk_count')}")
    print(f"missing_chunks={report.get('missing_chunk_count')}")
    print(f"failed_chunks={report.get('failed_chunk_count')}")
    print(f"raw_mode_count={report.get('raw_mode_count')}")
    print(f"deduped_mode_count={report.get('deduped_mode_count')}")
    print(f"final_aggregation_ready={report.get('final_aggregation_ready')}")
    print(f"terminal_status={terminal}")
    print("no solver executed")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="M4.4.1b-3/4: aggregate L_prod worker results (partial or full)."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--partial-ok",
        action="store_true",
        help="Allow aggregation when planned chunks are missing (partial mode).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing partial or full aggregation outputs.",
    )
    args = parser.parse_args(argv)

    if args.dry_run and args.execute:
        print("error: use --dry-run or --execute, not both", file=sys.stderr)
        return 2
    if not args.dry_run and not args.execute:
        print("error: specify --dry-run or --execute", file=sys.stderr)
        return 2

    repo_root = detect_repo_root(SCRIPT_DIR)
    run_root = args.run_dir if args.run_dir.is_absolute() else repo_root / args.run_dir
    run_root = run_root.resolve()

    if args.dry_run:
        return run_dry_run(repo_root=repo_root, run_root=run_root, partial_ok=bool(args.partial_ok))
    return run_execute(
        repo_root=repo_root,
        run_root=run_root,
        partial_ok=bool(args.partial_ok),
        force=bool(args.force),
    )


if __name__ == "__main__":
    raise SystemExit(main())
