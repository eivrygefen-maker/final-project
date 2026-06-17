#!/usr/bin/env python3
"""Post-aggregation shape-aware physical acceptance (advisory; does not block production)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
FEM_SCRIPTS = SCRIPT_DIR.parents[3] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(FEM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FEM_SCRIPTS))

from m4_shape_context import resolve_shape_context_from_sample_input  # noqa: E402
from m4_shape_registry import infer_shape_from_sample_id, normalize_shape_key  # noqa: E402
from m4_shape_validation_profile import (  # noqa: E402
    ShapeValidationProfile,
    resolve_shape_validation_profile,
)
from v2_b3_m4_freeze_first_e2e_run import AGG_STATUS_PASS  # noqa: E402
from v2_b3_m4_production_freeze import TERMINAL_PRODUCTION_COMPLETED  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

ACCEPTANCE_SCHEMA = "m4_shape_physical_acceptance_v1"
ACCEPTANCE_JSON_REL = "validation/shape_physical_acceptance.json"
ACCEPTANCE_MD_REL = "validation/shape_physical_acceptance.md"
RECOMMENDED_MIN_SAMPLES = 5


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


def _load_catalog_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except ValueError:
            continue
    return rows


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def _collect_metrics(
    *,
    agg: Mapping[str, Any],
    modes_summary: Mapping[str, Any],
    catalog_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    freqs = [_finite_or_none(r.get("frequency_hz")) for r in catalog_rows]
    freqs_ok = [f for f in freqs if f is not None]
    share_summary = modes_summary.get("share_summary") if isinstance(modes_summary.get("share_summary"), dict) else {}
    audio_summary = (
        modes_summary.get("audio_coupling_summary")
        if isinstance(modes_summary.get("audio_coupling_summary"), dict)
        else {}
    )
    return {
        "raw_mode_count": agg.get("raw_mode_count") or modes_summary.get("raw_mode_count"),
        "deduped_mode_count": agg.get("deduped_mode_count") or modes_summary.get("deduped_mode_count"),
        "completed_chunks": agg.get("completed_chunk_count"),
        "planned_chunks": agg.get("planned_chunk_count"),
        "frequency_min_hz": min(freqs_ok) if freqs_ok else None,
        "frequency_max_hz": max(freqs_ok) if freqs_ok else None,
        "coupling_class_counts": dict(modes_summary.get("coupling_class_counts") or {}),
        "top_share_stats": dict(share_summary.get("top_share") or share_summary.get("top_share_stats") or {}),
        "back_share_stats": dict(share_summary.get("back_share") or share_summary.get("back_share_stats") or {}),
        "air_share_stats": dict(share_summary.get("air_share") or share_summary.get("air_share_stats") or {}),
        "radiation_proxy_stats": dict(modes_summary.get("radiation_proxy_summary") or {}),
        "bridge_excitation_coupling_stats": {
            "bridge_coupling_available_count": modes_summary.get("bridge_coupling_available_count"),
            "audio_coupling_computed_count": modes_summary.get("audio_coupling_computed_count"),
            **(
                audio_summary.get("bridge_excitation_coupling_summary")
                if isinstance(audio_summary.get("bridge_excitation_coupling_summary"), dict)
                else {}
            ),
        },
        "mic_output_proxy_stats": {
            "mic_proxy_available_count": modes_summary.get("mic_proxy_available_count"),
            **(
                audio_summary.get("mic_output_proxy_summary")
                if isinstance(audio_summary.get("mic_output_proxy_summary"), dict)
                else {}
            ),
        },
        "participation_computed_count": modes_summary.get("participation_computed_count"),
        "dominant_region_counts": dict(modes_summary.get("dominant_region_counts") or {}),
    }


def evaluate_pipeline_integrity(
    *,
    run_root: Path,
    agg: Mapping[str, Any],
    sample_input: Mapping[str, Any],
    pipeline: Mapping[str, Any],
    shape_ctx: Mapping[str, Any],
    catalog_rows: Sequence[Mapping[str, Any]],
) -> Tuple[bool, List[str], List[str]]:
    failures: List[str] = []
    warnings: List[str] = []

    terminal = str(pipeline.get("terminal_status") or "")
    if terminal and terminal != TERMINAL_PRODUCTION_COMPLETED:
        warnings.append(f"terminal_status={terminal or 'missing'} (expected {TERMINAL_PRODUCTION_COMPLETED})")

    agg_status = str(agg.get("status") or "")
    if agg_status != AGG_STATUS_PASS:
        failures.append(f"aggregation_status={agg_status or 'missing'}")

    planned = agg.get("planned_chunk_count")
    completed = agg.get("completed_chunk_count")
    if planned is None or completed is None:
        failures.append("aggregation_missing_chunk_counts")
    elif int(planned) != int(completed):
        failures.append(f"aggregation_incomplete:planned={planned}:completed={completed}")

    failed_count = int(agg.get("failed_chunk_count") or len(agg.get("failed_chunks") or []))
    if failed_count > 0:
        failures.append(f"failed_chunks={agg.get('failed_chunks') or failed_count}")

    missing_count = int(agg.get("missing_chunk_count") or len(agg.get("missing_chunks") or []))
    if missing_count > 0:
        failures.append(f"missing_chunks={agg.get('missing_chunks') or missing_count}")

    if not bool(agg.get("final_aggregation_ready")):
        failures.append("final_aggregation_ready!=true")

    if not catalog_rows:
        failures.append("missing_modes_catalog_deduped")

    if not shape_ctx.get("shape_name"):
        failures.append("missing_shape_context")

    expected_shape = str(shape_ctx.get("shape_name") or "")
    sample_shape = str(sample_input.get("shape_name") or "")
    if expected_shape and sample_shape and expected_shape != sample_shape:
        failures.append(f"shape_context_mismatch:sample={sample_shape}:context={expected_shape}")

    for rel in ("scout/mesh", "lprod/mesh"):
        manifest = run_root / rel / ".mesh_manifest.json"
        if manifest.is_file():
            mdoc = _load_json(manifest)
            m_shape = str(mdoc.get("shape_name") or mdoc.get("geometry_shape_type") or "")
            if m_shape and expected_shape:
                norm_expected = normalize_shape_key(expected_shape)
                if m_shape.lower() not in {norm_expected, str(shape_ctx.get("geometry_shape_type") or "").lower()}:
                    if norm_expected == "classic" and m_shape.lower() in {"classical", "classic"}:
                        pass
                    elif m_shape.lower() != norm_expected:
                        warnings.append(f"mesh_manifest_shape_check:{rel}:{m_shape}")

    return len(failures) == 0, failures, warnings


def evaluate_numerical_acceptance(
    *,
    profile: ShapeValidationProfile,
    metrics: Mapping[str, Any],
    catalog_rows: Sequence[Mapping[str, Any]],
) -> Tuple[bool, List[str], List[str]]:
    failures: List[str] = []
    warnings: List[str] = []

    deduped = int(metrics.get("deduped_mode_count") or 0)
    if deduped < profile.deduped_mode_count_min:
        failures.append(
            f"deduped_mode_count={deduped}<min={profile.deduped_mode_count_min}"
        )
    elif deduped < profile.mode_count_warn_below:
        warnings.append(
            f"deduped_mode_count={deduped}<warn_below={profile.mode_count_warn_below}"
        )

    nan_freq = 0
    invalid_freq = 0
    for row in catalog_rows:
        f_hz = _finite_or_none(row.get("frequency_hz"))
        if row.get("frequency_hz") is not None and f_hz is None:
            nan_freq += 1
        elif f_hz is not None and f_hz <= 0:
            invalid_freq += 1
        for key in ("top_share", "back_share", "air_share", "mic_output_proxy", "radiation_proxy"):
            val = _finite_or_none(row.get(key))
            if row.get(key) is not None and val is None:
                warnings.append(f"non_finite_proxy:{key}")
                break

    if nan_freq:
        failures.append(f"nan_or_inf_frequency_count={nan_freq}")
    if invalid_freq:
        failures.append(f"invalid_frequency_count={invalid_freq}")

    participation = int(metrics.get("participation_computed_count") or 0)
    if catalog_rows and participation <= 0:
        warnings.append("participation_fields_missing_in_summary")

    mic_available = int((metrics.get("mic_output_proxy_stats") or {}).get("mic_proxy_available_count") or 0)
    if catalog_rows and mic_available <= 0:
        warnings.append("mic_output_proxy_unavailable_in_summary")

    return len(failures) == 0, failures, warnings


def _dominant_air_fraction(metrics: Mapping[str, Any]) -> float:
    dom = metrics.get("dominant_region_counts") or {}
    total = sum(int(v) for v in dom.values()) or 0
    if total <= 0:
        return 0.0
    air = int(dom.get("air") or dom.get("Air") or 0)
    return air / total


def evaluate_shape_physical_plausibility(
    *,
    profile: ShapeValidationProfile,
    metrics: Mapping[str, Any],
) -> Tuple[bool, List[str], List[str]]:
    failures: List[str] = []
    warnings: List[str] = []

    top_stats = metrics.get("top_share_stats") or {}
    air_stats = metrics.get("air_share_stats") or {}
    top_median = _finite_or_none(top_stats.get("median"))
    air_median = _finite_or_none(air_stats.get("median"))
    air_dominant_frac = _dominant_air_fraction(metrics)

    bridge_stats = metrics.get("bridge_excitation_coupling_stats") or {}
    bridge_available = int(bridge_stats.get("bridge_coupling_available_count") or 0)
    deduped = int(metrics.get("deduped_mode_count") or 0)

    policy = profile.profile_type
    if policy == "shape_relative_body_validation":
        if air_dominant_frac >= 0.4:
            warnings.append(
                f"box_relative:air_dominant_modes_fraction={air_dominant_frac:.2f} "
                "(expected for simple cavity/plate box bodies)"
            )
        if top_median is not None and top_median < 0.12:
            warnings.append(
                f"box_relative:low_top_share_median={top_median:.3f} "
                "(not classical guitar-like; informational only)"
            )
        if deduped > 0 and bridge_available < max(1, int(0.2 * deduped)):
            warnings.append(
                f"box_relative:sparse_bridge_coupling={bridge_available}/{deduped}"
            )
    elif policy in {"classical_reference", "acoustic_reference"}:
        if top_median is not None and top_median < 0.15:
            msg = f"guitar_reference:low_top_share_median={top_median:.3f}"
            if policy == "classical_reference":
                failures.append(msg)
            else:
                warnings.append(msg)
        if air_dominant_frac >= 0.5:
            msg = f"guitar_reference:high_air_dominant_fraction={air_dominant_frac:.2f}"
            if policy == "classical_reference":
                warnings.append(msg)
            else:
                warnings.append(msg)
        if profile.bridge_coupling_policy in {"stricter_guitar_like", "guitar_like"}:
            if deduped > 0 and bridge_available < int(0.5 * deduped):
                msg = f"guitar_reference:low_bridge_coupling_coverage={bridge_available}/{deduped}"
                if policy == "classical_reference":
                    failures.append(msg)
                else:
                    warnings.append(msg)
    else:
        warnings.append(f"unknown_profile_type={policy}:using_generic_relative_checks")
        if deduped > 0 and bridge_available < int(0.25 * deduped):
            warnings.append(f"generic:low_bridge_coupling={bridge_available}/{deduped}")

    plausibility_pass = len(failures) == 0
    return plausibility_pass, failures, warnings


def _overall_status(
    *,
    pipeline_ok: bool,
    numerical_ok: bool,
    plausibility_ok: bool,
    warnings: Sequence[str],
) -> str:
    if not pipeline_ok or not numerical_ok:
        return "FAIL"
    if not plausibility_ok or warnings:
        return "PASS_WITH_WARNING"
    return "PASS"


def evaluate_shape_physical_acceptance(
    *,
    run_root: Path,
    shape_key: Optional[str] = None,
    profile: Optional[ShapeValidationProfile] = None,
) -> Dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    sample_input = _load_json(run_root / "sample" / "sample_input.json")
    pipeline = _load_json(run_root / "pipeline_run_manifest.json")
    agg = _load_json(run_root / "aggregation" / "aggregation_result.json")
    modes_summary = _load_json(run_root / "aggregation" / "modes_summary.json")
    catalog_path = run_root / "aggregation" / "modes_catalog_deduped.jsonl"
    if not catalog_path.is_file():
        catalog_path = run_root / "aggregation" / "modes_catalog.jsonl"
    catalog_rows = _load_catalog_rows(catalog_path)

    sample_id = str(sample_input.get("sample_id") or agg.get("sample_id") or run_root.parent.parent.name)
    run_id = str(agg.get("run_id") or run_root.name)
    resolved_shape = str(
        shape_key
        or sample_input.get("shape_name")
        or (sample_input.get("shape_context") or {}).get("shape_name")
        or infer_shape_from_sample_id(sample_id)
    )
    shape_ctx_doc = dict(sample_input.get("shape_context") or {})
    if not shape_ctx_doc:
        try:
            shape_ctx_doc = resolve_shape_context_from_sample_input(sample_input).to_dict()
        except Exception:
            shape_ctx_doc = {"shape_name": normalize_shape_key(resolved_shape)}

    prof = profile or resolve_shape_validation_profile(resolved_shape)
    metrics = _collect_metrics(agg=agg, modes_summary=modes_summary, catalog_rows=catalog_rows)

    pipeline_ok, pipeline_failures, pipeline_warnings = evaluate_pipeline_integrity(
        run_root=run_root,
        agg=agg,
        sample_input=sample_input,
        pipeline=pipeline,
        shape_ctx=shape_ctx_doc,
        catalog_rows=catalog_rows,
    )
    numerical_ok, numerical_failures, numerical_warnings = evaluate_numerical_acceptance(
        profile=prof,
        metrics=metrics,
        catalog_rows=catalog_rows,
    )
    plausibility_ok, plausibility_failures, plausibility_warnings = evaluate_shape_physical_plausibility(
        profile=prof,
        metrics=metrics,
    )

    warnings = list(pipeline_warnings) + list(numerical_warnings) + list(plausibility_warnings)
    failures = list(pipeline_failures) + list(numerical_failures) + list(plausibility_failures)
    status = _overall_status(
        pipeline_ok=pipeline_ok,
        numerical_ok=numerical_ok,
        plausibility_ok=plausibility_ok,
        warnings=warnings,
    )

    recommendations: List[str] = [
        "musical_usefulness requires 5-10 completed samples per shape before final tuning",
        f"recommended_min_samples={RECOMMENDED_MIN_SAMPLES}",
    ]
    if prof.shape_name == "box":
        recommendations.append(
            "BOX modes are evaluated relative to box body expectations, not classical guitar thresholds"
        )
    if warnings:
        recommendations.append("review warnings in shape_physical_acceptance.md before ROM tuning")

    return {
        "schema": ACCEPTANCE_SCHEMA,
        "generated_utc": utc_now(),
        "sample_id": sample_id,
        "run_id": run_id,
        "shape_name": prof.shape_name,
        "profile_id": prof.profile_id,
        "profile_type": prof.profile_type,
        "status": status,
        "advisory_only": True,
        "blocks_production": False,
        "pipeline_integrity_pass": pipeline_ok,
        "numerical_acceptance_pass": numerical_ok,
        "shape_physical_plausibility_pass": plausibility_ok,
        "musical_usefulness_status": "NOT_EVALUATED_SINGLE_SAMPLE",
        "metrics": metrics,
        "warnings": warnings,
        "failures": failures,
        "layer_failures": {
            "pipeline_integrity": pipeline_failures,
            "numerical_acceptance": numerical_failures,
            "shape_physical_plausibility": plausibility_failures,
        },
        "recommendations": recommendations,
        "shape_context": shape_ctx_doc,
        "physical_acceptance_profile": prof.to_dict(),
    }


def render_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Shape physical acceptance",
        "",
        f"- sample_id: `{report.get('sample_id')}`",
        f"- run_id: `{report.get('run_id')}`",
        f"- shape: `{report.get('shape_name')}`",
        f"- profile: `{report.get('profile_id')}` ({report.get('profile_type')})",
        f"- status: **{report.get('status')}**",
        f"- advisory_only: `{report.get('advisory_only')}` (does not block freeze/compaction)",
        "",
        "## Layer results",
        f"- pipeline_integrity_pass: `{report.get('pipeline_integrity_pass')}`",
        f"- numerical_acceptance_pass: `{report.get('numerical_acceptance_pass')}`",
        f"- shape_physical_plausibility_pass: `{report.get('shape_physical_plausibility_pass')}`",
        f"- musical_usefulness_status: `{report.get('musical_usefulness_status')}`",
        "",
        "## Key metrics",
    ]
    metrics = report.get("metrics") or {}
    lines.extend(
        [
            f"- deduped_mode_count: {metrics.get('deduped_mode_count')}",
            f"- raw_mode_count: {metrics.get('raw_mode_count')}",
            f"- chunks: {metrics.get('completed_chunks')}/{metrics.get('planned_chunks')}",
            f"- frequency_range_hz: {metrics.get('frequency_min_hz')} – {metrics.get('frequency_max_hz')}",
            f"- dominant_region_counts: {metrics.get('dominant_region_counts')}",
        ]
    )
    notes = (report.get("physical_acceptance_profile") or {}).get("notes")
    if notes:
        lines.extend(["", "## Profile notes", str(notes)])
    warnings = list(report.get("warnings") or [])
    failures = list(report.get("failures") or [])
    if warnings:
        lines.extend(["", "## Warnings"] + [f"- {w}" for w in warnings])
    if failures:
        lines.extend(["", "## Failures"] + [f"- {f}" for f in failures])
    recs = list(report.get("recommendations") or [])
    if recs:
        lines.extend(["", "## Recommended next action"] + [f"- {r}" for r in recs])
    if str(report.get("shape_name")) == "box":
        lines.extend(
            [
                "",
                "## Interpretation",
                "BOX results are judged relative to box cavity/plate behavior. "
                "Dominant air/back modes or sparse guitar-like top/back balance are warnings, "
                "not automatic production failures.",
            ]
        )
    return "\n".join(lines) + "\n"


def write_shape_physical_acceptance(
    run_root: Path,
    report: Mapping[str, Any],
    *,
    copy_to_shared: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    Write advisory validation artifacts only under validation/ (and optional shared summaries).

    Does not modify aggregation, freeze, pipeline manifests, or other production artifacts.
    """
    validation_dir = run_root / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_root / ACCEPTANCE_JSON_REL
    md_path = run_root / ACCEPTANCE_MD_REL
    write_json_atomic(json_path, dict(report))
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    if copy_to_shared is not None:
        copy_to_shared.mkdir(parents=True, exist_ok=True)
        shape = str(report.get("shape_name") or "unknown")
        summary_dir = copy_to_shared / shape / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        sid = str(report.get("sample_id") or "sample")
        write_json_atomic(summary_dir / f"{sid}_shape_physical_acceptance.json", dict(report))
        (summary_dir / f"{sid}_shape_physical_acceptance.md").write_text(
            render_markdown_report(report),
            encoding="utf-8",
        )
    return json_path, md_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate shape-aware physical acceptance (advisory).")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--shape", type=str, default="")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate only; do not write any files.")
    parser.add_argument("--copy-to-shared", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_root = args.run_dir.expanduser().resolve()
    if not run_root.is_dir():
        print(f"error: run-dir not found: {run_root}", file=sys.stderr)
        return 2

    report = evaluate_shape_physical_acceptance(run_root=run_root, shape_key=args.shape or None)
    if args.dry_run:
        print("dry_run=true (no files written)")
    else:
        json_path, md_path = write_shape_physical_acceptance(
            run_root,
            report,
            copy_to_shared=args.copy_to_shared,
        )
        print(f"written={json_path}")
        print(f"written={md_path}")
    print(f"status={report.get('status')}")
    print(f"profile_id={report.get('profile_id')}")
    print(f"pipeline_integrity_pass={report.get('pipeline_integrity_pass')}")
    print(f"numerical_acceptance_pass={report.get('numerical_acceptance_pass')}")
    print(f"shape_physical_plausibility_pass={report.get('shape_physical_plausibility_pass')}")
    print(f"blocks_production={report.get('blocks_production')}")
    return 0 if report.get("status") != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
