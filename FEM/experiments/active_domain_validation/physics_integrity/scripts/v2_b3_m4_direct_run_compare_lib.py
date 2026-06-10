#!/usr/bin/env python3
"""Direct read-only numerical comparison of two completed M4 runs (no preconditions)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from v2_b3_m4_worker_run_lib import load_json  # noqa: E402

DIRECT_COMPARE_SCHEMA = "m4_direct_run_compare_v1"

FREQUENCY_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("60_150", 60.0, 150.0),
    ("150_250", 150.0, 250.0),
    ("250_350", 250.0, 350.0),
    ("350_450", 350.0, 450.0),
    ("450_550", 450.0, 550.0),
)

OPTIONAL_READ_PATHS: Tuple[str, ...] = (
    "aggregation/modes_catalog_deduped.jsonl",
    "aggregation/modes_summary.json",
    "aggregation/runtime_summary.json",
    "aggregation/aggregation_result.json",
    "m4_sample_runtime_provenance.json",
    "sample/sample_input.json",
    "freeze/freeze_manifest.json",
    "freeze/physics_identity_manifest.json",
)


def _load_optional(run_root: Path, rel: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    path = run_root / rel
    if not path.is_file():
        return None, False
    try:
        if path.suffix == ".jsonl":
            return None, True
        doc = load_json(path)
        return (doc if isinstance(doc, dict) else None), True
    except (OSError, ValueError, json.JSONDecodeError):
        return None, True


def _load_catalog(run_root: Path) -> Tuple[List[Dict[str, Any]], str, bool]:
    for rel in ("aggregation/modes_catalog_deduped.jsonl", "aggregation/modes_catalog.jsonl"):
        path = run_root / rel
        if not path.is_file():
            continue
        rows: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows, rel, True
    return [], "", False


def _mode_id(row: Mapping[str, Any]) -> str:
    return str(row.get("mode_id") or row.get("dedup_id") or row.get("frequency_hz"))


def greedy_monotonic_freq_pairs(
    ref_rows: Sequence[Mapping[str, Any]],
    cand_rows: Sequence[Mapping[str, Any]],
    *,
    max_distance_hz: float = 5.0,
) -> List[Tuple[int, int]]:
    """Monotonic nearest one-to-one frequency matching (reference frequency order)."""
    ref_freqs = [float(r.get("frequency_hz") or 0.0) for r in ref_rows]
    cand_freqs = [float(r.get("frequency_hz") or 0.0) for r in cand_rows]
    used_c: set[int] = set()
    pairs: List[Tuple[int, int]] = []
    for i in sorted(range(len(ref_freqs)), key=lambda k: ref_freqs[k]):
        best_j = None
        best_d = float("inf")
        for j, cf in enumerate(cand_freqs):
            if j in used_c:
                continue
            d = abs(cf - ref_freqs[i])
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is not None and best_d <= max_distance_hz:
            used_c.add(best_j)
            pairs.append((i, best_j))
    return pairs


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    rx = np.argsort(np.argsort(np.asarray(xs, dtype=float)))
    ry = np.argsort(np.argsort(np.asarray(ys, dtype=float)))
    corr = np.corrcoef(rx, ry)
    return float(corr[0, 1]) if corr.shape == (2, 2) else None


def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), pct))


def _stat_triplet(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"median": None, "p95": None, "maximum": None}
    arr = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "maximum": float(np.max(arr)),
    }


def _top10_overlap(key: str, ref_rows: Sequence[Mapping[str, Any]], cand_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ref_top = sorted(ref_rows, key=lambda r: float(r.get(key) or 0.0), reverse=True)[:10]
    cand_top = sorted(cand_rows, key=lambda r: float(r.get(key) or 0.0), reverse=True)[:10]
    ref_ids = [_mode_id(r) for r in ref_top]
    cand_ids = [_mode_id(r) for r in cand_top]
    return {
        "overlap_count": len(set(ref_ids) & set(cand_ids)),
        "reference_top10_ids": ref_ids,
        "candidate_top10_ids": cand_ids,
    }


def _proxy_metrics(
    ref_rows: Sequence[Mapping[str, Any]],
    cand_rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Tuple[int, int]],
    key: str,
) -> Dict[str, Any]:
    if not any(key in r for r in ref_rows) and not any(key in r for r in cand_rows):
        return {"status": "UNAVAILABLE", "field": key}
    ref_vals: List[float] = []
    cand_vals: List[float] = []
    ratios: List[float] = []
    for i, j in pairs:
        rv = float(ref_rows[i].get(key) or 0.0)
        cv = float(cand_rows[j].get(key) or 0.0)
        ref_vals.append(rv)
        cand_vals.append(cv)
        if rv != 0.0:
            ratios.append(cv / rv)
    abs_diffs = [abs(a - b) for a, b in zip(ref_vals, cand_vals)]
    return {
        "status": "AVAILABLE",
        "field": key,
        "matched_count": len(pairs),
        "spearman_rank_correlation": _spearman(ref_vals, cand_vals),
        "top10_overlap": _top10_overlap(key, ref_rows, cand_rows),
        "median_amplitude_ratio": _percentile(ratios, 50),
        "p95_amplitude_ratio": _percentile(ratios, 95),
        "median_abs_difference": _percentile(abs_diffs, 50),
        "p95_abs_difference": _percentile(abs_diffs, 95),
    }


def _share_metrics(
    ref_rows: Sequence[Mapping[str, Any]],
    cand_rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Tuple[int, int]],
) -> Dict[str, Any]:
    shares = ("top_share", "back_share", "air_share")
    if not any(share in r for r in ref_rows for share in shares):
        return {"status": "UNAVAILABLE"}
    out: Dict[str, Any] = {"status": "AVAILABLE"}
    all_diffs: List[float] = []
    for share in shares:
        diffs = [
            abs(float(cand_rows[j].get(share) or 0.0) - float(ref_rows[i].get(share) or 0.0))
            for i, j in pairs
        ]
        out[share] = {
            "median_abs_difference": _percentile(diffs, 50),
            "p95_abs_difference": _percentile(diffs, 95),
        }
        all_diffs.extend(diffs)
    out["aggregate"] = {
        "median_abs_difference": _percentile(all_diffs, 50),
        "p95_abs_difference": _percentile(all_diffs, 95),
    }
    return out


def _coupling_metrics(
    ref_rows: Sequence[Mapping[str, Any]],
    cand_rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Tuple[int, int]],
) -> Dict[str, Any]:
    if not any("coupling_class" in r for r in ref_rows):
        return {"status": "UNAVAILABLE"}
    agree = 0
    compared = 0
    dom_agree = 0
    dom_compared = 0
    for i, j in pairs:
        rc = str(ref_rows[i].get("coupling_class") or "")
        cc = str(cand_rows[j].get("coupling_class") or "")
        if rc and cc:
            compared += 1
            if rc == cc:
                agree += 1
        rd = str(ref_rows[i].get("dominant_region") or "")
        cd = str(cand_rows[j].get("dominant_region") or "")
        if rd and cd:
            dom_compared += 1
            if rd == cd:
                dom_agree += 1
    return {
        "status": "AVAILABLE",
        "coupling_class_agreement": (agree / compared) if compared else None,
        "coupling_pairs_compared": compared,
        "dominant_region_agreement": (dom_agree / dom_compared) if dom_compared else None,
        "dominant_region_pairs_compared": dom_compared,
    }


def _band_metrics(
    ref_rows: Sequence[Mapping[str, Any]],
    cand_rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Tuple[int, int]],
    matched: Sequence[Mapping[str, Any]],
    lo: float,
    hi: float,
) -> Dict[str, Any]:
    ref_idx = [i for i, r in enumerate(ref_rows) if lo <= float(r.get("frequency_hz") or 0) < hi]
    cand_idx = [j for j, r in enumerate(cand_rows) if lo <= float(r.get("frequency_hz") or 0) < hi]
    matched_ref = {i for i, _ in pairs}
    matched_in_band = sum(1 for i in ref_idx if i in matched_ref)
    rel_errors = [
        float(m["rel_error"])
        for m in matched
        if lo <= float(m["reference_hz"]) < hi
    ]
    recall = (matched_in_band / len(ref_idx)) if ref_idx else None
    return {
        "reference_count": len(ref_idx),
        "rom_count": len(cand_idx),
        "matched_count": matched_in_band,
        "recall": recall,
        "median_rel_frequency_error": _percentile(rel_errors, 50),
        "p95_rel_frequency_error": _percentile(rel_errors, 95),
    }


def _runtime_doc(run_root: Path) -> Dict[str, Any]:
    for rel in ("m4_sample_runtime_provenance.json", "aggregation/runtime_summary.json"):
        path = run_root / rel
        if path.is_file():
            try:
                doc = load_json(path)
                if isinstance(doc, dict):
                    return doc
            except (OSError, ValueError, json.JSONDecodeError):
                pass
    return {}


def _worker_count(run_root: Path, runtime: Mapping[str, Any], agg: Optional[Mapping[str, Any]]) -> Optional[int]:
    for key in ("workers_parallel_observed", "workers_actual_parallel", "worker_count"):
        val = runtime.get(key)
        if val is not None:
            return int(val)
    records = runtime.get("worker_resource_records") or []
    if records:
        return len(records)
    if agg:
        for key in ("planned_chunk_count", "completed_chunk_count"):
            val = agg.get(key)
            if val is not None:
                return int(val)
    return None


def _performance_metrics(ref_root: Path, cand_root: Path) -> Dict[str, Any]:
    ref_rt = _runtime_doc(ref_root)
    cand_rt = _runtime_doc(cand_root)
    ref_agg, _ = _load_optional(ref_root, "aggregation/aggregation_result.json")
    cand_agg, _ = _load_optional(cand_root, "aggregation/aggregation_result.json")
    ref_stage = ref_rt.get("stage_wall_times_s") or {}
    cand_stage = cand_rt.get("stage_wall_times_s") or {}
    ref_worker = float(ref_stage.get("stage5_workers") or ref_rt.get("workers_wall_time_s") or 0) or None
    cand_worker = float(cand_stage.get("stage5_workers") or cand_rt.get("workers_wall_time_s") or 0) or None
    ref_total = float(ref_rt.get("total_wall_time_s") or ref_rt.get("elapsed_s") or 0) or None
    cand_total = float(cand_rt.get("total_wall_time_s") or cand_rt.get("elapsed_s") or 0) or None
    if ref_total is None or ref_total <= 0:
        stage_sum = sum(float(ref_stage.get(s) or 0) for s in ("stage1_scout", "stage2_lprod_checkpoint", "stage5_workers"))
        ref_total = stage_sum or None
    if cand_total is None or cand_total <= 0:
        stage_sum = sum(float(cand_stage.get(s) or 0) for s in ("stage1_scout", "stage2_lprod_checkpoint", "stage5_workers"))
        cand_total = stage_sum or None
    speedup_worker = (ref_worker / cand_worker) if ref_worker and cand_worker and cand_worker > 0 else None
    speedup_total = (ref_total / cand_total) if ref_total and cand_total and cand_total > 0 else None
    ref_peaks = [
        int(r.get("peak_rss_bytes") or r.get("max_rss_bytes") or 0)
        for r in (ref_rt.get("worker_resource_records") or [])
        if r.get("peak_rss_bytes") or r.get("max_rss_bytes")
    ]
    cand_peaks = [
        int(r.get("peak_rss_bytes") or r.get("max_rss_bytes") or 0)
        for r in (cand_rt.get("worker_resource_records") or [])
        if r.get("peak_rss_bytes") or r.get("max_rss_bytes")
    ]
    ref_peak = max(ref_peaks) if ref_peaks else ref_rt.get("peak_rss_bytes_max_worker")
    cand_peak = max(cand_peaks) if cand_peaks else cand_rt.get("peak_rss_bytes_max_worker")
    return {
        "reference_worker_count": _worker_count(ref_root, ref_rt, ref_agg),
        "rom_worker_count": _worker_count(cand_root, cand_rt, cand_agg),
        "reference_worker_phase_s": ref_worker,
        "rom_worker_phase_s": cand_worker,
        "reference_total_pipeline_s": ref_total,
        "rom_total_pipeline_s": cand_total,
        "reference_peak_rss_bytes_per_worker": ref_peak,
        "rom_peak_rss_bytes_per_worker": cand_peak,
        "worker_phase_speedup": speedup_worker,
        "total_pipeline_speedup": speedup_total,
        "rom_samples_per_reference_sample_worker_phase": speedup_worker,
        "rom_samples_per_reference_sample_total_pipeline": speedup_total,
    }


def _mesh_scale(ref_identity: Optional[Mapping[str, Any]], cand_identity: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not ref_identity and not cand_identity:
        return {"status": "UNAVAILABLE"}

    def _extract(doc: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not doc:
            return {"nodes": None, "tetrahedra": None, "active_dimension": None}
        mesh = doc.get("mesh_components") or {}
        return {
            "nodes": doc.get("mesh_node_count") or mesh.get("n_nodes") or mesh.get("node_count"),
            "tetrahedra": doc.get("mesh_tetra_count") or mesh.get("n_tetra") or mesh.get("tetra_count"),
            "active_dimension": doc.get("active_dimension"),
        }

    ref_m = _extract(ref_identity)
    cand_m = _extract(cand_identity)
    node_reduction = None
    tetra_reduction = None
    active_reduction = None
    if ref_m["nodes"] and cand_m["nodes"]:
        node_reduction = (float(ref_m["nodes"]) - float(cand_m["nodes"])) / float(ref_m["nodes"])
    if ref_m["tetrahedra"] and cand_m["tetrahedra"]:
        tetra_reduction = (float(ref_m["tetrahedra"]) - float(cand_m["tetrahedra"])) / float(ref_m["tetrahedra"])
    if ref_m["active_dimension"] and cand_m["active_dimension"]:
        active_reduction = (
            float(ref_m["active_dimension"]) - float(cand_m["active_dimension"])
        ) / float(ref_m["active_dimension"])
    return {
        "status": "AVAILABLE",
        "reference": ref_m,
        "rom": cand_m,
        "node_count_reduction_percent": (node_reduction * 100.0) if node_reduction is not None else None,
        "tetrahedra_reduction_percent": (tetra_reduction * 100.0) if tetra_reduction is not None else None,
        "active_dimension_reduction_percent": (active_reduction * 100.0) if active_reduction is not None else None,
    }


def derive_practical_conclusion(report: Mapping[str, Any]) -> Dict[str, Any]:
    counts = report.get("mode_counts") or {}
    freq = report.get("frequency_matching") or {}
    perf = report.get("performance") or {}
    ref_n = counts.get("reference_deduped_modes")
    recall = freq.get("reference_recall")
    rel_med = (freq.get("relative_frequency_error") or {}).get("median")
    throughput = perf.get("worker_phase_speedup") or perf.get("total_pipeline_speedup")
    samples_per_ref = (
        perf.get("rom_samples_per_reference_sample_worker_phase")
        or perf.get("rom_samples_per_reference_sample_total_pipeline")
    )
    retention = recall
    if retention is None and ref_n:
        retention = (freq.get("matched_count") or 0) / ref_n if ref_n else None
    information_retention = retention
    if information_retention is not None and rel_med is not None:
        information_retention = float(information_retention) * max(0.0, 1.0 - float(rel_med))

    recommendation = "INSUFFICIENT_DATA"
    reason = "missing_catalog_or_timing"
    if throughput is not None and retention is not None:
        if throughput >= 1.25 and retention >= 0.85:
            recommendation = "FAVOR_ROM_FOR_DIVERSITY"
            reason = "throughput_gain_outweighs_moderate_per_sample_loss"
        elif throughput >= 1.5 and retention >= 0.70:
            recommendation = "FAVOR_ROM_WITH_MODERATE_LOSS"
            reason = "higher_sample_diversity_likely_worth_frequency_proxy_drift"
        elif throughput < 1.05:
            recommendation = "REFERENCE_PREFERRED_ON_THROUGHPUT"
            reason = "limited_runtime_advantage"
        elif retention < 0.60:
            recommendation = "REFERENCE_PREFERRED_ON_RETENTION"
            reason = "per_sample_information_loss_too_high"
        else:
            recommendation = "MIXED_TRADEOFF_REVIEW_BANDS"
            reason = "evaluate_band_recall_and_proxy_ranks_for_training_goal"

    return {
        "information_retention_per_sample": information_retention,
        "throughput_gain": throughput,
        "estimated_ROM_samples_per_reference_sample": samples_per_ref,
        "recommendation": recommendation,
        "recommendation_reason": reason,
        "guiding_question": (
            "Under the same compute-time budget, is the loss per lightweight sample "
            "outweighed by the increased number and diversity of training samples?"
        ),
    }


def compare_runs_direct(
    *,
    reference_run: Path,
    candidate_run: Path,
    match_tolerance_hz: float = 5.0,
) -> Dict[str, Any]:
    ref_root = reference_run.resolve()
    cand_root = candidate_run.resolve()
    artifacts: Dict[str, Any] = {"reference": {}, "rom": {}}
    for rel in OPTIONAL_READ_PATHS:
        ref_doc, ref_present = _load_optional(ref_root, rel)
        cand_doc, cand_present = _load_optional(cand_root, rel)
        artifacts["reference"][rel] = {"present": ref_present, "loaded": ref_doc is not None}
        artifacts["rom"][rel] = {"present": cand_present, "loaded": cand_doc is not None}

    ref_rows, ref_catalog_rel, ref_catalog_present = _load_catalog(ref_root)
    cand_rows, cand_catalog_rel, cand_catalog_present = _load_catalog(cand_root)

    report: Dict[str, Any] = {
        "schema": DIRECT_COMPARE_SCHEMA,
        "comparison_executed": True,
        "reference_run": str(ref_root),
        "rom_run": str(cand_root),
        "match_tolerance_hz": match_tolerance_hz,
        "artifacts": artifacts,
        "catalog_sources": {
            "reference": ref_catalog_rel or None,
            "rom": cand_catalog_rel or None,
        },
    }

    ref_n = len(ref_rows)
    cand_n = len(cand_rows)
    report["mode_counts"] = {
        "reference_deduped_modes": ref_n if ref_catalog_present else None,
        "rom_deduped_modes": cand_n if cand_catalog_present else None,
        "count_ratio_rom_over_reference": (cand_n / ref_n) if ref_n else None,
        "status": "AVAILABLE" if ref_catalog_present and cand_catalog_present else "PARTIAL" if ref_catalog_present or cand_catalog_present else "UNAVAILABLE",
    }

    if not ref_rows or not cand_rows:
        report["frequency_matching"] = {"status": "UNAVAILABLE", "reason": "missing_modes_catalog"}
        report["frequency_bands"] = {"status": "UNAVAILABLE"}
        report["participation"] = {"status": "UNAVAILABLE"}
        report["coupling_classification"] = {"status": "UNAVAILABLE"}
        report["proxies"] = {"status": "UNAVAILABLE"}
    else:
        pairs = greedy_monotonic_freq_pairs(ref_rows, cand_rows, max_distance_hz=match_tolerance_hz)
        matched: List[Dict[str, Any]] = []
        abs_errors: List[float] = []
        rel_errors: List[float] = []
        for i, j in pairs:
            rf = float(ref_rows[i].get("frequency_hz") or 0.0)
            cf = float(cand_rows[j].get("frequency_hz") or 0.0)
            abs_e = abs(cf - rf)
            rel_e = abs_e / rf if rf > 0 else 0.0
            abs_errors.append(abs_e)
            rel_errors.append(rel_e)
            matched.append(
                {
                    "reference_hz": rf,
                    "rom_hz": cf,
                    "abs_error_hz": abs_e,
                    "rel_error": rel_e,
                }
            )
        matched_ref = {i for i, _ in pairs}
        matched_cand = {j for _, j in pairs}
        report["frequency_matching"] = {
            "status": "AVAILABLE",
            "matched_count": len(pairs),
            "unmatched_reference_count": len(ref_rows) - len(matched_ref),
            "unmatched_rom_count": len(cand_rows) - len(matched_cand),
            "reference_recall": (len(matched_ref) / ref_n) if ref_n else None,
            "absolute_frequency_error_hz": _stat_triplet(abs_errors),
            "relative_frequency_error": _stat_triplet(rel_errors),
        }
        report["frequency_bands"] = {
            name: _band_metrics(ref_rows, cand_rows, pairs, matched, lo, hi)
            for name, lo, hi in FREQUENCY_BANDS
        }
        report["participation"] = _share_metrics(ref_rows, cand_rows, pairs)
        report["coupling_classification"] = _coupling_metrics(ref_rows, cand_rows, pairs)
        report["proxies"] = {
            key: _proxy_metrics(ref_rows, cand_rows, pairs, key)
            for key in ("bridge_excitation_abs", "mic_output_proxy", "radiation_proxy")
        }

    ref_identity, _ = _load_optional(ref_root, "freeze/physics_identity_manifest.json")
    cand_identity, _ = _load_optional(cand_root, "freeze/physics_identity_manifest.json")
    report["performance"] = _performance_metrics(ref_root, cand_root)
    report["mesh_scale"] = _mesh_scale(ref_identity, cand_identity)
    report["practical_conclusion"] = derive_practical_conclusion(report)
    return report


def render_markdown_direct(report: Mapping[str, Any]) -> str:
    counts = report.get("mode_counts") or {}
    freq = report.get("frequency_matching") or {}
    perf = report.get("performance") or {}
    conclusion = report.get("practical_conclusion") or {}
    lines = [
        "# Direct M4 run comparison",
        "",
        f"- reference: `{report.get('reference_run')}`",
        f"- ROM: `{report.get('rom_run')}`",
        f"- match tolerance: **{report.get('match_tolerance_hz')} Hz**",
        "",
        "## Mode counts",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| reference deduplicated modes | {counts.get('reference_deduped_modes', 'n/a')} |",
        f"| ROM deduplicated modes | {counts.get('rom_deduped_modes', 'n/a')} |",
        f"| count ratio (ROM/reference) | {counts.get('count_ratio_rom_over_reference', 'n/a')} |",
        "",
        "## Frequency matching",
        "",
    ]
    if freq.get("status") == "UNAVAILABLE":
        lines.append(f"_Unavailable: {freq.get('reason', 'missing data')}_")
    else:
        abs_e = freq.get("absolute_frequency_error_hz") or {}
        rel_e = freq.get("relative_frequency_error") or {}
        lines.extend(
            [
                "| metric | value |",
                "| --- | --- |",
                f"| matched | {freq.get('matched_count')} |",
                f"| unmatched reference | {freq.get('unmatched_reference_count')} |",
                f"| unmatched ROM | {freq.get('unmatched_rom_count')} |",
                f"| reference recall | {freq.get('reference_recall')} |",
                f"| abs error median / p95 / max (Hz) | {abs_e.get('median')} / {abs_e.get('p95')} / {abs_e.get('maximum')} |",
                f"| rel error median / p95 / max | {rel_e.get('median')} / {rel_e.get('p95')} / {rel_e.get('maximum')} |",
                "",
            ]
        )

    lines.extend(["## Frequency bands", "", "| band | ref | ROM | matched | recall | med rel err | p95 rel err |", "| --- | --- | --- | --- | --- | --- | --- |"])
    bands = report.get("frequency_bands") or {}
    if isinstance(bands, dict) and bands.get("status") == "UNAVAILABLE":
        lines.append("| n/a | | | | | | |")
    else:
        for name, lo, hi in FREQUENCY_BANDS:
            b = (bands or {}).get(name) or {}
            lines.append(
                f"| {lo:.0f}-{hi:.0f} Hz | {b.get('reference_count', 'n/a')} | {b.get('rom_count', 'n/a')} | "
                f"{b.get('matched_count', 'n/a')} | {b.get('recall', 'n/a')} | "
                f"{b.get('median_rel_frequency_error', 'n/a')} | {b.get('p95_rel_frequency_error', 'n/a')} |"
            )
    lines.extend(
        [
            "",
            "## Performance",
            "",
            "| metric | reference | ROM |",
            "| --- | --- | --- |",
            f"| worker count | {perf.get('reference_worker_count', 'n/a')} | {perf.get('rom_worker_count', 'n/a')} |",
            f"| worker phase (s) | {perf.get('reference_worker_phase_s', 'n/a')} | {perf.get('rom_worker_phase_s', 'n/a')} |",
            f"| total pipeline (s) | {perf.get('reference_total_pipeline_s', 'n/a')} | {perf.get('rom_total_pipeline_s', 'n/a')} |",
            f"| peak RSS / worker (bytes) | {perf.get('reference_peak_rss_bytes_per_worker', 'n/a')} | {perf.get('rom_peak_rss_bytes_per_worker', 'n/a')} |",
            f"| worker-phase speedup | {perf.get('worker_phase_speedup', 'n/a')} | |",
            f"| ROM samples per reference sample | {perf.get('rom_samples_per_reference_sample_worker_phase', 'n/a')} | |",
            "",
            "## Practical conclusion",
            "",
            "| field | value |",
            "| --- | --- |",
            f"| information_retention_per_sample | {conclusion.get('information_retention_per_sample', 'n/a')} |",
            f"| throughput_gain | {conclusion.get('throughput_gain', 'n/a')} |",
            f"| estimated_ROM_samples_per_reference_sample | {conclusion.get('estimated_ROM_samples_per_reference_sample', 'n/a')} |",
            f"| recommendation | **{conclusion.get('recommendation', 'n/a')}** |",
            "",
            f"_{conclusion.get('guiding_question', '')}_",
            "",
            f"Reason: {conclusion.get('recommendation_reason', 'n/a')}",
            "",
        ]
    )
    return "\n".join(lines)
