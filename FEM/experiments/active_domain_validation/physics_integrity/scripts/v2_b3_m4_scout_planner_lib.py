"""M4.3 — density zones, gapless L_prod target plan, worker chunk preview (no solver)."""
from __future__ import annotations

import math
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

ZONE_1 = "ZONE_1_dense"
ZONE_2 = "ZONE_2_medium"
ZONE_3 = "ZONE_3_sparse"

ZONE_SPACING_HZ: Dict[str, float] = {
    ZONE_1: 6.0,
    ZONE_2: 9.0,
    ZONE_3: 12.5,
}

CHUNK_PREF_LO = 20.0
CHUNK_PREF_HI = 40.0
CHUNK_MIN_HZ = 15.0
CHUNK_MAX_HZ = 50.0
LPROD_SEC_PER_TARGET = 95.0
UNIFORM_BASELINE_SPACING_HZ = 5.5


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * float(p) / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f]) * (c - k) + float(sorted_vals[c]) * (k - f)


def build_density_bins(
    unique_hz: Sequence[float],
    *,
    freq_min_hz: float,
    freq_max_hz: float,
    bin_width_hz: float,
) -> List[Dict[str, Any]]:
    edges: List[float] = []
    cur = float(freq_min_hz)
    hi = float(freq_max_hz)
    w = float(bin_width_hz)
    while cur < hi - 1e-9:
        edges.append(cur)
        cur += w
    edges.append(hi)

    freqs = sorted(float(x) for x in unique_hz)
    bins: List[Dict[str, Any]] = []
    for i in range(len(edges) - 1):
        b_lo, b_hi = edges[i], edges[i + 1]
        is_last = i == len(edges) - 2
        in_bin = [
            f
            for f in freqs
            if (b_lo <= f < b_hi - 1e-9) or (is_last and abs(f - b_hi) < 1e-6)
        ]
        count = len(in_bin)
        width = b_hi - b_lo
        bins.append(
            {
                "freq_lo_hz": round(b_lo, 6),
                "freq_hi_hz": round(b_hi, 6),
                "bin_width_hz": round(width, 6),
                "mode_count": count,
                "density_modes_per_hz": (count / width) if width > 0 else 0.0,
                "mode_frequencies_hz": in_bin,
                "zone_id": None,
                "recommended_lprod_spacing_hz": None,
            }
        )
    return bins


def classify_bins_percentile(
    bins: List[Dict[str, Any]],
    *,
    dense_percentile: float = 75.0,
    sparse_percentile: float = 35.0,
) -> Dict[str, Any]:
    """Top ~25% density → ZONE_1; bottom ~35% → ZONE_3; middle → ZONE_2."""
    densities = [float(b["density_modes_per_hz"]) for b in bins]
    sorted_d = sorted(densities)
    p_dense = _percentile(sorted_d, dense_percentile)
    p_sparse = _percentile(sorted_d, sparse_percentile)
    rule = (
        f"percentile_v1: zone_id=ZONE_1_dense if density>={dense_percentile}th pct ({p_dense:.6f}); "
        f"ZONE_3_sparse if density<={sparse_percentile}th pct ({p_sparse:.6f}); else ZONE_2_medium"
    )
    for b in bins:
        d = float(b["density_modes_per_hz"])
        if d >= p_dense:
            zid = ZONE_1
        elif d <= p_sparse:
            zid = ZONE_3
        else:
            zid = ZONE_2
        b["zone_id"] = zid
        b["recommended_lprod_spacing_hz"] = ZONE_SPACING_HZ[zid]
    return {
        "classification_rule": rule,
        "dense_percentile": dense_percentile,
        "sparse_percentile": sparse_percentile,
        "threshold_dense_ge": p_dense,
        "threshold_sparse_le": p_sparse,
        "median_density_modes_per_hz": statistics.median(densities) if densities else 0.0,
    }


def merge_zone_segments(bins: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not bins:
        return []
    segments: List[Dict[str, Any]] = []
    cur_zone = str(bins[0]["zone_id"])
    seg_lo = float(bins[0]["freq_lo_hz"])
    seg_hi = float(bins[0]["freq_hi_hz"])
    spacing = float(bins[0]["recommended_lprod_spacing_hz"])
    for b in bins[1:]:
        z = str(b["zone_id"])
        if z == cur_zone:
            seg_hi = float(b["freq_hi_hz"])
        else:
            segments.append(
                {
                    "freq_lo_hz": round(seg_lo, 6),
                    "freq_hi_hz": round(seg_hi, 6),
                    "zone_id": cur_zone,
                    "recommended_lprod_spacing_hz": spacing,
                }
            )
            cur_zone = z
            seg_lo = float(b["freq_lo_hz"])
            seg_hi = float(b["freq_hi_hz"])
            spacing = float(b["recommended_lprod_spacing_hz"])
    segments.append(
        {
            "freq_lo_hz": round(seg_lo, 6),
            "freq_hi_hz": round(seg_hi, 6),
            "zone_id": cur_zone,
            "recommended_lprod_spacing_hz": spacing,
        }
    )
    return segments


def _merge_intervals(
    windows: Sequence[Sequence[float]],
    *,
    gap_tolerance_hz: float,
) -> List[Tuple[float, float]]:
    intervals = sorted((float(w[0]), float(w[1])) for w in windows if len(w) >= 2)
    merged: List[Tuple[float, float]] = []
    for lo, hi in intervals:
        if not merged or lo > merged[-1][1] + gap_tolerance_hz:
            merged.append((lo, hi))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
    return merged


def _find_coverage_gaps(
    merged: Sequence[Tuple[float, float]],
    *,
    band_lo: float,
    band_hi: float,
    gap_tolerance_hz: float,
) -> List[Tuple[float, float]]:
    gaps: List[Tuple[float, float]] = []
    cursor = float(band_lo)
    for lo, hi in merged:
        if lo > cursor + gap_tolerance_hz:
            gaps.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < float(band_hi) - gap_tolerance_hz:
        gaps.append((cursor, float(band_hi)))
    return gaps


def verify_gapless_coverage(
    targets_hz: Sequence[float],
    target_windows_hz: Sequence[Sequence[float]],
    *,
    band_lo: float,
    band_hi: float,
    gap_tolerance_hz: float = 0.01,
    repair_targets_added: int = 0,
) -> Dict[str, Any]:
    if not targets_hz or not target_windows_hz:
        return {
            "pass": False,
            "band_hz": [band_lo, band_hi],
            "max_gap_hz": float("inf"),
            "gap_tolerance_hz": gap_tolerance_hz,
            "target_count": 0,
            "gap_count": 0,
            "gaps_hz": [],
            "repair_targets_added": repair_targets_added,
            "merged_window_count": 0,
            "notes": "empty target plan",
        }
    merged = _merge_intervals(target_windows_hz, gap_tolerance_hz=gap_tolerance_hz)
    gaps = _find_coverage_gaps(
        merged, band_lo=band_lo, band_hi=band_hi, gap_tolerance_hz=gap_tolerance_hz
    )
    max_gap = 0.0
    gaps_hz: List[List[float]] = []
    for g_lo, g_hi in gaps:
        span = g_hi - g_lo
        max_gap = max(max_gap, span)
        gaps_hz.append([round(g_lo, 6), round(g_hi, 6)])

    return {
        "pass": len(gaps) == 0,
        "band_hz": [band_lo, band_hi],
        "max_gap_hz": round(max_gap, 6),
        "gap_tolerance_hz": gap_tolerance_hz,
        "target_count": len(targets_hz),
        "gap_count": len(gaps),
        "gaps_hz": gaps_hz,
        "repair_targets_added": repair_targets_added,
        "merged_window_count": len(merged),
        "notes": "Union of half-width windows must cover band with no gaps > tolerance.",
    }


def _segment_targets_gapless(
    f_lo: float,
    f_hi: float,
    spacing: float,
    *,
    zone_id: str,
) -> List[Dict[str, Any]]:
    """
    Segment-local gapless grid: first target at lo+h, step by s, endpoint at hi-h if needed.
    Short segments use a single midpoint target with half-width covering [lo, hi].
    """
    s = float(spacing)
    if s <= 0:
        raise ValueError("spacing must be positive")
    lo = float(f_lo)
    hi = float(f_hi)
    h = s / 2.0
    width = hi - lo
    if width <= 0:
        return []

    if width <= 2.0 * h + 1e-9:
        t_mid = (lo + hi) / 2.0
        h_eff = width / 2.0 + 1e-6
        return [
            {
                "target_hz": round(t_mid, 6),
                "zone_id": zone_id,
                "spacing_hz": s,
                "half_width_hz": round(h_eff, 6),
                "source": "segment_short",
                "segment_lo_hz": lo,
                "segment_hi_hz": hi,
            }
        ]

    entries: List[Dict[str, Any]] = []

    def _add(t: float, *, source: str) -> None:
        t = round(t, 6)
        if entries and abs(entries[-1]["target_hz"] - t) < 1e-6:
            return
        entries.append(
            {
                "target_hz": t,
                "zone_id": zone_id,
                "spacing_hz": s,
                "half_width_hz": round(h, 6),
                "source": source,
                "segment_lo_hz": lo,
                "segment_hi_hz": hi,
            }
        )

    _add(lo + h, source="segment_grid")
    t = lo + h
    while t + s + h < hi - 1e-9:
        t += s
        _add(t, source="segment_grid")

    t_end = hi - h
    last_hi = entries[-1]["target_hz"] + h
    if last_hi < hi - 1e-9:
        _add(t_end, source="segment_endpoint")

    return entries


def _entries_to_plan_arrays(
    entries: List[Dict[str, Any]],
) -> Tuple[List[float], List[Dict[str, Any]], List[List[float]]]:
    targets: List[float] = []
    metadata: List[Dict[str, Any]] = []
    windows: List[List[float]] = []
    for e in entries:
        t = float(e["target_hz"])
        hw = float(e["half_width_hz"])
        targets.append(t)
        metadata.append(dict(e))
        windows.append([round(t - hw, 6), round(t + hw, 6)])
    return targets, metadata, windows


def _repair_coverage_gaps(
    entries: List[Dict[str, Any]],
    *,
    band_lo: float,
    band_hi: float,
    gap_tolerance_hz: float,
    max_repair_rounds: int = 32,
) -> Tuple[List[Dict[str, Any]], int]:
    """Insert midpoint repair targets for any uncovered gaps inside the band."""
    out = list(entries)
    repairs = 0
    for _ in range(max_repair_rounds):
        _, _, windows = _entries_to_plan_arrays(out)
        merged = _merge_intervals(windows, gap_tolerance_hz=gap_tolerance_hz)
        gaps = _find_coverage_gaps(
            merged, band_lo=band_lo, band_hi=band_hi, gap_tolerance_hz=gap_tolerance_hz
        )
        if not gaps:
            break
        g_lo, g_hi = gaps[0]
        mid = (g_lo + g_hi) / 2.0
        span = g_hi - g_lo
        hw_repair = span / 2.0 + gap_tolerance_hz
        zone_id = ZONE_2
        spacing = ZONE_SPACING_HZ[ZONE_2]
        if out:
            left = max((e for e in out if e["target_hz"] <= mid + 1e-6), key=lambda e: e["target_hz"], default=None)
            right = min((e for e in out if e["target_hz"] >= mid - 1e-6), key=lambda e: e["target_hz"], default=None)
            if left is not None:
                zone_id = str(left.get("zone_id") or zone_id)
                spacing = float(left.get("spacing_hz") or spacing)
            elif right is not None:
                zone_id = str(right.get("zone_id") or zone_id)
                spacing = float(right.get("spacing_hz") or spacing)
        out.append(
            {
                "target_hz": round(mid, 6),
                "zone_id": zone_id,
                "spacing_hz": spacing,
                "half_width_hz": round(hw_repair, 6),
                "source": "coverage_repair",
                "reason": "gap_between_zone_segments",
                "repair_gap_hz": [round(g_lo, 6), round(g_hi, 6)],
            }
        )
        out.sort(key=lambda e: float(e["target_hz"]))
        repairs += 1
    return out, repairs


def build_gapless_target_plan(
    segments: Sequence[Dict[str, Any]],
    *,
    sample_id: str,
    run_id: str,
    freq_min_hz: float,
    freq_max_hz: float,
    zone_policy_version: str = "v1",
    gap_tolerance_hz: float = 0.01,
) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for seg in segments:
        f_lo = float(seg["freq_lo_hz"])
        f_hi = float(seg["freq_hi_hz"])
        zone_id = str(seg["zone_id"])
        spacing = float(seg.get("recommended_lprod_spacing_hz") or ZONE_SPACING_HZ[zone_id])
        entries.extend(_segment_targets_gapless(f_lo, f_hi, spacing, zone_id=zone_id))

    entries.sort(key=lambda e: float(e["target_hz"]))
    entries, repair_count = _repair_coverage_gaps(
        entries,
        band_lo=freq_min_hz,
        band_hi=freq_max_hz,
        gap_tolerance_hz=gap_tolerance_hz,
    )

    targets, metadata, windows = _entries_to_plan_arrays(entries)
    coverage = verify_gapless_coverage(
        targets,
        windows,
        band_lo=freq_min_hz,
        band_hi=freq_max_hz,
        gap_tolerance_hz=gap_tolerance_hz,
        repair_targets_added=repair_count,
    )
    return {
        "schema": "m4_lprod_target_plan_v1",
        "will_execute": False,
        "sample_id": sample_id,
        "run_id": run_id,
        "generated_utc": _utc_now(),
        "zone_policy_version": zone_policy_version,
        "target_generation_policy": "gapless_grid_v2_segment_endpoint_plus_coverage_repair",
        "frequency_range_hz": [freq_min_hz, freq_max_hz],
        "mesh_level": "L_prod",
        "targets_hz": targets,
        "target_metadata": metadata,
        "target_windows_hz": windows,
        "coverage_check": coverage,
    }


def estimate_runtime_summary(
    *,
    target_count: int,
    scout_wall_seconds: Optional[float],
    workers: int,
    freq_min_hz: float,
    freq_max_hz: float,
) -> Dict[str, Any]:
    band = float(freq_max_hz) - float(freq_min_hz)
    uniform_count = max(1, int(math.floor(band / UNIFORM_BASELINE_SPACING_HZ)) + 1)
    adaptive_seconds = float(target_count) * LPROD_SEC_PER_TARGET
    uniform_seconds = float(uniform_count) * LPROD_SEC_PER_TARGET
    scout_overhead = float(scout_wall_seconds or 0.0)

    def _wall_for(n_workers: int) -> float:
        n_workers = max(1, int(n_workers))
        return scout_overhead + adaptive_seconds / n_workers

    return {
        "target_count": target_count,
        "estimated_seconds_per_target": LPROD_SEC_PER_TARGET,
        "estimated_total_seconds_adaptive": adaptive_seconds,
        "estimated_total_seconds_uniform_5p5hz": uniform_seconds,
        "uniform_baseline_target_count": uniform_count,
        "uniform_baseline_spacing_hz": UNIFORM_BASELINE_SPACING_HZ,
        "scout_measured_overhead_seconds": scout_wall_seconds,
        "workers": {
            str(w): {
                "estimated_wall_seconds": round(_wall_for(w), 1),
                "note": f"scout overhead + adaptive L_prod / {w} FCFS workers (planning estimate)",
            }
            for w in (1, 2, 3)
            if w <= max(3, workers)
        },
        "placeholder": False,
    }


def build_worker_chunk_preview(
    segments: Sequence[Dict[str, Any]],
    *,
    sample_id: str,
    run_id: str,
    freq_min_hz: float,
    freq_max_hz: float,
    targets_hz: Sequence[float],
    target_windows_hz: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """Chunk plan over band; status PLANNED_NOT_EXECUTED (no worker execution)."""
    chunks: List[Dict[str, Any]] = []
    idx = 1
    for seg in segments:
        seg_lo = float(seg["freq_lo_hz"])
        seg_hi = float(seg["freq_hi_hz"])
        zone_id = str(seg["zone_id"])
        cursor = seg_lo
        while cursor < seg_hi - 1e-9:
            width = CHUNK_PREF_HI
            chunk_hi = min(seg_hi, cursor + width)
            span = chunk_hi - cursor
            if span > CHUNK_MAX_HZ:
                chunk_hi = cursor + CHUNK_MAX_HZ
            elif span < CHUNK_MIN_HZ and chunk_hi < seg_hi:
                chunk_hi = min(seg_hi, cursor + CHUNK_MIN_HZ)
            t_in = [t for t in targets_hz if cursor - 1e-6 <= t <= chunk_hi + 1e-6]
            w_in = [
                list(w)
                for t, w in zip(targets_hz, target_windows_hz)
                if cursor - 1e-6 <= t <= chunk_hi + 1e-6
            ]
            chunks.append(
                {
                    "chunk_id": f"{sample_id}_chunk_{idx:02d}",
                    "freq_range_hz": [round(cursor, 6), round(chunk_hi, 6)],
                    "zone_ids": [zone_id],
                    "targets_hz": t_in,
                    "target_windows_hz": w_in,
                    "target_count": len(t_in),
                    "estimated_cost": {
                        "target_count": len(t_in),
                        "estimated_seconds": len(t_in) * LPROD_SEC_PER_TARGET,
                        "relative_weight": len(t_in) * LPROD_SEC_PER_TARGET,
                    },
                    "status": "PLANNED_NOT_EXECUTED",
                    "assigned_worker_id": None,
                    "priority": 0,
                }
            )
            cursor = chunk_hi
            idx += 1
    return {
        "schema": "m4_worker_chunk_plan_v1",
        "will_execute": False,
        "status": "PLANNED_NOT_EXECUTED",
        "sample_id": sample_id,
        "run_id": run_id,
        "generated_utc": _utc_now(),
        "chunk_policy_version": "v1",
        "frequency_range_hz": [freq_min_hz, freq_max_hz],
        "chunk_policy": {
            "preferred_width_hz": [CHUNK_PREF_LO, CHUNK_PREF_HI],
            "min_width_hz_soft": CHUNK_MIN_HZ,
            "max_width_hz_soft": CHUNK_MAX_HZ,
            "respect_zone_boundaries": True,
            "note": "Chunk width limits are scheduling preferences, not physics constraints.",
        },
        "lprod_target_plan_path": "lprod/lprod_target_plan.json",
        "chunks": chunks,
        "warnings": _chunk_preview_warnings(chunks),
    }


def _chunk_preview_warnings(chunks: Sequence[Dict[str, Any]]) -> List[str]:
    warnings = ["Preview only — workers not executed in M4.3."]
    for c in chunks:
        fr = c.get("freq_range_hz") or []
        if len(fr) < 2:
            continue
        span = float(fr[1]) - float(fr[0])
        if span < CHUNK_MIN_HZ - 1e-6:
            warnings.append(
                f"{c.get('chunk_id')}: width {span:.2f} Hz below soft minimum {CHUNK_MIN_HZ} Hz (allowed)"
            )
        elif span > CHUNK_MAX_HZ + 1e-6:
            warnings.append(
                f"{c.get('chunk_id')}: width {span:.2f} Hz above soft maximum {CHUNK_MAX_HZ} Hz (allowed)"
            )
    return warnings


def build_density_zones_document(
    *,
    sample_id: str,
    run_id: str,
    bins: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    classification_meta: Dict[str, Any],
    unique_hz: Sequence[float],
    freq_min_hz: float,
    freq_max_hz: float,
    bin_width_hz: float,
    density_result_path: str,
) -> Dict[str, Any]:
    return {
        "schema": "m4_density_zones_v1",
        "will_execute": False,
        "sample_id": sample_id,
        "run_id": run_id,
        "generated_utc": _utc_now(),
        "zone_policy_version": "v1",
        "scout_policy_version": "v1",
        "frequency_range_hz": [freq_min_hz, freq_max_hz],
        "bin_width_hz": bin_width_hz,
        "dedupe_tolerance_hz": 0.05,
        "unique_mode_count": len(unique_hz),
        "classification_rule": classification_meta.get("classification_rule"),
        "classification_thresholds": classification_meta,
        "zone_spacing_hz": dict(ZONE_SPACING_HZ),
        "bins": bins,
        "segments": segments,
        "density_result_path": density_result_path,
    }


def render_density_zones_md(doc: Dict[str, Any]) -> str:
    lines = [
        f"# Density zones — {doc.get('sample_id')}",
        "",
        f"- run_id: `{doc.get('run_id')}`",
        f"- unique modes: **{doc.get('unique_mode_count')}**",
        f"- bins: **{doc.get('bin_width_hz')} Hz** over **{doc.get('frequency_range_hz')}**",
        "",
        f"Classification: {doc.get('classification_rule')}",
        "",
        "| bin (Hz) | count | density/Hz | zone | L_prod spacing |",
        "|----------|-------|------------|------|----------------|",
    ]
    for b in doc.get("bins") or []:
        lines.append(
            f"| {b['freq_lo_hz']}-{b['freq_hi_hz']} | {b['mode_count']} | "
            f"{b['density_modes_per_hz']:.4f} | {b['zone_id']} | {b['recommended_lprod_spacing_hz']} |"
        )
    lines.append("")
    lines.append("## Merged segments")
    lines.append("")
    for s in doc.get("segments") or []:
        lines.append(
            f"- [{s['freq_lo_hz']}, {s['freq_hi_hz']}] Hz → **{s['zone_id']}** "
            f"(spacing {s['recommended_lprod_spacing_hz']} Hz)"
        )
    return "\n".join(lines) + "\n"


def render_target_plan_md(plan: Dict[str, Any], runtime: Dict[str, Any]) -> str:
    cov = plan.get("coverage_check") or {}
    lines = [
        f"# L_prod target plan — {plan.get('sample_id')}",
        "",
        f"- run_id: `{plan.get('run_id')}`",
        f"- targets: **{len(plan.get('targets_hz') or [])}**",
        f"- coverage pass: **{cov.get('pass')}** (max gap {cov.get('max_gap_hz')} Hz, "
        f"repair targets {cov.get('repair_targets_added', 0)})",
        f"- policy: {plan.get('target_generation_policy')}",
        "",
        "## Runtime estimate (planning)",
        "",
        f"- per target: **{runtime.get('estimated_seconds_per_target')} s**",
        f"- adaptive total (1 worker): **{runtime.get('estimated_total_seconds_adaptive')} s**",
        f"- uniform 5.5 Hz baseline targets: **{runtime.get('uniform_baseline_target_count')}** "
        f"→ **{runtime.get('estimated_total_seconds_uniform_5p5hz')} s**",
        f"- scout measured overhead: **{runtime.get('scout_measured_overhead_seconds')} s**",
        "",
        "| workers | est. wall (s) |",
        "|---------|---------------|",
    ]
    for w, row in (runtime.get("workers") or {}).items():
        lines.append(f"| {w} | {row.get('estimated_wall_seconds')} |")
    lines.append("")
    lines.append("## Zone segments (first/last targets)")
    lines.append("")
    targets = plan.get("targets_hz") or []
    if targets:
        lines.append(f"- first: {targets[0]} Hz")
        lines.append(f"- last: {targets[-1]} Hz")
    return "\n".join(lines) + "\n"


def render_chunk_preview_md(preview: Dict[str, Any]) -> str:
    lines = [
        f"# Worker chunk preview — {preview.get('sample_id')}",
        "",
        f"- chunks: **{len(preview.get('chunks') or [])}**",
        f"- status: **{preview.get('status')}** (not executed)",
        "",
        "| chunk | range (Hz) | targets | zone |",
        "|-------|------------|---------|------|",
    ]
    for c in preview.get("chunks") or []:
        fr = c.get("freq_range_hz") or []
        lines.append(
            f"| {c.get('chunk_id')} | {fr[0]}-{fr[1]} | {c.get('target_count')} | "
            f"{','.join(c.get('zone_ids') or [])} |"
        )
    return "\n".join(lines) + "\n"
