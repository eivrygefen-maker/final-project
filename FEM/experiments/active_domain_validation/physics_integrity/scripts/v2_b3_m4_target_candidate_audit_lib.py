#!/usr/bin/env python3
"""Per-target candidate diagnostics for modal discovery audit (additive; no solve changes)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

TARGET_CANDIDATE_AUDIT_FILENAME = "target_candidate_audit.jsonl"
MERGED_AUDIT_REL_AGG = "aggregation/target_candidate_audit_merged.jsonl"
MERGED_AUDIT_REL_VALIDATION = "validation/target_candidate_audit_merged.jsonl"
DURABLE_CHUNK_AUDIT_DIR_REL = "validation/target_candidate_audit"

REQUIRED_AUDIT_KEYS: Tuple[str, ...] = (
    "chunk_id",
    "target_hz",
    "target_window_hz",
    "solver_factor",
    "requested_eigenpairs",
    "candidate_count_raw",
    "candidate_count_after_residual",
    "candidate_count_after_window",
    "candidate_count_after_physical_filters",
    "accepted_mode_count",
    "rejected_candidate_count",
    "rejection_reasons",
    "min_residual",
    "accepted_frequencies_hz",
    "raw_candidate_frequencies_hz",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def target_candidate_audit_path(chunk_dir: Path) -> Path:
    return chunk_dir / TARGET_CANDIDATE_AUDIT_FILENAME


def merged_audit_path_agg(run_root: Path) -> Path:
    return run_root / MERGED_AUDIT_REL_AGG


def merged_audit_path_validation(run_root: Path) -> Path:
    return run_root / MERGED_AUDIT_REL_VALIDATION


def durable_chunk_audit_path(run_root: Path, chunk_id: str) -> Path:
    return run_root / DURABLE_CHUNK_AUDIT_DIR_REL / f"{chunk_id}.jsonl"


def _freq_in_window(freq: float, win: Optional[Sequence[float]]) -> bool:
    if win is None or len(win) != 2:
        return True
    return float(win[0]) <= float(freq) <= float(win[1])


def _raw_candidate_frequencies(target_row: Mapping[str, Any]) -> List[float]:
    freqs: List[float] = []
    for m in target_row.get("converged_modes") or []:
        if isinstance(m, dict) and m.get("frequency_hz") is not None:
            try:
                freqs.append(float(m["frequency_hz"]))
            except (TypeError, ValueError):
                continue
    return sorted(freqs)


def _count_in_window(target_row: Mapping[str, Any], win: Optional[Sequence[float]]) -> Optional[int]:
    freqs = _raw_candidate_frequencies(target_row)
    if not freqs:
        return None if target_row.get("converged_mode_count") is None else 0
    return sum(1 for f in freqs if _freq_in_window(f, win))


def normalize_target_candidate_audit_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Ensure all required keys exist (null when unavailable)."""
    base: Dict[str, Any] = {
        "chunk_id": None,
        "target_hz": None,
        "target_window_hz": None,
        "solver_factor": None,
        "requested_eigenpairs": None,
        "candidate_count_raw": None,
        "candidate_count_after_residual": None,
        "candidate_count_after_window": None,
        "candidate_count_after_physical_filters": None,
        "accepted_mode_count": None,
        "rejected_candidate_count": None,
        "rejection_reasons": {},
        "min_residual": None,
        "accepted_frequencies_hz": [],
        "raw_candidate_frequencies_hz": [],
    }
    base.update(dict(row))
    if base.get("rejection_reasons") is None:
        base["rejection_reasons"] = {}
    if base.get("accepted_frequencies_hz") is None:
        base["accepted_frequencies_hz"] = []
    if base.get("raw_candidate_frequencies_hz") is None:
        base["raw_candidate_frequencies_hz"] = []
    return base


def build_target_candidate_audit_row(
    *,
    chunk_id: str,
    target_row: Mapping[str, Any],
    target_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one JSONL audit record from a solver per-target row (post-solve)."""
    meta = dict(target_meta or {})
    target_hz = target_row.get("target_frequency_hz") or target_row.get("target_hz")
    win = target_row.get("per_target_acceptance_window_hz")
    if win is None and meta.get("window_hz") is not None:
        raw_win = meta["window_hz"]
        if isinstance(raw_win, (list, tuple)) and len(raw_win) == 2:
            win = [float(raw_win[0]), float(raw_win[1])]
        elif target_hz is not None:
            wh = float(raw_win)
            thz = float(target_hz)
            win = [thz - wh, thz + wh]
    converged = target_row.get("converged_mode_count")
    accepted_n = target_row.get("accepted_mode_count_in_interval")
    rejection_tally = dict(target_row.get("candidate_rejection_tally") or {})
    rejected = sum(int(v) for v in rejection_tally.values()) if rejection_tally else None
    if rejected is None and converged is not None and accepted_n is not None:
        try:
            rejected = max(0, int(converged) - int(accepted_n))
        except (TypeError, ValueError):
            rejected = None

    accepted_freqs = list(target_row.get("accepted_frequencies_hz") or [])
    if not accepted_freqs and isinstance(target_row.get("accepted_modes"), list):
        accepted_freqs = [
            float(m["frequency_hz"])
            for m in target_row["accepted_modes"]
            if isinstance(m, dict) and m.get("frequency_hz") is not None
        ]

    raw_freqs = _raw_candidate_frequencies(target_row)
    after_window = _count_in_window(target_row, win)

    min_residual = None
    for key in ("min_eps_compute_error_relative", "min_residual"):
        if target_row.get(key) is not None:
            min_residual = target_row.get(key)
            break
    if min_residual is None and raw_freqs:
        errs = [
            float(m.get("eps_compute_error_relative"))
            for m in (target_row.get("converged_modes") or [])
            if isinstance(m, dict) and m.get("eps_compute_error_relative") is not None
        ]
        if errs:
            min_residual = min(errs)

    return normalize_target_candidate_audit_row(
        {
            "chunk_id": chunk_id,
            "target_hz": float(target_hz) if target_hz is not None else None,
            "target_window_hz": win,
            "zone_id": meta.get("zone_id") or target_row.get("zone_id"),
            "spacing_hz": meta.get("spacing_hz") or target_row.get("spacing_hz"),
            "solve_status": target_row.get("status"),
            "solver_factor": target_row.get("factor_solver") or target_row.get("factor_solver_effective"),
            "requested_eigenpairs": target_row.get("nev"),
            "requested_ncv": target_row.get("ncv"),
            "candidate_count_raw": converged,
            "candidate_count_after_residual": converged,
            "candidate_count_after_window": after_window,
            "candidate_count_after_physical_filters": accepted_n,
            "accepted_mode_count": accepted_n,
            "rejected_candidate_count": rejected,
            "rejection_reasons": rejection_tally or {},
            "acceptance_policy": target_row.get("acceptance_policy"),
            "acceptance_freq_lo_hz": target_row.get("acceptance_freq_lo_hz"),
            "acceptance_freq_hi_hz": target_row.get("acceptance_freq_hi_hz"),
            "min_residual": min_residual,
            "accepted_frequencies_hz": accepted_freqs,
            "raw_candidate_frequencies_hz": raw_freqs,
            "targets_passed_means_solve_success": True,
            "note": (
                "targets_passed on worker_result means numerical target solve PASS, "
                "not guaranteed mode discovery."
            ),
        }
    )


def append_target_candidate_audit_row(chunk_dir: Path, row: Mapping[str, Any]) -> Path:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    path = target_candidate_audit_path(chunk_dir)
    normalized = normalize_target_candidate_audit_row(row)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(normalized, sort_keys=True) + "\n")
    return path


def write_target_candidate_audit_jsonl(
    chunk_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    path = target_candidate_audit_path(chunk_dir)
    text = "".join(
        json.dumps(normalize_target_candidate_audit_row(r), sort_keys=True) + "\n" for r in rows
    )
    path.write_text(text, encoding="utf-8")
    return path


def load_target_candidate_audit_rows(chunk_dir: Path) -> list[Dict[str, Any]]:
    path = target_candidate_audit_path(chunk_dir)
    if not path.is_file():
        return []
    rows: list[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
            if isinstance(doc, dict):
                rows.append(normalize_target_candidate_audit_row(doc))
        except ValueError:
            continue
    return rows


def load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
            if isinstance(doc, dict):
                rows.append(normalize_target_candidate_audit_row(doc))
        except ValueError:
            continue
    return rows


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(normalize_target_candidate_audit_row(r), sort_keys=True) + "\n" for r in rows
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def collect_chunk_target_candidate_rows(
    run_root: Path,
    chunk_id: str,
) -> List[Dict[str, Any]]:
    """Load per-chunk rows from worker dir, durable copy, or merged file."""
    run_root = run_root.resolve()
    worker_rows = load_target_candidate_audit_rows(run_root / "worker_results" / chunk_id)
    if worker_rows:
        return worker_rows
    durable_rows = load_jsonl_rows(durable_chunk_audit_path(run_root, chunk_id))
    if durable_rows:
        return durable_rows
    return []


def copy_chunk_audit_to_durable(run_root: Path, chunk_id: str) -> Optional[Path]:
    rows = collect_chunk_target_candidate_rows(run_root, chunk_id)
    if not rows:
        return None
    return write_jsonl_atomic(durable_chunk_audit_path(run_root, chunk_id), rows)


def merge_target_candidate_audit_for_run(
    run_root: Path,
    chunk_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Merge per-chunk target_candidate_audit.jsonl into durable run-level files.

    Writes:
      aggregation/target_candidate_audit_merged.jsonl
      validation/target_candidate_audit_merged.jsonl
      validation/target_candidate_audit/<chunk_id>.jsonl (per chunk)
    """
    run_root = run_root.resolve()
    if chunk_ids is None:
        chunk_ids = []
        worker_root = run_root / "worker_results"
        if worker_root.is_dir():
            chunk_ids = sorted(p.name for p in worker_root.iterdir() if p.is_dir())
        durable_dir = run_root / DURABLE_CHUNK_AUDIT_DIR_REL
        if durable_dir.is_dir():
            for p in durable_dir.glob("*.jsonl"):
                cid = p.stem
                if cid not in chunk_ids:
                    chunk_ids.append(cid)

    merged: List[Dict[str, Any]] = []
    per_chunk_counts: Dict[str, int] = {}
    sources: Dict[str, str] = {}

    for chunk_id in chunk_ids:
        worker_path = run_root / "worker_results" / chunk_id / TARGET_CANDIDATE_AUDIT_FILENAME
        rows = load_target_candidate_audit_rows(run_root / "worker_results" / chunk_id)
        if rows:
            sources[chunk_id] = str(worker_path)
        else:
            rows = load_jsonl_rows(durable_chunk_audit_path(run_root, chunk_id))
            if rows:
                sources[chunk_id] = str(durable_chunk_audit_path(run_root, chunk_id))
        if rows:
            copy_chunk_audit_to_durable(run_root, chunk_id)
            per_chunk_counts[chunk_id] = len(rows)
            merged.extend(rows)

    meta = {
        "schema": "m4_target_candidate_audit_merged_v1",
        "generated_utc": utc_now(),
        "chunk_count": len(per_chunk_counts),
        "target_row_count": len(merged),
        "per_chunk_target_row_count": per_chunk_counts,
        "sources": sources,
    }

    if merged:
        write_jsonl_atomic(merged_audit_path_agg(run_root), merged)
        write_jsonl_atomic(merged_audit_path_validation(run_root), merged)

    return meta


def ensure_target_candidate_audit_durable(run_root: Path) -> Dict[str, Any]:
    """Idempotent: merge worker chunk audits into compaction-safe paths before worker_results deletion."""
    run_root = run_root.resolve()
    agg_merged = merged_audit_path_agg(run_root)
    if agg_merged.is_file() and agg_merged.stat().st_size > 0:
        rows = load_jsonl_rows(agg_merged)
        if rows:
            write_jsonl_atomic(merged_audit_path_validation(run_root), rows)
            by_chunk: Dict[str, List[Dict[str, Any]]] = {}
            for row in rows:
                cid = str(row.get("chunk_id") or "unknown")
                by_chunk.setdefault(cid, []).append(row)
            for cid, chunk_rows in by_chunk.items():
                write_jsonl_atomic(durable_chunk_audit_path(run_root, cid), chunk_rows)
            return {
                "action": "already_present",
                "target_row_count": len(rows),
                "merged_path": str(agg_merged),
            }
    return merge_target_candidate_audit_for_run(run_root)


def load_merged_target_candidate_audit_rows(run_root: Path) -> List[Dict[str, Any]]:
    run_root = run_root.resolve()
    for path in (merged_audit_path_agg(run_root), merged_audit_path_validation(run_root)):
        rows = load_jsonl_rows(path)
        if rows:
            return rows
    return []


def collect_run_target_candidate_rows(
    run_root: Path,
    chunk_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Return (rows, source_label).

    Prefer merged durable file, then per-chunk durable/worker paths.
    """
    run_root = run_root.resolve()
    merged = load_merged_target_candidate_audit_rows(run_root)
    if merged:
        return merged, MERGED_AUDIT_REL_AGG

    if chunk_ids is None:
        chunk_ids = []
        plan_path = run_root / "lprod" / "worker_chunk_plan.preview.json"
        if plan_path.is_file():
            try:
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                chunk_ids = [
                    str(c.get("chunk_id"))
                    for c in (plan.get("chunks") or [])
                    if c.get("chunk_id")
                ]
            except (OSError, ValueError, json.JSONDecodeError):
                chunk_ids = []

    rows: List[Dict[str, Any]] = []
    for chunk_id in chunk_ids or []:
        rows.extend(collect_chunk_target_candidate_rows(run_root, chunk_id))
    if rows:
        return rows, "per_chunk_audit"
    return [], "none"


def audit_instrumentation_complete(rows: Sequence[Mapping[str, Any]]) -> Tuple[bool, List[str]]:
    """True when rows have real candidate instrumentation (not all-null counts)."""
    if not rows:
        return False, ["no_target_candidate_rows"]
    missing: List[str] = []
    has_any_raw = any(r.get("candidate_count_raw") is not None for r in rows)
    if not has_any_raw:
        missing.append("candidate_count_raw_all_null")
    for key in REQUIRED_AUDIT_KEYS:
        if key not in rows[0]:
            missing.append(f"missing_key:{key}")
    return has_any_raw and not missing, missing


def build_candidate_loss_analysis(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-target candidate audit rows for modal discovery audit."""
    complete, incomplete_reasons = audit_instrumentation_complete(rows)
    hist: Dict[str, int] = {}
    total_raw = 0
    total_accepted = 0
    total_rejected = 0
    zero_raw_targets = 0
    raw_but_zero_accepted = 0
    nev_values: List[int] = []
    solver_factors: Dict[str, int] = {}
    window_samples: List[List[float]] = []

    for row in rows:
        raw_c = row.get("candidate_count_raw")
        if raw_c is not None:
            total_raw += int(raw_c)
            if int(raw_c) == 0:
                zero_raw_targets += 1
        acc = row.get("accepted_mode_count")
        if acc is not None:
            total_accepted += int(acc)
        rej = row.get("rejected_candidate_count")
        if rej is not None:
            total_rejected += int(rej)
        elif raw_c is not None and acc is not None:
            total_rejected += max(0, int(raw_c) - int(acc))
        if raw_c is not None and int(raw_c) > 0 and (acc is None or int(acc) == 0):
            raw_but_zero_accepted += 1
        for reason, count in (row.get("rejection_reasons") or {}).items():
            hist[str(reason)] = hist.get(str(reason), 0) + int(count)
        nev = row.get("requested_eigenpairs")
        if nev is not None:
            try:
                nev_values.append(int(nev))
            except (TypeError, ValueError):
                pass
        sf = str(row.get("solver_factor") or "unknown")
        solver_factors[sf] = solver_factors.get(sf, 0) + 1
        win = row.get("target_window_hz")
        if isinstance(win, list) and len(win) == 2:
            window_samples.append([float(win[0]), float(win[1])])

    nev_summary: Dict[str, Any] = {}
    if nev_values:
        nev_summary = {
            "count": len(nev_values),
            "min": min(nev_values),
            "max": max(nev_values),
            "unique": sorted(set(nev_values)),
        }

    window_summary: Dict[str, Any] = {}
    if window_samples:
        widths = [w[1] - w[0] for w in window_samples]
        window_summary = {
            "sample_count": len(window_samples),
            "median_half_width_hz": sorted(widths)[len(widths) // 2] / 2.0,
            "min_half_width_hz": min(widths) / 2.0,
            "max_half_width_hz": max(widths) / 2.0,
        }

    loss_classification = "UNKNOWN"
    if not complete:
        loss_classification = "AUDIT_INCOMPLETE"
    elif zero_raw_targets > len(rows) // 3:
        loss_classification = "SOLVER_RETURNS_TOO_FEW_CANDIDATES"
    elif raw_but_zero_accepted > len(rows) // 4 or sum(hist.values()) > total_accepted:
        loss_classification = "CANDIDATE_FILTER_TOO_STRICT"
    elif total_accepted > 0 and total_raw > 0:
        loss_classification = "ACCEPTANCE_OR_AGGREGATION_LOSS"

    return {
        "audit_completeness": "COMPLETE" if complete else "AUDIT_INCOMPLETE",
        "audit_incomplete_reasons": incomplete_reasons,
        "total_targets": len(rows),
        "total_raw_candidates": total_raw,
        "total_accepted_modes": total_accepted,
        "total_rejected_candidates": total_rejected,
        "rejection_reason_histogram": hist,
        "targets_with_zero_raw_candidates": zero_raw_targets,
        "targets_with_raw_candidates_zero_accepted": raw_but_zero_accepted,
        "requested_eigenpairs_summary": nev_summary,
        "solver_factor_summary": solver_factors,
        "acceptance_window_summary": window_summary,
        "loss_classification": loss_classification,
    }
