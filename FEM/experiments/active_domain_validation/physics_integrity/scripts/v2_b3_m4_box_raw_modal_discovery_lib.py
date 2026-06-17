#!/usr/bin/env python3
"""BOX raw/unfiltered modal discovery diagnostic mode (additive; CLASSIC unchanged)."""
from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

RAW_DIAGNOSTIC_ENV = "BOX_RAW_MODAL_DISCOVERY"
WORKER_DIAGNOSTIC_JSONL = "raw_modal_diagnostic.jsonl"

RAW_SOLVER_CATALOG_AGG = "aggregation/raw_solver_candidate_catalog.json"
RAW_SOLVER_CATALOG_CSV_AGG = "aggregation/raw_solver_candidate_catalog.csv"
UNFILTERED_CATALOG_AGG = "aggregation/unfiltered_mode_catalog.json"
UNFILTERED_CATALOG_CSV_AGG = "aggregation/unfiltered_mode_catalog.csv"
ACCEPTED_FILTERED_CATALOG_AGG = "aggregation/accepted_filtered_mode_catalog.json"

RAW_SOLVER_CATALOG_VAL = "validation/raw_solver_candidate_catalog.json"
UNFILTERED_CATALOG_VAL = "validation/unfiltered_mode_catalog.json"
ACCEPTED_FILTERED_CATALOG_VAL = "validation/accepted_filtered_mode_catalog.json"

CATALOG_SCHEMA = "m4_box_raw_modal_catalog_v1"

CATALOG_CSV_FIELDS: Tuple[str, ...] = (
    "sample_id",
    "run_id",
    "shape",
    "chunk_id",
    "target_hz",
    "frequency_hz",
    "source_stage",
    "solver_factor",
    "requested_eigenpairs",
    "candidate_rank",
    "residual",
    "residual_status",
    "inside_target_window",
    "target_window_hz",
    "would_pass_normal_filters",
    "normal_filter_rejection_reasons",
    "bridge_excitation_proxy",
    "radiation_proxy",
    "mic_output_proxy",
    "dominant_region",
    "top_share",
    "back_share",
    "air_share",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def box_raw_modal_discovery_enabled(*, shape_name: Optional[str] = None) -> bool:
    """True when BOX raw diagnostic env is set and shape is box (never classic/acoustic)."""
    val = os.environ.get(RAW_DIAGNOSTIC_ENV, "").strip().lower()
    if val not in ("1", "true", "yes", "on"):
        return False
    shape = (shape_name or os.environ.get("SHAPE") or "").strip().lower()
    if not shape:
        return True
    return shape == "box"


def resolve_worker_shape_name(
    chunk_targets: Optional[Mapping[str, Any]] = None,
) -> str:
    env_shape = os.environ.get("SHAPE", "").strip().lower()
    if env_shape:
        return env_shape
    sid = str((chunk_targets or {}).get("sample_id") or "")
    if sid.startswith("box_"):
        return "box"
    if sid.startswith("acoustic_"):
        return "acoustic"
    return "classic"


def _share_from_participation(mode: Mapping[str, Any], key: str) -> Optional[float]:
    for name in (f"{key}_share", f"{key}_participation"):
        val = mode.get(name)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def build_catalog_row(
    *,
    sample_id: str,
    run_id: str,
    shape: str,
    chunk_id: str,
    target_hz: float,
    target_row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_rank: int,
) -> Dict[str, Any]:
    win = candidate.get("target_window_hz")
    if win is None:
        win = target_row.get("per_target_acceptance_window_hz")
    return {
        "sample_id": sample_id,
        "run_id": run_id,
        "shape": shape,
        "chunk_id": chunk_id,
        "target_hz": float(target_hz),
        "frequency_hz": candidate.get("frequency_hz"),
        "source_stage": "worker_target_solve",
        "solver_factor": target_row.get("factor_solver_effective") or target_row.get("factor_solver"),
        "requested_eigenpairs": target_row.get("nev"),
        "candidate_rank": int(candidate_rank),
        "residual": candidate.get("residual"),
        "residual_status": candidate.get("residual_status") or "UNKNOWN",
        "inside_target_window": candidate.get("inside_target_window"),
        "target_window_hz": win,
        "would_pass_normal_filters": candidate.get("would_pass_normal_filters"),
        "normal_filter_rejection_reasons": list(candidate.get("normal_filter_rejection_reasons") or []),
        "passes_numerical_sanity": candidate.get("passes_numerical_sanity"),
        "bridge_excitation_proxy": candidate.get("bridge_excitation_coupling"),
        "radiation_proxy": candidate.get("radiation_proxy"),
        "mic_output_proxy": candidate.get("mic_output_proxy"),
        "dominant_region": candidate.get("dominant_region"),
        "top_share": _share_from_participation(candidate, "top"),
        "back_share": _share_from_participation(candidate, "back"),
        "air_share": _share_from_participation(candidate, "air"),
    }


def append_worker_diagnostic_row(chunk_dir: Path, row: Mapping[str, Any]) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    path = chunk_dir / WORKER_DIAGNOSTIC_JSONL
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_worker_diagnostic_from_solver_targets(
    *,
    output_dir: Path,
    chunk_targets: Mapping[str, Any],
    solver_targets: Sequence[Mapping[str, Any]],
    shape_name: str,
) -> int:
    """Persist per-chunk diagnostic JSONL from solver per-target rows."""
    sample_id = str(chunk_targets.get("sample_id") or "")
    run_id = str(chunk_targets.get("run_id") or "")
    chunk_id = str(chunk_targets.get("chunk_id") or "")
    count = 0
    jsonl_path = output_dir / WORKER_DIAGNOSTIC_JSONL
    if jsonl_path.is_file():
        jsonl_path.unlink()
    for target_row in solver_targets:
        target_hz = target_row.get("target_frequency_hz")
        if target_hz is None:
            continue
        candidates = target_row.get("diagnostic_candidates") or []
        for rank, cand in enumerate(candidates):
            row = build_catalog_row(
                sample_id=sample_id,
                run_id=run_id,
                shape=shape_name,
                chunk_id=chunk_id,
                target_hz=float(target_hz),
                target_row=target_row,
                candidate=cand,
                candidate_rank=rank,
            )
            append_worker_diagnostic_row(output_dir, row)
            count += 1
    return count


def load_worker_diagnostic_rows(chunk_dir: Path) -> List[Dict[str, Any]]:
    path = chunk_dir / WORKER_DIAGNOSTIC_JSONL
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
            if isinstance(doc, dict):
                rows.append(doc)
        except ValueError:
            continue
    return rows


def _write_catalog_json(path: Path, *, schema: str, rows: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema": schema,
        "generated_utc": utc_now(),
        "row_count": len(rows),
        **dict(meta),
        "rows": list(rows),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_catalog_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CATALOG_CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("target_window_hz", "normal_filter_rejection_reasons"):
                if key in out and not isinstance(out[key], str):
                    out[key] = json.dumps(out[key], sort_keys=True)
            writer.writerow(out)


def build_accepted_filtered_rows(
    *,
    sample_id: str,
    run_id: str,
    shape: str,
    accepted_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in accepted_records:
        rows.append(
            {
                "sample_id": sample_id,
                "run_id": run_id,
                "shape": shape,
                "chunk_id": rec.get("chunk_id"),
                "target_hz": rec.get("target_hz"),
                "frequency_hz": rec.get("frequency_hz"),
                "source_stage": rec.get("source") or "aggregation_accepted",
                "zone_id": rec.get("zone_id"),
                "would_pass_normal_filters": True,
                "normal_filter_rejection_reasons": [],
            }
        )
    return rows


def merge_box_raw_catalogs_for_run(
    run_root: Path,
    *,
    sample_id: str,
    run_id: str,
    shape_name: str,
    chunk_ids: Sequence[str],
    accepted_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Merge per-chunk worker diagnostics into run-level catalogs."""
    run_root = run_root.resolve()
    raw_rows: List[Dict[str, Any]] = []
    for chunk_id in chunk_ids:
        raw_rows.extend(load_worker_diagnostic_rows(run_root / "worker_results" / chunk_id))

    unfiltered_rows = [r for r in raw_rows if r.get("passes_numerical_sanity")]
    accepted_rows = build_accepted_filtered_rows(
        sample_id=sample_id,
        run_id=run_id,
        shape=shape_name,
        accepted_records=accepted_records,
    )

    meta = {
        "sample_id": sample_id,
        "run_id": run_id,
        "shape": shape_name,
        "chunk_count": len(chunk_ids),
        "raw_solver_candidate_count": len(raw_rows),
        "unfiltered_mode_count": len(unfiltered_rows),
        "accepted_filtered_mode_count": len(accepted_rows),
        "diagnostic_mode": "BOX_RAW_MODAL_DISCOVERY",
    }

    if raw_rows:
        _write_catalog_json(run_root / RAW_SOLVER_CATALOG_AGG, schema=CATALOG_SCHEMA, rows=raw_rows, meta=meta)
        _write_catalog_csv(run_root / RAW_SOLVER_CATALOG_CSV_AGG, raw_rows)
        _write_catalog_json(run_root / RAW_SOLVER_CATALOG_VAL, schema=CATALOG_SCHEMA, rows=raw_rows, meta=meta)

    if unfiltered_rows:
        _write_catalog_json(run_root / UNFILTERED_CATALOG_AGG, schema=CATALOG_SCHEMA, rows=unfiltered_rows, meta=meta)
        _write_catalog_csv(run_root / UNFILTERED_CATALOG_CSV_AGG, unfiltered_rows)
        _write_catalog_json(run_root / UNFILTERED_CATALOG_VAL, schema=CATALOG_SCHEMA, rows=unfiltered_rows, meta=meta)

    if accepted_rows:
        _write_catalog_json(
            run_root / ACCEPTED_FILTERED_CATALOG_AGG,
            schema=CATALOG_SCHEMA,
            rows=accepted_rows,
            meta=meta,
        )
        _write_catalog_json(
            run_root / ACCEPTED_FILTERED_CATALOG_VAL,
            schema=CATALOG_SCHEMA,
            rows=accepted_rows,
            meta=meta,
        )

    return meta


def load_catalog_rows(run_root: Path, rel_path: str) -> List[Dict[str, Any]]:
    path = run_root / rel_path
    if not path.is_file():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    rows = doc.get("rows") or []
    return [r for r in rows if isinstance(r, dict)]


def build_raw_vs_filtered_analysis(
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    unfiltered_rows: Sequence[Mapping[str, Any]],
    accepted_rows: Sequence[Mapping[str, Any]],
    deduped_mode_count: int,
    chunk_audits: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    hist: Counter[str] = Counter()
    for row in raw_rows:
        if row.get("would_pass_normal_filters"):
            continue
        for reason in row.get("normal_filter_rejection_reasons") or []:
            hist[str(reason)] += 1

    total_raw = len(raw_rows)
    total_unfiltered = len(unfiltered_rows)
    total_accepted = len(accepted_rows)
    total_rejected = sum(1 for r in raw_rows if not r.get("would_pass_normal_filters"))

    targets_many_raw_zero_accepted: List[float] = []
    by_target_raw: Dict[float, int] = {}
    by_target_accepted: Dict[float, int] = {}
    for row in raw_rows:
        thz = row.get("target_hz")
        if thz is None:
            continue
        t = float(thz)
        by_target_raw[t] = by_target_raw.get(t, 0) + 1
        if row.get("would_pass_normal_filters"):
            by_target_accepted[t] = by_target_accepted.get(t, 0) + 1
    for t, raw_c in by_target_raw.items():
        if raw_c >= 2 and by_target_accepted.get(t, 0) == 0:
            targets_many_raw_zero_accepted.append(t)

    chunks_raw_no_accept: List[str] = []
    if chunk_audits:
        for ch in chunk_audits:
            per = ch.get("per_target_diagnostics") or []
            raw_c = sum(int(r.get("candidate_count_raw") or 0) for r in per)
            acc = sum(int(r.get("accepted_mode_count") or 0) for r in per)
            if raw_c > 0 and acc == 0:
                chunks_raw_no_accept.append(str(ch.get("chunk_id")))

    zero_solver_targets = sum(1 for t, c in by_target_raw.items() if c == 0)
    targets_zero_solver = [
        float(t) for t in by_target_raw if by_target_raw[t] == 0
    ]

    rejected_sorted = sorted(
        [r for r in raw_rows if not r.get("would_pass_normal_filters")],
        key=lambda r: (
            -(len(r.get("normal_filter_rejection_reasons") or [])),
            float(r.get("frequency_hz") or 0.0),
        ),
    )[:20]

    def _freq_span(rows: Sequence[Mapping[str, Any]]) -> Optional[List[float]]:
        freqs = [float(r["frequency_hz"]) for r in rows if r.get("frequency_hz") is not None]
        if not freqs:
            return None
        return [min(freqs), max(freqs)]

    pct_kept = None
    if total_raw > 0:
        pct_kept = round(100.0 * total_accepted / total_raw, 2)

    classification = "RAW_DIAGNOSTIC_INCOMPLETE"
    if total_raw == 0:
        classification = "RAW_DIAGNOSTIC_INCOMPLETE"
    elif total_raw < max(20, int(0.35 * max(len(by_target_raw), 1))):
        classification = "SOLVER_RETURNS_TOO_FEW_CANDIDATES"
    elif total_unfiltered > max(total_accepted * 2, total_accepted + 5):
        if hist.get("outside_acceptance_window", 0) > hist.get("support_participation_fail", 0):
            classification = "TARGET_WINDOW_TOO_STRICT"
        elif hist.get("support_participation_fail", 0) >= 3:
            classification = "CLASSIC_SHAPE_ASSUMPTION_SUSPECTED"
        else:
            classification = "FILTERS_REJECT_VALID_BOX_MODES"
    elif total_accepted > 0:
        classification = "ACCEPTANCE_OR_AGGREGATION_LOSS"

    dominant_reason = hist.most_common(1)[0][0] if hist else None
    if dominant_reason in ("support_participation_fail",) and classification == "FILTERS_REJECT_VALID_BOX_MODES":
        classification = "CLASSIC_SHAPE_ASSUMPTION_SUSPECTED"

    return {
        "total_solver_candidates": total_raw,
        "total_unfiltered_candidates": total_unfiltered,
        "total_normally_accepted_modes": total_accepted,
        "deduped_final_modes": deduped_mode_count,
        "total_rejected_by_normal_filters": total_rejected,
        "rejection_reason_histogram": dict(hist),
        "percentage_kept": pct_kept,
        "targets_with_many_raw_zero_accepted": sorted(targets_many_raw_zero_accepted)[:20],
        "chunks_with_raw_candidates_zero_accepted": chunks_raw_no_accept[:20],
        "targets_with_zero_solver_candidates": targets_zero_solver[:20],
        "targets_with_zero_solver_candidate_count": zero_solver_targets,
        "top_rejected_candidates": [
            {
                "frequency_hz": r.get("frequency_hz"),
                "target_hz": r.get("target_hz"),
                "chunk_id": r.get("chunk_id"),
                "rejection_reasons": r.get("normal_filter_rejection_reasons"),
            }
            for r in rejected_sorted
        ],
        "raw_catalog_frequency_span_hz": _freq_span(raw_rows),
        "filtered_catalog_frequency_span_hz": _freq_span(accepted_rows),
        "loss_classification": classification,
        "diagnostic_mode": "BOX_RAW_MODAL_DISCOVERY",
    }


def write_raw_diagnostic_plots(
    *,
    agg_dir: Path,
    raw_rows: Sequence[Mapping[str, Any]],
    accepted_rows: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
    except ImportError:
        return out

    agg_dir.mkdir(parents=True, exist_ok=True)
    if raw_rows:
        targets = [float(r["target_hz"]) for r in raw_rows if r.get("target_hz") is not None]
        freqs = [float(r["frequency_hz"]) for r in raw_rows if r.get("frequency_hz") is not None]
        if targets and freqs:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.scatter(targets[: len(freqs)], freqs, s=10, alpha=0.6, c="#3498db")
            ax.plot([min(targets), max(targets)], [min(targets), max(targets)], "k--", alpha=0.3, lw=1)
            ax.set_xlabel("target_hz")
            ax.set_ylabel("raw candidate frequency_hz")
            ax.set_title("BOX raw diagnostic: frequency vs target")
            fig.tight_layout()
            p = agg_dir / "raw_frequency_vs_target.png"
            fig.savefig(p, dpi=120)
            plt.close(fig)
            out["raw_frequency_vs_target.png"] = p

        if freqs:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(freqs, bins=min(40, max(10, len(freqs) // 3)), color="#9b59b6", alpha=0.8)
            ax.set_xlabel("frequency_hz")
            ax.set_ylabel("count")
            ax.set_title("BOX raw diagnostic: frequency histogram")
            fig.tight_layout()
            p = agg_dir / "raw_frequency_histogram.png"
            fig.savefig(p, dpi=120)
            plt.close(fig)
            out["raw_frequency_histogram.png"] = p

        acc_freqs = [float(r["frequency_hz"]) for r in accepted_rows if r.get("frequency_hz") is not None]
        if freqs and acc_freqs:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.scatter(freqs, [1.0] * len(freqs), s=8, alpha=0.4, c="#95a5a6", label="raw")
            ax.scatter(acc_freqs, [1.2] * len(acc_freqs), s=12, alpha=0.8, c="#e74c3c", label="accepted")
            ax.set_xlabel("frequency_hz")
            ax.set_ylabel("lane")
            ax.set_title("BOX raw vs accepted frequency overlay")
            ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            p = agg_dir / "raw_vs_accepted_frequency_overlay.png"
            fig.savefig(p, dpi=120)
            plt.close(fig)
            out["raw_vs_accepted_frequency_overlay.png"] = p

    hist = analysis.get("rejection_reason_histogram") or {}
    if hist:
        fig, ax = plt.subplots(figsize=(10, 4))
        labels = list(hist.keys())
        vals = [hist[k] for k in labels]
        ax.barh(labels, vals, color="#e67e22")
        ax.set_xlabel("count")
        ax.set_title("BOX raw diagnostic: rejection reason histogram")
        fig.tight_layout()
        p = agg_dir / "rejection_reason_histogram.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        out["rejection_reason_histogram.png"] = p

    return out
