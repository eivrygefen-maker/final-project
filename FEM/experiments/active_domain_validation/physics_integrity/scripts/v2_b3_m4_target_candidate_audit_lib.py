#!/usr/bin/env python3
"""Per-target candidate diagnostics for modal discovery audit (additive; no solve changes)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

TARGET_CANDIDATE_AUDIT_FILENAME = "target_candidate_audit.jsonl"


def target_candidate_audit_path(chunk_dir: Path) -> Path:
    return chunk_dir / TARGET_CANDIDATE_AUDIT_FILENAME


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

    min_residual = None
    for key in ("min_eps_compute_error_relative", "min_residual"):
        if target_row.get(key) is not None:
            min_residual = target_row.get(key)
            break

    return {
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
        "candidate_count_after_physical_filters": accepted_n,
        "accepted_mode_count": accepted_n,
        "rejected_candidate_count": rejected,
        "rejection_reasons": rejection_tally or {},
        "acceptance_policy": target_row.get("acceptance_policy"),
        "acceptance_freq_lo_hz": target_row.get("acceptance_freq_lo_hz"),
        "acceptance_freq_hi_hz": target_row.get("acceptance_freq_hi_hz"),
        "min_residual": min_residual,
        "accepted_frequencies_hz": accepted_freqs,
        "targets_passed_means_solve_success": True,
        "note": (
            "targets_passed on worker_result means numerical target solve PASS, "
            "not guaranteed mode discovery."
        ),
    }


def append_target_candidate_audit_row(chunk_dir: Path, row: Mapping[str, Any]) -> Path:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    path = target_candidate_audit_path(chunk_dir)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return path


def write_target_candidate_audit_jsonl(
    chunk_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    path = target_candidate_audit_path(chunk_dir)
    text = "".join(json.dumps(dict(r), sort_keys=True) + "\n" for r in rows)
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
                rows.append(doc)
        except ValueError:
            continue
    return rows
