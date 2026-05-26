#!/usr/bin/env python3
"""Diagnostic-only persistence helpers for mapping-fixed preserve-all candidate banks."""
from __future__ import annotations

import math
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fem_mode_array_utils import MODE_VECTOR_FILE_SUFFIX, dense_to_csr_f32_column, save_mode_csr

VERDICT_PERSISTENCE_FAILURE = (
    "MAPPING_FIXED_UNREGULARIZED_BASELINE_CANDIDATE_PERSISTENCE_FAILURE"
)

CANDIDATE_FILENAME_TEMPLATE = "candidate_eps_slot_{index:04d}"


def candidate_slot_path(modes_dir: Path, slot_index: int) -> Path:
    return modes_dir / f"{CANDIDATE_FILENAME_TEMPLATE.format(index=int(slot_index))}{MODE_VECTOR_FILE_SUFFIX}"


def load_preserve_all_bank_from_config(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect in-memory preserve-all bank attached during _solve_coupled_evp."""
    sc = cfg.get("solver") if isinstance(cfg.get("solver"), dict) else {}
    bank = list(cfg.get("_eps_diagnostic_candidate_bank_records") or [])
    if not bank and isinstance(sc, dict):
        bank = list(sc.get("_eps_diagnostic_candidate_bank_records") or [])
    if not bank:
        eps = cfg.get("_eps_batch_diagnostics") or {}
        bank = list(eps.get("_eps_diagnostic_candidate_bank_records") or [])
    return bank


def bank_records_from_eigvecs(
    freqs_hz: List[float],
    eigvecs: np.ndarray,
    *,
    eps_diag: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Fallback when bank metadata was not propagated but worker returned column vectors."""
    n = int(eigvecs.shape[1]) if eigvecs.ndim == 2 else 0
    sem = str(eps_diag.get("eps_eigenvalue_semantics", "slepc_backtransformed"))
    leg = bool(eps_diag.get("legacy_double_shift_mapping_disabled", True))
    sigma = float(eps_diag.get("st_sigma_hz_used", float("nan")))
    out: List[Dict[str, Any]] = []
    for j in range(n):
        f_hz = float(freqs_hz[j]) if j < len(freqs_hz) else float("nan")
        lam = (2.0 * math.pi * f_hz) ** 2 if math.isfinite(f_hz) and f_hz > 0 else None
        out.append(
            {
                "eps_slot_index": int(j),
                "candidate_index": int(j),
                "mu_raw": None,
                "lam_phys": lam,
                "lam_map_tag": "from_eigvec_fallback",
                "reported_frequency_hz": f_hz if math.isfinite(f_hz) else None,
                "sigma_used_hz": sigma,
                "st_type": None,
                "eps_eigenvalue_semantics": sem,
                "legacy_double_shift_mapping_disabled": leg,
                "vector": np.asarray(eigvecs[:, j], dtype=np.float64),
            }
        )
    return out


def persist_candidate_bank(
    case_dir: Path,
    bank_records: List[Dict[str, Any]],
    *,
    save_vector_fn,
) -> Tuple[int, List[Dict[str, Any]], List[str]]:
    """
    Save every bank record vector to modes/candidate_eps_slot_*.smx.npz.

    ``save_vector_fn(vec, mode_path) -> row_dict`` performs optional energy diagnostics.
    Returns (num_saved, saved_rows, save_errors).
    """
    modes_dir = case_dir / "modes"
    modes_dir.mkdir(parents=True, exist_ok=True)
    saved_rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for rec in bank_records:
        slot = int(rec.get("eps_slot_index", rec.get("candidate_index", len(saved_rows))))
        vec = rec.get("vector")
        if vec is None:
            errors.append(f"slot_{slot}:missing_vector_in_bank_record")
            continue
        mode_path = candidate_slot_path(modes_dir, slot)
        try:
            row = save_vector_fn(np.asarray(vec, dtype=np.float64).ravel(), mode_path, rec)
            row["candidate_index"] = slot
            row["eps_slot_index"] = slot
            row["vector_file"] = str(mode_path.relative_to(case_dir)).replace("\\", "/")
            row["vector_path"] = row["vector_file"]
            row["persistence_status"] = "saved"
            for key in (
                "mu_raw",
                "lam_phys",
                "lam_map_tag",
                "reported_frequency_hz",
                "sigma_used_hz",
                "st_type",
                "eps_eigenvalue_semantics",
                "legacy_double_shift_mapping_disabled",
            ):
                if key in rec and key not in row:
                    row[key] = rec.get(key)
            saved_rows.append(row)
        except Exception as exc:
            errors.append(f"slot_{slot}:{type(exc).__name__}:{exc}")
    return len(saved_rows), saved_rows, errors


def pressure_block_mapping_metadata(
    *,
    p_to_W: Optional[np.ndarray],
    source: str,
) -> Dict[str, Any]:
    p = np.asarray(p_to_W if p_to_W is not None else [], dtype=np.int32).ravel()
    return {
        "source": str(source),
        "p_to_W": p.tolist(),
        "p_to_W_length": int(p.size),
        "p_to_W_crc32": int(zlib.crc32(p.tobytes()) & 0xFFFFFFFF) if p.size else 0,
    }


def write_eps_candidate_bank_json(
    case_dir: Path,
    *,
    bank_records: List[Dict[str, Any]],
    saved_rows: List[Dict[str, Any]],
    nconv_marked: int,
    save_errors: List[str],
    pressure_block_mapping: Optional[Dict[str, Any]] = None,
) -> Path:
    path = case_dir / "diagnostics" / "eps_candidate_bank.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    payload = {
        "candidates": [
            {k: v for k, v in rec.items() if k != "vector"} for rec in bank_records
        ],
        "saved_mode_rows": saved_rows,
        "nconv_marked": int(nconv_marked),
        "eps_diagnostic_candidate_bank_count": len(bank_records),
        "num_vectors_saved": len(saved_rows),
        "save_errors": list(save_errors),
        "pressure_block_mapping": dict(pressure_block_mapping or {}),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def check_persistence_gate(
    *,
    nconv_marked: int,
    bank_count: int,
    num_vectors_saved: int,
    save_errors: List[str],
) -> Optional[Dict[str, Any]]:
    """Return persistence-failure payload if gates fail; else None."""
    if int(nconv_marked) <= 0:
        return None
    if int(num_vectors_saved) == 0:
        return {
            "diagnostic_verdict": VERDICT_PERSISTENCE_FAILURE,
            "verdict_reason": "nconv_marked_positive_but_num_vectors_saved_zero",
            "nconv_marked": int(nconv_marked),
            "eps_diagnostic_candidate_bank_count": int(bank_count),
            "num_vectors_saved": int(num_vectors_saved),
            "save_errors": list(save_errors),
        }
    if int(bank_count) > 0 and int(num_vectors_saved) != int(bank_count):
        return {
            "diagnostic_verdict": VERDICT_PERSISTENCE_FAILURE,
            "verdict_reason": "num_vectors_saved_not_equal_to_eps_diagnostic_candidate_bank_count",
            "nconv_marked": int(nconv_marked),
            "eps_diagnostic_candidate_bank_count": int(bank_count),
            "num_vectors_saved": int(num_vectors_saved),
            "save_errors": list(save_errors),
        }
    if int(num_vectors_saved) < int(nconv_marked):
        return {
            "diagnostic_verdict": VERDICT_PERSISTENCE_FAILURE,
            "verdict_reason": "num_vectors_saved_less_than_nconv_marked",
            "nconv_marked": int(nconv_marked),
            "eps_diagnostic_candidate_bank_count": int(bank_count),
            "num_vectors_saved": int(num_vectors_saved),
            "save_errors": list(save_errors),
        }
    return None
