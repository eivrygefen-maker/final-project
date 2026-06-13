#!/usr/bin/env python3
"""Stage 5.1 ROM readiness report from actual manifests and datasets."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _sample_index(sample_id: str) -> Optional[int]:
    try:
        return int(str(sample_id).split("_")[-1])
    except ValueError:
        return None


def _sorted_sample_ids(ids: Sequence[str]) -> List[str]:
    return sorted(set(str(x) for x in ids if x), key=lambda s: (_sample_index(s) or -1, s))


def _gaps_in_sequence(nums: Sequence[int]) -> List[int]:
    if not nums:
        return []
    lo, hi = min(nums), max(nums)
    present = set(nums)
    return [n for n in range(lo, hi + 1) if n not in present]


def load_lhs_pool_entries(repo_root: Path) -> List[Dict[str, Any]]:
    path = repo_root / "ROM" / "classic" / "lhs_pool.json"
    if not path.is_file():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(doc, list):
        return list(doc)
    if isinstance(doc, dict):
        entries = doc.get("entries")
        if isinstance(entries, list):
            return list(entries)
    return []


def deterministic_holdout_split(
    sample_ids: Sequence[str],
    *,
    val_fraction: float = 0.20,
    holdout_latest_n: int = 12,
) -> Dict[str, Any]:
    ids = _sorted_sample_ids(sample_ids)
    if not ids:
        return {"train": [], "validation": [], "strategy": "empty"}
    # Prefer latest-N holdout when enough samples for extrapolation test
    if len(ids) >= holdout_latest_n + 8:
        val = ids[-holdout_latest_n:]
        train = ids[:-holdout_latest_n]
        return {
            "strategy": "holdout_latest_n",
            "holdout_latest_n": holdout_latest_n,
            "train": train,
            "validation": val,
        }
    n_val = max(1, int(round(len(ids) * val_fraction)))
    val = ids[-n_val:]
    train = ids[:-n_val]
    return {
        "strategy": "fraction_tail",
        "val_fraction": val_fraction,
        "train": train,
        "validation": val,
    }


def assess_readiness(
    *,
    jsonl_ids: Sequence[str],
    manifest_ids: Sequence[str],
    lhs_count: int,
    manifest_jsonl_agree: bool,
    has_surrogate_npz: bool,
    gaps_jsonl: Sequence[int],
) -> Tuple[str, str]:
    n_jsonl = len(jsonl_ids)
    if n_jsonl < 20 or gaps_jsonl or not has_surrogate_npz:
        return "NOT_READY", "Continue classical-guitar sampling — insufficient registered ROM entries or surrogate missing."
    if not manifest_jsonl_agree or n_jsonl < 30:
        return (
            "READY_FOR_MORE_CLASSICAL_SAMPLES",
            "Run additional LHS validation batch and align manifest with jsonl before next body family.",
        )
    if n_jsonl >= 30 and has_surrogate_npz:
        return (
            "READY_FOR_NEXT_BODY_FAMILY_DIAGNOSTIC",
            "ROM coverage adequate for diagnostic work on next body family; STK V4.1 endpoint should be listening-validated first.",
        )
    return "NOT_READY", "Insufficient data."


def build_rom_readiness_report(
    repo_root: Path,
    *,
    out_json: Optional[Path] = None,
    out_md: Optional[Path] = None,
) -> Dict[str, Any]:
    rom_dir = repo_root / "ROM" / "classic"
    jsonl_path = rom_dir / "official_rom_dataset.jsonl"
    manifest_path = rom_dir / "rom_model_manifest.json"
    retrain_path = rom_dir / "rom_retrain_state.json"
    sur_json = rom_dir / "m4_modal_surrogate.json"
    sur_npz = rom_dir / "m4_modal_surrogate.npz"

    jsonl_rows = _read_jsonl(jsonl_path)
    jsonl_ids = _sorted_sample_ids([str(r.get("sample_id") or "") for r in jsonl_rows])
    jsonl_nums = [n for n in (_sample_index(s) for s in jsonl_ids) if n is not None]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest_ids = _sorted_sample_ids(list(manifest.get("training_sample_ids") or []))
    manifest_nums = [n for n in (_sample_index(s) for s in manifest_ids) if n is not None]

    retrain = json.loads(retrain_path.read_text(encoding="utf-8")) if retrain_path.is_file() else {}

    lhs_entries = load_lhs_pool_entries(repo_root)
    lhs_ids = _sorted_sample_ids([str(e.get("id") or e.get("sample_id") or "") for e in lhs_entries])

    jsonl_set = set(jsonl_ids)
    manifest_set = set(manifest_ids)
    only_manifest = sorted(manifest_set - jsonl_set)
    only_jsonl = sorted(jsonl_set - manifest_set)

    range_check = {}
    for label, lo, hi in (
        ("sample_037_to_066_in_jsonl", 37, 66),
        ("sample_037_to_066_in_manifest", 37, 66),
    ):
        present = [f"sample_{n:03d}" for n in range(lo, hi + 1) if f"sample_{n:03d}" in (jsonl_set if "jsonl" in label else manifest_set)]
        range_check[label] = {
            "present_count": len(present),
            "present_ids": present[:20],
            "missing_count": (hi - lo + 1) - len(present),
        }

    holdout = deterministic_holdout_split(jsonl_ids or manifest_ids)
    readiness, recommendation_detail = assess_readiness(
        jsonl_ids=jsonl_ids,
        manifest_ids=manifest_ids,
        lhs_count=len(lhs_ids),
        manifest_jsonl_agree=(not only_manifest and not only_jsonl and len(manifest_ids) == len(jsonl_ids)),
        has_surrogate_npz=sur_npz.is_file(),
        gaps_jsonl=_gaps_in_sequence(jsonl_nums),
    )

    recommendation_map = {
        "NOT_READY": "A. Continue classical-guitar sampling",
        "READY_FOR_MORE_CLASSICAL_SAMPLES": "B. Run a new LHS validation batch",
        "READY_FOR_NEXT_BODY_FAMILY_DIAGNOSTIC": "C. Start next body family diagnostic",
        "READY_FOR_NEXT_BODY_FAMILY_PRODUCTION": "D. Start box/acoustic-body prototype",
    }

    report: Dict[str, Any] = {
        "schema_version": "stage51_rom_readiness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_rom_dataset_jsonl": {
            "path": str(jsonl_path),
            "entry_count": len(jsonl_rows),
            "sample_ids": jsonl_ids,
            "min_sample_id": jsonl_ids[0] if jsonl_ids else None,
            "max_sample_id": jsonl_ids[-1] if jsonl_ids else None,
            "missing_gaps_in_sequence": _gaps_in_sequence(jsonl_nums),
        },
        "rom_model_manifest": {
            "path": str(manifest_path),
            "training_sample_count": len(manifest_ids),
            "training_sample_ids": manifest_ids,
            "min_sample_id": manifest_ids[0] if manifest_ids else None,
            "max_sample_id": manifest_ids[-1] if manifest_ids else None,
            "missing_gaps_in_sequence": _gaps_in_sequence(manifest_nums),
            "model_version": manifest.get("model_version"),
            "generated_utc": manifest.get("generated_utc"),
        },
        "rom_retrain_state": {
            "path": str(retrain_path),
            "present": retrain_path.is_file(),
            "last_registered_run_id": retrain.get("last_registered_run_id"),
            "last_registered_utc": retrain.get("last_registered_utc"),
            "new_samples_since_last_train": retrain.get("new_samples_since_last_train"),
        },
        "lhs_pool": {
            "path": str(rom_dir / "lhs_pool.json"),
            "entry_count": len(lhs_entries),
            "max_sample_id": lhs_ids[-1] if lhs_ids else None,
        },
        "manifest_jsonl_diff": {
            "only_in_manifest_not_jsonl": only_manifest,
            "only_in_jsonl_not_manifest": only_jsonl,
            "counts_agree": len(manifest_ids) == len(jsonl_ids),
        },
        "range_sample_037_066": range_check,
        "m4_surrogate": {
            "json_present": sur_json.is_file(),
            "npz_present": sur_npz.is_file(),
        },
        "holdout_split": holdout,
        "validation_split_adequate": len(holdout.get("validation") or []) >= 6,
        "readiness_status": readiness,
        "recommendation": recommendation_map.get(readiness, readiness),
        "recommendation_detail": recommendation_detail,
        "fem_launched": False,
    }

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if out_md:
        out_md.write_text(_render_md(report), encoding="utf-8")
    return report


def _render_md(report: Mapping[str, Any]) -> str:
    j = report.get("official_rom_dataset_jsonl") or {}
    m = report.get("rom_model_manifest") or {}
    lines = [
        "# Stage 5.1 — ROM readiness report",
        "",
        f"- **jsonl entries:** {j.get('entry_count')} ({j.get('min_sample_id')} → {j.get('max_sample_id')})",
        f"- **manifest training samples:** {m.get('training_sample_count')} ({m.get('min_sample_id')} → {m.get('max_sample_id')})",
        f"- **jsonl gaps:** {j.get('missing_gaps_in_sequence')}",
        f"- **manifest-only ids:** {(report.get('manifest_jsonl_diff') or {}).get('only_in_manifest_not_jsonl')}",
        f"- **jsonl-only ids:** {(report.get('manifest_jsonl_diff') or {}).get('only_in_jsonl_not_manifest')}",
        f"- **sample_037–066 in jsonl:** {(report.get('range_sample_037_066') or {}).get('sample_037_to_066_in_jsonl', {}).get('present_count', 0)}",
        f"- **M4 surrogate npz:** {(report.get('m4_surrogate') or {}).get('npz_present')}",
        "",
        f"## Readiness: **{report.get('readiness_status')}**",
        f"Recommendation: **{report.get('recommendation')}**",
        "",
        report.get("recommendation_detail", ""),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    out = repo / "audio" / "debug_reports"
    build_rom_readiness_report(
        repo,
        out_json=out / "stage51_rom_readiness_report.json",
        out_md=out / "stage51_rom_readiness_report.md",
    )
    print("Wrote stage51_rom_readiness_report")
