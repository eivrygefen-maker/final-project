#!/usr/bin/env python3
"""Official ROM-mesh training dataset policy (L_rom_prod only, no legacy mixing)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from v2_b3_m4_lhs_pool_bridge import AGG_PASS, read_run_production_summary  # noqa: E402
from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
    DATASET_VERSION_ROM,
    LEVEL_ROM_PROD,
    MESH_PROFILE_ROM,
)
from v2_b3_m4_rom_fom_compare_lib import load_fom_modes_catalog_deduped  # noqa: E402
from v2_b3_m4_rom_scalar_fields import (  # noqa: E402
    ACCURACY_BAND_HZ_DEFAULT,
    INTENSITY_LOG_EPSILON,
    NORMALIZATION_PERCENTILE,
    enrich_catalog_intensity_derivatives,
)
from v2_b3_m4_worker_run_lib import load_json, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

OFFICIAL_INITIAL_RUN_IDS: Tuple[str, ...] = (
    "sample_000_rom_official_v1",
    "sample_001_rom_official_v1",
    "sample_002_rom_official_v1",
    "sample_003_rom_official_v1",
    "sample_004_rom_official_v1",
)

EXCLUDED_RUN_ID_FRAGMENTS: Tuple[str, ...] = (
    "rom_prod_004",
    "m4prod2",
    "strict_val",
    "reference",
    "_legacy",
    "_fixture",
    "_test",
)

OFFICIAL_DATASET_REGISTRY_REL = "ROM/classic/official_rom_dataset.jsonl"
OFFICIAL_DATASET_SCHEMA = "m4_official_rom_dataset_entry_v1"
FEATURE_SCHEMA_VERSION = "m4_lhs_geometry_wood_v1"
TARGET_SCHEMA_VERSION = "m4_modal_surrogate_v2_1_intensity"

MATURITY_INTEGRATION_ONLY = "integration_only"


def official_dataset_registry_path(repo_root: Path, shape_name: str = "classic") -> Path:
    if shape_name != "classic":
        return repo_root / "ROM" / shape_name / "official_rom_dataset.jsonl"
    return repo_root / OFFICIAL_DATASET_REGISTRY_REL


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_sample_input_profile(run_root: Path) -> Dict[str, Any]:
    path = run_root / "sample" / "sample_input.json"
    if not path.is_file():
        return {}
    try:
        doc = load_json(path)
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _read_cleanup_barrier(run_root: Path) -> Dict[str, Any]:
    path = run_root / "cleanup" / "sample_cleanup_barrier.json"
    if not path.is_file():
        return {}
    try:
        doc = load_json(path)
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _read_compaction_manifest(run_root: Path) -> Dict[str, Any]:
    path = run_root / "compaction" / "compaction_manifest.json"
    if not path.is_file():
        return {}
    try:
        doc = load_json(path)
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def is_excluded_run_id(run_id: str) -> bool:
    rid = str(run_id or "").strip().lower()
    if not rid:
        return True
    return any(frag in rid for frag in EXCLUDED_RUN_ID_FRAGMENTS)


def evaluate_official_rom_run_eligibility(
    run_root: Path,
    *,
    run_id: Optional[str] = None,
    require_initial_allowlist: bool = False,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Return (eligible, reasons, evidence) for official ROM-mesh training."""
    reasons: List[str] = []
    evidence: Dict[str, Any] = {"run_root": str(run_root)}
    run_root = run_root.resolve()
    if not run_root.is_dir():
        return False, ["missing_run_root"], evidence

    rid = str(run_id or _read_sample_input_profile(run_root).get("run_id") or run_root.name)
    evidence["run_id"] = rid
    if is_excluded_run_id(rid):
        reasons.append(f"excluded_run_id:{rid}")
    if require_initial_allowlist and rid not in OFFICIAL_INITIAL_RUN_IDS:
        reasons.append(f"not_in_initial_allowlist:{rid}")

    sample_doc = _read_sample_input_profile(run_root)
    evidence["mesh_profile"] = sample_doc.get("mesh_profile")
    evidence["mesh_level_id"] = sample_doc.get("mesh_level_id")
    evidence["dataset_version"] = sample_doc.get("dataset_version")
    if str(sample_doc.get("mesh_profile") or "") != MESH_PROFILE_ROM:
        reasons.append("mesh_profile_not_rom")
    if str(sample_doc.get("mesh_level_id") or "") != LEVEL_ROM_PROD:
        reasons.append("mesh_level_not_L_rom_prod")
    if str(sample_doc.get("dataset_version") or "") != DATASET_VERSION_ROM:
        reasons.append("dataset_version_mismatch")

    summary = read_run_production_summary(run_root)
    evidence["terminal_status"] = summary.get("terminal_status")
    evidence["aggregation_status"] = summary.get("aggregation_status")
    if str(summary.get("terminal_status") or "") != "COMPLETED":
        reasons.append("terminal_status_not_completed")
    if str(summary.get("aggregation_status") or "") != AGG_PASS:
        reasons.append("aggregation_not_pass")

    freeze_path = run_root / "freeze" / "freeze_manifest.json"
    acceptance_pass = None
    if freeze_path.is_file():
        try:
            freeze = load_json(freeze_path)
            acceptance_pass = bool(freeze.get("production_acceptance_pass"))
            evidence["production_acceptance_pass"] = acceptance_pass
        except (OSError, ValueError, json.JSONDecodeError):
            acceptance_pass = None
    if acceptance_pass is not True:
        phys = run_root / "freeze" / "physics_identity_manifest.json"
        if phys.is_file():
            try:
                identity = load_json(phys)
                acceptance_pass = bool(identity.get("production_acceptance_pass"))
                evidence["production_acceptance_pass"] = acceptance_pass
            except (OSError, ValueError, json.JSONDecodeError):
                pass
    if acceptance_pass is not True:
        reasons.append("production_acceptance_not_pass")

    compaction = _read_compaction_manifest(run_root)
    compaction_status = str(compaction.get("status") or "")
    evidence["compaction_status"] = compaction_status or None
    if compaction_status not in ("completed", "already_compacted"):
        reasons.append(f"compaction_not_completed:{compaction_status or 'missing'}")

    barrier = _read_cleanup_barrier(run_root)
    cleanup_status = str(barrier.get("status") or "")
    evidence["cleanup_status"] = cleanup_status or None
    if cleanup_status != "completed":
        reasons.append(f"cleanup_not_completed:{cleanup_status or 'missing'}")

    verify = barrier.get("verification") if isinstance(barrier.get("verification"), dict) else {}
    verification_pass = bool(verify.get("pass")) if verify else bool(barrier.get("verification_pass"))
    evidence["cleanup_verification_pass"] = verification_pass
    if not verification_pass:
        reasons.append("cleanup_verification_not_pass")

    catalog_deduped = run_root / "aggregation" / "modes_catalog_deduped.jsonl"
    catalog_raw = run_root / "aggregation" / "modes_catalog.jsonl"
    catalog_path = catalog_deduped if catalog_deduped.is_file() else catalog_raw
    if not catalog_path.is_file():
        reasons.append("missing_modes_catalog")
    else:
        evidence["catalog_path"] = rel(catalog_path, repo_root=run_root.parent.parent.parent.parent.parent)

    return len(reasons) == 0, reasons, evidence


def guitars_root(repo_root: Path) -> Path:
    return (
        repo_root
        / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
    )


def collect_official_rom_training_rows(
    *,
    repo_root: Path,
    exclude_sample_ids: Optional[Sequence[str]] = None,
    exclude_run_ids: Optional[Sequence[str]] = None,
    require_initial_allowlist: bool = False,
    allowed_run_ids: Optional[Sequence[str]] = None,
    min_mode_count: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collect training rows from official ROM-mesh runs only."""
    training: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    excluded_sids = {str(s).strip() for s in (exclude_sample_ids or []) if str(s).strip()}
    excluded_rids = {str(r).strip() for r in (exclude_run_ids or []) if str(r).strip()}
    allow_rids: Optional[Set[str]] = None
    if allowed_run_ids is not None:
        allow_rids = {str(r).strip() for r in allowed_run_ids if str(r).strip()}
    elif require_initial_allowlist:
        allow_rids = set(OFFICIAL_INITIAL_RUN_IDS)

    root = guitars_root(repo_root)
    if not root.is_dir():
        return training, [{"reason": "guitars_root_missing", "path": rel(root, repo_root=repo_root)}]

    candidates: List[Tuple[str, str, Path]] = []
    for sample_dir in sorted(root.iterdir()):
        if not sample_dir.is_dir() or not sample_dir.name.startswith("sample_"):
            continue
        runs_dir = sample_dir / "runs"
        if not runs_dir.is_dir():
            continue
        for run_dir in sorted(runs_dir.iterdir()):
            if run_dir.is_dir():
                candidates.append((sample_dir.name, run_dir.name, run_dir))

    for sid, rid, run_root in candidates:
        if sid in excluded_sids or rid in excluded_rids:
            skipped.append({"sample_id": sid, "run_id": rid, "reason": "excluded"})
            continue
        if allow_rids is not None and rid not in allow_rids:
            skipped.append({"sample_id": sid, "run_id": rid, "reason": "not_in_allowed_run_ids"})
            continue
        eligible, reasons, evidence = evaluate_official_rom_run_eligibility(
            run_root,
            run_id=rid,
            require_initial_allowlist=require_initial_allowlist and allow_rids is None,
        )
        if not eligible:
            skipped.append(
                {
                    "sample_id": sid,
                    "run_id": rid,
                    "reason": "ineligible",
                    "reasons": reasons,
                    "evidence": evidence,
                }
            )
            continue

        catalog_path = run_root / "aggregation" / "modes_catalog_deduped.jsonl"
        if not catalog_path.is_file():
            catalog_path = run_root / "aggregation" / "modes_catalog.jsonl"
        try:
            raw_modes, modes, dedupe_meta = load_fom_modes_catalog_deduped(catalog_path)
            freqs = [float(m["frequency_hz"]) for m in modes]
        except (OSError, ValueError, FileNotFoundError) as exc:
            skipped.append({"sample_id": sid, "run_id": rid, "reason": "catalog_read_error", "error": str(exc)})
            continue
        if len(freqs) < int(min_mode_count):
            skipped.append(
                {"sample_id": sid, "run_id": rid, "reason": "insufficient_modes", "mode_count": len(freqs)}
            )
            continue

        sample_doc = _read_sample_input_profile(run_root)
        params = dict(sample_doc.get("parameters") or {})
        enriched_catalog, p95_map = enrich_catalog_intensity_derivatives(
            modes,
            band=ACCURACY_BAND_HZ_DEFAULT,
            epsilon=INTENSITY_LOG_EPSILON,
            percentile=NORMALIZATION_PERCENTILE,
        )
        catalog_sha = _sha256_file(catalog_path)
        training.append(
            {
                "sample_id": sid,
                "lhs_row_index": sample_doc.get("lhs_row_index"),
                "run_id": rid,
                "run_root": rel(run_root, repo_root=repo_root),
                "catalog_path": rel(catalog_path, repo_root=repo_root),
                "catalog_sha256": catalog_sha,
                "shape_name": str(sample_doc.get("shape_name") or "classic"),
                "parameters": params,
                "frequencies_hz": freqs,
                "mode_catalog": enriched_catalog,
                "mode_count": len(freqs),
                "raw_mode_count": int(dedupe_meta.get("raw_mode_count") or len(raw_modes)),
                "deduped_mode_count": int(dedupe_meta.get("deduped_mode_count") or len(modes)),
                "intensity_p95_map": p95_map,
                "mesh_profile": MESH_PROFILE_ROM,
                "mesh_level_id": LEVEL_ROM_PROD,
                "dataset_version": DATASET_VERSION_ROM,
            }
        )

    training.sort(key=lambda r: (str(r.get("sample_id") or ""), str(r.get("run_id") or "")))
    return training, skipped


def load_official_dataset_registry(repo_root: Path, shape_name: str = "classic") -> List[Dict[str, Any]]:
    path = official_dataset_registry_path(repo_root, shape_name)
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
            if isinstance(doc, dict):
                rows.append(doc)
        except json.JSONDecodeError:
            continue
    return rows


def register_official_rom_dataset_entry(
    *,
    repo_root: Path,
    sample_id: str,
    run_id: str,
    run_root: Path,
    shape_name: str = "classic",
) -> Dict[str, Any]:
    """Append accepted FOM sample to official ROM dataset registry (idempotent per run_id)."""
    eligible, reasons, evidence = evaluate_official_rom_run_eligibility(run_root, run_id=run_id)
    if not eligible:
        raise ValueError(f"cannot_register_ineligible_run:{reasons}")

    registry_path = official_dataset_registry_path(repo_root, shape_name)
    existing = load_official_dataset_registry(repo_root, shape_name)
    if any(str(r.get("run_id")) == run_id for r in existing):
        return next(r for r in existing if str(r.get("run_id")) == run_id)

    catalog_path = run_root / "aggregation" / "modes_catalog_deduped.jsonl"
    if not catalog_path.is_file():
        catalog_path = run_root / "aggregation" / "modes_catalog.jsonl"
    entry = {
        "schema": OFFICIAL_DATASET_SCHEMA,
        "registered_utc": utc_now(),
        "sample_id": sample_id,
        "run_id": run_id,
        "shape_name": shape_name,
        "mesh_profile": MESH_PROFILE_ROM,
        "mesh_level_id": LEVEL_ROM_PROD,
        "dataset_version": DATASET_VERSION_ROM,
        "catalog_path": rel(catalog_path, repo_root=repo_root),
        "catalog_sha256": _sha256_file(catalog_path) if catalog_path.is_file() else None,
        "evidence": evidence,
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def official_training_run_ids_from_registry(
    repo_root: Path,
    shape_name: str = "classic",
) -> List[str]:
    return [str(r["run_id"]) for r in load_official_dataset_registry(repo_root, shape_name) if r.get("run_id")]


def build_initial_five_run_dataset_report(
    *,
    repo_root: Path,
    training_rows: Sequence[Mapping[str, Any]],
    skipped_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema": "m4_official_rom_initial_dataset_report_v1",
        "generated_utc": utc_now(),
        "allowed_run_ids": list(OFFICIAL_INITIAL_RUN_IDS),
        "training_run_ids": [str(r["run_id"]) for r in training_rows],
        "training_sample_ids": [str(r["sample_id"]) for r in training_rows],
        "training_row_count": len(training_rows),
        "skipped_row_count": len(skipped_rows),
        "catalog_sha256_values": [r.get("catalog_sha256") for r in training_rows],
        "mesh_profile": MESH_PROFILE_ROM,
        "mesh_level_id": LEVEL_ROM_PROD,
        "dataset_version": DATASET_VERSION_ROM,
        "skipped": list(skipped_rows),
    }


def _git_head_sha(repo_root: Path) -> Optional[str]:
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def write_official_rom_model_manifest(
    repo_root: Path,
    shape_name: str,
    *,
    model_version: str,
    training_rows: Sequence[Mapping[str, Any]],
    surrogate_schema: str,
    prediction_method: str,
    maturity: str = MATURITY_INTEGRATION_ONLY,
    production_accuracy_validated: bool = False,
) -> Path:
    manifest_path = repo_root / "ROM" / shape_name / "rom_model_manifest.json"
    surrogate_json = repo_root / "ROM" / shape_name / "m4_modal_surrogate.json"
    body = {
        "schema": "m4_rom_model_manifest_v2",
        "generated_utc": utc_now(),
        "shape_name": shape_name,
        "active_backend": "m4_surrogate",
        "m4_surrogate_json": "m4_modal_surrogate.json",
        "m4_surrogate_npz": "m4_modal_surrogate.npz",
        "model_version": model_version,
        "training_sample_ids": [str(r["sample_id"]) for r in training_rows],
        "training_run_ids": [str(r["run_id"]) for r in training_rows],
        "training_dataset_version": DATASET_VERSION_ROM,
        "mesh_profile": MESH_PROFILE_ROM,
        "mesh_level_id": LEVEL_ROM_PROD,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "target_schema_version": TARGET_SCHEMA_VERSION,
        "training_sample_count": len(training_rows),
        "surrogate_schema": surrogate_schema,
        "prediction_method": prediction_method,
        "maturity": maturity,
        "production_accuracy_validated": bool(production_accuracy_validated),
        "legacy_basis_npz": None,
        "git_commit_sha": _git_head_sha(repo_root),
        "notes": "Official ROM-mesh surrogate only; no legacy POD basis.",
    }
    write_json_atomic(manifest_path, body)
    body["model_manifest_sha256"] = _sha256_file(manifest_path)
    if surrogate_json.is_file():
        body["surrogate_json_sha256"] = _sha256_file(surrogate_json)
    write_json_atomic(manifest_path, body)
    return manifest_path
