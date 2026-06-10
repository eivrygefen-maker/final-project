#!/usr/bin/env python3
"""M4 production mesh profile resolution (reference vs ROM)."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from v2_mesh_convergence_common import load_manifest  # noqa: E402
from v2_mesh_convergence_mesh import effective_controls_from_level_def  # noqa: E402

MESH_PROFILE_REFERENCE = "reference"
MESH_PROFILE_ROM = "rom"
MESH_PROFILES: Tuple[str, ...] = (MESH_PROFILE_REFERENCE, MESH_PROFILE_ROM)

LEVEL_L_PROD_LEGACY = "L_prod"
LEVEL_PROD_REFERENCE = "L_prod_reference"
LEVEL_ROM_PROD = "L_rom_prod"
PRODUCTION_MESH_LEVELS: Tuple[str, ...] = (LEVEL_PROD_REFERENCE, LEVEL_ROM_PROD)

DATASET_VERSION_LEGACY = "m4_geometry_corrected_v1"
DATASET_VERSION_REFERENCE = "m4_geometry_corrected_reference_v1"
DATASET_VERSION_ROM = "m4_geometry_corrected_rommesh_v1"

STRICT_DATASET_VERSIONS = frozenset(
    {
        DATASET_VERSION_LEGACY,
        DATASET_VERSION_REFERENCE,
        DATASET_VERSION_ROM,
    }
)

REFERENCE_CONTROLS_M: Dict[str, float] = {
    "wood_thickness_size_m": 0.0010,
    "wood_surface_size_m": 0.0070,
    "air_threshold_size_min_m": 0.0040,
    "air_threshold_size_max_m": 0.0500,
    "air_threshold_dist_min_m": 0.015,
    "air_threshold_dist_max_m": 0.25,
}

ROM_CONTROLS_M: Dict[str, float] = {
    "wood_thickness_size_m": 0.00125,
    "wood_surface_size_m": 0.0085,
    "air_threshold_size_min_m": 0.0055,
    "air_threshold_size_max_m": 0.0600,
    "air_threshold_dist_min_m": 0.015,
    "air_threshold_dist_max_m": 0.25,
}

PROFILE_TO_LEVEL: Dict[str, str] = {
    MESH_PROFILE_REFERENCE: LEVEL_PROD_REFERENCE,
    MESH_PROFILE_ROM: LEVEL_ROM_PROD,
}

PROFILE_TO_DATASET: Dict[str, str] = {
    MESH_PROFILE_REFERENCE: DATASET_VERSION_REFERENCE,
    MESH_PROFILE_ROM: DATASET_VERSION_ROM,
}

LEVEL_ALIASES: Dict[str, str] = {
    LEVEL_L_PROD_LEGACY: LEVEL_PROD_REFERENCE,
}

# Profile mesh levels (L_prod_reference / L_rom_prod) select build controls and run-tree paths.
# Stage A / v2_b3_checkpoint_export.py uses a separate fixed export contract level:
CHECKPOINT_EXPORT_MESH_LEVEL = LEVEL_L_PROD_LEGACY


def checkpoint_export_mesh_level() -> str:
    """
    Internal Stage A / checkpoint exporter --mesh-level contract.

    Profile selection does not change this value. The selected operator mesh is passed
    explicitly via --operator-mesh-file; profile identity lives in mesh_profile /
    mesh_level_id provenance stamped after export.
    """
    return CHECKPOINT_EXPORT_MESH_LEVEL


class MeshProfileError(ValueError):
    """Invalid mesh profile / dataset pairing or reuse mismatch."""


@dataclass(frozen=True)
class MeshProfileResolved:
    mesh_profile: str
    mesh_level_id: str
    dataset_version: str
    effective_controls_m: Dict[str, float]
    allow_baseline_mesh_reuse: bool

    def provenance_fields(self) -> Dict[str, Any]:
        return {
            "mesh_profile": self.mesh_profile,
            "mesh_level_id": self.mesh_level_id,
            "dataset_version": self.dataset_version,
            "effective_controls_m": dict(self.effective_controls_m),
        }


def canonical_mesh_level_id(level_id: str) -> str:
    lid = str(level_id or "").strip()
    return LEVEL_ALIASES.get(lid, lid)


def is_production_mesh_level(level_id: str) -> bool:
    return canonical_mesh_level_id(level_id) in PRODUCTION_MESH_LEVELS


def normalize_mesh_profile(profile: Optional[str]) -> str:
    p = str(profile or MESH_PROFILE_ROM).strip().lower()
    if p not in MESH_PROFILES:
        raise MeshProfileError(f"unsupported mesh_profile={profile!r}; expected reference|rom")
    return p


def canonical_dataset_version_for_profile(profile: str) -> str:
    """Canonical dataset for new production runs (no legacy ambiguity)."""
    return PROFILE_TO_DATASET[normalize_mesh_profile(profile)]


def default_dataset_version_for_profile(profile: str) -> str:
    return canonical_dataset_version_for_profile(profile)


def resolve_mesh_profile(
    *,
    mesh_profile: Optional[str] = None,
    dataset_version: Optional[str] = None,
    allow_legacy_dataset: bool = False,
) -> MeshProfileResolved:
    """Resolve profile → level, dataset, controls. Default = ROM production profile."""
    profile = normalize_mesh_profile(mesh_profile)
    level_id = PROFILE_TO_LEVEL[profile]

    if dataset_version is None or not str(dataset_version).strip():
        ds = canonical_dataset_version_for_profile(profile)
    else:
        ds = str(dataset_version).strip()

    validate_profile_dataset_pairing(profile, ds, allow_legacy_dataset=allow_legacy_dataset)

    manifest = load_manifest()
    level_def = (manifest.get("mesh_levels") or {}).get(level_id)
    if not level_def:
        raise MeshProfileError(f"missing mesh_levels.{level_id} in v2_mesh_convergence_manifest.json")

    controls = effective_controls_from_level_def(level_def)
    if not controls:
        controls = dict(REFERENCE_CONTROLS_M if profile == MESH_PROFILE_REFERENCE else ROM_CONTROLS_M)

    return MeshProfileResolved(
        mesh_profile=profile,
        mesh_level_id=level_id,
        dataset_version=ds,
        effective_controls_m=controls,
        allow_baseline_mesh_reuse=(profile == MESH_PROFILE_REFERENCE),
    )


def validate_profile_dataset_pairing(
    mesh_profile: str,
    dataset_version: str,
    *,
    allow_legacy_dataset: bool = False,
) -> None:
    profile = normalize_mesh_profile(mesh_profile)
    ds = str(dataset_version).strip()
    expected = canonical_dataset_version_for_profile(profile)
    if profile == MESH_PROFILE_REFERENCE:
        if ds == DATASET_VERSION_LEGACY and not allow_legacy_dataset:
            raise MeshProfileError(
                f"mesh_profile=reference cannot use legacy dataset {DATASET_VERSION_LEGACY!r} "
                f"for new runs; required {DATASET_VERSION_REFERENCE!r}"
            )
        if ds not in (DATASET_VERSION_REFERENCE, DATASET_VERSION_LEGACY):
            raise MeshProfileError(
                f"mesh_profile=reference requires dataset_version={DATASET_VERSION_REFERENCE!r}; got {ds!r}"
            )
        if ds == DATASET_VERSION_LEGACY and allow_legacy_dataset:
            return
        if ds != expected:
            raise MeshProfileError(
                f"mesh_profile=reference requires dataset_version={expected!r}; got {ds!r}"
            )
        return
    if profile == MESH_PROFILE_ROM:
        if ds != DATASET_VERSION_ROM:
            raise MeshProfileError(
                f"mesh_profile=rom requires dataset_version={DATASET_VERSION_ROM}; got {ds!r}"
            )
        return
    raise MeshProfileError(f"unsupported mesh_profile={mesh_profile!r}")


def assert_l_prod_alias_controls_match() -> None:
    """Startup guard: L_prod alias must not drift from L_prod_reference numerically."""
    manifest = load_manifest()
    levels = manifest.get("mesh_levels") or {}
    ref_def = levels.get(LEVEL_PROD_REFERENCE)
    alias_def = levels.get(LEVEL_L_PROD_LEGACY)
    if not ref_def or not alias_def:
        raise MeshProfileError("missing L_prod_reference or L_prod in mesh manifest")
    ref_ctrl = effective_controls_from_level_def(ref_def)
    alias_ctrl = effective_controls_from_level_def(alias_def)
    if not _controls_equal(ref_ctrl, alias_ctrl):
        raise MeshProfileError(
            f"L_prod alias controls drift from L_prod_reference: "
            f"alias={alias_ctrl} reference={ref_ctrl}"
        )


def resolve_mesh_profile_from_mapping(
    data: Optional[Mapping[str, Any]],
    *,
    fallback_dataset_version: Optional[str] = None,
    allow_legacy_dataset: bool = False,
) -> MeshProfileResolved:
    if not data:
        return resolve_mesh_profile(
            mesh_profile=MESH_PROFILE_REFERENCE,
            dataset_version=fallback_dataset_version,
            allow_legacy_dataset=allow_legacy_dataset,
        )
    profile = data.get("mesh_profile")
    ds = data.get("dataset_version")
    if ds is None and fallback_dataset_version and not data.get("mesh_profile"):
        # Do not silently inherit pool legacy dataset for profile-aware runs.
        ds = None
    if profile:
        return resolve_mesh_profile(
            mesh_profile=profile,
            dataset_version=ds,
            allow_legacy_dataset=allow_legacy_dataset,
        )
    level = data.get("mesh_level_id") or data.get("mesh_level")
    if level:
        canon = canonical_mesh_level_id(str(level))
        for prof, lid in PROFILE_TO_LEVEL.items():
            if lid == canon:
                return resolve_mesh_profile(
                    mesh_profile=prof,
                    dataset_version=ds,
                    allow_legacy_dataset=allow_legacy_dataset,
                )
    return resolve_mesh_profile(
        mesh_profile=MESH_PROFILE_REFERENCE,
        dataset_version=ds,
        allow_legacy_dataset=allow_legacy_dataset,
    )


def apply_mesh_profile_to_sample_input(
    sample_input: Dict[str, Any],
    resolved: MeshProfileResolved,
) -> Dict[str, Any]:
    sample_input["mesh_profile"] = resolved.mesh_profile
    sample_input["mesh_level_id"] = resolved.mesh_level_id
    sample_input["dataset_version"] = resolved.dataset_version
    sample_input["effective_controls_m"] = dict(resolved.effective_controls_m)
    return sample_input


def production_mesh_levels_for_cleanup() -> Tuple[str, ...]:
    return ("L_scout_coarse", LEVEL_PROD_REFERENCE, LEVEL_ROM_PROD, LEVEL_L_PROD_LEGACY)


def run_tree_lprod_mesh_path(run_root: Path, sample_id: str, mesh_level_id: str) -> Path:
    level = canonical_mesh_level_id(mesh_level_id)
    return run_root / "lprod" / "mesh" / level / f"{sample_id}.msh"


def convergence_mesh_path(mesh_level_id: str, sample_id: str) -> Path:
    from v2_mesh_convergence_common import mesh_path  # noqa: WPS433

    return mesh_path(canonical_mesh_level_id(mesh_level_id), sample_id)


def mesh_profile_from_artifacts(
    *,
    sample_input: Optional[Mapping[str, Any]] = None,
    built_metadata: Optional[Mapping[str, Any]] = None,
    pipeline_manifest: Optional[Mapping[str, Any]] = None,
) -> Optional[MeshProfileResolved]:
    for src in (sample_input, built_metadata, pipeline_manifest):
        if not src:
            continue
        mp = src.get("mesh_profile")
        if mp:
            try:
                return resolve_mesh_profile(
                    mesh_profile=str(mp),
                    dataset_version=src.get("dataset_version"),
                )
            except MeshProfileError:
                pass
        ml = src.get("mesh_level_id") or src.get("mesh_level")
        if ml and is_production_mesh_level(str(ml)):
            canon = canonical_mesh_level_id(str(ml))
            for profile, level in PROFILE_TO_LEVEL.items():
                if level == canon:
                    return resolve_mesh_profile(
                        mesh_profile=profile,
                        dataset_version=src.get("dataset_version"),
                    )
    return None


def derive_reference_controls_from_durable(
    run_root: Path,
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """
    Read reference mesh controls from durable post-cleanup artifacts only.

    Returns (controls, source_labels, errors). Does not read deleted checkpoints/meshes.
    """
    errors: List[str] = []
    sources: List[str] = []
    candidates: List[Tuple[str, Path, str]] = [
        ("freeze/freeze_manifest.json", run_root / "freeze" / "freeze_manifest.json", "effective_controls_m"),
        (
            "freeze/physics_identity_manifest.json",
            run_root / "freeze" / "physics_identity_manifest.json",
            "effective_controls_m",
        ),
        ("pipeline_run_manifest.json", run_root / "pipeline_run_manifest.json", "effective_controls_m"),
        ("sample/sample_input.json", run_root / "sample" / "sample_input.json", "effective_controls_m"),
    ]
    for label, path, key in candidates:
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append(f"{label}:unreadable")
            continue
        raw = doc.get(key)
        if not isinstance(raw, dict) or not raw:
            continue
        try:
            controls = {str(k): float(v) for k, v in raw.items()}
        except (TypeError, ValueError):
            errors.append(f"{label}:{key}:non_numeric")
            continue
        sources.append(f"{label}:{key}")
        return controls, sources, errors

    errors.append(
        "missing:effective_controls_m "
        "(freeze/freeze_manifest.json, freeze/physics_identity_manifest.json, "
        "pipeline_run_manifest.json, sample/sample_input.json)"
    )
    return {}, sources, errors


def _controls_equal(a: Mapping[str, Any], b: Mapping[str, Any], *, rtol: float = 1.0e-9) -> bool:
    keys = set(a) | set(b)
    for k in keys:
        try:
            av = float(a.get(k, float("nan")))
            bv = float(b.get(k, float("nan")))
        except (TypeError, ValueError):
            return False
        if abs(av - bv) > rtol * max(1.0, abs(av), abs(bv)):
            return False
    return True


def validate_mesh_profile_reuse(
    *,
    expected: MeshProfileResolved,
    existing: Mapping[str, Any],
    context: str = "reuse",
) -> List[str]:
    """Hard-fail list when resume/reuse profile identity does not match."""
    errors: List[str] = []
    ex_profile = str(existing.get("mesh_profile") or "").strip()
    if ex_profile:
        ex_level = canonical_mesh_level_id(str(existing.get("mesh_level_id") or ""))
    else:
        ex_level = canonical_mesh_level_id(
            str(existing.get("mesh_level_id") or existing.get("mesh_level") or "")
        )
    ex_ds = str(existing.get("dataset_version") or "").strip()

    if ex_profile and ex_profile != expected.mesh_profile:
        errors.append(f"{context}:mesh_profile_mismatch:{ex_profile}!={expected.mesh_profile}")
    if ex_profile:
        if not ex_level:
            errors.append(f"{context}:missing_mesh_level_id")
        elif ex_level != expected.mesh_level_id:
            errors.append(f"{context}:mesh_level_id_mismatch:{ex_level}!={expected.mesh_level_id}")
    elif ex_level and ex_level != expected.mesh_level_id:
        errors.append(f"{context}:mesh_level_id_mismatch:{ex_level}!={expected.mesh_level_id}")
    if ex_ds and ex_ds != expected.dataset_version:
        errors.append(f"{context}:dataset_version_mismatch:{ex_ds}!={expected.dataset_version}")

    ex_controls = existing.get("effective_controls_m") or existing.get("mesh_controls_m")
    if isinstance(ex_controls, dict) and ex_controls:
        if not _controls_equal(ex_controls, expected.effective_controls_m):
            errors.append(f"{context}:effective_controls_m_mismatch")

    for hash_key in ("generated_mesh_sha256", "operator_mesh_sha256"):
        ex_hash = str(existing.get(hash_key) or "").strip()
        exp_hash = str(existing.get(f"expected_{hash_key}") or "").strip()
        if exp_hash and ex_hash and ex_hash != exp_hash:
            errors.append(f"{context}:{hash_key}_mismatch")

    return errors


def legacy_run_without_profile(sample_input: Mapping[str, Any], built: Mapping[str, Any]) -> bool:
    """True for pre-profile runs (e.g. in-flight L_prod) — do not auto-upgrade."""
    if sample_input.get("mesh_profile"):
        return False
    if built.get("mesh_profile"):
        return False
    level = str(built.get("mesh_level") or built.get("mesh_level_id") or "")
    return level in ("", LEVEL_L_PROD_LEGACY)


def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_target_plan_file(path: Path) -> Tuple[Dict[str, Any], str]:
    """Load explicit validation target plan; return body and sha256."""
    if not path.is_file():
        raise MeshProfileError(f"target plan file missing: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MeshProfileError(f"invalid target plan JSON: {path}: {exc}") from exc
    if not isinstance(body, dict):
        raise MeshProfileError(f"target plan must be JSON object: {path}")
    if not body.get("targets_hz"):
        raise MeshProfileError(f"target plan missing targets_hz: {path}")
    return body, digest


VALIDATION_INPUT_PACKAGE_SCHEMA = "m4_mesh_validation_input_package_v1"
EXTERNAL_VALIDATION_INPUT_PACKAGE_SCHEMA_V1 = "m4_external_validation_input_package_v1"
VALIDATION_INPUT_PACKAGE_REL = "validation/mesh_profile_inputs"
VALIDATION_INPUT_MANIFEST_NAME = "validation_input_manifest.json"
DURABLE_VALIDATION_INPUT_REL: Tuple[str, ...] = (
    f"{VALIDATION_INPUT_PACKAGE_REL}/target_plan.json",
    f"{VALIDATION_INPUT_PACKAGE_REL}/validation_input_manifest.json",
    f"{VALIDATION_INPUT_PACKAGE_REL}/scout_intrinsic_summary.json",
    f"{VALIDATION_INPUT_PACKAGE_REL}/worker_chunk_plan.json",
)


def materialize_validation_input_package(
    *,
    run_root: Path,
    source_path: Path,
    input_name: str,
    sample_id: str,
    run_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    Copy shared validation input into a dedicated read-only package inside run_root.
    Returns (package_manifest, input_body, sha256).
    """
    from v2_b3_petsc_util import write_json_atomic  # noqa: WPS433

    run_root = run_root.resolve()
    source_path = source_path.resolve()
    body, digest = load_target_plan_file(source_path)

    pkg_dir = run_root / VALIDATION_INPUT_PACKAGE_REL
    pkg_dir.mkdir(parents=True, exist_ok=True)
    package_rel = f"{VALIDATION_INPUT_PACKAGE_REL}/{input_name}.json"
    package_path = run_root / package_rel
    write_json_atomic(package_path, body)

    manifest_path = pkg_dir / VALIDATION_INPUT_MANIFEST_NAME
    prior: Dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            prior = {}

    targets = body.get("targets_hz") or []
    freq_range = body.get("frequency_range_hz") or [
        min(targets) if targets else None,
        max(targets) if targets else None,
    ]
    geom_fp = None
    try:
        from v2_b3_m4_lprod_interfaces import extract_geometry_dict, geometry_fingerprint  # noqa: WPS433

        si_path = run_root / "sample" / "sample_input.json"
        if si_path.is_file():
            si = json.loads(si_path.read_text(encoding="utf-8"))
            geom = extract_geometry_dict(si)
            if geom:
                geom_fp = geometry_fingerprint(geom)
    except (OSError, ValueError, json.JSONDecodeError, ImportError):
        geom_fp = None

    entry = {
        "name": input_name,
        "source_path": str(source_path),
        "package_path": package_rel,
        "sha256": digest,
        "read_only": True,
        "sample_id": sample_id,
        "run_id": run_id,
        "geometry_fingerprint": geom_fp,
        "target_count": len(targets),
        "frequency_range_hz": freq_range,
        "targets_hz": list(targets),
        "chunk_plan_identity": body.get("chunk_plan_id") or body.get("worker_chunk_plan_sha256"),
        "creation_reason": "mesh_profile_validation_input",
        "materialized_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    inputs: List[Dict[str, Any]] = list(prior.get("inputs") or [])
    inputs = [row for row in inputs if str(row.get("name")) != input_name]
    inputs.append(entry)
    manifest = {
        "schema": VALIDATION_INPUT_PACKAGE_SCHEMA,
        "sample_id": sample_id,
        "run_id": run_id,
        "inputs": inputs,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest, body, digest


def install_explicit_target_plan(
    *,
    run_root: Path,
    target_plan_path: Path,
    sample_id: str,
    run_id: str,
) -> Dict[str, Any]:
    """Validation-only: materialize frozen target plan into read-only validation package."""
    _manifest, body, digest = materialize_validation_input_package(
        run_root=run_root,
        source_path=target_plan_path,
        input_name="target_plan",
        sample_id=sample_id,
        run_id=run_id,
    )
    plan_sample = str(body.get("sample_id") or "")
    plan_run = str(body.get("run_id") or "")
    if plan_sample and plan_sample != sample_id:
        raise MeshProfileError(
            f"target plan sample_id={plan_sample!r} does not match run sample_id={sample_id!r}"
        )
    if plan_run and plan_run not in ("", run_id):
        # Source plan may name the reference run_id; candidate run_id is allowed to differ.
        pass

    package_rel = f"{VALIDATION_INPUT_PACKAGE_REL}/target_plan.json"
    out = dict(body)
    out["explicit_target_plan"] = True
    out["validation_input_package"] = package_rel
    out["validation_input_name"] = "target_plan"
    out["validation_input_sha256"] = digest
    out["validation_only"] = True
    out.setdefault("sample_id", sample_id)
    out.setdefault("run_id", run_id)
    return out


def validation_input_manifest_path(run_root: Path) -> Path:
    return run_root / VALIDATION_INPUT_PACKAGE_REL / VALIDATION_INPUT_MANIFEST_NAME


def durable_target_plan_path(run_root: Path) -> Path:
    return run_root / VALIDATION_INPUT_PACKAGE_REL / "target_plan.json"


def preserve_target_plan_before_cleanup(
    *,
    run_root: Path,
    sample_id: str,
    run_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Copy transient lprod target plan into durable validation package before cleanup.
    Idempotent if package already exists with matching content.
    """
    durable = durable_target_plan_path(run_root)
    if durable.is_file():
        body, digest = load_target_plan_file(durable)
        return {"status": "already_preserved", "sha256": digest, "target_count": len(body.get("targets_hz") or [])}

    transient = run_root / "lprod" / "lprod_target_plan.json"
    if not transient.is_file():
        return None

    manifest, body, digest = materialize_validation_input_package(
        run_root=run_root,
        source_path=transient,
        input_name="target_plan",
        sample_id=sample_id,
        run_id=run_id,
    )
    return {
        "status": "preserved",
        "sha256": digest,
        "target_count": len(body.get("targets_hz") or []),
        "manifest": manifest,
    }


@dataclass(frozen=True)
class ExternalValidationInputPackage:
    """Read-only external immutable validation-input package (not copied into run trees)."""

    package_root: Path
    target_plan: Dict[str, Any]
    target_plan_sha256: str
    manifest: Dict[str, Any]
    manifest_entry: Dict[str, Any]


def _resolve_external_validation_manifest_entry(
    manifest: Mapping[str, Any],
    *,
    plan_body: Mapping[str, Any],
    plan_sha: str,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Resolve manifest entry for nested in-run schema or flat external v1 schema.

    Does not mutate the manifest.
    """
    errors: List[str] = []
    nested = [r for r in (manifest.get("inputs") or []) if str(r.get("name")) == "target_plan"]
    if nested:
        entry = dict(nested[0])
        manifest_sha = str(entry.get("sha256") or "")
        if manifest_sha and manifest_sha != plan_sha:
            errors.append("external_validation_input_sha256_mismatch")
        return (entry if not errors else None), errors

    schema = str(manifest.get("schema") or "")
    if schema != EXTERNAL_VALIDATION_INPUT_PACKAGE_SCHEMA_V1:
        return None, ["external_validation_input_manifest_missing_target_plan_entry"]

    manifest_sha = str(manifest.get("target_plan_sha256") or "")
    if not manifest_sha:
        return None, ["external_validation_input_missing_target_plan_sha256"]
    if manifest_sha != plan_sha:
        errors.append("external_validation_input_sha256_mismatch")

    plan_targets = [float(x) for x in (plan_body.get("targets_hz") or [])]
    manifest_targets = [float(x) for x in (manifest.get("targets_hz") or [])]
    if manifest_targets and plan_targets and manifest_targets != plan_targets:
        errors.append("external_validation_input_targets_hz_mismatch")
    if manifest.get("target_count") is not None and plan_targets:
        if int(manifest.get("target_count")) != len(plan_targets):
            errors.append("external_validation_input_target_count_mismatch")
    if manifest_targets and not plan_targets:
        errors.append("external_validation_input_missing_targets_hz_in_plan_file")

    if errors:
        return None, errors

    entry = {
        "name": "target_plan",
        "sha256": manifest_sha,
        "sample_id": manifest.get("sample_id"),
        "targets_hz": manifest_targets or plan_targets,
        "target_count": manifest.get("target_count") or len(plan_targets),
        "frequency_range_hz": manifest.get("frequency_range_hz") or plan_body.get("frequency_range_hz"),
        "geometry_fingerprint": manifest.get("geometry_fingerprint"),
        "material_fingerprint": manifest.get("material_fingerprint"),
        "physics_identity_hash": manifest.get("physics_identity_hash"),
        "chunk_count": manifest.get("chunk_count"),
        "schema": schema,
    }
    return entry, []


def load_external_validation_package(
    package_root: Path,
) -> Tuple[Optional[ExternalValidationInputPackage], List[str]]:
    """
    Load authoritative validation package from an external directory.

    Expected layout:
      <package_root>/target_plan.json
      <package_root>/validation_input_manifest.json

    Supports nested `inputs[]` manifests and flat `m4_external_validation_input_package_v1`.
    """
    errors: List[str] = []
    root = package_root.expanduser().resolve()
    plan_path = root / "target_plan.json"
    man_path = root / VALIDATION_INPUT_MANIFEST_NAME
    if not plan_path.is_file():
        errors.append("missing_external_target_plan")
    if not man_path.is_file():
        errors.append("missing_external_validation_input_manifest")
    if errors:
        return None, errors

    try:
        plan_body, plan_sha = load_target_plan_file(plan_path)
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError, MeshProfileError) as exc:
        return None, [f"external_package_unreadable:{exc}"]

    if not isinstance(manifest, dict):
        return None, ["external_validation_input_manifest_not_object"]

    entry, entry_errors = _resolve_external_validation_manifest_entry(
        manifest,
        plan_body=plan_body,
        plan_sha=plan_sha,
    )
    if entry_errors:
        return None, entry_errors
    if entry is None:
        return None, ["external_validation_input_manifest_missing_target_plan_entry"]

    return (
        ExternalValidationInputPackage(
            package_root=root,
            target_plan=plan_body,
            target_plan_sha256=plan_sha,
            manifest=manifest,
            manifest_entry=entry,
        ),
        [],
    )


def load_durable_target_plan(run_root: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str], List[str]]:
    """
    Post-cleanup target plan loader. Never reads lprod/lprod_target_plan.json.
    Returns (body, sha256, errors).
    """
    errors: List[str] = []
    path = durable_target_plan_path(run_root)
    if not path.is_file():
        return None, None, ["TARGET_PLAN_UNAVAILABLE"]
    try:
        body, digest = load_target_plan_file(path)
    except MeshProfileError as exc:
        return None, None, [str(exc)]
    man_path = validation_input_manifest_path(run_root)
    if man_path.is_file():
        try:
            man = json.loads(man_path.read_text(encoding="utf-8"))
            entries = [r for r in (man.get("inputs") or []) if r.get("name") == "target_plan"]
            if entries and str(entries[0].get("sha256")) != digest:
                errors.append("validation_input_sha256_mismatch")
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append("validation_input_manifest_unreadable")
    return body, digest, errors


def evaluate_legacy_reference_compatibility(
    *,
    run_root: Path,
    repo_root: Optional[Path] = None,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Read-only audit: accept completed pre-profile reference run for comparison only.
    Does not mutate manifests.
    """
    from v2_b3_m4_sample_cleanup_barrier import (  # noqa: WPS433
        require_cleanup_barrier_passed_for_validation,
    )

    errors: List[str] = []
    meta: Dict[str, Any] = {
        "legacy_reference_compatibility": False,
        "resolved_reference_profile": None,
    }
    run_root = run_root.resolve()
    sample_input_path = run_root / "sample" / "sample_input.json"
    sample_in: Dict[str, Any] = {}
    if sample_input_path.is_file():
        try:
            sample_in = json.loads(sample_input_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append("sample_input_unreadable")

    if sample_in.get("mesh_profile"):
        errors.append("mesh_profile_already_present_not_legacy")

    identity_path = run_root / "freeze" / "physics_identity_manifest.json"
    if not identity_path.is_file():
        errors.append("missing_physics_identity_manifest")
        return False, meta, errors
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        errors.append("physics_identity_unreadable")
        return False, meta, errors

    if not bool(identity.get("production_acceptance_pass")):
        errors.append("production_acceptance_pass!=true")

    freeze_path = run_root / "freeze" / "freeze_manifest.json"
    if freeze_path.is_file():
        try:
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            if not bool(freeze.get("production_acceptance_pass")):
                errors.append("freeze_production_acceptance_pass!=true")
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append("freeze_manifest_unreadable")
    else:
        errors.append("missing_freeze_manifest")

    if not identity.get("generated_mesh_sha256"):
        errors.append("missing_generated_mesh_sha256")
    if not identity.get("operator_mesh_sha256"):
        errors.append("missing_operator_mesh_sha256")
    if not bool(identity.get("operator_mesh_matches_generated")):
        errors.append("operator_mesh_matches_generated!=true")

    controls, control_sources, control_errors = derive_reference_controls_from_durable(run_root)
    meta["reference_controls_sources"] = control_sources
    errors.extend(control_errors)
    if controls and not _controls_equal(controls, REFERENCE_CONTROLS_M):
        errors.append("effective_controls_m_not_reference_full_mesh")
    meta["reference_controls_m"] = controls or None

    from v2_b3_m4_lprod_interfaces import extract_geometry_dict  # noqa: WPS433

    geom = extract_geometry_dict(sample_in) if sample_in else {}
    if not geom:
        errors.append("geometry_payload_missing")
    elif not identity.get("geometry_fingerprint"):
        errors.append("geometry_fingerprint_missing_in_identity")

    if repo_root is not None:
        ok, barrier_meta, barrier_errors = require_cleanup_barrier_passed_for_validation(
            repo_root=repo_root,
            run_root=run_root,
            label="legacy_reference",
        )
        meta["cleanup_barrier"] = barrier_meta
        errors.extend(barrier_errors)

    if not errors:
        meta["legacy_reference_compatibility"] = True
        meta["resolved_reference_profile"] = MESH_PROFILE_REFERENCE
        meta["resolved_mesh_level_id"] = LEVEL_PROD_REFERENCE
        meta["resolved_dataset_version"] = identity.get("dataset_version") or DATASET_VERSION_LEGACY
    return len(errors) == 0, meta, errors
