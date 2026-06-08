#!/usr/bin/env python3
"""Shared helpers for rich modal export v1 (active basis + W prolongation)."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from v2_b3_petsc_util import write_json_atomic


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v

RICH_MODAL_SCHEMA = "b3_rich_modal_manifest_v1"
SYNTHESIS_METADATA_SCHEMA = "b3_synthesis_metadata_v1"
RICH_MODAL_POST_SCHEMA = "b3_rich_modal_post_v1"
REGION_DOF_LAYOUT = "B3_W_global_row_indices_via_u_idx_p_idx"
UNAVAILABLE_REGION_INDICES_STATUS = "unavailable_region_indices"
REGION_PARTICIPATION_STATUS_AVAILABLE = "available"
STRUCTURAL_REGION_KEYS = ("u_idx_top", "u_idx_back", "u_idx_ribs", "u_idx_soundhole")

SYNTHESIS_METADATA_JSON = "synthesis_metadata.json"
REGION_DOF_INDICES_NPZ = "region_dof_indices.npz"
REGION_DOF_METADATA_JSON = "region_dof_metadata.json"
REGION_DOF_METADATA_SCHEMA = "b3_region_dof_metadata_v1"

REGION_INDEX_KEYS = frozenset(
    {
        "u_idx_top",
        "u_idx_back",
        "u_idx_ribs",
        "u_idx_soundhole",
        "p_idx_air",
        "p_idx_all",
        "p_idx_aperture",
        "u_idx_all",
    }
)

REGION_NPZ_STRING_META_KEYS = frozenset(
    {
        "layout",
        "region_dof_source",
        "region_dof_mesh_file",
        "aperture_selection_method",
        "mic_output_method",
    }
)

REGION_NPZ_SCALAR_META_KEYS = frozenset(
    {
        "back_includes_ribs",
        "p_idx_aperture_count",
        "aperture_facet_count",
        "adjacent_air_cell_count",
    }
)

REGION_NPZ_FLOAT_META_KEYS = frozenset(
    {
        "aperture_coordinate_bounds_min",
        "aperture_coordinate_bounds_max",
    }
)

REGION_NPZ_ALL_META_KEYS = (
    REGION_NPZ_STRING_META_KEYS | REGION_NPZ_SCALAR_META_KEYS | REGION_NPZ_FLOAT_META_KEYS
)


class RegionDofSchemaError(ValueError):
    """Raised when region_dof_indices.npz violates the index/metadata contract."""


# Backward-compatible alias (superset of legacy string scalars in mixed NPZ files).
_NPZ_SCALAR_META_KEYS = REGION_NPZ_STRING_META_KEYS | frozenset({"back_includes_ribs"})
RICH_MODAL_DIRNAME = "rich_modal"
RICH_MODAL_MANIFEST_JSON = "rich_modal_manifest.json"
MODES_ACTIVE_NPZ = "modes_active.npz"
MODES_CATALOG_JSONL = "modes_catalog.jsonl"

TAG_PROTOCOL_V1 = {
    "Top_Plate": 1,
    "Soundhole": 2,
    "Back_Plate": 3,
    "Ribs_Sides": 4,
    "wood_fix": 5,
    "Air_Internal": 10,
}


def normalization_convention_v1() -> Dict[str, Any]:
    return {
        "problem_type": "GNHEP",
        "eps_type": "KRYLOVSCHUR",
        "st_type": "SINVERT",
        "eigenvector_basis": "active_reduced_A_active",
        "eigenpair_storage": "real_part_only",
        "eigenvector_dtype": "float64",
        "st_shift_applied_to": "shift_invert_center_lambda_sq",
        "physical_frequency_hz": "sqrt(lambda_real)/(2*pi) when lambda_imag≈0",
        "damping_in_eigensolve": False,
        "damping_applied_in": "STK_or_audio_post",
        "note": (
            "Vectors are SLEPc Ritz coefficients in active DOF layout; not mass-normalized. "
            "Do not label outputs as microphone pressure; use audio_output_proxy wording only."
        ),
    }


def prolongate_active_to_W(x_active: np.ndarray, built: Dict[str, Any]) -> np.ndarray:
    """Map active-vector coefficients to full B3 W row layout (length n_w)."""
    active_local = np.asarray(built["active_local"], dtype=np.int32).ravel()
    free_rows = np.asarray(built["free_rows"], dtype=np.int32).ravel()
    n_w = int(built["n_w"])
    n_free = int(free_rows.size)
    x_active = np.asarray(x_active, dtype=np.float64).ravel()
    if int(x_active.size) != int(active_local.size):
        raise ValueError(f"active vector length {x_active.size} != active_local {active_local.size}")
    x_free = np.zeros(n_free, dtype=np.float64)
    x_free[active_local] = x_active
    x_full = np.zeros(n_w, dtype=np.float64)
    x_full[free_rows] = x_free
    return x_full


def participation_energy_fraction(
    x_full: np.ndarray,
    indices: np.ndarray,
    *,
    unavailable_if_empty: bool = False,
) -> Optional[float]:
    """‖x[indices]‖² / ‖x‖² with safe denominator. Returns None when indices unavailable."""
    idx = np.asarray(indices, dtype=np.int32).ravel()
    if idx.size == 0:
        return None if unavailable_if_empty else 0.0
    x = np.asarray(x_full, dtype=np.float64).ravel()
    total = float(np.dot(x, x))
    if total <= 0.0:
        return None if unavailable_if_empty else 0.0
    part = x[idx]
    return float(np.dot(part, part) / total)


def structural_region_indices_available(region: Dict[str, np.ndarray], *, npz_present: bool) -> bool:
    if not npz_present:
        return False
    top_n = np.asarray(region.get("u_idx_top", []), dtype=np.int32).size
    back_n = np.asarray(region.get("u_idx_back", []), dtype=np.int32).size
    ribs_n = np.asarray(region.get("u_idx_ribs", []), dtype=np.int32).size
    return top_n > 0 and (back_n > 0 or ribs_n > 0)


def pressure_region_indices_available(region: Dict[str, np.ndarray]) -> bool:
    return np.asarray(region.get("p_idx_air", []), dtype=np.int32).size > 0


_NPZ_SCALAR_META_KEYS = frozenset(
    {"layout", "region_dof_source", "region_dof_mesh_file", "back_includes_ribs"}
)


def _npz_scalar_str(z: Any, key: str, *, default: Optional[str] = None) -> Optional[str]:
    if key not in z.files:
        return default
    try:
        arr = np.asarray(z[key]).ravel()
    except (KeyError, TypeError, ValueError):
        return default
    if arr.size == 0:
        return default
    item = arr[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return str(item)


def _npz_scalar_bool(z: Any, key: str, *, default: bool = False) -> bool:
    if key not in z.files:
        return default
    try:
        arr = np.asarray(z[key]).ravel()
    except (KeyError, TypeError, ValueError):
        return default
    if arr.size == 0:
        return default
    return bool(arr[0])


def _npz_scalar_int(z: Any, key: str, *, default: Optional[int] = None) -> Optional[int]:
    if key not in z.files:
        return default
    try:
        arr = np.asarray(z[key]).ravel()
    except (KeyError, TypeError, ValueError):
        return default
    if arr.size == 0:
        return default
    try:
        return int(arr[0])
    except (TypeError, ValueError):
        return default


def _npz_float_array(z: Any, key: str) -> Optional[List[float]]:
    if key not in z.files:
        return None
    arr = np.asarray(z[key], dtype=np.float64).ravel()
    if arr.size == 0:
        return None
    return [float(v) for v in arr.tolist()]


def coerce_region_index_array(key: str, raw: Any) -> np.ndarray:
    """Coerce one NPZ entry to int32 index vector; fail with key name on non-integer data."""
    arr = np.asarray(raw)
    if arr.dtype.kind in ("U", "S"):
        raise RegionDofSchemaError(
            f"region index key {key!r} stores string data {arr.ravel()[:1]!r}; expected integer indices"
        )
    if arr.dtype == object:
        raise RegionDofSchemaError(
            f"region index key {key!r} has object dtype; expected integer indices"
        )
    try:
        out = np.asarray(arr, dtype=np.int32).ravel().copy()
    except (TypeError, ValueError) as exc:
        raise RegionDofSchemaError(
            f"region index key {key!r}: cannot coerce to int32 (value sample={arr.ravel()[:3]!r})"
        ) from exc
    return out


def _classify_npz_key(key: str, arr: np.ndarray) -> str:
    if key in REGION_INDEX_KEYS:
        return "index"
    if key in REGION_NPZ_ALL_META_KEYS:
        return "metadata"
    if np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool:
        return "index"
    if np.issubdtype(arr.dtype, np.floating):
        return "metadata"
    if arr.dtype.kind in ("U", "S", "O"):
        return "metadata"
    return "unknown"


def _load_region_metadata_from_npz(z: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key in REGION_NPZ_STRING_META_KEYS:
        val = _npz_scalar_str(z, key)
        if val is not None:
            metadata[key] = val
    if "back_includes_ribs" in z.files:
        metadata["back_includes_ribs"] = _npz_scalar_bool(z, "back_includes_ribs", default=True)
    for key in ("p_idx_aperture_count", "aperture_facet_count", "adjacent_air_cell_count"):
        val = _npz_scalar_int(z, key)
        if val is not None:
            metadata[key] = val
    bounds_min = _npz_float_array(z, "aperture_coordinate_bounds_min")
    bounds_max = _npz_float_array(z, "aperture_coordinate_bounds_max")
    if bounds_min is not None and bounds_max is not None:
        metadata["aperture_coordinate_bounds"] = {"min": bounds_min, "max": bounds_max}
    return metadata


def _load_region_indices_from_npz(z: Any) -> Dict[str, np.ndarray]:
    region: Dict[str, np.ndarray] = {}
    for key in z.files:
        arr = np.asarray(z[key])
        kind = _classify_npz_key(key, arr)
        if kind == "metadata":
            continue
        if kind == "index" or key in REGION_INDEX_KEYS:
            region[key] = coerce_region_index_array(key, arr)
            continue
        raise RegionDofSchemaError(f"unrecognized region_dof_indices.npz key {key!r} dtype={arr.dtype!r}")
    return region


def load_region_dof_metadata_file(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise RegionDofSchemaError(f"{path}: expected JSON object for region DOF metadata")
    return data


def write_region_dof_metadata_json(checkpoint: Path, metadata: Mapping[str, Any]) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    body = dict(metadata)
    body.setdefault("schema", REGION_DOF_METADATA_SCHEMA)
    body.setdefault("generated_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    out_path = checkpoint / REGION_DOF_METADATA_JSON
    write_json_atomic(out_path, body)
    return out_path


def build_region_dof_metadata_from_operator_build(region_dof_build: Mapping[str, Any]) -> Dict[str, Any]:
    p_idx_aperture = np.asarray(region_dof_build.get("p_idx_aperture") or [], dtype=np.int32).ravel()
    mesh_file = str(region_dof_build.get("region_dof_mesh_file") or "")
    source = str(region_dof_build.get("region_dof_source") or "operator_build_context")
    metadata: Dict[str, Any] = {
        "schema": REGION_DOF_METADATA_SCHEMA,
        "layout": str(region_dof_build.get("layout") or REGION_DOF_LAYOUT),
        "region_dof_source": source,
        "region_dof_mesh_file": mesh_file,
        "back_includes_ribs": bool(region_dof_build.get("back_includes_ribs", True)),
        "aperture_selection_method": str(region_dof_build.get("aperture_selection_method") or ""),
        "mic_output_method": str(
            region_dof_build.get("mic_output_method") or "aperture_pressure_rms_proxy_v1"
        ),
        "p_idx_aperture_count": int(p_idx_aperture.size),
        "aperture_facet_count": int(region_dof_build.get("aperture_facet_count") or 0),
        "adjacent_air_cell_count": int(region_dof_build.get("adjacent_air_cell_count") or 0),
    }
    bounds = region_dof_build.get("aperture_coordinate_bounds")
    if isinstance(bounds, dict):
        metadata["aperture_coordinate_bounds"] = {
            "min": list(bounds.get("min") or [0.0, 0.0, 0.0]),
            "max": list(bounds.get("max") or [0.0, 0.0, 0.0]),
        }
    return metadata


def validate_region_dof_contract(
    region: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    built_meta: Mapping[str, Any],
    *,
    require_aperture: bool = False,
) -> List[str]:
    """Return validation failure messages (empty list = pass)."""
    failures: List[str] = []
    raw_ap = region.get("p_idx_aperture")
    p_ap = (
        np.asarray([], dtype=np.int32)
        if raw_ap is None
        else np.asarray(raw_ap, dtype=np.int32).ravel()
    )
    if require_aperture or "p_idx_aperture" in region:
        if p_ap.size <= 0:
            failures.append("p_idx_aperture_count<=0")
        elif not np.issubdtype(p_ap.dtype, np.integer):
            failures.append(f"p_idx_aperture_dtype_not_integer:{p_ap.dtype}")
        else:
            n_w = int(built_meta.get("n_w") or 0)
            if n_w > 0 and int(p_ap.max(initial=-1)) >= n_w:
                failures.append(f"p_idx_aperture_out_of_range:n_w={n_w},max={int(p_ap.max())}")
        method = str(metadata.get("aperture_selection_method") or "").strip()
        if p_ap.size > 0 and not method:
            failures.append("aperture_selection_method_missing")
        meta_count = metadata.get("p_idx_aperture_count")
        if p_ap.size > 0 and meta_count is not None and int(meta_count) != int(p_ap.size):
            failures.append(
                f"p_idx_aperture_count_mismatch:metadata={meta_count},actual={int(p_ap.size)}"
            )
    return failures


def split_region_dof_mixed_npz_inplace(checkpoint: Path) -> Dict[str, Any]:
    """One-time migration: move metadata out of mixed NPZ into region_dof_metadata.json."""
    checkpoint = checkpoint.expanduser().resolve()
    npz_path = checkpoint / REGION_DOF_INDICES_NPZ
    if not npz_path.is_file():
        raise FileNotFoundError(f"missing {npz_path}")
    with np.load(npz_path, allow_pickle=False) as z:
        region = _load_region_indices_from_npz(z)
        metadata = _load_region_metadata_from_npz(z)
        mixed_meta_in_npz = any(k in z.files for k in REGION_NPZ_ALL_META_KEYS)
    if not mixed_meta_in_npz and (checkpoint / REGION_DOF_METADATA_JSON).is_file():
        return {"status": "SKIP", "reason": "already_split"}
    np.savez_compressed(npz_path, **region)
    meta_path = write_region_dof_metadata_json(checkpoint, metadata)
    return {
        "status": "PASS",
        "region_dof_indices_npz": str(npz_path),
        "region_dof_metadata_json": str(meta_path),
        "index_keys": sorted(region.keys()),
        "metadata_keys": sorted(metadata.keys()),
    }


def load_region_dof_bundle(
    checkpoint: Path,
    built_meta: Dict[str, Any],
    *,
    validate_aperture: bool = True,
) -> Dict[str, Any]:
    """Load region index sets; distinguish facet structural (npz) vs pressure (built_meta ok)."""
    checkpoint = checkpoint.expanduser().resolve()
    npz_path = checkpoint / REGION_DOF_INDICES_NPZ
    meta_path = checkpoint / REGION_DOF_METADATA_JSON
    npz_present = npz_path.is_file()
    region: Dict[str, np.ndarray]
    metadata: Dict[str, Any] = {}
    source = "built_metadata_pressure_only"
    back_includes_ribs = True
    contract_failures: List[str] = []
    if npz_present:
        with np.load(npz_path, allow_pickle=False) as z:
            region = _load_region_indices_from_npz(z)
            metadata = _load_region_metadata_from_npz(z)
        if meta_path.is_file():
            file_meta = load_region_dof_metadata_file(meta_path)
            metadata.update({k: v for k, v in file_meta.items() if k != "schema"})
        source = str(metadata.get("region_dof_source") or "region_dof_indices_npz")
        back_includes_ribs = bool(metadata.get("back_includes_ribs", True))
        if validate_aperture and "p_idx_aperture" in region:
            contract_failures = validate_region_dof_contract(
                region,
                metadata,
                built_meta,
                require_aperture=True,
            )
            if contract_failures:
                raise RegionDofSchemaError("; ".join(contract_failures))
    else:
        u_idx = np.asarray(built_meta.get("u_idx") or [], dtype=np.int32).ravel()
        p_idx = np.asarray(built_meta.get("p_idx") or [], dtype=np.int32).ravel()
        region = {
            "u_idx_top": np.asarray([], dtype=np.int32),
            "u_idx_back": np.asarray([], dtype=np.int32),
            "u_idx_ribs": np.asarray([], dtype=np.int32),
            "u_idx_soundhole": np.asarray([], dtype=np.int32),
            "p_idx_air": p_idx.copy(),
            "p_idx_all": p_idx.copy(),
            "u_idx_all": u_idx.copy(),
        }

    structural_ok = structural_region_indices_available(region, npz_present=npz_present)
    pressure_ok = pressure_region_indices_available(region)
    return {
        "region": region,
        "metadata": metadata,
        "npz_present": npz_present,
        "region_dof_metadata_present": meta_path.is_file(),
        "region_dof_source": source,
        "back_includes_ribs": back_includes_ribs,
        "aperture_selection_method": metadata.get("aperture_selection_method"),
        "p_idx_aperture_count": int(np.asarray(region.get("p_idx_aperture", [])).size)
        if "p_idx_aperture" in region
        else metadata.get("p_idx_aperture_count"),
        "contract_validation_failures": contract_failures,
        "structural_region_participation_status": (
            REGION_PARTICIPATION_STATUS_AVAILABLE
            if structural_ok
            else UNAVAILABLE_REGION_INDICES_STATUS
        ),
        "structural_audio_proxy_status": (
            REGION_PARTICIPATION_STATUS_AVAILABLE
            if structural_ok
            else UNAVAILABLE_REGION_INDICES_STATUS
        ),
        "pressure_region_status": (
            REGION_PARTICIPATION_STATUS_AVAILABLE if pressure_ok else UNAVAILABLE_REGION_INDICES_STATUS
        ),
        "structural_indices_available": structural_ok,
        "pressure_indices_available": pressure_ok,
    }


def audio_output_proxies_v1(
    x_full: np.ndarray,
    region: Dict[str, np.ndarray],
    *,
    structural_available: bool,
    pressure_available: bool,
) -> Dict[str, Any]:
    u_top = region.get("u_idx_top", np.asarray([], dtype=np.int32))
    u_sh = region.get("u_idx_soundhole", np.asarray([], dtype=np.int32))
    p_air = region.get("p_idx_air", np.asarray([], dtype=np.int32))

    proxies: Dict[str, Any] = {
        "proxy_type": "audio_output_proxy_v1",
        "not_microphone_pressure": True,
        "structural_audio_proxy_status": (
            REGION_PARTICIPATION_STATUS_AVAILABLE
            if structural_available
            else UNAVAILABLE_REGION_INDICES_STATUS
        ),
        "pressure_audio_proxy_status": (
            REGION_PARTICIPATION_STATUS_AVAILABLE
            if pressure_available
            else UNAVAILABLE_REGION_INDICES_STATUS
        ),
        "top_plate_displacement_rms_proxy_v1": None,
        "soundhole_facet_displacement_rms_proxy_v1": None,
        "cavity_pressure_max_proxy_v1": None,
    }
    if structural_available:
        top_vals = x_full[np.asarray(u_top, dtype=np.int32)]
        sh_vals = x_full[np.asarray(u_sh, dtype=np.int32)]
        proxies["top_plate_displacement_rms_proxy_v1"] = (
            float(np.sqrt(np.mean(top_vals**2))) if top_vals.size else None
        )
        proxies["soundhole_facet_displacement_rms_proxy_v1"] = (
            float(np.sqrt(np.mean(sh_vals**2))) if sh_vals.size else None
        )
    if pressure_available:
        p_vals = x_full[np.asarray(p_air, dtype=np.int32)]
        proxies["cavity_pressure_max_proxy_v1"] = (
            float(np.max(np.abs(p_vals))) if p_vals.size else None
        )
    return proxies


def build_mode_synthesis_row(
    *,
    catalog_index: int,
    x_active: np.ndarray,
    built: Dict[str, Any],
    region_ctx: Dict[str, Any],
    scalars: Dict[str, Any],
) -> Dict[str, Any]:
    region = region_ctx["region"]
    structural_ok = bool(region_ctx["structural_indices_available"])
    pressure_ok = bool(region_ctx["pressure_indices_available"])
    x_full = prolongate_active_to_W(x_active, built)
    unavail = UNAVAILABLE_REGION_INDICES_STATUS

    row: Dict[str, Any] = {
        "catalog_index": int(catalog_index),
        "frequency_hz": scalars["frequency_hz"],
        "lambda_real": scalars["lambda_real"],
        "lambda_imag": scalars["lambda_imag"],
        "st_shift_target_hz": scalars["st_shift_target_hz"],
        "target_index": scalars["target_index"],
        "eps_slot_index": scalars["eps_slot_index"],
        "eps_compute_error_relative": scalars["eps_compute_error_relative"],
        "u_norm_W": scalars["u_norm_W"],
        "p_norm_W": scalars["p_norm_W"],
        "p_support": scalars["p_support"],
        "structural_region_participation_status": (
            REGION_PARTICIPATION_STATUS_AVAILABLE if structural_ok else unavail
        ),
        "structural_audio_proxy_status": (
            REGION_PARTICIPATION_STATUS_AVAILABLE if structural_ok else unavail
        ),
        "participation_top": (
            participation_energy_fraction(x_full, region.get("u_idx_top", []))
            if structural_ok
            else None
        ),
        "participation_back": (
            participation_energy_fraction(x_full, region.get("u_idx_back", []))
            if structural_ok
            else None
        ),
        "participation_ribs": (
            participation_energy_fraction(x_full, region.get("u_idx_ribs", []))
            if structural_ok
            else None
        ),
        "participation_soundhole": (
            participation_energy_fraction(x_full, region.get("u_idx_soundhole", []))
            if structural_ok
            else None
        ),
        "participation_air_p": (
            participation_energy_fraction(x_full, region.get("p_idx_air", []))
            if pressure_ok
            else None
        ),
        "audio_output_proxies": audio_output_proxies_v1(
            x_full,
            region,
            structural_available=structural_ok,
            pressure_available=pressure_ok,
        ),
    }
    return row


def scalar_block_norms(x_full: np.ndarray, built: Dict[str, Any]) -> Dict[str, float]:
    u_idx = np.asarray(built["u_idx"], dtype=np.int32).ravel()
    p_idx = np.asarray(built["p_idx"], dtype=np.int32).ravel()
    u_norm = float(np.linalg.norm(np.abs(x_full[u_idx])))
    p_norm = float(np.linalg.norm(np.abs(x_full[p_idx])))
    x_norm = float(np.linalg.norm(x_full))
    p_support = p_norm / max(x_norm, 1.0e-30)
    return {
        "u_norm_W": u_norm,
        "p_norm_W": p_norm,
        "x_norm_W": x_norm,
        "p_support": p_support,
    }


class RichModalCollector:
    """Accumulate accepted modes across ST targets for one solve run."""

    def __init__(self) -> None:
        self._columns: List[np.ndarray] = []
        self._catalog: List[Dict[str, Any]] = []

    def add_mode(
        self,
        *,
        x_active: np.ndarray,
        target_index: int,
        target_hz: float,
        record: Dict[str, Any],
    ) -> None:
        self._columns.append(np.asarray(x_active, dtype=np.float64).ravel().copy())
        entry = dict(record)
        entry["target_index"] = int(target_index)
        entry["st_shift_target_hz"] = float(target_hz)
        entry["catalog_index"] = len(self._catalog)
        self._catalog.append(entry)

    @property
    def mode_count(self) -> int:
        return len(self._columns)

    def write_bundle(
        self,
        rich_dir: Path,
        *,
        checkpoint_dir: Path,
        solve_output_dir: Path,
        factor_solver: str,
        nev: int,
        ncv: int,
        target_set: str,
        targets_hz: Sequence[float],
        acceptance_interval_hz: Sequence[float],
        synthesis_metadata_path: Path,
    ) -> Dict[str, Any]:
        rich_dir.mkdir(parents=True, exist_ok=True)
        if not self._columns:
            manifest = {
                "schema": RICH_MODAL_SCHEMA,
                "mode_count": 0,
                "status": "empty",
            }
            write_json_atomic(rich_dir / RICH_MODAL_MANIFEST_JSON, manifest)
            return manifest

        n_active = int(self._columns[0].size)
        mat = np.column_stack(self._columns)
        cat = self._catalog
        np.savez_compressed(
            rich_dir / MODES_ACTIVE_NPZ,
            eigenvectors_active=mat,
            active_dimension=np.int64(n_active),
            mode_index=np.asarray([c.get("eps_slot_index", c.get("mode_index", -1)) for c in cat], dtype=np.int32),
            frequency_hz=np.asarray([c["frequency_hz"] for c in cat], dtype=np.float64),
            lambda_real=np.asarray([c.get("lambda_real", np.nan) for c in cat], dtype=np.float64),
            lambda_imag=np.asarray([c.get("lambda_imag", np.nan) for c in cat], dtype=np.float64),
            st_shift_target_hz=np.asarray([c["st_shift_target_hz"] for c in cat], dtype=np.float64),
            target_index=np.asarray([c["target_index"] for c in cat], dtype=np.int32),
            eps_slot_index=np.asarray([c.get("eps_slot_index", c.get("mode_index", -1)) for c in cat], dtype=np.int32),
            eps_compute_error_relative=np.asarray(
                [c.get("eps_compute_error_relative", np.nan) for c in cat], dtype=np.float64
            ),
            u_norm_W=np.asarray([c.get("u_norm_W", np.nan) for c in cat], dtype=np.float64),
            p_norm_W=np.asarray([c.get("p_norm_W", np.nan) for c in cat], dtype=np.float64),
            p_support=np.asarray([c.get("p_support", np.nan) for c in cat], dtype=np.float64),
            x_norm_W=np.asarray([c.get("x_norm_W", np.nan) for c in cat], dtype=np.float64),
        )
        catalog_path = rich_dir / MODES_CATALOG_JSONL
        with catalog_path.open("w", encoding="utf-8") as fh:
            for row in cat:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

        freqs = [float(c["frequency_hz"]) for c in cat]
        manifest: Dict[str, Any] = {
            "schema": RICH_MODAL_SCHEMA,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "solve_output_dir": str(solve_output_dir.resolve()),
            "factor_solver": factor_solver,
            "nev": int(nev),
            "ncv": int(ncv),
            "target_set": str(target_set),
            "targets_hz": list(targets_hz),
            "acceptance_interval_hz": list(acceptance_interval_hz),
            "normalization_convention": normalization_convention_v1(),
            "synthesis_metadata_ref": str(synthesis_metadata_path.resolve()),
            "modes_active_npz": str((rich_dir / MODES_ACTIVE_NPZ).resolve()),
            "modes_catalog_jsonl": str(catalog_path.resolve()),
            "mode_count": len(cat),
            "unique_frequency_count": len({round(f, 6) for f in freqs}),
            "duplicate_frequency_count": len(freqs) - len({round(f, 6) for f in freqs}),
            "deduplication_note": (
                "Duplicates across shifts are retained in v1; use catalog for MAC/frequency dedupe."
            ),
        }
        write_json_atomic(rich_dir / RICH_MODAL_MANIFEST_JSON, manifest)
        return manifest


def frequency_dedupe_report(
    modes: Sequence[Dict[str, Any]],
    *,
    tol_hz: float,
) -> Dict[str, Any]:
    """Report duplicate frequencies without removing modes silently."""
    groups: Dict[float, List[int]] = {}
    for i, row in enumerate(modes):
        f = float(row["frequency_hz"])
        key = round(f / tol_hz) * tol_hz if tol_hz > 0 else round(f, 6)
        groups.setdefault(float(key), []).append(int(i))
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    return {
        "tolerance_hz": float(tol_hz),
        "unique_frequency_groups": len(groups),
        "duplicate_groups": len(dup_groups),
        "duplicate_group_keys_hz": [safe_float(k) for k in sorted(dup_groups.keys())],
        "duplicate_indices_by_group": dup_groups,
    }
