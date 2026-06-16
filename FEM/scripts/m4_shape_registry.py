#!/usr/bin/env python3
"""M4 FOM/ROM shape registry — resolves LHS, geometry, GMSH, export, and ROM namespaces."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
ROM_SHAPES_PATH = REPO_ROOT / "FEM" / "configs" / "rom_shapes.json"

# Production aperture / sound-hole policy (global solver contract; shape-specific metadata).
APERTURE_SELECTION_METHOD = "facet_adjacent_air_cell_dofs_v1"
PRODUCTION_MIC_METHOD = "aperture_pressure_rms_proxy_v1"

# Scout Stage-2 density acceptance policies (see v2_b3_m4_scout_intrinsic_coverage.py).
SCOUT_DENSITY_POLICY_CLASSIC = "intrinsic_discovered_modes_v1"
SCOUT_DENSITY_POLICY_BOX = "box_discovered_modes_v1"
SCOUT_DENSITY_POLICY_ACOUSTIC = "acoustic_discovered_modes_v1"
REGISTERED_SCOUT_DENSITY_POLICIES: tuple[str, ...] = (
    SCOUT_DENSITY_POLICY_CLASSIC,
    SCOUT_DENSITY_POLICY_BOX,
    SCOUT_DENSITY_POLICY_ACOUSTIC,
)


@dataclass(frozen=True)
class M4ShapeConfig:
    shape_key: str
    display_name: str
    lhs_pool_rel: str
    rom_dir_rel: str
    shared_export_key: str
    geometry_shape_type: str
    gmsh_shape_type: str
    sample_id_prefix: str
    default_lhs_count: int
    has_soundhole: bool
    requires_aperture_mask: bool
    soundhole_note: str
    scout_density_policy: str = SCOUT_DENSITY_POLICY_CLASSIC
    base_config_rel: str = "FEM/configs/guitar_3d.json"

    @property
    def sample_id_pattern(self) -> str:
        return f"{self.sample_id_prefix}{{index:03d}}"

    def sample_id(self, index: int) -> str:
        return self.sample_id_pattern.format(index=int(index))

    def lhs_pool_path(self, repo_root: Optional[Path] = None) -> Path:
        root = Path(repo_root or REPO_ROOT)
        return root / self.lhs_pool_rel

    def rom_dir(self, repo_root: Optional[Path] = None) -> Path:
        root = Path(repo_root or REPO_ROOT)
        return root / self.rom_dir_rel

    def pipeline_index_dir_rel(self) -> str:
        return (
            "FEM/experiments/active_domain_validation/physics_integrity/"
            f"pipeline_runs/index/{self.shape_key}"
        )

    def acoustic_opening_policy(self) -> Dict[str, Any]:
        return {
            "shape_key": self.shape_key,
            "has_soundhole": self.has_soundhole,
            "requires_aperture_mask": self.requires_aperture_mask,
            "aperture_selection_method": APERTURE_SELECTION_METHOD,
            "production_mic_method": PRODUCTION_MIC_METHOD,
            "soundhole_note": self.soundhole_note,
            "policy_scope": "global_m4_production_contracts",
        }

    def shape_context_fields(self, *, lhs_path: str = "") -> Dict[str, Any]:
        return {
            "shape_name": self.shape_key,
            "geometry_shape_type": self.geometry_shape_type,
            "gmsh_shape_type": self.gmsh_shape_type,
            "lhs_path": lhs_path,
            "acoustic_opening_policy": self.acoustic_opening_policy(),
        }


def _load_rom_shapes_doc() -> Dict[str, Any]:
    if not ROM_SHAPES_PATH.is_file():
        return {"shapes": {}}
    return json.loads(ROM_SHAPES_PATH.read_text(encoding="utf-8"))


def _geometry_shape_type_from_rom_entry(entry: Mapping[str, Any]) -> str:
    sweep = entry.get("parameter_sweep") or {}
    values = sweep.get("geometry.shape_type")
    if isinstance(values, list) and values:
        return str(values[0])
    if isinstance(values, str):
        return values
    return "Classical"


# Static registry — rom_shapes.json supplies geometry.shape_type sweep values.
_SHAPE_REGISTRY: Dict[str, M4ShapeConfig] = {
    "classic": M4ShapeConfig(
        shape_key="classic",
        display_name="Classical",
        lhs_pool_rel="ROM/classic/lhs_pool.json",
        rom_dir_rel="ROM/classic",
        shared_export_key="classic",
        geometry_shape_type="Classical",
        gmsh_shape_type="Classical",
        sample_id_prefix="sample_",
        default_lhs_count=500,
        has_soundhole=True,
        requires_aperture_mask=True,
        soundhole_note="Classical soundhole via geometry.hole_radius; aperture mask enforced in production.",
        scout_density_policy=SCOUT_DENSITY_POLICY_CLASSIC,
    ),
    "box": M4ShapeConfig(
        shape_key="box",
        display_name="Box",
        lhs_pool_rel="ROM/box/lhs_pool.json",
        rom_dir_rel="ROM/box",
        shared_export_key="box",
        geometry_shape_type="Box",
        gmsh_shape_type="Box",
        sample_id_prefix="box_sample_",
        default_lhs_count=100,
        has_soundhole=True,
        requires_aperture_mask=True,
        soundhole_note="Box LHS includes geometry.hole_radius; same global aperture mask policy as classic.",
        scout_density_policy=SCOUT_DENSITY_POLICY_BOX,
    ),
    "acoustic": M4ShapeConfig(
        shape_key="acoustic",
        display_name="Acoustic",
        lhs_pool_rel="ROM/acoustic/lhs_pool.json",
        rom_dir_rel="ROM/acoustic",
        shared_export_key="acoustic",
        geometry_shape_type="Acoustic",
        gmsh_shape_type="Acoustic",
        sample_id_prefix="acoustic_sample_",
        default_lhs_count=100,
        has_soundhole=True,
        requires_aperture_mask=True,
        soundhole_note="Acoustic/dreadnought body uses acoustic.step; hole_radius swept in LHS.",
        scout_density_policy=SCOUT_DENSITY_POLICY_ACOUSTIC,
    ),
}


def is_registered_scout_density_policy(policy: str) -> bool:
    return str(policy or "") in REGISTERED_SCOUT_DENSITY_POLICIES


def scout_density_policy_for_shape(shape_key: str) -> str:
    return resolve_shape_config(shape_key).scout_density_policy


def registered_shape_keys() -> tuple[str, ...]:
    return tuple(_SHAPE_REGISTRY.keys())


def normalize_shape_key(value: str) -> str:
    raw = str(value or "classic").strip().lower()
    aliases = {
        "classical": "classic",
        "classic": "classic",
        "box": "box",
        "rect": "box",
        "acoustic": "acoustic",
        "dreadnought": "acoustic",
        "dread": "acoustic",
    }
    key = aliases.get(raw, raw)
    if key not in _SHAPE_REGISTRY:
        raise ValueError(f"unknown shape {value!r}; expected one of {registered_shape_keys()}")
    return key


def resolve_shape_config(shape_key: str) -> M4ShapeConfig:
    key = normalize_shape_key(shape_key)
    cfg = _SHAPE_REGISTRY[key]
    rom_doc = _load_rom_shapes_doc()
    rom_entry = (rom_doc.get("shapes") or {}).get(key) or {}
    if rom_entry:
        geom_type = _geometry_shape_type_from_rom_entry(rom_entry)
        base = str(rom_entry.get("base_config") or cfg.base_config_rel)
        return M4ShapeConfig(
            shape_key=cfg.shape_key,
            display_name=cfg.display_name,
            lhs_pool_rel=cfg.lhs_pool_rel,
            rom_dir_rel=cfg.rom_dir_rel,
            shared_export_key=cfg.shared_export_key,
            geometry_shape_type=geom_type,
            gmsh_shape_type=geom_type,
            sample_id_prefix=cfg.sample_id_prefix,
            default_lhs_count=cfg.default_lhs_count,
            has_soundhole=cfg.has_soundhole,
            requires_aperture_mask=cfg.requires_aperture_mask,
            soundhole_note=cfg.soundhole_note,
            scout_density_policy=cfg.scout_density_policy,
            base_config_rel=base,
        )
    return cfg


def infer_shape_from_lhs_path(lhs_path: Path) -> Optional[str]:
    parts = {p.lower() for p in Path(lhs_path).parts}
    for key in registered_shape_keys():
        if key in parts:
            return key
    return None


def infer_shape_from_sample_id(sample_id: str) -> str:
    sid = str(sample_id or "")
    if sid.startswith("box_sample_"):
        return "box"
    if sid.startswith("acoustic_sample_"):
        return "acoustic"
    return "classic"


def shape_from_pool(pool: Mapping[str, Any]) -> str:
    return normalize_shape_key(str(pool.get("shape_name") or "classic"))


def resolve_geometry_shape_type(
    *,
    pool: Optional[Mapping[str, Any]] = None,
    parameters: Optional[Mapping[str, Any]] = None,
    sample_input: Optional[Mapping[str, Any]] = None,
) -> str:
    if sample_input:
        if sample_input.get("geometry_shape_type"):
            return str(sample_input["geometry_shape_type"])
        params = sample_input.get("parameters")
        if isinstance(params, dict) and params.get("geometry.shape_type"):
            return str(params["geometry.shape_type"])
        pool_shape = sample_input.get("shape_name")
        if pool_shape:
            return resolve_shape_config(str(pool_shape)).geometry_shape_type
    if parameters and parameters.get("geometry.shape_type"):
        return str(parameters["geometry.shape_type"])
    if pool:
        return resolve_shape_config(shape_from_pool(pool)).geometry_shape_type
    return resolve_shape_config("classic").geometry_shape_type


def ensure_parameters_shape_type(
    parameters: Dict[str, Any],
    *,
    shape_key: str,
) -> Dict[str, Any]:
    cfg = resolve_shape_config(shape_key)
    out = dict(parameters)
    out["geometry.shape_type"] = cfg.geometry_shape_type
    return out


def lhs_bounds_for_shape(shape_key: str) -> Dict[str, Any]:
    """Return LHS sweep bounds for box/acoustic; classic uses regenerate_lhs_pool."""
    key = normalize_shape_key(shape_key)
    woods = ["spruce", "cedar", "mahogany", "rosewood", "maple"]
    if key == "box":
        return {
            "geometry.length": {"min": 0.40, "max": 0.52},
            "geometry.width": {"min": 0.32, "max": 0.42},
            "geometry.depth": {"min": 0.06, "max": 0.16},
            "geometry.top_thickness": {"min": 0.0025, "max": 0.0035},
            "geometry.hole_radius": {"min": 0.035, "max": 0.048},
            "top_wood_id": woods,
            "back_wood_id": woods,
        }
    if key == "acoustic":
        return {
            "geometry.length": {"min": 0.45, "max": 0.70},
            "geometry.width": {"min": 0.30, "max": 0.55},
            "geometry.depth": {"min": 0.10, "max": 0.20},
            "geometry.top_thickness": {"min": 0.0025, "max": 0.0035},
            "geometry.hole_radius": {"min": 0.035, "max": 0.055},
            "top_wood_id": woods,
            "back_wood_id": woods,
        }
    return {}
