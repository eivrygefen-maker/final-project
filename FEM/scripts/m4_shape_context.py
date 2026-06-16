#!/usr/bin/env python3
"""Resolved shape context for the shared M4 FOM/ROM pipeline (all body shapes)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from m4_shape_registry import (  # noqa: E402
    M4ShapeConfig,
    resolve_geometry_shape_type,
    resolve_shape_config,
    shape_from_pool,
)


class ShapeContextError(ValueError):
    """Raised when shape metadata is missing or inconsistent."""


@dataclass(frozen=True)
class ShapeContext:
    shape_name: str
    geometry_shape_type: str
    gmsh_shape_type: str
    lhs_path: str
    rom_output_root: str
    shared_export_key: str
    scout_density_policy: str
    base_config_rel: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape_name": self.shape_name,
            "geometry_shape_type": self.geometry_shape_type,
            "gmsh_shape_type": self.gmsh_shape_type,
            "lhs_path": self.lhs_path,
            "rom_output_root": self.rom_output_root,
            "shared_export_key": self.shared_export_key,
            "scout_density_policy": self.scout_density_policy,
            "base_config_rel": self.base_config_rel,
        }

    def to_sample_input_fields(self) -> Dict[str, Any]:
        return {
            "shape_name": self.shape_name,
            "geometry_shape_type": self.geometry_shape_type,
            "gmsh_shape_type": self.gmsh_shape_type,
            "lhs_path": self.lhs_path,
            "scout_density_policy": self.scout_density_policy,
        }

    def to_mesh_metadata(self) -> Dict[str, str]:
        return {
            "shape_name": self.shape_name,
            "geometry_shape_type": self.geometry_shape_type,
            "gmsh_shape_type": self.gmsh_shape_type,
        }


def _shape_context_from_config(
    cfg: M4ShapeConfig,
    *,
    lhs_path: str = "",
    repo_root: Optional[Path] = None,
) -> ShapeContext:
    rom_root = str(cfg.rom_dir(repo_root))
    return ShapeContext(
        shape_name=cfg.shape_key,
        geometry_shape_type=cfg.geometry_shape_type,
        gmsh_shape_type=cfg.gmsh_shape_type,
        lhs_path=lhs_path,
        rom_output_root=rom_root,
        shared_export_key=cfg.shared_export_key,
        scout_density_policy=cfg.scout_density_policy,
        base_config_rel=cfg.base_config_rel,
    )


def resolve_shape_context(
    shape_key: str,
    *,
    lhs_path: str = "",
    repo_root: Optional[Path] = None,
) -> ShapeContext:
    cfg = resolve_shape_config(shape_key)
    return _shape_context_from_config(cfg, lhs_path=lhs_path, repo_root=repo_root)


def _coerce_shape_name(
    sample_input: Mapping[str, Any],
    *,
    pool: Optional[Mapping[str, Any]],
    legacy_classic_default: bool,
) -> str:
    shape_name = sample_input.get("shape_name")
    if shape_name:
        return str(shape_name)
    if pool is not None:
        return shape_from_pool(pool)
    if legacy_classic_default:
        return "classic"
    raise ShapeContextError(
        "shape_name missing from sample_input; cannot resolve shape context"
    )


def _validate_non_classic_geometry(ctx: ShapeContext) -> None:
    if ctx.shape_name == "classic":
        return
    cfg = resolve_shape_config(ctx.shape_name)
    if ctx.geometry_shape_type == "Classical" and cfg.geometry_shape_type != "Classical":
        raise ShapeContextError(
            f"shape={ctx.shape_name!r} resolved to geometry_shape_type=Classical; "
            f"expected {cfg.geometry_shape_type!r}"
        )
    if ctx.geometry_shape_type != cfg.geometry_shape_type:
        raise ShapeContextError(
            f"shape={ctx.shape_name!r} geometry_shape_type={ctx.geometry_shape_type!r} "
            f"does not match registry {cfg.geometry_shape_type!r}"
        )
    if ctx.gmsh_shape_type != cfg.gmsh_shape_type:
        raise ShapeContextError(
            f"shape={ctx.shape_name!r} gmsh_shape_type={ctx.gmsh_shape_type!r} "
            f"does not match registry {cfg.gmsh_shape_type!r}"
        )


def resolve_shape_context_from_sample_input(
    sample_input: Mapping[str, Any],
    *,
    pool: Optional[Mapping[str, Any]] = None,
    legacy_classic_default: bool = True,
    repo_root: Optional[Path] = None,
) -> ShapeContext:
    """Resolve canonical shape context from a production sample_input manifest."""
    shape_name = _coerce_shape_name(
        sample_input,
        pool=pool,
        legacy_classic_default=legacy_classic_default,
    )
    cfg = resolve_shape_config(shape_name)
    lhs_path = str(
        sample_input.get("lhs_path")
        or sample_input.get("lhs_source_path")
        or cfg.lhs_pool_rel
    )
    geometry_shape_type = str(
        sample_input.get("geometry_shape_type")
        or resolve_geometry_shape_type(
            pool=pool,
            sample_input=sample_input,
            legacy_classic_default=legacy_classic_default,
        )
    )
    gmsh_shape_type = str(
        sample_input.get("gmsh_shape_type") or cfg.gmsh_shape_type
    )
    ctx = ShapeContext(
        shape_name=cfg.shape_key,
        geometry_shape_type=geometry_shape_type,
        gmsh_shape_type=gmsh_shape_type,
        lhs_path=lhs_path,
        rom_output_root=str(cfg.rom_dir(repo_root)),
        shared_export_key=cfg.shared_export_key,
        scout_density_policy=str(
            sample_input.get("scout_density_policy") or cfg.scout_density_policy
        ),
        base_config_rel=cfg.base_config_rel,
    )
    if not legacy_classic_default or shape_name != "classic":
        _validate_non_classic_geometry(ctx)
    return ctx


def apply_shape_context_to_resolved_config(
    resolved: Dict[str, Any],
    ctx: ShapeContext,
    *,
    geometry_numeric: Optional[Mapping[str, Any]] = None,
) -> None:
    """Apply resolved shape context to a core config (Stage 0 / L_prod)."""
    geom = resolved.setdefault("geometry", {})
    if isinstance(geom, dict):
        geom["shape_type"] = ctx.geometry_shape_type
        if geometry_numeric:
            for key, val in geometry_numeric.items():
                geom[key] = val

    for top_key, val in ctx.to_sample_input_fields().items():
        if val:
            resolved[str(top_key)] = str(val)

    m4_meta = resolved.setdefault("m4_run_metadata", {})
    if isinstance(m4_meta, dict):
        m4_meta.update(
            {
                "shape_name": ctx.shape_name,
                "geometry_shape_type": ctx.geometry_shape_type,
                "gmsh_shape_type": ctx.gmsh_shape_type,
                "scout_density_policy": ctx.scout_density_policy,
            }
        )
        if ctx.lhs_path:
            m4_meta["lhs_path"] = ctx.lhs_path

    if geometry_numeric:
        resolved["geometry_numeric_parameters"] = {
            k: float(v) for k, v in geometry_numeric.items()
        }

    resolved["shape_context"] = ctx.to_dict()
