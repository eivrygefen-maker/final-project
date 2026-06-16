#!/usr/bin/env python3
"""Inspect shape mesh geometry tags and soundhole↔air connectivity for GMSH visual proof."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
FEM_SCRIPTS = (REPO_ROOT / "FEM" / "scripts").resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(FEM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FEM_SCRIPTS))
from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
    LEVEL_ROM_PROD,
    convergence_mesh_path,
    run_tree_lprod_mesh_path,
)
from v2_b3_m4_physics_identity_lib import CAVITY_AIR_TAG, WOOD_VOLUME_TAGS  # noqa: E402
from v2_b3_m4_worker_run_lib import detect_repo_root, rel, utc_now  # noqa: E402
from v2_b3_m4_mesh_manifest_lib import load_mesh_manifest, mesh_manifest_path  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

# Shared facet/volume tag conventions (classical + box production meshes).
SOUNDHOLE_FACET_TAG = 2
TOP_FACET_TAG = 1
BACK_FACET_TAG = 3
RIBS_FACET_TAG = 4

MESH_LEVEL_FALLBACKS: Tuple[str, ...] = (
    LEVEL_ROM_PROD,
    "L_scout_coarse",
    "L_prod_reference",
    "L_prod",
)

GUITARS_REL = Path(
    "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
)


def resolve_run_root(repo_root: Path, *, sample_id: str, run_id: str) -> Path:
    return repo_root / GUITARS_REL / sample_id / "runs" / run_id


def _candidate_mesh_paths(
    *,
    repo_root: Path,
    sample_id: str,
    mesh_level: str,
    run_root: Optional[Path],
) -> List[Path]:
    candidates: List[Path] = []
    if mesh_level:
        candidates.append(convergence_mesh_path(mesh_level, sample_id))
    for level in MESH_LEVEL_FALLBACKS:
        p = convergence_mesh_path(level, sample_id)
        if p not in candidates:
            candidates.append(p)
    if run_root is not None:
        for level in (mesh_level,) + MESH_LEVEL_FALLBACKS if mesh_level else MESH_LEVEL_FALLBACKS:
            if not level:
                continue
            rp = run_tree_lprod_mesh_path(run_root, sample_id, level)
            if rp not in candidates:
                candidates.append(rp)
        scout_mesh = run_root / "scout" / "mesh" / "L_scout_coarse" / f"{sample_id}.msh"
        if scout_mesh not in candidates:
            candidates.append(scout_mesh)
    return candidates


def locate_mesh_path(
    *,
    repo_root: Path,
    sample_id: str,
    mesh_level: str,
    run_root: Optional[Path],
) -> Tuple[Optional[Path], List[str]]:
    tried: List[str] = []
    for path in _candidate_mesh_paths(
        repo_root=repo_root,
        sample_id=sample_id,
        mesh_level=mesh_level,
        run_root=run_root,
    ):
        tried.append(str(rel(path, repo_root=repo_root)))
        if path.is_file():
            return path, tried
    return None, tried


def _face_key(nodes: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(n) for n in nodes))


def _collect_tet_faces(tets: Iterable[Sequence[int]]) -> Set[Tuple[int, ...]]:
    faces: Set[Tuple[int, ...]] = set()
    for tet in tets:
        a, b, c, d = (int(x) for x in tet)
        for face in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            faces.add(_face_key(face))
    return faces


def audit_mesh_aperture(mesh_path: Path) -> Dict[str, Any]:
    import meshio

    m = meshio.read(str(mesh_path))
    phys = m.cell_data_dict.get("gmsh:physical", {})

    volume_blocks: List[Tuple[str, Any, Any]] = []
    for block in m.cells:
        tags = phys.get(block.type)
        if tags is None:
            continue
        volume_blocks.append((block.type, block.data, tags))

    facet_blocks: List[Tuple[str, Any, Any]] = []
    for block in m.cells:
        if block.type not in ("triangle", "quad"):
            continue
        tags = phys.get(block.type)
        if tags is None:
            continue
        facet_blocks.append((block.type, block.data, tags))

    volume_tag_counts: Counter[int] = Counter()
    facet_tag_counts: Counter[int] = Counter()
    air_tets: List[Sequence[int]] = []
    soundhole_facets: List[Sequence[int]] = []

    for _ctype, data, tags in volume_blocks:
        flat_tags = tags.reshape(-1)
        for idx, row in enumerate(data):
            tag = int(flat_tags[idx]) if idx < len(flat_tags) else int(flat_tags[0])
            volume_tag_counts[tag] += 1
            if tag == CAVITY_AIR_TAG:
                air_tets.append(row)

    for _ctype, data, tags in facet_blocks:
        flat_tags = tags.reshape(-1)
        for idx, row in enumerate(data):
            tag = int(flat_tags[idx]) if idx < len(flat_tags) else int(flat_tags[0])
            facet_tag_counts[tag] += 1
            if tag == SOUNDHOLE_FACET_TAG:
                nodes = row.tolist() if hasattr(row, "tolist") else list(row)
                if len(nodes) >= 3:
                    soundhole_facets.append(nodes[:3])

    air_faces = _collect_tet_faces(air_tets)
    air_nodes: Set[int] = set()
    for tet in air_tets:
        air_nodes.update(int(n) for n in tet)

    adjacent_to_air = 0
    on_air_boundary = 0
    for tri_nodes in soundhole_facets:
        key = _face_key(tri_nodes)
        if key in air_faces:
            on_air_boundary += 1
        if all(int(n) in air_nodes for n in tri_nodes):
            adjacent_to_air += 1

    soundhole_count = int(facet_tag_counts.get(SOUNDHOLE_FACET_TAG, 0))
    air_volume_count = int(volume_tag_counts.get(CAVITY_AIR_TAG, 0))
    wood_volume_count = sum(int(volume_tag_counts.get(t, 0)) for t in WOOD_VOLUME_TAGS)

    checks = {
        "soundhole_facets_present": soundhole_count > 0,
        "air_volume_present": air_volume_count > 0,
        "soundhole_facets_on_air_boundary": on_air_boundary > 0,
        "soundhole_facets_adjacent_to_air_nodes": adjacent_to_air > 0,
        "air_path_connected": on_air_boundary > 0 and air_volume_count > 0,
    }
    failures: List[str] = []
    if not checks["soundhole_facets_present"]:
        failures.append("missing_soundhole_facet_tag2")
    if not checks["air_volume_present"]:
        failures.append("missing_air_volume_tag10")
    if not checks["soundhole_facets_on_air_boundary"]:
        failures.append("soundhole_not_on_air_volume_boundary")
    if not checks["soundhole_facets_adjacent_to_air_nodes"]:
        failures.append("soundhole_not_adjacent_to_air_nodes")

    status = "PASS" if not failures else "FAIL"
    return {
        "schema": "m4_shape_mesh_aperture_audit_v1",
        "mesh_path": str(mesh_path),
        "volume_tag_counts": {str(k): v for k, v in sorted(volume_tag_counts.items())},
        "facet_tag_counts": {str(k): v for k, v in sorted(facet_tag_counts.items())},
        "soundhole_facet_count": soundhole_count,
        "air_volume_cell_count": air_volume_count,
        "wood_volume_cell_count": wood_volume_count,
        "soundhole_facets_on_air_boundary": on_air_boundary,
        "soundhole_facets_adjacent_to_air_nodes": adjacent_to_air,
        "checks": checks,
        "failures": failures,
        "status": status,
    }


def render_gmsh_report_md(
    *,
    shape_key: str,
    sample_id: str,
    mesh_path: Path,
    audit: Dict[str, Any],
    shape_cfg: Any,
) -> str:
    rel_mesh = mesh_path.as_posix()
    lines = [
        f"# Mesh aperture inspection — {sample_id}",
        "",
        f"- shape_name: **{shape_key}**",
        f"- geometry_shape_type: **{shape_cfg.geometry_shape_type}**",
        f"- gmsh_shape_type: **{shape_cfg.gmsh_shape_type}**",
        f"- mesh_path: `{rel_mesh}`",
        f"- audit_status: **{audit.get('status')}**",
        "",
        "## Tag summary",
        "",
        f"- soundhole/aperture facet tag **{SOUNDHOLE_FACET_TAG}**: "
        f"{audit.get('soundhole_facet_count')} facets",
        f"- air volume tag **{CAVITY_AIR_TAG}**: {audit.get('air_volume_cell_count')} cells",
        f"- wood volume tags **{WOOD_VOLUME_TAGS}**: {audit.get('wood_volume_cell_count')} cells",
        "",
        "## Air path / aperture connectivity",
        "",
        f"- soundhole facets on air-volume boundary: **{audit.get('soundhole_facets_on_air_boundary')}**",
        f"- soundhole facets adjacent to air nodes: **{audit.get('soundhole_facets_adjacent_to_air_nodes')}**",
        f"- air_path_connected: **{audit.get('checks', {}).get('air_path_connected')}**",
        "",
        "## Open in GMSH",
        "",
        "```bash",
        f"gmsh {rel_mesh}",
        "```",
        "",
        "Enable these physical groups in GMSH to visually verify geometry:",
        "",
        f"- Tag **{TOP_FACET_TAG}**: top plate facets",
        f"- Tag **{BACK_FACET_TAG}**: back plate facets",
        f"- Tag **{RIBS_FACET_TAG}**: side/rib facets (if present)",
        f"- Tag **{SOUNDHOLE_FACET_TAG}**: soundhole / aperture boundary",
        f"- Tag **{CAVITY_AIR_TAG}**: air cavity volume",
        f"- Wood body volumes: tags **{WOOD_VOLUME_TAGS}**",
        "",
        "For BOX: confirm the outer box body encloses the air volume and that tag-2 facets",
        "form an opening connected to tag-10 air cells (not an isolated surface tag only).",
        "",
    ]
    failures = audit.get("failures") or []
    if failures:
        lines.extend(["## Failures", ""] + [f"- {f}" for f in failures] + [""])
    return "\n".join(lines)


def inspect_shape_mesh_aperture(
    *,
    repo_root: Path,
    shape_key: str,
    sample_id: str,
    mesh_level: str = "",
    run_id: str = "",
    write_run_report: bool = True,
) -> Dict[str, Any]:
    from m4_shape_registry import resolve_shape_config  # noqa: WPS433

    repo_root = repo_root.expanduser().resolve()
    shape_cfg = resolve_shape_config(shape_key)
    run_root = resolve_run_root(repo_root, sample_id=sample_id, run_id=run_id) if run_id else None
    mesh_path, tried = locate_mesh_path(
        repo_root=repo_root,
        sample_id=sample_id,
        mesh_level=mesh_level,
        run_root=run_root if run_root and run_root.is_dir() else None,
    )
    out: Dict[str, Any] = {
        "shape_name": shape_key,
        "sample_id": sample_id,
        "geometry_shape_type": shape_cfg.geometry_shape_type,
        "gmsh_shape_type": shape_cfg.gmsh_shape_type,
        "mesh_level_requested": mesh_level or None,
        "mesh_candidates_tried": tried,
        "generated_utc": utc_now(),
    }
    if mesh_path is None:
        out["status"] = "FAIL"
        out["error"] = "mesh_not_found"
        return out

    out["mesh_path"] = str(rel(mesh_path, repo_root=repo_root))
    manifest = load_mesh_manifest(mesh_path)
    if manifest:
        out["mesh_manifest_shape"] = manifest.get("geometry_shape_type")
        out["mesh_manifest_gmsh_shape_type"] = manifest.get("gmsh_shape_type")
        out["mesh_manifest_path"] = str(rel(mesh_manifest_path(mesh_path), repo_root=repo_root))
    audit = audit_mesh_aperture(mesh_path)
    out.update(audit)

    if write_run_report and run_root and run_root.is_dir():
        report_dir = run_root / "mesh_inspection"
        report_dir.mkdir(parents=True, exist_ok=True)
        md_path = report_dir / f"{sample_id}_mesh_aperture_report.md"
        json_path = report_dir / f"{sample_id}_mesh_aperture_report.json"
        md_path.write_text(
            render_gmsh_report_md(
                shape_key=shape_key,
                sample_id=sample_id,
                mesh_path=mesh_path,
                audit=audit,
                shape_cfg=shape_cfg,
            ),
            encoding="utf-8",
        )
        write_json_atomic(json_path, out)
        out["run_report_md"] = str(rel(md_path, repo_root=repo_root))
        out["run_report_json"] = str(rel(json_path, repo_root=repo_root))

    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect shape mesh tags and aperture↔air connectivity.")
    parser.add_argument("--shape", required=True, help="Shape key: classic|box|acoustic")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--mesh-level", default="", help="Preferred mesh level (e.g. L_rom_prod, L_scout_coarse)")
    parser.add_argument("--run-id", default="", help="Optional run id for run-tree mesh/report output")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--no-run-report", action="store_true")
    args = parser.parse_args(argv)

    repo_root = (args.repo_root or detect_repo_root(SCRIPT_DIR)).resolve()
    report = inspect_shape_mesh_aperture(
        repo_root=repo_root,
        shape_key=str(args.shape),
        sample_id=str(args.sample_id),
        mesh_level=str(args.mesh_level or ""),
        run_id=str(args.run_id or ""),
        write_run_report=not bool(args.no_run_report),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"audit_status={report.get('status')}")
    if report.get("mesh_path"):
        print(f"mesh_path={report.get('mesh_path')}")
    if report.get("run_report_md"):
        print(f"report_md={report.get('run_report_md')}")
    return 0 if report.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
