#!/usr/bin/env python3
"""
Soundhole facet (tag 2) ↔ air volume (tag 10) adjacency audit for physics-integrity gate.

Experiment-only: does not modify mesh, production solver, or BCs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "FEM" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

AIR_VOLUME_TAG = 10
SOUNDHOLE_FACET_TAG = 2
TOP_FACET_TAG = 1
BACK_FACET_TAG = 3
RIBS_FACET_TAG = 4
WOOD_VOLUME_TAGS = (1, 2, 3)

# Validation mesh profile (matches prepare_validation_mesh / FEM_VALIDATION_MESH).
EXPECTED_HOLE_RADIUS_M = 0.047
EXPECTED_LENGTH_M = 0.48
EXPECTED_WIDTH_M = 0.325
EXPECTED_DEPTH_M = 0.10
EXPECTED_SOUNDHOLE_FROM_NECK = 0.5


def _expected_soundhole_center_m() -> Tuple[float, float, float]:
    """Classical validation hole centre (matches build_3d_guitar defaults)."""
    L = EXPECTED_LENGTH_M
    W = EXPECTED_WIDTH_M
    D = EXPECTED_DEPTH_M
    hole_x = (0.5 - EXPECTED_SOUNDHOLE_FROM_NECK) * L - 0.5 * L
    hole_y = 0.0
    hole_z = 0.5 * D
    return float(hole_x), float(hole_y), float(hole_z)


def _facet_centroid(msh, facet_index: int) -> np.ndarray:
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    verts = f2v.links(int(facet_index))
    coords = msh.geometry.x[np.asarray(verts, dtype=np.int32)]
    return np.mean(coords, axis=0)


def _audit_dolfinx(
    mesh_path: Path,
) -> Tuple[Dict[str, Any], Any, Any, Any]:
    from mpi4py import MPI

    import fem_main_3d as fem3d
    from dolfinx import fem
    from basix.ufl import element

    msh, cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_path)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    msh.topology.create_connectivity(fdim, 0)
    f2c = msh.topology.connectivity(fdim, tdim)

    tag2_facets = np.asarray(facet_tags.find(SOUNDHOLE_FACET_TAG), dtype=np.int32)
    p_el = element("Lagrange", msh.basix_cell(), 1)
    V_p = fem.functionspace(msh, p_el)
    air_cells = np.asarray(cell_tags.find(AIR_VOLUME_TAG), dtype=np.int32)
    p_air_v = fem3d._locate_air_volume_pressure_dofs(V_p, msh, cell_tags)
    soundhole_facets = np.asarray(facet_tags.find(SOUNDHOLE_FACET_TAG), dtype=np.int32)
    p_sh_v = fem3d._locate_soundhole_pressure_release_dofs(V_p, soundhole_facets)
    p_sh_active = np.intersect1d(p_sh_v, p_air_v)

    per_facet: List[Dict[str, Any]] = []
    class_counts = Counter()
    n_invalid = 0

    for fi in tag2_facets:
        try:
            adj_cells = np.asarray(f2c.links(int(fi)), dtype=np.int32)
        except Exception:
            adj_cells = np.array([], dtype=np.int32)
        if adj_cells.size == 0:
            n_invalid += 1
            class_counts["no_adjacent_volume"] += 1
            per_facet.append(
                {
                    "facet_index": int(fi),
                    "n_adjacent_cells": 0,
                    "adjacent_volume_tags": [],
                    "touches_air_tag_10": False,
                    "touches_wood_only": False,
                    "classification": "no_adjacent_volume",
                }
            )
            continue

        vol_tags = [int(cell_tags.values[int(c)]) for c in adj_cells]
        touches_air = AIR_VOLUME_TAG in vol_tags
        touches_wood = any(t in WOOD_VOLUME_TAGS for t in vol_tags)
        touches_wood_only = touches_wood and not touches_air

        if touches_air:
            cls = "air_adjacent"
        elif touches_wood_only:
            cls = "wood_only_adjacent"
        else:
            cls = "other_non_air"
        class_counts[cls] += 1

        centroid = _facet_centroid(msh, int(fi))
        hx, hy, hz = _expected_soundhole_center_m()
        dist_xy = float(math.hypot(centroid[0] - hx, centroid[1] - hy))
        per_facet.append(
            {
                "facet_index": int(fi),
                "n_adjacent_cells": int(adj_cells.size),
                "adjacent_volume_tags": vol_tags,
                "touches_air_tag_10": touches_air,
                "touches_wood_only": touches_wood_only,
                "classification": cls,
                "centroid_m": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
                "dist_xy_from_expected_hole_m": dist_xy,
                "dist_z_from_expected_top_m": float(abs(centroid[2] - hz)),
            }
        )

    n_total = int(tag2_facets.size)
    n_air_adj = int(class_counts.get("air_adjacent", 0))
    n_wood_only = int(class_counts.get("wood_only_adjacent", 0))
    n_other = int(class_counts.get("other_non_air", 0))
    n_no_vol = int(class_counts.get("no_adjacent_volume", 0))

    centroids = np.array([p["centroid_m"] for p in per_facet if p.get("centroid_m")], dtype=np.float64)
    geom = {}
    if centroids.size > 0:
        geom = {
            "centroid_mean_m": centroids.mean(axis=0).tolist(),
            "centroid_std_m": centroids.std(axis=0).tolist(),
            "max_xy_spread_m": float(
                np.max(np.linalg.norm(centroids[:, :2] - centroids[:, :2].mean(axis=0), axis=1))
            ),
            "expected_hole_center_m": list(_expected_soundhole_center_m()),
            "expected_hole_radius_m": EXPECTED_HOLE_RADIUS_M,
            "fraction_within_hole_radius_xy": float(
                np.mean(
                    np.linalg.norm(
                        centroids[:, :2] - np.array(_expected_soundhole_center_m()[:2]),
                        axis=1,
                    )
                    <= EXPECTED_HOLE_RADIUS_M * 1.5
                )
            )
            if centroids.shape[0] > 0
            else 0.0,
        }

    dof_audit = {
        "n_p_full": int(V_p.dofmap.index_map.size_global),
        "n_p_air_supported": int(p_air_v.size),
        "n_p_soundhole_full": int(p_sh_v.size),
        "n_p_soundhole_in_air_subgraph": int(p_sh_active.size),
        "fraction_soundhole_in_air_subgraph": (
            float(p_sh_active.size) / float(p_sh_v.size) if p_sh_v.size else 0.0
        ),
    }

    diagnosis = _diagnose(n_total, n_air_adj, n_wood_only, n_no_vol, dof_audit, geom)

    summary = {
        "mesh_file": str(mesh_path.resolve()),
        "audit_backend": "dolfinx",
        "facet_tag_soundhole": SOUNDHOLE_FACET_TAG,
        "volume_tag_air": AIR_VOLUME_TAG,
        "facet_counts": {
            "tag2_total": n_total,
            "tag2_adjacent_to_air_10": n_air_adj,
            "tag2_adjacent_wood_only": n_wood_only,
            "tag2_adjacent_other_non_air": n_other,
            "tag2_no_adjacent_volume": n_no_vol,
        },
        "fractions": {
            "adjacent_to_air": (n_air_adj / n_total if n_total else 0.0),
            "adjacent_wood_only": (n_wood_only / n_total if n_total else 0.0),
        },
        "geometry_soundhole_tag2": geom,
        "pressure_dof_audit": dof_audit,
        "diagnosis": diagnosis,
        "per_facet_sample": per_facet[:20],
        "per_facet_note": f"First 20 of {len(per_facet)} tag-2 facets; see JSON for full list.",
        "per_facet_all": per_facet,
        "coupled_baseline_implication": _coupled_implication(n_air_adj, n_total, dof_audit),
    }
    return summary, msh, cell_tags, facet_tags


def _diagnose(
    n_total: int,
    n_air_adj: int,
    n_wood_only: int,
    n_no_vol: int,
    dof_audit: Dict[str, Any],
    geom: Dict[str, Any],
) -> Dict[str, Any]:
    codes: List[str] = []
    if n_total == 0:
        codes.append("C")
    if n_air_adj == 0 and n_total > 0:
        codes.append("B")
    if n_wood_only > 0 and n_air_adj == 0:
        codes.append("B")
    if dof_audit.get("n_p_soundhole_in_air_subgraph", 0) == 0 and dof_audit.get(
        "n_p_soundhole_full", 0
    ) > 0:
        if n_air_adj > 0:
            codes.append("A")
        else:
            codes.append("B")
    if n_air_adj > 0 and dof_audit.get("n_p_soundhole_in_air_subgraph", 0) == 0:
        codes.append("A")
    if n_air_adj > int(0.8 * n_total) and dof_audit.get("n_p_soundhole_in_air_subgraph", 0) > 0:
        codes.append("OK")
    if not codes:
        codes.append("D")

    primary = "D"
    if "C" in codes:
        primary = "C"
    elif "A" in codes and "B" not in codes:
        primary = "A"
    elif "B" in codes and "A" not in codes:
        primary = "B"
    elif "A" in codes and "B" in codes:
        primary = "D"
    elif "OK" in codes:
        primary = "OK"

    narratives = {
        "A": (
            "Tag-2 facets touch air volume cells but soundhole pressure DOFs are not in the "
            "air-supported subgraph — reduced acoustic branch mapping/BC restriction issue."
        ),
        "B": (
            "Tag-2 facets are not adjacent to air tag 10 (wood/top exterior only) — soundhole "
            "surface is not topologically connected to the cavity mesh used for acoustic DOFs."
        ),
        "C": "No tag-2 soundhole facets in mesh — CAD/mesh missing opening surface group.",
        "D": "Mixed or ambiguous tagging — review per-facet table and visualization.",
        "OK": "Tag-2 facets adjoin air and soundhole DOFs intersect air-supported pressure DOFs.",
    }
    return {
        "primary_hypothesis": primary,
        "hypothesis_codes": codes,
        "narrative": narratives.get(primary, ""),
    }


def _coupled_implication(n_air_adj: int, n_total: int, dof_audit: Dict[str, Any]) -> Dict[str, str]:
    n_sh = int(dof_audit.get("n_p_soundhole_full", 0))
    n_active = int(dof_audit.get("n_p_soundhole_in_air_subgraph", 0))
    return {
        "coupled_pressure_release": (
            f"Coupled baseline applies p=0 on {n_sh} full-volume pressure DOFs (facet tag 2), "
            "independent of the air-only subgraph."
        ),
        "effectiveness": (
            "If n_p_soundhole_in_air_subgraph=0, the pressure-release BC is enforced on DOFs "
            "that are NOT in the air-supported set used by acoustic-only restriction. "
            "Those rows still exist in the coupled mixed operator (algebraic Dirichlet on "
            "global p DOFs), but they are not the same DOFs that participate in ∫_air a_pp/m_pp. "
            "Cavity opening control in coupled runs may be ineffective for the air subspace "
            "until tag-2 facets adjoin air tag 10 and soundhole DOFs overlap air-supported nodes."
            if n_active == 0
            else "Soundhole DOFs overlap air-supported nodes; coupled BC likely affects cavity."
        ),
        "facet_air_fraction": (
            f"{n_air_adj}/{n_total} tag-2 facets have an adjacent air (tag 10) cell."
            if n_total
            else "no tag-2 facets"
        ),
    }


def _export_visualization_meshio(
    mesh_path: Path,
    out_dir: Path,
    per_facet: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Write ParaView-friendly XDMF and CSV (Gmsh physical groups on volume/shell triangles)."""
    import csv
    import meshio

    out_dir.mkdir(parents=True, exist_ok=True)
    m = meshio.read(str(mesh_path))
    paths: Dict[str, str] = {}
    phys = m.cell_data_dict.get("gmsh:physical", {})

    tet_blocks = [b.data for b in m.cells if b.type == "tetra"]
    if tet_blocks:
        tet_tags = np.asarray(phys.get("tetra", []), dtype=np.int32).ravel()
        vol_path = out_dir / "validation_mesh_volume_tags.xdmf"
        meshio.write(
            str(vol_path),
            meshio.Mesh(
                m.points,
                [("tetra", np.vstack(tet_blocks))],
                cell_data={"physical_tag": [tet_tags]},
            ),
        )
        paths["volume_tags_xdmf"] = str(vol_path)

    tri_blocks = [b.data for b in m.cells if b.type == "triangle"]
    if tri_blocks:
        tri_tags = np.asarray(phys.get("triangle", []), dtype=np.int32).ravel()
        shell_mask = np.isin(
            tri_tags,
            [TOP_FACET_TAG, SOUNDHOLE_FACET_TAG, BACK_FACET_TAG, RIBS_FACET_TAG],
        )
        if np.any(shell_mask):
            shell_path = out_dir / "shell_facets_tags_1_2_3_4.xdmf"
            tris = np.vstack(tri_blocks)
            meshio.write(
                str(shell_path),
                meshio.Mesh(
                    m.points,
                    [("triangle", tris[shell_mask])],
                    cell_data={"physical_tag": [tri_tags[shell_mask]]},
                ),
            )
            paths["shell_facets_xdmf"] = str(shell_path)
        sh_mask = tri_tags == SOUNDHOLE_FACET_TAG
        if np.any(sh_mask):
            sh_path = out_dir / "soundhole_facets_tag2.xdmf"
            tris = np.vstack(tri_blocks)
            meshio.write(
                str(sh_path),
                meshio.Mesh(
                    m.points,
                    [("triangle", tris[sh_mask])],
                    cell_data={"physical_tag": [tri_tags[sh_mask]]},
                ),
            )
            paths["soundhole_facets_tag2_xdmf"] = str(sh_path)

    csv_path = out_dir / "soundhole_facet_adjacency_table.csv"
    if per_facet:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "facet_index",
                    "n_adjacent_cells",
                    "adjacent_volume_tags",
                    "touches_air_tag_10",
                    "touches_wood_only",
                    "classification",
                    "dist_xy_from_expected_hole_m",
                ],
            )
            writer.writeheader()
            for row in per_facet:
                writer.writerow(
                    {
                        "facet_index": row["facet_index"],
                        "n_adjacent_cells": row["n_adjacent_cells"],
                        "adjacent_volume_tags": ",".join(str(t) for t in row["adjacent_volume_tags"]),
                        "touches_air_tag_10": row["touches_air_tag_10"],
                        "touches_wood_only": row["touches_wood_only"],
                        "classification": row["classification"],
                        "dist_xy_from_expected_hole_m": row.get("dist_xy_from_expected_hole_m", ""),
                    }
                )
        paths["facet_table_csv"] = str(csv_path)

    gmsh_script = out_dir / "VIEWING.md"
    gmsh_script.write_text(
        "\n".join(
            [
                "# Mesh tagging visualization",
                "",
                "## ParaView",
                f"- Open `{paths.get('volume_tags_xdmf', 'validation_mesh_volume_tags.xdmf')}` "
                "→ color by `physical_tag` (air=10, wood volumes 1–3).",
                f"- Open `{paths.get('shell_facets_xdmf', 'shell_facets_tags_1_2_3_4.xdmf')}` "
                "→ tag 1=top, 2=soundhole, 3=back, 4=ribs.",
                f"- Open `{paths.get('soundhole_facets_tag2_xdmf', 'soundhole_facets_tag2.xdmf')}` "
                "→ all Gmsh physical tag-2 triangles.",
                f"- Table `{paths.get('facet_table_csv', 'soundhole_facet_adjacency_table.csv')}` "
                "→ dolfinx per-facet air/wood adjacency (authoritative for diagnosis).",
                "",
                "## Gmsh",
                f"gmsh {mesh_path.name}  # from experiment mesh directory",
                "Visibility → Physical Groups → Surface 2 (Soundhole), Volume 10 (Air_Internal).",
                "",
            ]
        ),
        encoding="utf-8",
    )
    paths["viewing_notes"] = str(gmsh_script)
    return paths


def _write_markdown_report(summary: Dict[str, Any], md_path: Path) -> None:
    fc = summary["facet_counts"]
    fr = summary["fractions"]
    dof = summary["pressure_dof_audit"]
    diag = summary["diagnosis"]
    impl = summary["coupled_baseline_implication"]
    lines = [
        "# Soundhole ↔ air adjacency audit",
        "",
        f"Mesh: `{summary['mesh_file']}`",
        "",
        "## Facet tag 2 summary",
        "",
        f"| Metric | Count |",
        f"|--------|------:|",
        f"| Total tag-2 facets | {fc['tag2_total']} |",
        f"| Adjacent to air volume (tag 10) | {fc['tag2_adjacent_to_air_10']} |",
        f"| Adjacent to wood only (no air) | {fc['tag2_adjacent_wood_only']} |",
        f"| Other non-air | {fc['tag2_adjacent_other_non_air']} |",
        f"| No adjacent volume | {fc['tag2_no_adjacent_volume']} |",
        "",
        f"- Fraction air-adjacent: **{fr['adjacent_to_air']:.4f}**",
        f"- Fraction wood-only: **{fr['adjacent_wood_only']:.4f}**",
        "",
        "## Pressure DOF overlap (same maps as acoustic-only solver)",
        "",
        f"- `n_p_full`: {dof['n_p_full']}",
        f"- `n_p_air_supported`: {dof['n_p_air_supported']}",
        f"- `n_p_soundhole_full`: {dof['n_p_soundhole_full']}",
        f"- `n_p_soundhole_in_air_subgraph`: **{dof['n_p_soundhole_in_air_subgraph']}**",
        "",
        "## Diagnosis",
        "",
        f"**Primary hypothesis: {diag['primary_hypothesis']}** — {diag['narrative']}",
        "",
        f"Codes considered: {', '.join(diag['hypothesis_codes'])}",
        "",
        "## Coupled baseline implication",
        "",
        f"- {impl['coupled_pressure_release']}",
        f"- {impl['effectiveness']}",
        f"- {impl['facet_air_fraction']}",
        "",
        "## Geometry (tag-2 centroids)",
        "",
        json.dumps(summary.get("geometry_soundhole_tag2", {}), indent=2),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Soundhole–air adjacency audit")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=EXPERIMENT_ROOT / "mesh" / "validation_tiny_guitar_3d.msh",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PHYSICS_ROOT / "diagnostics" / "soundhole_air_audit",
    )
    parser.add_argument(
        "--skip-xdmf",
        action="store_true",
        help="Skip meshio XDMF export (topology audit still runs)",
    )
    args = parser.parse_args()

    mesh_path = args.mesh.resolve()
    if not mesh_path.is_file():
        print(f"[audit] Mesh not found: {mesh_path}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        summary, _msh, _ct, _ft = _audit_dolfinx(mesh_path)
    except Exception as exc:
        print(f"[audit] dolfinx audit failed: {exc}", file=sys.stderr)
        return 2

    per_facet = summary.pop("per_facet_all", [])
    json_path = args.out_dir / "soundhole_air_adjacency_report.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown_report(summary, args.out_dir / "soundhole_air_adjacency_report.md")

    viz_paths = {}
    if not args.skip_xdmf:
        try:
            viz_paths = _export_visualization_meshio(mesh_path, args.out_dir, per_facet)
        except Exception as exc:
            summary["visualization_error"] = str(exc)
            json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["visualization"] = viz_paths
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    diag = summary["diagnosis"]
    fc = summary["facet_counts"]
    dof = summary["pressure_dof_audit"]
    print("[soundhole_air_audit] Done.")
    print(
        f"  tag2_total={fc['tag2_total']} air_adjacent={fc['tag2_adjacent_to_air_10']} "
        f"wood_only={fc['tag2_adjacent_wood_only']}"
    )
    print(
        f"  soundhole_p: full={dof['n_p_soundhole_full']} "
        f"in_air_subgraph={dof['n_p_soundhole_in_air_subgraph']}"
    )
    print(f"  primary_hypothesis={diag['primary_hypothesis']}: {diag['narrative']}")
    print(f"  report={json_path}")
    for k, v in viz_paths.items():
        print(f"  {k}={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
