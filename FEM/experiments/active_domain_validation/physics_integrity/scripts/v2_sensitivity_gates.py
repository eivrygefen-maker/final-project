#!/usr/bin/env python3
"""Run mesh gates for a v2 sensitivity sample (no eigen solve)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PHYSICS_ROOT / "scripts"


def run_mesh_gates(
    mesh_path: Path,
    *,
    hole_radius_m: float,
    gates_dir: Path,
) -> Dict[str, Any]:
    gates_dir.mkdir(parents=True, exist_ok=True)
    aperture_dir = gates_dir / "soundhole_aperture_audit"
    adjacency_dir = gates_dir / "soundhole_air_audit"
    aperture_dir.mkdir(parents=True, exist_ok=True)
    adjacency_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    aperture_rc = subprocess.run(
        [
            py,
            str(SCRIPT_DIR / "audit_soundhole_aperture_geometry.py"),
            "--mesh",
            str(mesh_path.resolve()),
            "--hole-radius",
            str(float(hole_radius_m)),
            "--out-dir",
            str(aperture_dir),
        ],
        cwd=str(REPO_ROOT),
        check=False,
    ).returncode
    adjacency_rc = subprocess.run(
        [
            py,
            str(SCRIPT_DIR / "audit_soundhole_air_adjacency.py"),
            "--mesh",
            str(mesh_path.resolve()),
            "--out-dir",
            str(adjacency_dir),
            "--skip-xdmf",
        ],
        cwd=str(REPO_ROOT),
        check=False,
    ).returncode

    aperture_json = aperture_dir / "soundhole_aperture_geometry_report.json"
    adjacency_json = adjacency_dir / "soundhole_air_adjacency_report.json"
    aperture = (
        json.loads(aperture_json.read_text(encoding="utf-8"))
        if aperture_json.is_file()
        else {"gate_pass": False, "error": "missing aperture report"}
    )
    adjacency = (
        json.loads(adjacency_json.read_text(encoding="utf-8"))
        if adjacency_json.is_file()
        else {"error": "missing adjacency report"}
    )
    fc = adjacency.get("facet_counts") or {}
    dof = adjacency.get("pressure_dof_audit") or {}
    combined = (
        aperture_rc == 0
        and adjacency_rc == 0
        and bool(aperture.get("gate_pass"))
        and int(fc.get("tag2_adjacent_to_air_10", 0)) > 0
        and int(fc.get("tag2_adjacent_wood_only", 0)) == 0
        and int(dof.get("n_p_soundhole_in_air_subgraph", 0)) > 0
    )
    return {
        "mesh_file": str(mesh_path.resolve()),
        "expected_hole_radius_m": float(hole_radius_m),
        "aperture_audit_exit": int(aperture_rc),
        "adjacency_audit_exit": int(adjacency_rc),
        "aperture_gate_pass": bool(aperture.get("gate_pass")),
        "tag2_adjacent_to_air_10": int(fc.get("tag2_adjacent_to_air_10", 0)),
        "n_p_soundhole_in_air_subgraph": int(dof.get("n_p_soundhole_in_air_subgraph", 0)),
        "combined_mesh_gate_pass": bool(combined),
        "aperture_summary": {
            "tag2_total_area_m2": aperture.get("tag2_total_area_m2"),
            "area_ratio_vs_pi_r2": aperture.get("area_ratio_vs_pi_r2"),
            "radial_max_m": aperture.get("radial_max_m"),
        },
        "adjacency_diagnosis": (adjacency.get("diagnosis") or {}).get("primary_hypothesis"),
    }
