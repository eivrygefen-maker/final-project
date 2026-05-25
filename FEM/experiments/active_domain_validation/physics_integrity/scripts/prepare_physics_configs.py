#!/usr/bin/env python3
"""Write physics_integrity JSON configs from the validation mesh profile."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PHYSICS_ROOT / "configs"
VALIDATION_MESH = (EXPERIMENT_ROOT / "mesh" / "validation_tiny_guitar_3d.msh").resolve()
SOURCE = REPO_ROOT / "FEM" / "configs" / "guitar_3d.json"

CASE_NAMES = (
    "coupled_nominal",
    "structural_only",
    "acoustic_only",
    "coupled_low_frequency",
)
CASE_SUBDIRS = ("logs", "results", "modes", "diagnostics", "timing", "sorting")

CONFIG_FILES = (
    "coupled_nominal_202hz.json",
    "structural_only.json",
    "acoustic_only.json",
    "coupled_low_frequency.json",
    "operator_audit.json",
)


def ensure_physics_integrity_output_dirs() -> None:
    """Create all case and comparison output trees before shell runners use tee."""
    for case in CASE_NAMES:
        for sub in CASE_SUBDIRS:
            (PHYSICS_ROOT / case / sub).mkdir(parents=True, exist_ok=True)
    (PHYSICS_ROOT / "comparison").mkdir(parents=True, exist_ok=True)
    (PHYSICS_ROOT / "comparison" / "plots").mkdir(parents=True, exist_ok=True)


def _base() -> dict:
    if not VALIDATION_MESH.is_file():
        print(
            f"[prepare_physics_configs] ERROR: validation mesh not found: {VALIDATION_MESH}",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = json.loads(SOURCE.read_text(encoding="utf-8"))
    geom = dict(cfg.get("geometry", {}))
    geom.update(
        {
            "shape_type": "Classical",
            "length": 0.48,
            "width": 0.325,
            "depth": 0.10,
            "top_thickness": 0.003,
            "back_thickness": 0.0033,
            "hole_radius": 0.047,
            "mesh_mode": "fom",
        }
    )
    solver = dict(cfg.get("solver", {}))
    solver.update(
        {
            "mesh_file": str(VALIDATION_MESH),
            "soundhole_bc": "pressure_release",
            "pressure_gauge": "none",
            "couple_fluid": True,
            "clamp_ribs": False,
            "adaptive_mode_sifter": False,
            "gnhep_block_frobenius_normalize": True,
            "eps_pin_fix_tag5": True,
            "eps_algebraic_bc_zero_columns": True,
            "eps_band_solver": "shift_invert",
            "eps_reject_sigma_spurious": False,
            "eps_reject_target_locked": False,
            "eps_reject_decoupled_u_only": False,
            "eps_harvest_allow_weak_coupling": True,
            "physics_integrity_capture": True,
            "st_pc_type": "lu",
            "eigs_maxiter": 3000,
        }
    )
    solver.pop("active_domain_experiment", None)
    return {"geometry": geom, "materials": cfg.get("materials", {}), "solver": solver}


def main() -> None:
    ensure_physics_integrity_output_dirs()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    base = _base()

    nominal = json.loads(json.dumps(base))
    nominal["solver"].update(
        {
            "num_modes": 8,
            "shift_invert_target_hz": 202.0,
            "eps_broad_search_hz": 46.0,
            "_worker_harvest_lo_hz": 156.0,
            "_worker_harvest_hi_hz": 248.0,
        }
    )
    (CONFIG_DIR / "coupled_nominal_202hz.json").write_text(
        json.dumps(nominal, indent=2), encoding="utf-8"
    )

    struct = json.loads(json.dumps(base))
    struct["solver"].update(
        {
            "structural_only_diagnosis": True,
            "couple_fluid": False,
            "structural_only_num_modes": 30,
            "structural_shift_target_hz": 202.0,
            "structural_expected_hz_min": 140.0,
            "structural_expected_hz_max": 260.0,
        }
    )
    (CONFIG_DIR / "structural_only.json").write_text(json.dumps(struct, indent=2), encoding="utf-8")

    acoustic = json.loads(json.dumps(base))
    acoustic["solver"].update(
        {
            "acoustic_cavity_only_diagnosis": True,
            "structural_only_diagnosis": False,
            "couple_fluid": False,
            "acoustic_cavity_num_modes": 24,
            "acoustic_shift_target_hz": 120.0,
            "acoustic_min_mode_hz": 60.0,
            "acoustic_max_mode_hz": 250.0,
        }
    )
    (CONFIG_DIR / "acoustic_only.json").write_text(json.dumps(acoustic, indent=2), encoding="utf-8")

    low = json.loads(json.dumps(nominal))
    low["solver"].update(
        {
            "num_modes": 8,
            "shift_invert_target_hz": 120.0,
            "eps_broad_search_hz": 80.0,
            "_worker_harvest_lo_hz": 60.0,
            "_worker_harvest_hi_hz": 200.0,
            "_worker_target_hz": 120.0,
        }
    )
    (CONFIG_DIR / "coupled_low_frequency.json").write_text(
        json.dumps(low, indent=2), encoding="utf-8"
    )

    audit = json.loads(json.dumps(nominal))
    audit["solver"]["num_modes"] = 0
    (CONFIG_DIR / "operator_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"[prepare_physics_configs] Wrote configs under {CONFIG_DIR}")
    print(f"[prepare_physics_configs] validation_mesh={VALIDATION_MESH}")
    for name in CONFIG_FILES:
        path = CONFIG_DIR / name
        mesh = json.loads(path.read_text(encoding="utf-8"))["solver"]["mesh_file"]
        ok = Path(mesh).resolve() == VALIDATION_MESH and Path(mesh).is_file()
        print(f"  {name}: mesh_file={mesh} exists={ok}")


if __name__ == "__main__":
    main()
