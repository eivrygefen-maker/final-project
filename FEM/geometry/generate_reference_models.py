#!/usr/bin/env python3
"""
Write placeholder STEP reference bodies into FEM/geometry/models/.

Run once before the first sketch/FOM mesh build if you do not have custom CAD yet:

    python3 FEM/geometry/generate_reference_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import gmsh

MODELS_DIR = Path(__file__).resolve().parent / "models"


def _write_step(name: str, builder) -> None:
    gmsh.model.add(name)
    builder(gmsh.model.occ)
    gmsh.model.occ.synchronize()
    out = MODELS_DIR / name
    gmsh.write(str(out))
    print(f"Wrote {out}")
    gmsh.model.remove()


def _classic(occ) -> None:
    """Nominal 0.50 x 0.36 x 0.10 m; neck at +x."""
    lx, ly, lz = 0.50, 0.36, 0.10
    # Rounded box proxy (valid closed solid for importShapes).
    occ.addBox(-lx / 2.0, -ly / 2.0, -lz / 2.0, lx, ly, lz)


def _acoustic(occ) -> None:
    """Nominal dreadnought proxy: slightly wider."""
    lx, ly, lz = 0.50, 0.40, 0.10
    occ.addBox(-lx / 2.0, -ly / 2.0, -lz / 2.0, lx, ly, lz)


def _box(occ) -> None:
    lx, ly, lz = 0.48, 0.37, 0.10
    occ.addBox(-lx / 2.0, -ly / 2.0, -lz / 2.0, lx, ly, lz)


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    gmsh.initialize([sys.argv[0], "-nopopup"])
    try:
        _write_step("classic.step", _classic)
        _write_step("acoustic.step", _acoustic)
        _write_step("box.step", _box)
    finally:
        gmsh.finalize()
    print(f"Reference models ready in {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
