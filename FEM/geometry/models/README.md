# CAD reference models (STEP)

The mesh builder loads **closed B-rep solids** from this folder instead of procedural 2D perimeters.

## Required files

| File | `shape_type` in config | Nominal size (m) | Notes |
|------|------------------------|------------------|-------|
| `classic.step` | `Classical` (default) | 0.50 × 0.36 × 0.10 (L×W×D) | Torres / classical body |
| `acoustic.step` | `Dreadnought`, `Acoustic` | 0.50 × 0.40 × 0.10 | Broader shoulders |
| `box.step` | `Box` | 0.48 × 0.37 × 0.10 | Rectangular debug body |

**Frame convention (mandatory):**

- **+x** = neck end (`x = +L/2` after scaling)
- **−x** = tail (`x = −L/2`)
- **y** = lateral half-width
- **z** = plate normal (top ≈ `+z`)

Export from FreeCAD, Onshape, or Fusion as STEP AP214. The builder scales each model to the simulator `length`, `width`, and `depth` with `gmsh.model.occ.dilate`.

## Generate placeholders (no external CAD)

From the repo root (with `gmsh` in your venv):

```bash
python3 FEM/geometry/generate_reference_models.py
```

This writes `classic.step`, `acoustic.step`, and `box.step` into this directory:

- **classic** / **acoustic**: fixed classical / dreadnought point templates (flat neck & tail, B-spline bouts), extruded to a solid. No polar or Gaussian envelope math.
- **box**: simple rectangular solid for debug.

Replace with external CAD later if you need higher-fidelity bracing or arching.

## Sketch vs engineering

- **`mesh_mode=sketch`** (`FEM_ALLOW_PREVIEW=1`): import + scale only (fast GUI preview).
- **Display / FOM**: additionally hollow shell (`outer − inner`) and soundhole `occ.cut` on the reference solid.
