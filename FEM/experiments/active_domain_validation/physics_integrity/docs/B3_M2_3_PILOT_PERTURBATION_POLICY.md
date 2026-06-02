# B3 M2.3 - physical perturbation policy for 3-sample pilot (planning-only)

## 1. Purpose and constraints

Define a safe, near-baseline perturbation policy so the M2.1 3-sample pilot becomes physically meaningful (not placeholder-only) while keeping risk low and interpretation clear.

This document is planning/spec only:

- no Stage A/B/C execution,
- no runtime manifest creation,
- no orchestrator execution implementation,
- no cleanup/move/archive actions.

Fixed pilot constraints:

- `mesh_level = L_prod` for all 3 samples,
- `target_set = full9` for all 3 samples,
- low-dimensional perturbations (1 variable per sample for first physical pass).

## 2. Safest-first perturbation strategy (recommended)

Recommendation for the first physical pilot:

- **Use the same baseline L_prod mesh for all 3 samples (no per-sample remesh).**
- Apply **material/config perturbations only** in this first physical pass.

Why this is safest:

- isolates physics sensitivity from meshing variability,
- keeps A/B/C behavior directly comparable to validated baseline flow,
- minimizes risk of introducing geometry-induced discretization artifacts too early.

## 3. Safe-to-perturb first (near baseline)

Use small changes (about 1-3%) in parameters that do not require geometry regeneration:

1. `materials.top.density`
2. `materials.back.density`
3. `materials.top.E_L` (if needed in a later pilot, but not combined with many other changes)
4. `solver.fsi_coupling_gain` (small adjustment only; solver/config perturbation)

For this first 3-sample pilot, prefer only density perturbations:

- physically interpretable,
- low coupling to solver stability compared with broad stiffness tensor edits,
- no mesh rebuild required.

## 4. Do-not-perturb-yet list

Defer these until after first physical mini-pilot PASS:

Geometry-affecting (requires remesh, defer for now):

- `geometry.length`
- `geometry.width`
- `geometry.depth`
- `geometry.hole_radius`
- `geometry.top_thickness`
- `geometry.back_thickness`

High-risk multi-parameter / solver behavior:

- simultaneous changes to multiple orthotropic constants (`E_*`, `G_*`, `nu_*`) in one sample,
- harvest window shifts (`_worker_harvest_lo_hz`, `_worker_harvest_hi_hz`) during first physical pilot,
- solver algorithm toggles (ST/EPS strategy flags, filtering policy toggles).

## 5. Concrete payloads for current 3 samples

All values are relative to current baseline in `coupled_physical_core_v2.json`.

### `lhs_pilot_001_timing`

- intent: timing baseline with physically meaningful slight softening of top mass
- perturbation:
  - `material_delta.top.density = 445.5` (baseline `450.0`, about -1.0%)
- mesh regeneration: **not required** (material-only)

### `lhs_pilot_002_timing`

- intent: second timing point with slight increase in back mass
- perturbation:
  - `material_delta.back.density = 842.45` (baseline `830.0`, about +1.5%)
- mesh regeneration: **not required** (material-only)

### `lhs_pilot_003_synthesis`

- intent: synthesis subset with slight top mass increase
- perturbation:
  - `material_delta.top.density = 456.75` (baseline `450.0`, about +1.5%)
- mesh regeneration: **not required** (material-only)

## 6. Mesh regeneration policy for M2.3

Decision for first physical pilot:

- **Do not regenerate mesh per sample.**
- Reuse the same `L_prod` mesh source and meshing controls.

When remesh becomes required (future phase):

- any non-empty `geometry_delta` that changes CAD/shape dimensions,
- any change to meshing controls in `v2_mesh_convergence_manifest.json`.

## 7. Mapping rules to config/manifests and Stage A input

### 7.1 `guitar_3d.json`

Role in M2.3:

- remains baseline/reference for canonical geometry/material defaults and UI-facing config.

Mapping rule:

- `parameter_payload.geometry_delta` and `parameter_payload.material_delta` are interpreted as per-sample overrides relative to baseline values represented by canonical config state.
- For this first physical pilot, `geometry_delta` remains empty in all samples.

### 7.2 `v2_mesh_convergence_manifest.json`

Role in M2.3:

- authoritative mesh-level policy (`L_prod`) and meshing controls.

Mapping rule:

- sample must keep `mesh_level = L_prod`.
- no per-sample mesh control override in M2.3 first pass.
- if future sample has non-empty geometry delta, mark `requires_mesh_regeneration = true` in planning payload and regenerate mesh before Stage A.

### 7.3 `coupled_physical_core_v2.json`

Role in M2.3:

- baseline operational physics/solver config for Stage A input preparation.

Mapping rule:

- apply `material_delta` as a shallow-per-field override onto baseline material subtree.
- avoid broad solver flag overrides in first physical pilot.
- keep target policy unchanged (`target_set = full9`).

### 7.4 Stage A inputs

Stage A should receive an effective sample config equivalent to:

1. baseline config (`coupled_physical_core_v2.json`),
2. plus allowed per-sample deltas from pilot row payload,
3. with `mesh_level = L_prod` and unchanged target policy.

For M2.3 first pass:

- Stage A input differences are material-only,
- no CAD/mesh regeneration branch should be triggered.

## 8. Suggested JSONL payload shape (planning contract)

Per row `parameter_payload` should follow:

```json
{
  "note": "m2_3_near_baseline_material_only",
  "geometry_delta": {},
  "material_delta": {
    "top": {"density": 445.5}
  },
  "requires_mesh_regeneration": false
}
```

Use only fields relevant to each sample (top or back density in this M2.3 set).

## 9. Final recommendation

Safest first physical pilot strategy:

- **Material-only, single-variable, near-baseline perturbations on shared `L_prod` mesh** across all 3 samples.

This gives a physically meaningful mini-LHS step with minimal new risk before introducing geometry perturbations and per-sample remeshing in a later phase.
