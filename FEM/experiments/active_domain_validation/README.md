# Active-domain validation experiment

Isolated comparison of **full-volume mixed** baseline vs **algebraic active-domain restriction**
on a shared coarse validation mesh. Safe to delete this entire folder when finished.

## Layout

- `configs/` — experiment JSON only (not production `FEM/SORTING`)
- `mesh/` — shared `validation_tiny_guitar_3d.msh` + `mesh_audit.json`
- `baseline/` — full-volume run outputs
- `active_domain/` — reduced-DOF run outputs
- `comparison/` — reports after `compare_results.py`
- `scripts/` — mesh prep, solves, comparison

## Opt-in production code

Enabled only when `solver.active_domain_experiment.enabled` is `true`:

- `FEM/scripts/fem_active_domain.py` (new)
- `FEM/scripts/fem_main_3d.py` — hook after BC application; eigenvector prolongation
- `FEM/scripts/fem_worker_single.py` — optional `bypass_worker_mode_cap` for experiments
- `FEM/geometry/build_3d_guitar.py` — `FEM_VALIDATION_MESH=1` experiment mesh profile only

## Validation mesh profile

Activated with `FEM_VALIDATION_MESH=1` (not used by production `FEM_ALLOW_FOM`):

- Wood surface lc ≈ 14 mm, air min ≈ 9 mm, air max ≈ 40 mm
- Soundhole band 12 mm (same idea as FOM — local, not volume-filling)

Target: **20k–80k nodes**. Inspect `mesh/mesh_audit.json` before long solves.

## Physical settings (both runs)

- `soundhole_bc = pressure_release`, `pressure_gauge = none`
- FSI enabled, tag-5 pin, 202 Hz shift, harvest [156, 248] Hz, `eps_broad_search_hz = 46`
- `num_modes = 8`

## Candidate method

**Option A — algebraic restriction:** assemble the same UFL operators on the parent mesh,
then extract the coupled subgraph of `A`/`M` containing air-pressure seeds, shell/FSI
displacement seeds, and Dirichlet rows. SLEPc solves the reduced system; eigenvectors are
prolongated to the full mixed layout for export and MAC comparison.

Operator reuse across frequency bands is **intentionally deferred** until this experiment passes.
