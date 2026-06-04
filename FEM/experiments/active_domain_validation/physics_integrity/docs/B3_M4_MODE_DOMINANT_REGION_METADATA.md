# M4 mode dominant-region metadata (lightweight)

Per accepted ST mode, the worker solve path records which subsystem dominates (top / back / air) for later STK/audio damping — **without** rich modal export or stored mode shapes.

## Fields (per mode)

| Field | Type | Meaning |
|-------|------|---------|
| `dominant_region` | `top` \| `back` \| `air` \| `unknown` | Argmax of available participation fractions |
| `top_participation` | float \| null | ‖x[u_idx_top]‖² / ‖x‖² |
| `back_participation` | float \| null | ‖x[u_idx_back ∪ u_idx_ribs]‖² / ‖x‖² |
| `air_participation` | float \| null | ‖x[p_idx_air]‖² / ‖x‖² |
| `participation_method` | string | How fractions were computed |
| `participation_status` | `computed` \| `not_available` | |
| `participation_detail` | string | e.g. `region_dof_source` or missing-data reason |

## Where it is written

| Artifact | Path |
|----------|------|
| Solver | `worker_results/<chunk>/solver_result.json` → `targets[].accepted_modes[]` |
| Worker | `worker_results/<chunk>/worker_result.json` → `accepted_mode_records[]` |
| Aggregation | `aggregation/modes_catalog.jsonl`, `modes_summary.json` (`dominant_region_counts`) |

## Implementation

- Module: `scripts/v2_b3_mode_region_participation.py`
- Solve hook: `collect_accepted_st_modes()` in `v2_b3_st_sinvert_solver_lib.py` (uses `x_active` already in memory; **does not** set `export_vectors=True`)
- Region indices: `load_region_dof_bundle()` from `v2_b3_rich_modal_lib.py`

## Root cause (sample_000 blocker — fixed)

`best_effort` was enabled, but export still looked up **`mesh_path(L_prod, baseline_coupled_v2)`** instead of the M4 mesh at `lprod/mesh/L_prod/<sample_id>.msh` from `lprod/resolved_core_config.json`. Missing baseline mesh → `deferred_to_stage_c`, no npz.

**Fix (mesh):** `resolve_region_dof_mesh_file()` uses per-sample `lprod/resolved_core_config.json` mesh path.

**Fix (indices):** B3 `u_idx = arange(n_u_b3)` — shell trace facet DOF rows map **directly** to W u-block rows (not via `parent_index_per_trace_dof`). Status `BEST_EFFORT_PASS`. **Back includes ribs** in `back_participation`.

**Fix (imports):** isolated `v2_b3_synthesis_region_dof_worker.py` bootstraps `FEM/scripts` on `sys.path` / `PYTHONPATH` (same as B3 audit) so `import fem_main_3d` works in the production subprocess.

## Availability estimate

| Input | Available in target-list solve? |
|-------|--------------------------------|
| Accepted mode vector `x_active` | **Yes** — built for every converged mode before acceptance |
| `built` row maps (`u_idx`, `p_idx`, `free_rows`, …) | **Yes** — from checkpoint `built_metadata.json` |
| Top/back facet DOF indices | **Yes** when npz exists (per-sample mesh) |

`region_dof_indices.npz` is produced when L_prod checkpoint runs `--B3-synthesis-region-dofs best_effort` (**M4 default:** `LPROD_SYNTHESIS_REGION_DOFS_DEFAULT`).

If the npz is missing:

- `air_participation` may still be computed from `p_idx` in built metadata.
- `top_participation` / `back_participation` are null; `dominant_region` may be `air` or `unknown`.
- Solver logs a one-line warning at chunk start.

## Methods

| `participation_method` | When |
|------------------------|------|
| `structural_pressure_energy_fraction_v1` | npz present (top, back+ribs, air) |
| `pressure_energy_fraction_v1` | pressure indices only |
| `pressure_displacement_norm_proxy_v1` | fallback from stored norms only (no vector) |
| `not_available` | no indices and no norms |

Reused checkpoints created before this default may lack the npz; re-run Stage 4 with `--force` or use worker air/norm fallback until refreshed.

No heavy post-process or Stage C required.
