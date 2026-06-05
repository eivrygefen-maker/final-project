# M4 mode dominant-region metadata (lightweight)

Per accepted ST mode, the worker solve path records which subsystem dominates (top / back / air) for later STK/audio damping — **without** rich modal export or stored mode shapes.

## Raw scores vs normalized shares

| Field | Type | Meaning |
|-------|------|---------|
| `top_participation` | float \| null | **Legacy name.** Raw score ‖x[u_idx_top]‖² / ‖x‖² (not a normalized fraction). |
| `back_participation` | float \| null | **Legacy name.** Raw score ‖x[u_idx_back ∪ u_idx_ribs]‖² / ‖x‖² |
| `air_participation` | float \| null | **Legacy name.** Raw score ‖x[p_idx_air]‖² / ‖x‖² |
| `top_participation_score` | float \| null | Alias of `top_participation` (aggregation catalog). |
| `back_participation_score` | float \| null | Alias of `back_participation` (aggregation catalog). |
| `air_participation_score` | float \| null | Alias of `air_participation` (aggregation catalog). |
| `participation_scores_semantics` | string | Documents that scores overlap and may sum to > 1. |

Scores are **non-partitioning energy fractions** (regions overlap). Values may exceed 1.0 individually or in sum.

## Normalized shares (STK damping weights)

Computed at **aggregation** from raw scores (no re-solve):

```
top_share  = top_score  / (top_score + back_score + air_score)
back_share = back_score / (top_score + back_score + air_score)
air_share  = air_score  / (top_score + back_score + air_score)
```

| Field | Type | Meaning |
|-------|------|---------|
| `top_share` | float \| null | Normalized share in [0, 1]; sums to 1 with siblings when scores present. |
| `back_share` | float \| null | Same |
| `air_share` | float \| null | Same |
| `share_denominator` | float | Sum of available scores (0 if none). |

## Classification fields

| Field | Type | Meaning |
|-------|------|---------|
| `dominant_region` | `top` \| `back` \| `air` \| `unknown` | Argmax of **normalized shares** (display/summary label). |
| `secondary_region` | `top` \| `back` \| `air` \| null | Second-largest share ≥ 0.1, excluding dominant. |
| `coupling_class` | string | Mixed/dominant STK routing hint (see rules below). |
| `participation_method` | string | How fractions were computed |
| `participation_status` | `computed` \| `fallback` \| `not_available` | |
| `participation_detail` | string | e.g. `region_dof_source` or missing-data reason |

### `coupling_class` rules

```
if top_share >= 0.25 and back_share >= 0.25:
    coupling_class = "top_back_mixed"
elif air_share >= 0.5:
    coupling_class = "air_dominant"
elif dominant_region in (top, back, air):
    coupling_class = f"{dominant_region}_dominant"
else:
    coupling_class = "weak_or_unknown"
```

## STK / audio damping guidance

**Do not use hard `dominant_region` alone for damping.**

Weight Q/damping contributions by `top_share`, `back_share`, `air_share`. Use `coupling_class` for routing hints (e.g. `top_back_mixed` modes need blended wood damping). `dominant_region` is a summary label only.

For excitation/output scalars (`bridge_excitation_coupling`, `radiation_proxy`, `mic_output_proxy`, `modal_norm`), see [B3_M4_MODE_AUDIO_COUPLING_METADATA.md](B3_M4_MODE_AUDIO_COUPLING_METADATA.md).

## Where it is written

| Artifact | Path |
|----------|------|
| Solver | `worker_results/<chunk>/solver_result.json` → `targets[].accepted_modes[]` (raw scores only) |
| Worker | `worker_results/<chunk>/worker_result.json` → `accepted_mode_records[]` |
| Aggregation | `aggregation/modes_catalog.jsonl` (scores + shares + coupling), `modes_summary.json` |

### `modes_summary.json` aggregation fields

| Field | Meaning |
|-------|---------|
| `dominant_region_counts` | Argmax counts on normalized shares (same as `normalized_dominant_region_counts`) |
| `normalized_dominant_region_counts` | Explicit alias for share-based argmax |
| `coupling_class_counts` | Histogram of `coupling_class` |
| `share_summary` | median/mean/max/min per share + `top_share_ge_0.25_count` |
| `stk_damping_guidance` | Short text reminder for downstream audio |

Re-aggregation on existing worker results refreshes shares without re-running the solver.

## Implementation

- Module: `scripts/v2_b3_mode_region_participation.py`
  - Solve: raw scores via `compute_mode_dominant_region_metadata()`
  - Aggregation: `enrich_participation_catalog_metadata()`, `summarize_participation_shares()`
- Solve hook: `collect_accepted_st_modes()` in `v2_b3_st_sinvert_solver_lib.py` (uses `x_active` already in memory; **does not** set `export_vectors=True`)
- Aggregation: `v2_b3_m4_aggregate_worker_results.py` → `_enrich_catalog_participation()`
- Region indices: `load_region_dof_bundle()` from `v2_b3_rich_modal_lib.py`

## Root cause (sample_000 blocker — fixed)

`best_effort` was enabled, but export still looked up **`mesh_path(L_prod, baseline_coupled_v2)`** instead of the M4 mesh at `lprod/mesh/L_prod/<sample_id>.msh` from `lprod/resolved_core_config.json`. Missing baseline mesh → `deferred_to_stage_c`, no npz.

**Fix (mesh):** `resolve_region_dof_mesh_file()` uses per-sample `lprod/resolved_core_config.json` mesh path.

**Fix (indices):** B3 `u_idx = arange(n_u_b3)` — shell trace facet DOF rows map **directly** to W u-block rows (not via `parent_index_per_trace_dof`). Status `BEST_EFFORT_PASS`. **Back includes ribs** in `back_participation`.

**Fix (production path):** `best_effort` writes `region_dof_indices.npz` **in-process** from Stage A operator build (`built["region_dof_build"]` captured on trace shell mesh). No isolated PETSc/MPI subprocess. `region_dof_source=operator_build_context`. Indices are B3 W u-block rows `0..n_u_b3-1` (same layout as `x_full` in `collect_accepted_st_modes()`).

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
