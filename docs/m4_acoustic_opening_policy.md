# M4 acoustic opening / sound-hole policy

**Scope:** FOM/M4 production — not STK.

## Where the fix lives

The production sound-hole / active acoustic domain fix is **global** to M4 strict production, not classical-only:

| Component | Path | Role |
|-----------|------|------|
| Aperture mask export | `v2_b3_aperture_pressure_mask.py` | Builds `p_idx_aperture` from facet-adjacent air DOFs |
| Checkpoint region DOFs | `run_v2_B3_trace_coupled_operator_and_seed_transfer_audit.py` | Exports `region_dof_indices.npz` with aperture indices |
| Production contracts | `v2_b3_m4_production_contracts.py` | Requires `p_idx_aperture_count > 0` when `B3_REQUIRE_APERTURE_MASK=1` (default in strict mode) |
| Mic proxy | `v2_b3_m4_production_contracts.py` | `PRODUCTION_MIC_METHOD = aperture_pressure_rms_proxy_v1` |
| Aggregation | `v2_b3_m4_aggregate_worker_results.py` | Audio coupling uses aperture-based proxy |

Environment flag: `B3_REQUIRE_APERTURE_MASK` (default `"1"` in strict production).

## Per-shape policy (registry)

Defined in `FEM/scripts/m4_shape_registry.py` → `M4ShapeConfig.acoustic_opening_policy()`:

| Shape | `has_soundhole` | `requires_aperture_mask` | Notes |
|-------|-----------------|--------------------------|-------|
| classic | yes | yes | `geometry.hole_radius` in LHS |
| box | yes | yes | `geometry.hole_radius` swept in LHS |
| acoustic | yes | yes | `geometry.hole_radius` swept; GMSH uses `acoustic.step` |

All shapes use the **same** global aperture selection method: `facet_adjacent_air_cell_dofs_v1`.

## Manifest fields

Each `sample/sample_input.json` includes:

```json
"acoustic_opening_policy": {
  "shape_key": "...",
  "has_soundhole": true,
  "requires_aperture_mask": true,
  "aperture_selection_method": "facet_adjacent_air_cell_dofs_v1",
  "production_mic_method": "aperture_pressure_rms_proxy_v1",
  "soundhole_note": "..."
}
```

`run_m4_production_pipeline.py` logs `acoustic_opening_policy=...` at batch start.

## Static validation (no FEM)

Smoke test `tools/run_shape_fom_smoke.sh` asserts all three shapes expose `requires_aperture_mask` and `aperture_selection_method` via the registry.

## Historical note

See `FEM/experiments/active_domain_validation/physics_integrity/docs/M4_OPERATOR_MESH_AND_SOUNDHOLE_ROOT_CAUSE.md` for the pre-fix root cause (cavity-max mic fallback, zero aperture DOFs). Current production path requires aperture mask export and rejects `cavity_pressure_max_proxy` for corrected datasets.
