# M4 production fallback and default audit

**Dataset marker:** `m4_geometry_corrected_v1`  
**Date:** 2026-06-02  
**Scope:** Production path only (validation trees unchanged).

## Summary

| Class | Count | Action |
|-------|-------|--------|
| UNSAFE_SILENT → FAIL_FAST / REMOVED | 8 | Fixed in this promotion |
| SAFE_PRODUCTION | 6 | Retained |
| SAFE_DIAGNOSTIC | 5 | Retained, env-gated |

---

## Findings

| Location | Fallback / default | Class | Resolution |
|----------|-------------------|-------|------------|
| `run_v2_B3_trace...audit.py` `_b3_build_corrected_structural_active_operators` | `mesh_path(mesh_level, baseline_coupled_v2)` when `operator_mesh_file` omitted | **UNSAFE_SILENT** | **REMOVED** for M4 L_prod: `v2_b3_checkpoint_export.py` requires `--operator-mesh-file` / `solver.mesh_file` |
| `v2_build_coupled_acoustic_seed.py` `_assemble_reduced_coupled_replay` | `sc["mesh_file"] = mesh_path` from operator build arg | SAFE_PRODUCTION | Correct when operator mesh is sample-specific |
| `v2_b3_synthesis_export.resolve_region_dof_mesh_file` | `baseline_fallback` candidate | **UNSAFE_SILENT** | **FAIL_FAST**: skipped when `operator_mesh_matches_generated=true` |
| `v2_b3_mode_audio_coupling._mic_output_from_proxies` | `cavity_pressure_max_proxy_v1` when `u_idx_soundhole` empty | **UNSAFE_SILENT** | **FAIL_FAST**: production prefers `p_idx_aperture`; `B3_REQUIRE_APERTURE_MASK=1` (default) fails sample |
| `v2_b3_mode_audio_coupling` structural soundhole mask | `u_idx_soundhole` displacement proxy | SAFE_DIAGNOSTIC | Not primary; aperture pressure preferred |
| `v2_b3_aperture_pressure_mask` `cfg.get(...) or []` on ndarray | NumPy truth-value crash / wrong branch | **UNSAFE_SILENT** | **REMOVED**: `_as_int32_index_map()` |
| `v2_b3_synthesis_export.export_region_dof_indices_from_operator_build` | export without `p_idx_aperture` | **UNSAFE_SILENT** | **FAIL_FAST**: returns `FAIL` if empty |
| `v2_b3_m4_lprod_checkpoint_run` region DOF warning | "production continues" on missing top/back masks | SAFE_PRODUCTION | Warn only; aperture mask now hard-required |
| `v2_b3_m4_lhs_production_batch` ROM prepredict | loads `ROM/classic` model | **UNSAFE_SILENT** | **FAIL_FAST**: `maybe_run_rom_prepredict` skips when `pool.dataset_version != m4_geometry_corrected_v1` |
| `lhs_pool.json` completion skip | `--skip-completed` reuses pre-fix runs | **UNSAFE_SILENT** | Migration resets 000–035 to pending; new `dataset_version` marker |
| `v2_b3_m4_aggregate_worker_results` | copies `mic_output_method` from solve (no recompute) | SAFE_PRODUCTION | Solve-time contract enforced |
| `resolve_lprod_core_config` | copies scout config + mesh path | SAFE_PRODUCTION | Now injects `geometry` + `dataset_version` |
| `v2_mesh_convergence_common.mesh_path` | baseline path helper | SAFE_DIAGNOSTIC | Retained for convergence studies, not M4 operator build |
| Checkpoint reuse across samples | identical CSR structure under baseline mesh | **UNSAFE_SILENT** | **REMOVED** via per-sample operator mesh |
| Aggregation raw vs deduped | 0.05 Hz dedupe in catalog | SAFE_PRODUCTION | Unchanged; audit duplicate accepts per chunk separately |
| Material / geometry defaults | `coupled_physical_core_v2.json` canonical materials | SAFE_PRODUCTION | Per-sample overlay via `resolved_core_config.json` |
| `B3_ALLOW_CAVITY_MAX_MIC_FALLBACK=1` | diagnostic cavity max proxy | SAFE_DIAGNOSTIC | Opt-in only; not for ROM-training production |

---

## Production contracts (post-promotion)

Required before sample `COMPLETED`:

```text
operator_mesh_matches_generated = true
geometry_fingerprint present
generated_mesh_sha256 present
active_dimension > 0
p_idx_aperture_count > 0
mic_output_method = aperture_pressure_rms_proxy_v1 (catalog modes)
aggregation PASS
dataset_version = m4_geometry_corrected_v1
```

Enforced by: `v2_b3_m4_production_contracts.evaluate_production_acceptance()` in production batch.

---

## Environment flags

| Variable | Default | Purpose |
|----------|---------|---------|
| `B3_REQUIRE_APERTURE_MASK` | `1` | Fail solve-time audio if `p_idx_aperture` empty |
| `B3_ALLOW_CAVITY_MAX_MIC_FALLBACK` | unset | Diagnostic only; allows legacy cavity max |
| `B3_DIAGNOSTIC_MIC_FALLBACK_ONLY` | unset | Validation / audit scripts |
| `M4_REQUIRED_ROM_DATASET_VERSION` | `m4_geometry_corrected_v1` | Block ROM prepredict on old pool |
