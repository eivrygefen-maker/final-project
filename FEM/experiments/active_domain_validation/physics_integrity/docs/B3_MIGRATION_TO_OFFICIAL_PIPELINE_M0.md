# B3 migration control (M0)

## 1. Purpose

Promote the validated A+B+C checkpoint/rich pipeline from experiment reference to the official path for future LHS, ROM, and STK workflows.

This document is a control spec for migration governance only. It does not authorize destructive actions.

## 2. Current validated reference (frozen)

- Stage A PASS checkpoint: `v2_mesh_convergence/diagnostics/st_worker_scaling_L_prod_rich_safe_20260601T164739Z`
- Stage B rich PASS solve: `v2_mesh_convergence/diagnostics/solver_benchmarks/checkpoint_solve_mkl_pardiso_full9_20260601T203438Z`
- Stage C PASS output: `.../checkpoint_solve_mkl_pardiso_full9_20260601T203438Z/rich_modal_post/`
- Validated values:
  - `mode_count = 115`
  - `schema = b3_rich_modal_post_v1`
- Official archive (verified): `~/final-project-archives/archive_official_A_B_C_rich_PASS_20260601T203438Z.tar.gz`

## 3. Official entrypoints (to be treated as canonical)

- `scripts/v2_b3_checkpoint_export.py`
- `scripts/v2_b3_checkpoint_solve.py`
- `scripts/v2_b3_checkpoint_solver_multi_benchmark.py`
- `scripts/v2_b3_rich_modal_post.py`
- `scripts/v2_b3_synthesis_export.py`
- `scripts/v2_b3_synthesis_region_dof_worker.py`
- `scripts/v2_b3_rich_modal_lib.py`
- `scripts/v2_b3_checkpoint_pipeline_lib.py`
- `scripts/v2_b3_checkpoint_metadata_lib.py`
- `scripts/v2_b3_operator_checkpoint_portable.py`
- `scripts/v2_b3_st_sinvert_solver_lib.py`
- `scripts/run_v2_B3_trace_coupled_operator_and_seed_transfer_audit.py` (current Stage A operator-build dependency)

## 4. Required inputs to reproduce Stage A

- `configs/v2_mesh_convergence_manifest.json`
- `v2_mesh_convergence/mesh/L_prod/`
- `configs/coupled_physical_core_v2.json` (minimal canonical version)
- Current Stage A dependency chain used by `v2_b3_checkpoint_export.py`

## 5. Future LHS policy (official intent)

- All LHS points:
  - Stage A checkpoint export
  - Stage B solve (timing/frequency/scalar summaries)
- Selected subset only:
  - Stage B rich export (`--B3-export-rich-modal-data`)
- Synthesis / ROM / STK subset:
  - Stage C rich modal post
- Rich export remains opt-in (not default global behavior)

## 6. Future output structure proposal

Use a normalized run-root for future wiring:

- `pipeline_runs/checkpoints/`
- `pipeline_runs/solves/`
- `pipeline_runs/synthesis/`
- `pipeline_runs/logs/`
- `pipeline_runs/manifests/`
- `pipeline_runs/archives/`

This is a migration target layout; it does not move current validated artifacts.

## 7. Legacy policy

- Old direct experiment runners remain legacy/diagnostic history.
- Old target density/alignment experiments are historical references.
- Old monolithic worker-scaling path is not the default official route.
- No deletion now. Archive/deprecate only after replacement wiring is complete and documented.

## 8. Cleanup relationship

- Cleanup must not drive migration decisions.
- Migration decisions determine what becomes obsolete.
- Official A+B+C PASS artifacts are P0 protected.
- `mesh/L_prod` and `v2_mesh_convergence_manifest.json` are P0 reproduction inputs.

## 9. Next migration phases

- **M1**: path normalization and wrapper entrypoints
- **M2**: LHS integration on Stage A/B default flow
- **M3**: ROM/STK synthesis ingestion from Stage C outputs
- **M4**: legacy deprecation and archive execution

## 10. Explicit safety rule

No destructive or structural action (delete, move, overwrite, large path migration, or archive pruning) without explicit review and approval.

---

Related references:

- `docs/B3_OFFICIAL_RICH_PIPELINE_COMMANDS.md`
- `docs/B3_RICH_MODAL_EXPORT_TODO.md`
- `environment/solver-mkl/README.md`
