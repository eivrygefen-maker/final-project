# B3 M4 production promotion audit

**Date:** 2026-06-03  
**Status:** First safe integration pass (M4 validated E2E on `sample_001`–`sample_004`).  
**Principle:** Functional migration — replace responsibilities, not filenames. No blind deletes.

## Validation basis

| Sample | Result | Notes |
|--------|--------|-------|
| `sample_001` | AGGREGATION_PASS | Frozen reference; excluded from production batches |
| `sample_002`–`004` | AGGREGATION_PASS | M4.5 small-batch validation |

Pipeline stages validated: sample input → scout → density zones → adaptive L_prod targets → L_prod mesh/checkpoint → workers → aggregation → freeze.

---

## New production entrypoints (REPLACE_WITH_M4)

| Component | Path | Role |
|-----------|------|------|
| **Production batch runner** | `scripts/v2_b3_m4_lhs_production_batch.py` | Default multi-sample LHS execution |
| **Single-sample runner** | `scripts/v2_b3_m4_run_one_sample.py` | One guitar; `--production-mode` or `--m45-batch-mode` |
| **Compatibility wrapper** | `scripts/v2_b3_lhs_production_run.py` | Delegates to M4 batch runner |
| **Batch spec template** | `pipeline_runs/specs/m4_lhs_production_batch.template.json` | Copy for real 500+ LHS lists |
| **Runtime hygiene** | `docs/B3_PIPELINE_RUNTIME_HYGIENE.md` | Git/IDE ignore policy |

---

## Component audit

### M4 pipeline (production — KEEP_ACTIVE)

| Old path / component | Current role | M4 replacement / dependency | Decision | Reason | Risk | Action |
|---------------------|--------------|------------------------------|----------|--------|------|--------|
| — | Full LHS per guitar | `v2_b3_m4_run_one_sample.py` | **KEEP_ACTIVE** | Validated E2E orchestrator | Low | Default single-sample path |
| — | Multi-guitar LHS | `v2_b3_m4_lhs_production_batch.py` | **KEEP_ACTIVE** | Production batch | Low | Default batch path |
| — | Scout stages 0–3 | `v2_b3_m4_pipeline_run_scout.py` | **KEEP_ACTIVE** | Adaptive zones + targets | Low | Called by orchestrator |
| — | L_prod checkpoint | `v2_b3_m4_lprod_checkpoint_run.py` | **KEEP_ACTIVE** | Stage 4 | Low | |
| — | Worker execution | `v2_b3_m4_worker_run_remaining.py` | **KEEP_ACTIVE** | Stage 5 | Low | |
| — | Aggregation | `v2_b3_m4_aggregate_worker_results.py` | **KEEP_ACTIVE** | Stage 6 | Low | |
| — | Freeze / summary | `v2_b3_m4_freeze_first_e2e_run.py` | **KEEP_ACTIVE** | Post-PASS milestone | Low | |
| — | Scout planner lib | `v2_b3_m4_scout_planner_lib.py` | **KEEP_USED_BY_M4** | Zone/target math | Low | |
| — | L_prod interfaces | `v2_b3_m4_lprod_interfaces.py` | **KEEP_USED_BY_M4** | Chunk/mesh contracts | Low | |
| — | Worker lib | `v2_b3_m4_worker_run_lib.py` | **KEEP_USED_BY_M4** | Subprocess solver | Low | |
| — | Pipeline dry-run | `v2_b3_m4_pipeline_dry_run.py` | **KEEP_ACTIVE** | Planning only | Low | |
| — | Small batch dry-run | `v2_b3_m4_small_batch_dry_run.py` | **KEEP_ACTIVE** | Run-tree bootstrap | Low | Used by production batch |
| — | Schemas | `pipeline_runs/schemas/m4/` | **KEEP_ACTIVE** | Contracts | Low | Stay in Git |

### Stage primitives (KEEP_USED_BY_M4)

| Old path / component | Current role | M4 replacement | Decision | Reason | Risk | Action |
|---------------------|--------------|----------------|----------|--------|------|--------|
| `v2_b3_checkpoint_export.py` | Stage A export | Used by scout/checkpoint | **KEEP_USED_BY_M4** | Env-isolated export | Low | Keep |
| `v2_b3_checkpoint_solve_target_list.py` | Multi-target solve | Worker subprocess | **KEEP_USED_BY_M4** | Core solver path | Low | Keep |
| `v2_b3_checkpoint_solve.py` | Single-target solve | Worker primitive | **KEEP_USED_BY_M4** | | Low | Keep |
| `v2_b3_resolve_pilot_core_config.py` | Config overlay resolve | Stage 0 | **KEEP_USED_BY_M4** | | Low | Keep |
| `v2_b3_st_sinvert_solver_lib.py` | PETSc solver lib | Workers | **KEEP_USED_BY_M4** | | Low | Keep |
| `v2_b3_petsc_util.py` | JSON I/O utilities | All M4 scripts | **KEEP_USED_BY_M4** | | Low | Keep |
| `run_v2_B3_scout_coarse_mesh_build.py` | L_scout mesh build | Scout stage 1 | **KEEP_USED_BY_M4** | | Low | Keep |
| `v2_b3_m4_lprod_mesh_build.py` | L_prod mesh build | Checkpoint | **KEEP_USED_BY_M4** | | Low | Keep |
| `v2_b3_synthesis_region_dof_worker.py` | Region DOF export | Stage A subprocess | **KEEP_USED_BY_M4** | | Low | Keep |

### Legacy LHS orchestration (DEPRECATE_LEGACY → REPLACE_WITH_M4)

| Old path / component | Current role | M4 replacement | Decision | Reason | Risk | Action |
|---------------------|--------------|----------------|----------|--------|------|--------|
| `v2_b3_m3_orchestrator_run_one.py` | M3.3 L_prod timing A+B (no full scout/adaptive plan) | `v2_b3_m4_run_one_sample.py` | **DEPRECATE_LEGACY** | Superseded by M4 | Med | LEGACY stderr banner; do not use for new LHS |
| `v2_b3_m3_orchestrator_dry_run.py` | M3.2 dry-run preview | `v2_b3_m4_pipeline_dry_run.py` | **DEPRECATE_LEGACY** | Planning only | Low | Mark legacy in docs |
| `v2_b3_run_coarse_scout_lhs_batch.py` | M3.4 coarse scout overnight batch | M4 scout + production batch | **DEPRECATE_LEGACY** | Scout-only, fixed coarse mesh | Med | LEGACY banner; keep for historical scout reports |
| `v2_b3_lhs_orchestrator_preview.py` | M2 command preview | `v2_b3_m4_pipeline_dry_run.py` | **DEPRECATE_LEGACY** | No execution | Low | LEGACY banner |
| `v2_b3_lhs_orchestrator_run.py` | Planned M3 runner | `v2_b3_m4_lhs_production_batch.py` | **REMOVE_RUNTIME_ARTIFACT** | Never implemented | Low | N/A (file absent) |
| `v2_b3_frequency_coarse_planner.py` | Fixed coarse frequency plan | M4 adaptive `lprod_target_plan` | **DEPRECATE_LEGACY** | Fixed-plan route obsolete | Low | Keep for M3.4 demos only |
| `v2_b3_coarse_mesh_scout_plan.py` | M3.4 scout planner dry-run | `v2_b3_m4_pipeline_run_scout.py` | **DEPRECATE_LEGACY** | | Low | Non-default |

### M4 dev / validation (DEPRECATE_LEGACY for production)

| Old path / component | Current role | M4 replacement | Decision | Reason | Risk | Action |
|---------------------|--------------|----------------|----------|--------|------|--------|
| `v2_b3_m4_worker_smoke_test.py` | Single-chunk smoke | Production workers | **DEPRECATE_LEGACY** | Dev only | Low | Not production entry |
| `v2_b3_m4_worker_minibatch.py` | Partial worker test | `v2_b3_m4_worker_run_remaining.py` | **DEPRECATE_LEGACY** | Dev only | Low | |
| `v2_b3_m4_aggregation_dry_run.py` | Aggregation validator | `v2_b3_m4_aggregate_worker_results.py` | **DEPRECATE_LEGACY** | Dry-run only | Low | |
| `v2_b3_m4_lprod_worker_dry_run.py` | Worker plan dry-run | Orchestrator stage | **KEEP_USED_BY_M4** | Called in pipeline | Low | |

### Diagnostics / benchmarks (NEEDS_REVIEW — non-production)

| Old path / component | Current role | Decision | Action |
|---------------------|--------------|----------|--------|
| `v2_b3_st_worker_scaling_benchmark.py` | Dev benchmark | **NEEDS_REVIEW** | Keep; not LHS |
| `checkpoint_*_smoke.py`, `rich_modal_*`, `operator_*` | Stage diagnostics | **DEPRECATE_LEGACY** | Non-default |
| `v2_b3_pipeline_run_manifest.py` | M1 manifest helper | **KEEP_ACTIVE** | Infra |

### Specs and inputs (KEEP_ACTIVE)

| Path | Decision | Action |
|------|----------|--------|
| `pipeline_runs/specs/m4_5_small_lhs_batch_first3.json` | **KEEP_ACTIVE** | Validation reference |
| `pipeline_runs/specs/m4_lhs_production_batch.template.json` | **KEEP_ACTIVE** | Production template |
| `pipeline_runs/specs/m3_4_coarse_scout_lhs_batch.jsonl` | **DEPRECATE_LEGACY** | Historical scout list |
| `pipeline_runs/specs/m2_1_pilot_3_samples.jsonl` | **DEPRECATE_LEGACY** | M2 pilot |
| `pipeline_runs/specs/m3_2_dry_run_samples.jsonl` | **DEPRECATE_LEGACY** | M3 dry-run |
| `pipeline_runs/specs/previews/*.json` | **KEEP_ACTIVE** | Small intentional fixtures (<10 KB) |
| `pipeline_runs/specs/frequency_plans/m3_4_*` | **DEPRECATE_LEGACY** | Fixed-plan demos only |

### Runtime trees (REMOVE_RUNTIME_ARTIFACT — Git + IDE ignore)

| Path | Decision | Action |
|------|----------|--------|
| `pipeline_runs/guitars/**` | **REMOVE_RUNTIME_ARTIFACT** | `.gitignore`; `git rm --cached` done |
| `pipeline_runs/batches/**` | **REMOVE_RUNTIME_ARTIFACT** | Ignored; batch summaries local only |
| `pipeline_runs/scout_density_reports/**` | **REMOVE_RUNTIME_ARTIFACT** | Ignored |
| `pipeline_runs/config_overlays/**` | **REMOVE_RUNTIME_ARTIFACT** | Ignored; regenerable |
| `pipeline_runs/logs/**` | **REMOVE_RUNTIME_ARTIFACT** | Ignored |
| `**/worker_results/**`, `**/aggregation/**`, `**/freeze/**` | **REMOVE_RUNTIME_ARTIFACT** | Under run trees |
| `physics_integrity/*/results/`, `diagnostics/`, etc. | **REMOVE_RUNTIME_ARTIFACT** | Already in `.gitignore` |

**Do not** commit `aggregation_result.json`, `worker_result.json`, `modes_catalog.jsonl`, mesh `.msh`, or freeze manifests unless explicitly requested for a milestone export.

---

## Responsibility mapping (1-to-1 conceptual)

| Legacy responsibility | M4 implementation |
|----------------------|-------------------|
| Fixed coarse frequency plan | Adaptive `lprod_target_plan.json` from scout zones |
| Manual zone assignment | `density_zones.json` from scout discovery |
| M3.3 single-guitar L_prod timing | Full M4 pipeline with scout + workers + aggregation |
| M3.4 overnight scout-only batch | Scout sub-pipeline inside each M4 run |
| M2 command preview JSON | `v2_b3_m4_pipeline_dry_run.py` / production batch `--dry-run` |
| Ad-hoc per-stage manual commands | `v2_b3_m4_run_one_sample.py` or production batch |
| Stage C / rich modal / audio | **Not in default production** (unchanged policy) |

---

## First integration pass (completed)

1. Added `v2_b3_m4_lhs_production_batch.py` with required CLI.
2. Added `v2_b3_lhs_production_run.py` compatibility wrapper.
3. Extended `v2_b3_m4_run_one_sample.py` with `--production-mode`.
4. Legacy stderr banners on M2/M3 entry scripts.
5. Runtime hygiene enforced (see `B3_PIPELINE_RUNTIME_HYGIENE.md`).
6. Untracked `pipeline_runs/guitars/`, `batches/`, `scout_density_reports/` from Git index.

## Not done in this pass (intentional)

- Full 500-sample LHS spec generation (operator supplies `samples-json`).
- Deleting legacy source files (deprecated, not removed).
- Stage C, rich modal, cleanup, or promotion automation.
- Re-running validated samples.

---

## VM commands

**Dry-run slice (plan only):**

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_lhs_production_batch.py \
  --samples-json FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/specs/m4_lhs_production_batch.template.json \
  --batch-id lhs_prod_m4_001 \
  --start-index 0 \
  --max-samples 1 \
  --workers 3 \
  --dry-run
```

**Production execute (example — replace `samples-json` with your real LHS spec):**

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_lhs_production_batch.py \
  --samples-json FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/specs/<real_lhs_samples>.json \
  --batch-id lhs_prod_m4_001 \
  --start-index 5 \
  --max-samples 10 \
  --workers 3 \
  --execute \
  --continue-on-fail
```

**Single sample (same spec file):**

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_run_one_sample.py \
  --run-dir FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_005/runs/sample_005_m4prod1 \
  --execute --workers 3 \
  --production-mode \
  --production-samples-json FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/specs/<real_lhs_samples>.json
```

Batch summary: `pipeline_runs/batches/<batch_id>/batch_execution_summary.json` (local, gitignored).
