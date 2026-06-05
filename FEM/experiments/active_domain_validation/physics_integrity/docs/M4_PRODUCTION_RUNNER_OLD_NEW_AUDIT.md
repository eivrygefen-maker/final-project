# M4 production runner — old vs new audit

**Date:** 2026-06-03  
**Purpose:** Safe migration from legacy `RUN_PIPELINE` / M3 orchestration to permanent M4 production flow.  
**Rule:** Classify before delete. No source removal in this pass.

---

## Executive summary

| Flow | Role today | Decision |
|------|------------|----------|
| **`run_m4_production_pipeline.py`** | User-facing LHS → M4 E2E | **KEEP_ACTIVE** — default production entry |
| **`v2_b3_m4_lhs_production_batch.py`** | Batch executor (spec-driven) | **KEEP_ACTIVE** — called by new runner |
| **`v2_b3_m4_run_one_sample.py`** | Per-guitar stage orchestrator | **KEEP_ACTIVE** |
| **`run_pipeline.sh` + `FEM/scripts/run_pipeline.py`** | ROM snapshot / fem_master LHS | **DEPRECATE_WITH_WARNING** — different physics path |
| **M3 orchestrators / scout-only batches** | Pre-M4 validation | **DEPRECATE_WITH_WARNING** |

Validated production outputs (sample_000 repair, sample_005 clean run): `AGGREGATION_PASS`, participation + audio coupling metadata, `workers_actual_parallel=3`, `region_dof_indices.npz` from `operator_build_context`.

---

## 1. Old runner files and responsibilities

### Legacy ROM / FEM LHS (`RUN_PIPELINE`)

| File | Responsibility |
|------|----------------|
| `run_pipeline.sh` | Bash wrapper: marathon mode, env defaults, calls `run_pipeline.py` |
| `FEM/scripts/run_pipeline.py` | ROM workflow: fem_master → tuner → package_rom → update pool status in `lhs_samples.json` / pool |
| `FEM/scripts/rom_pipeline.py` | `collect` / `build-basis` for ROM snapshots |
| `FEM/rom/rom_manager.py` | LHS pool lifecycle (`pending` → `running` → `completed`) for ROM |
| `gui/app.py` | Streamlit UI; ROM collect/build, not B3 M4 physics |

**Not equivalent to M4:** no scout density zones, no adaptive L_prod target plan, no B3 checkpoint/workers/aggregation, no participation/audio coupling metadata.

### Legacy B3 orchestration (M2/M3)

| File | Responsibility |
|------|----------------|
| `v2_b3_m3_orchestrator_run_one.py` | M3.3 timing-only L_prod A+B (no full scout/adaptive) |
| `v2_b3_m3_orchestrator_dry_run.py` | M3.2 command preview |
| `v2_b3_run_coarse_scout_lhs_batch.py` | M3.4 overnight coarse scout batch |
| `v2_b3_lhs_orchestrator_preview.py` | M2 command preview |
| `v2_b3_frequency_coarse_planner.py` | Fixed coarse frequency plan |
| `v2_b3_coarse_mesh_scout_plan.py` | M3.4 scout planner dry-run |

### Legacy compatibility wrapper

| File | Responsibility |
|------|----------------|
| `v2_b3_lhs_production_run.py` | Thin delegate to M4 batch runner (stderr deprecation) |

---

## 2. New runner files and responsibilities

### User-facing production

| File | Responsibility |
|------|----------------|
| **`run_m4_production_pipeline.py`** | Read `ROM/classic/lhs_pool.json`, select pending samples, auto-generate specs, update status index, invoke batch runner |
| **`v2_b3_m4_lhs_pool_bridge.py`** | LHS → `m4_sample_input_v1`, batch spec builder, `lhs_pool_status.json`, `lhs_production_runs_index.jsonl` |

### Execution engine (unchanged core, hardened)

| File | Responsibility |
|------|----------------|
| `v2_b3_m4_lhs_production_batch.py` | Multi-sample sequential execute, batch summaries under `pipeline_runs/batches/` |
| `v2_b3_m4_run_one_sample.py` | Scout → worker plan → checkpoint → workers → aggregate → freeze |
| `v2_b3_m4_pipeline_run_scout.py` | Stages 0–3 |
| `v2_b3_m4_lprod_checkpoint_run.py` | Stage 4 |
| `v2_b3_m4_worker_run_remaining.py` | Stage 5 (FCFS parallel workers) |
| `v2_b3_m4_aggregate_worker_results.py` | Stage 6 + modes_summary / runtime_summary |
| `v2_b3_m4_freeze_first_e2e_run.py` | Post-PASS freeze milestone |

### Metadata modules (production-relevant)

| File | Responsibility |
|------|----------------|
| `v2_b3_mode_region_participation.py` | Participation scores + normalized shares + `coupling_class` |
| `v2_b3_mode_audio_coupling.py` | ROM/STK/audio coupling scalars |
| `v2_b3_m4_runtime_provenance.py` | `m4_sample_runtime_provenance.json` |

### Generated artifacts (gitignored)

| Path | Role |
|------|------|
| `pipeline_runs/index/lhs_pool_status.json` | Per-sample run state (source of truth for resume) |
| `pipeline_runs/index/lhs_production_runs_index.jsonl` | Append-only event log |
| `pipeline_runs/specs/generated/*.json` | Auto batch + per-sample production specs |
| `pipeline_runs/guitars/<sample>/runs/<run_id>/` | Full run tree |
| `pipeline_runs/batches/<batch_id>/` | Batch plan/summary/log |

**Immutable design pool:** `ROM/classic/lhs_pool.json` is not modified by M4 production.

---

## 3. Old features covered by new flow

| Old expectation | New coverage |
|-----------------|--------------|
| Run N LHS samples overnight | `--max-samples N --continue-on-fail` |
| Know which samples finished | `lhs_pool_status.json` + batch summary |
| Per-sample parameters | Auto from `lhs_pool.json` `entries[].parameters` |
| Frequency coverage 60–550 Hz | `frequency_policy` in generated batch spec |
| Parallel workers | `--workers 3`, FCFS, `workers_actual_parallel` in provenance |
| Mode catalog for ROM/STK | `modes_catalog.jsonl`, shares, audio coupling |
| Resume after interrupt | `--resume`, `skip-completed`, partial run trees |
| Skip completed samples | `--skip-completed` (default on) |
| Reference sample policy | `--exclude-reference` / `--include-reference` (no silent skip) |

---

## 4. Old features missing — port status

| Feature | Status | Notes |
|---------|--------|-------|
| GUI one-click production | **Not ported** | GUI still ROM path; needs separate bridge |
| Cross-guitar dashboard | **Future** | Batch summary JSON only |
| ROM pool `status` field sync | **Not ported** | M4 uses sidecar status; ROM pool stays design-only |
| `runs_index.jsonl` (M3 style) | **Replaced** | `lhs_production_runs_index.jsonl` |
| Marathon bash UX | **Replaced** | `run_m4_production_pipeline.py` batch loop |
| True bridge DOF mask | **Future** | `top_plate_proxy` for `bridge_excitation_*` |
| External mic sampling | **Future** | `mic_output_proxy` uses soundhole/pressure proxy |
| Mode plot coupling overlay | **Optional** | Plot exists; color-by-`coupling_class` not blocking |
| Parallel multi-guitar | **Not in v1** | Sequential samples; workers parallel within sample |

---

## 5. Files — KEEP_ACTIVE

- `run_m4_production_pipeline.py`
- `v2_b3_m4_lhs_pool_bridge.py`
- `v2_b3_m4_lhs_production_batch.py`
- `v2_b3_m4_run_one_sample.py`
- All M4 stage scripts listed in §2
- `v2_b3_mode_region_participation.py`, `v2_b3_mode_audio_coupling.py`
- `pipeline_runs/schemas/m4/`
- `pipeline_runs/specs/m4_lhs_production_batch.template.json`
- `pipeline_runs/specs/m4_5_small_lhs_batch_first3.json` (validation reference)
- `docs/B3_PIPELINE_RUNTIME_HYGIENE.md`

---

## 6. Files — KEEP_COMPAT_ALIAS

| File | Delegates to |
|------|----------------|
| `v2_b3_lhs_production_run.py` | `v2_b3_m4_lhs_production_batch.py` (stderr deprecation → prefer `run_m4_production_pipeline.py`) |

---

## 7. Files — DEPRECATE_WITH_WARNING

| File | Replacement |
|------|-------------|
| `run_pipeline.sh` | `run_m4_production_pipeline.py` (usage banner added) |
| `FEM/scripts/run_pipeline.py` | M4 production (ROM path still valid for ROM-only work) |
| `v2_b3_m3_orchestrator_run_one.py` | `v2_b3_m4_run_one_sample.py` |
| `v2_b3_m3_orchestrator_dry_run.py` | `v2_b3_m4_pipeline_dry_run.py` |
| `v2_b3_run_coarse_scout_lhs_batch.py` | `run_m4_production_pipeline.py` |
| `v2_b3_lhs_orchestrator_preview.py` | `v2_b3_m4_pipeline_dry_run.py` |
| `v2_b3_frequency_coarse_planner.py` | M4 adaptive `lprod_target_plan.json` |
| `v2_b3_coarse_mesh_scout_plan.py` | `v2_b3_m4_pipeline_run_scout.py` |

---

## 8. Files — REMOVE_RUNTIME_ARTIFACT (safe to delete/ignore)

| Path | Action |
|------|--------|
| `pipeline_runs/guitars/**` | Gitignored; delete locally to reclaim disk |
| `pipeline_runs/batches/**` | Gitignored |
| `pipeline_runs/specs/generated/**` | Gitignored; regenerable |
| `pipeline_runs/index/lhs_*` | Gitignored; regenerable from re-runs |
| `pipeline_runs/config_overlays/**` | Gitignored |
| Old M3/M4 dry-run trees no longer referenced | Archive/delete per hygiene doc |
| Hand-written `lhs_prod_m4_sample_*.json` on VM only | Superseded by auto-generated specs |

**Do not delete** committed specs/schemas/docs.

---

## 9. Files — DELETE_OBSOLETE_SOURCE (only after explicit approval)

| File | Reason | Risk if deleted early |
|------|--------|----------------------|
| `v2_b3_lhs_orchestrator_run.py` | Never implemented | None (absent) |
| M3 orchestrators | Superseded | Historical repro scripts |
| M3.4 scout batch | Superseded | Scout report comparisons |

**Recommendation:** Keep deprecated sources for one production LHS tranche (≥50 samples), then remove in a dedicated cleanup PR with grep verification.

---

## 10. Production command (canonical)

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --max-samples 10 \
  --workers 3 \
  --execute \
  --continue-on-fail
```

Dry-run plan:

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --max-samples 5 \
  --workers 3 \
  --dry-run
```

Force one sample (e.g. rerun sample_001 with new `run_id_suffix`):

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --force-sample sample_001 \
  --include-reference \
  --run-id-suffix m4prod1 \
  --workers 3 \
  --execute --force
```

---

## 11. Sample 1–4 policy

| Sample | Prior use | Production policy |
|--------|-----------|-------------------|
| `sample_000` | Repair/debug gate | Rerun with current pipeline + `m4prod1` |
| `sample_001` | Frozen reference (M4.5) | **Included by default**; use `--exclude-reference` to skip; `--include-reference` for explicit rerun |
| `sample_002`–`004` | M4.5 validation | Rerun cleanly; not excluded |

No silent exclusion of LHS rows 0–4.

---

## 12. Outputs preserved per sample

```
aggregation/modes_catalog.jsonl
aggregation/modes_summary.json
aggregation/runtime_summary.json
aggregation/aggregation_result.json
aggregation/mode_frequency_plot.png
aggregation/warnings_and_failures.json
lprod/checkpoint/region_dof_indices.npz
lprod/checkpoint/checkpoint_export_manifest.json
worker_results/*/solver_result.json
worker_results/*/worker_result.json
m4_sample_runtime_provenance.json
```

---

## 13. Changes in this integration pass

1. Added `run_m4_production_pipeline.py` + `v2_b3_m4_lhs_pool_bridge.py`
2. LHS sidecar status + runs index (gitignored)
3. Auto-generated specs under `pipeline_runs/specs/generated/`
4. Fixed reference exclusion: only when `exclude_from_batch` / `--exclude-reference`
5. Batch runner: `production_mode=True`, `force_stages` wired, audio coupling in summaries
6. Deprecation banners on legacy entry points

**Not changed:** FEM physics, solver math, Stage C default off, no full mode-shape storage.
