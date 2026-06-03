# B3 M4 — JSON schema contracts (M4.1)

Planning-only data contracts for the full LHS pipeline orchestrator. Authoritative stage flow: `B3_M4_FULL_LHS_PIPELINE_ORCHESTRATOR_CONTRACT.md`.

**Status:** M4.1 schemas done; M4.2 dry-run tree done; **M4.3** scout execution (`v2_b3_m4_pipeline_run_scout.py`) fills real `density_zones.json` and gapless `lprod_target_plan.json`.

**Schema root:** `pipeline_runs/schemas/m4/`  
**Examples:** `pipeline_runs/schemas/m4/examples/`  
**Validator:** `scripts/v2_b3_m4_schema_validate_examples.py`  
**Dry-run planner:** `scripts/v2_b3_m4_pipeline_dry_run.py`  
**Scout executor (Stages 0–3):** `scripts/v2_b3_m4_pipeline_run_scout.py`  
**L_prod worker dry-run (M4.4-pre / M4.4.1a):** `scripts/v2_b3_m4_lprod_worker_dry_run.py`  
**Target-list worker solve:** `scripts/v2_b3_checkpoint_solve_target_list.py`  
**Aggregation dry-run:** `scripts/v2_b3_m4_aggregation_dry_run.py`

---

## Policy constants (v1, frozen in examples)

| Policy | Value |
|--------|--------|
| Band | 60–550 Hz |
| Scout spacing / half-width | 7.5 / 3.75 Hz |
| Scout mesh | `L_scout_coarse` |
| L_prod mesh | `L_prod` |
| Zone spacing | ZONE_1_dense **6**, ZONE_2_medium **9**, ZONE_3_sparse **12.5** Hz |
| Chunk width | prefer 20–40 Hz; min ~15; max ~50; respect zone boundaries when practical |

Zone IDs (v1 only): `ZONE_1_dense`, `ZONE_2_medium`, `ZONE_3_sparse` — no transition zones.

---

## Schema index

| Schema file | Purpose | Produced by | Consumed by |
|-------------|---------|-------------|-------------|
| `sample_input.schema.json` | One LHS guitar row (geometry/material) | External pool / batch spec | Stage 0 resolve |
| `sample_resolved_config_manifest.schema.json` | Post-resolve paths, mesh slots, readiness | Stage 0 | Stages 1–6, manifest |
| `pipeline_run_manifest.schema.json` | Terminal run lifecycle + stage status | Orchestrator (updated each stage) | M4.2 dry-run, M4.5 batch reporting |
| `scout_result.schema.json` | Scout mesh + checkpoint + discovery summary | Stages 1–2 | Stage 3 zone planner |
| `density_zones.schema.json` | Binned density + zone segments | Stage 3 | `lprod_target_plan`, chunker |
| `lprod_target_plan.schema.json` | Gapless L_prod target grid | Stage 3 | Stage 5 workers, chunk plan |
| `worker_chunk_plan.schema.json` | FCFS chunk queue | Stage 3 (plan) / 5 (assign) | Workers |
| `worker_result.schema.json` | Per-chunk solver output | Stage 5 worker | Stage 6 aggregate |
| `aggregation_result.schema.json` | Deduped catalog + paths | Stage 6 | Manifest, batch rollup |
| `runtime_summary.schema.json` | Timing rollup | Stage 6+ / batch | Reporting (M4.5) |

---

## M4.2 dry-run — required fields

M4.2 writes `*.placeholder.json` files with `will_execute: false` and `status: pending_scout` / `pending_target_plan` where scout data is absent.

**Command (example):**

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_pipeline_dry_run.py \
  --sample-json FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/schemas/m4/examples/sample_input.example.json \
  --run-id sample_001_m4dry1 \
  --freq-min-hz 60 --freq-max-hz 550 \
  --scout-spacing-hz 7.5 --scout-half-width-hz 3.75 \
  --zone-spacing-dense-hz 6 --zone-spacing-medium-hz 9 --zone-spacing-sparse-hz 12.5 \
  --workers 3 --dry-run --force
```

Planning outputs (no solver execution):

1. **sample_input** — copied to `sample/sample_input.json`; `sample_id`, `shape_name`, `parameters` or nested geometry.
2. **sample_resolved_config_manifest** — `sample/sample_resolved_config_manifest.json`; mesh paths for `L_scout_coarse` / `L_prod`; `status: PLANNED`.
3. **density_zones.placeholder** — `bins[]` with required bin keys; `zone_id: pending_scout` until M4.3.
4. **lprod_target_plan.placeholder** — empty `targets_hz`; `coverage_check.pending_scout: true` (not gapless until scout).
5. **worker_chunk_plan.placeholder** — `chunks[]` over 60–550 Hz; `status: pending_target_plan`; empty `targets_hz`.
6. **pipeline_run_manifest.json** — `mode: dry_run`, `will_execute: false`, all stages `PLANNED`, command previews + env profiles.

Dry-run may omit real `worker_result` bodies; aggregation/runtime are placeholders only.

---

## M4.4 / M4.5 — placeholders (filled at execution)

| Field / artifact | Milestone |
|------------------|-----------|
| `scout_result.discovery.density_result_path`, real mode counts | M4.3 scout run |
| `sample_resolved_config_manifest.core_config_sha256`, mesh file hashes | M4.3–M4.4 |
| `lprod_target_plan.estimated_runtime.*` (non-placeholder) | M4.4 after timing probes |
| `worker_chunk_plan.chunks[].status` → ASSIGNED/RUNNING/PASS | M4.4 worker pool |
| `worker_result.*` timing and `accepted_modes` | M4.4 per chunk |
| `aggregation_result.modal_npz_path`, `plots.*` on disk | M4.4–M4.5 |
| `runtime_summary` full stage wall times | M4.5 batch |
| `pipeline_run_manifest.provenance.mesh_hashes.L_prod` | M4.4 L_prod mesh build |

---

## Examples

| Example | Aligns with schema |
|---------|-------------------|
| `sample_input.example.json` | Mahogany classic LHS row (project geometry floats) |
| `lprod_target_plan.example.json` | 6 / 9 / 12.5 Hz zones; `coverage_check.pass` for 60–550 Hz |
| `worker_chunk_plan.example.json` | Six chunks, 15–50 Hz widths, zone-aware |
| `worker_result.example.json` | Chunk 01 on `L_prod`, PASS stub |
| `aggregation_result.example.json` | PARTIAL merge (schema example only) |
| `pipeline_run_manifest.example.json` | `will_execute: false`, stages through plan |

---

## Validation

```bash
python -m py_compile FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_schema_validate_examples.py
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_schema_validate_examples.py
```

Expected:

```text
schemas/examples parse OK
required key checks PASS
no execution performed
```

Uses `jsonschema` when importable; otherwise required-key and nested checks only.

---

## Non-goals (M4.1)

- No orchestrator implementation
- No Stage A/B/C execution, mesh build, or L_prod solves
- No cleanup or production promotion

### M4.3 scout execution (VM)

Preview:

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_pipeline_run_scout.py \
  --run-dir FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1 \
  --dry-run
```

Execute (after review):

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_pipeline_run_scout.py \
  --run-dir FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1 \
  --execute-scout
```

Writes: `scout/scout_result.json`, `scout/density_zones.json`, `lprod/lprod_target_plan.json`, `lprod/worker_chunk_plan.preview.json` (status `PLANNED_NOT_EXECUTED`). Manifest terminal status: `SCOUT_PASS_TARGET_PLAN_READY`.

### M4.4-pre — L_prod worker execution dry-run

Requires `terminal_status=SCOUT_PASS_TARGET_PLAN_READY`, `lprod/lprod_target_plan.json`, `lprod/worker_chunk_plan.preview.json`.

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_lprod_worker_dry_run.py \
  --run-dir FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1 \
  --workers 3 --dry-run --force
```

Writes under `lprod/`: `lprod_execution_plan.json`, `worker_commands.json`, `aggregation_plan.json` (+ `.md`). Preview manifest: `pipeline_run_manifest.m4_4_dry_run_preview.json` (`LPROD_WORKER_PLAN_READY`, Stages 4–6 `PLANNED_READY`). Does not modify Stage 0–3 PASS artifacts.

### M4.4.1a — L_prod execution interfaces (dry-run)

| Artifact | Schema / path | Notes |
|----------|---------------|--------|
| Per-chunk targets | `m4_worker_chunk_targets_v1` → `worker_results/<chunk_id>/chunk_targets.json` | Preserves `window_hz`, `zone_id`, `spacing_hz` per target |
| Worker command | `worker_results/<chunk_id>/worker_command.sh` | Uses `v2_b3_checkpoint_solve_target_list.py --targets-json` |
| Worker / solver stubs | `worker_result.json`, `solver_result.json`, `log.txt` | `will_execute=false`, `DRY_RUN_PLANNED` in M4.4.1a |
| Mesh readiness | `lprod/lprod_mesh_checkpoint_readiness.json` | `lprod_mesh_status`, `lprod_checkpoint_status`, geometry fingerprints |

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_lprod_worker_dry_run.py \
  --run-dir .../sample_001_m4dry1 --workers 3 --dry-run --force

python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_aggregation_dry_run.py \
  --run-dir .../sample_001_m4dry1 --dry-run
```

### M4.4.1b-0 — L_prod checkpoint execution

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_lprod_checkpoint_run.py \
  --run-dir .../sample_001_m4dry1 --execute
```

Writes: `lprod/resolved_core_config.json`, `lprod/mesh/L_prod/<sample_id>.msh`, `lprod/checkpoint/*`, updated `lprod_mesh_checkpoint_readiness.json`. Terminal: `LPROD_CHECKPOINT_READY`.

### M4.4.1b-1 — single-chunk worker smoke

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_worker_smoke_test.py \
  --run-dir .../sample_001_m4dry1 --chunk-id sample_001_chunk_04 --execute
```

Writes per chunk: `env_probe.json`, real `solver_result.json`, `worker_result.json`, `worker_smoke_manifest.json`. Preview: `pipeline_run_manifest.m4_4_worker_smoke_preview.json`.

Next: **M4.4.1b** FCFS all chunks + aggregation (checkpoint must be PASS).
