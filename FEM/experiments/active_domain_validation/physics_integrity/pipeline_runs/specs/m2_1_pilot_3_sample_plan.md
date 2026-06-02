# M2.1 — 3-sample pilot manifest plan (planning-only)

This document defines a **review-only** pilot plan for the official A+B/C pipeline.

No Stage A/B/C execution is performed by this plan.
No runtime manifests are created by this plan.

## Pilot objective

Prepare a minimal, reviewable first LHS-style batch with:

- 2 timing-only samples (`A+B`, no rich, `C=SKIPPED`)
- 1 rich+synthesis sample (`A+B rich`, `C requested`)

Policy baseline:

- `mesh_level = L_prod`
- `target_set = full9`
- manual JSONL sample list
- no advanced target planner/zones in this pilot

## Sample set

| sample_id | selection_reason | timing_only | rich_requested | synthesis_requested |
|---|---|---:|---:|---:|
| `lhs_pilot_001_timing` | `lhs_timing_baseline` | true | false | false |
| `lhs_pilot_002_timing` | `lhs_timing_baseline` | true | false | false |
| `lhs_pilot_003_synthesis` | `lhs_synthesis_subset_v0` | false | true | true |

Source references for all three:

- `FEM/experiments/active_domain_validation/physics_integrity/configs/v2_mesh_convergence_manifest.json`
- `FEM/experiments/active_domain_validation/physics_integrity/configs/coupled_physical_core_v2.json`

## Per-sample fields (contract)

Each sample entry must include:

- `sample_id`
- `mesh_level` (`L_prod`)
- `target_set` (`full9`)
- `selection_reason`
- policy flags:
  - `timing_only`
  - `rich_requested`
  - `synthesis_requested`
- `source_refs` (manifest + canonical core config)
- `parameter_payload` (placeholder/minimal deltas for now)

## Initial expected stage statuses (pre-execution)

Timing samples (`lhs_pilot_001_timing`, `lhs_pilot_002_timing`):

- `A = PENDING`
- `B = PENDING`
- `C = SKIPPED`

Synthesis sample (`lhs_pilot_003_synthesis`):

- `A = PENDING`
- `B = PENDING`
- `C = PENDING`

## Future manifest registration commands (preview only)

Do **not** execute yet. These are templates for later approval.

### 1) Timing sample registration (no rich, C skipped)

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_pipeline_run_manifest.py \
  --run-id lhs_pilot_001_timing \
  --mode timing \
  --mesh-level L_prod \
  --selection-reason lhs_timing_baseline \
  --append-index
```

Repeat for `lhs_pilot_002_timing` with same mode/reason.

### 2) Synthesis sample registration (rich + C requested)

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_pipeline_run_manifest.py \
  --run-id lhs_pilot_003_synthesis \
  --mode synthesis \
  --mesh-level L_prod \
  --selection-reason lhs_synthesis_subset_v0 \
  --append-index
```

## Official reference run note

The already registered official reference:

- `official_abc_rich_pass_20260601T203438Z`

remains a reference only and is **not rerun** as part of this pilot.

## Safety gates before any execution (required)

- git tracked source is clean
- official archive exists and is verified:
  - `~/final-project-archives/archive_official_A_B_C_rich_PASS_20260601T203438Z.tar.gz`
- disk space check completed
- environments verified:
  - Stage A/C in production `.venv`
  - Stage B in `solver-mkl`
- command previews reviewed
- no cleanup/deletion/archive action tied to pilot run

## Open questions (must be resolved before execution)

1. **Parameter deltas**: what geometry/material perturbations should be used for the three pilot samples?
2. **Perturbation magnitude**: should first pilot use near-baseline perturbations only?
3. **Mesh policy in first pilot**: should all three samples reuse current `L_prod` mesh for orchestration testing, or require regenerated meshes per sample from parameter deltas?

---

This file is a planning spec under `pipeline_runs/specs/` and is intended to be tracked.
