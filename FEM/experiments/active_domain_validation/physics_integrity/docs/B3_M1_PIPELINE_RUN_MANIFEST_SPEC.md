# B3 M1 pipeline run manifest spec

## 1. Purpose (M1)

Establish a canonical pipeline run manifest and run-root layout for future official Stage A/B/C and LHS operations.

M1 is documentation/specification only. It does not change runtime behavior.

## 2. Future run-root layout

Canonical root:

`FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/`

Proposed subfolders:

- `manifests/` - per-run pipeline manifests (`run_<run_id>.json`)
- `logs/` - stage logs for wrapper/orchestration runs
- `index/` - optional append-only run index (`runs_index.jsonl`)
- `specs/` - schema/examples/checklists

M1 does not require moving any existing artifacts into this layout.

## 3. Manifest schema (`b3_pipeline_run_manifest_v1`)

Required top-level keys:

- `schema` (string): must be `b3_pipeline_run_manifest_v1`
- `run_id` (string): globally unique run identifier
- `created_utc` (string, ISO-8601 UTC)
- `source` (object): source inputs and references
- `policy` (object): run intent (timing/rich/synthesis selection)
- `stages` (object): `A`, `B`, `C` stage records
- `environment` (object): environment intent per stage

Recommended minimal structure:

```json
{
  "schema": "b3_pipeline_run_manifest_v1",
  "run_id": "20260602T120000Z_lhs_sample_0001",
  "created_utc": "2026-06-02T12:00:00Z",
  "source": {
    "mesh_level": "L_prod",
    "mesh_convergence_manifest": "FEM/experiments/active_domain_validation/physics_integrity/configs/v2_mesh_convergence_manifest.json",
    "core_config": "FEM/experiments/active_domain_validation/physics_integrity/configs/coupled_physical_core_v2.json"
  },
  "policy": {
    "rich_export": false,
    "stage_c_requested": false,
    "selection_reason": "timing|lhs_candidate|synthesis_candidate"
  },
  "stages": {
    "A": {
      "status": "PENDING",
      "script": "scripts/v2_b3_checkpoint_export.py",
      "command": null,
      "checkpoint_dir": null,
      "export_manifest": null
    },
    "B": {
      "status": "PENDING",
      "script": "scripts/v2_b3_checkpoint_solve.py",
      "command": null,
      "solve_dir": null,
      "result_json": null,
      "rich_modal_requested": false,
      "rich_modal_dir": null
    },
    "C": {
      "status": "SKIPPED",
      "script": "scripts/v2_b3_rich_modal_post.py",
      "command": null,
      "synthesis_dir": null,
      "modes_synthesis_json": null
    }
  },
  "environment": {
    "stage_a_env": "production_venv",
    "stage_b_env": "solver_mkl",
    "stage_c_env": "production_venv"
  }
}
```

## 4. Stage linking rules

- Stage B must reference a successful Stage A checkpoint (`stages.A.status=PASS`, valid `checkpoint_dir`).
- Stage C must reference a Stage B run with rich modal outputs (`stages.B.rich_modal_requested=true` and `rich_modal_dir` exists).
- Rich export is opt-in only (`policy.rich_export=true` or explicit Stage B rich request).
- Stage C is valid only when rich modal data exists.

## 5. Status model

Allowed stage statuses:

- `PENDING`
- `PASS`
- `FAIL`
- `SKIPPED`

M1 rule: stage status is explicit and never inferred from missing keys.

## 6. LHS policy contract

- All LHS points: Stage A + Stage B timing/scalar summaries (no rich by default)
- Selected subset: Stage B rich (`--B3-export-rich-modal-data`)
- Synthesis/ROM/STK subset: Stage C post on rich-modal outputs

## 7. Example manifests

### A) Timing run (A+B only, no rich, C skipped)

```json
{
  "schema": "b3_pipeline_run_manifest_v1",
  "run_id": "20260602T120100Z_timing_0001",
  "policy": {"rich_export": false, "stage_c_requested": false},
  "stages": {
    "A": {"status": "PASS", "checkpoint_dir": ".../st_worker_scaling_L_prod_..."},
    "B": {"status": "PASS", "rich_modal_requested": false, "solve_dir": ".../checkpoint_solve_mkl_pardiso_full9_..."},
    "C": {"status": "SKIPPED"}
  }
}
```

### B) Rich run (A+B rich, C skipped)

```json
{
  "schema": "b3_pipeline_run_manifest_v1",
  "run_id": "20260602T120200Z_rich_0001",
  "policy": {"rich_export": true, "stage_c_requested": false},
  "stages": {
    "A": {"status": "PASS", "checkpoint_dir": ".../st_worker_scaling_L_prod_..."},
    "B": {"status": "PASS", "rich_modal_requested": true, "rich_modal_dir": ".../rich_modal"},
    "C": {"status": "SKIPPED"}
  }
}
```

### C) Full synthesis run (A+B rich + C)

```json
{
  "schema": "b3_pipeline_run_manifest_v1",
  "run_id": "20260602T120300Z_synthesis_0001",
  "policy": {"rich_export": true, "stage_c_requested": true},
  "stages": {
    "A": {"status": "PASS", "checkpoint_dir": ".../st_worker_scaling_L_prod_..."},
    "B": {"status": "PASS", "rich_modal_requested": true, "rich_modal_dir": ".../rich_modal"},
    "C": {"status": "PASS", "synthesis_dir": ".../rich_modal_post", "modes_synthesis_json": ".../modes_synthesis.json"}
  }
}
```

## 8. Safety rules (M1)

- M1 does not move old outputs.
- M1 does not modify validated scripts.
- M1 does not delete anything.
- Any future wrapper must call existing Stage scripts unchanged first.

## 9. Relationship to M0

This M1 spec extends:

`docs/B3_MIGRATION_TO_OFFICIAL_PIPELINE_M0.md`

M0 defines governance and migration boundaries. M1 defines the concrete run-manifest contract.

## 10. Next step after M1 (do not implement yet)

Implement a tiny manifest-only helper that:

- creates `run_<run_id>.json` with this schema,
- records stage commands and paths,
- updates stage statuses (`PENDING`/`PASS`/`FAIL`/`SKIPPED`),
- does not replace Stage scripts,
- does not move or delete any artifacts.
