# B3 M3 orchestrator contract (planning-only)

**Status:** M3.1 specification — no implementation, no Stage A/B/C execution, no cleanup.  
**Extends:** [`B3_M1_PIPELINE_RUN_MANIFEST_SPEC.md`](B3_M1_PIPELINE_RUN_MANIFEST_SPEC.md), [`B3_M2_PILOT_CLOSURE_AND_M3_ORCHESTRATOR_PLAN.md`](B3_M2_PILOT_CLOSURE_AND_M3_ORCHESTRATOR_PLAN.md)

---

## 1. M3 scope

M3 is **orchestration only**. It wraps the already validated Stage A/B/C entry scripts with:

- correct **Python executable** per stage,
- correct **environment variables** per stage,
- deterministic **output paths** keyed by `run_id`,
- **manifest lifecycle** (PENDING → per-stage PASS/FAIL → terminal run state),
- **fail-stop** behavior and **no overwrite** of existing run directories.

M3 does **not**:

- redesign physics, weak forms, or solver algorithms,
- change Stage A/B/C script internals (only invokes them with documented argv/env),
- change overlay resolver semantics (M2.4.1) or material-delta policy (M2.3),
- implement full LHS sampling, batch pools, or MAC dedupe,
- execute cleanup, archival, or production promotion.

The orchestrator is a **thin wrapper** whose job is to remove manual mistakes observed in M2 (wrong env, preview vs actual paths, missing manifest updates, accidental dir reuse).

---

## 2. Inputs

### 2.1 Orchestrator input record

Each run is driven by one **input record** (JSON object, JSONL row, or CLI-derived struct). Required and recommended fields:

| Field | Required | Description |
|-------|----------|-------------|
| `sample_id` | yes | Logical LHS / pilot identity (stable across retries) |
| `run_id` | yes | Physical execution identity; keys all output directories and manifest filename |
| `mode` | yes | `timing` \| `rich` \| `synthesis` — derived from policy flags if omitted |
| `mesh_level` | yes | e.g. `L_prod` for current pilot |
| `target_set` | yes | e.g. `full9` for current pilot |
| `selection_reason` | recommended | e.g. `lhs_timing_baseline`, `lhs_synthesis_subset_v0` |
| `resolved_core_config` | yes | Repo-relative path to `pipeline_runs/config_overlays/<sample_id>/resolved_core_config.json` |
| `checkpoint_dir` | yes | Stage A output root (must contain `run_id` in path) |
| `solve_dir` | yes | Stage B output root (must contain `run_id` in path) |
| `synthesis_dir` | if synthesis | Stage C output; typically `<solve_dir>/rich_modal_post` |
| `rich_requested` | yes | Policy: Stage B should export rich modal data |
| `synthesis_requested` | yes | Policy: run Stage C after rich Stage B |
| `stage_c_requested` | yes | Explicit gate for Stage C (true iff `synthesis_requested`) |

Optional extensions (M3.2+):

| Field | Description |
|-------|-------------|
| `samples_jsonl` | Source spec file path |
| `mesh_convergence_manifest` | Canonical mesh suite reference |
| `canonical_core_config` | Baseline path for provenance only |
| `synthesis_region_dofs_mode` | `off` (pilot default) or `best_effort` (separate quality decision) |

### 2.2 `sample_id` vs `run_id`

| Concept | Keying rule | Example |
|---------|-------------|---------|
| `sample_id` | Logical identity, overlays, perturbation policy | `lhs_pilot_001_timing` |
| `run_id` | Checkpoint dir, solve dir, manifest `run_<run_id>.json`, index line | `lhs_pilot_001_timing_retry1` |

Rules:

- **Output paths must use `run_id`, not `sample_id` alone.**
- **Config overlays remain keyed by `sample_id`:**  
  `pipeline_runs/config_overlays/<sample_id>/resolved_core_config.json`
- Retries: **same `sample_id`, new `run_id`** (e.g. `_retry1`, `_retry2`, or timestamp suffix).
- Orchestrator must **refuse** to run if `checkpoint_dir` or `solve_dir` already exists on disk (unless a future explicit `--force-new-run-id` workflow is approved).

### 2.3 Mode and policy mapping

| `mode` | `rich_requested` | `synthesis_requested` | `stage_c_requested` | Stages executed |
|------|------------------|----------------------|---------------------|-----------------|
| `timing` | false | false | false | A, B |
| `rich` | true | false | false | A, B (+ rich flag) |
| `synthesis` | true | true | true | A, B (+ rich), C |

Normative rule: `synthesis_requested=true` implies `rich_requested=true` for command generation (M2.2 policy).

Initial status before execution:

| Stage | timing | rich | synthesis |
|-------|--------|------|-----------|
| A | PENDING | PENDING | PENDING |
| B | PENDING | PENDING | PENDING |
| C | SKIPPED | SKIPPED | PENDING |

---

## 3. Stage command contract

All commands assume `cwd` = repository root (`final-project`). Scripts are repo-relative under `FEM/experiments/active_domain_validation/physics_integrity/scripts/`.

The orchestrator records the **full argv string** (with chosen Python prefix) in `stages.<X>.command` before execution.

### 3.1 Stage A — checkpoint export

**Environment:** production FEM (see §4.1).

**Script:** `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_export.py`

**Required arguments:**

| Argument | Value |
|----------|--------|
| `--mesh-level` | `<mesh_level>` (e.g. `L_prod`) |
| `--B3-block-compose-backend` | `csr_bulk` |
| `--B3-synthesis-region-dofs` | `off` (pilot default; timing/rich/synthesis unless separate quality approval) |
| `--core-config` | `<resolved_core_config>` |
| `--output-dir` | `<checkpoint_dir>` |

**Example (pattern only):**

```bash
/home/vboxuser/final-project/.venv/bin/python \
  FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_export.py \
  --mesh-level L_prod \
  --B3-block-compose-backend csr_bulk \
  --B3-synthesis-region-dofs off \
  --core-config "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/config_overlays/lhs_pilot_001_timing/resolved_core_config.json" \
  --output-dir "FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/st_worker_scaling_L_prod_<run_id>"
```

**PASS artifacts (minimum):** `checkpoint_export_manifest.json` with `status: PASS`, `A_active.petsc.bin`, `M_active.petsc.bin`, CSR exports, `built_metadata.json`, `synthesis_metadata.json`.

### 3.2 Stage B — timing (no rich)

**Environment:** solver-mkl (see §4.2).

**Script:** `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_solve.py`

**Required arguments:**

| Argument | Value |
|----------|--------|
| `--checkpoint-dir` | `<checkpoint_dir>` |
| `--factor-solver` | `mkl_pardiso` |
| `--target-set` | `full9` (or `<target_set>` from input) |
| `--output-dir` | `<solve_dir>` |

**Must not include:** `--B3-export-rich-modal-data`

**Example (pattern only):**

```bash
/home/vboxuser/solver-mkl/venv/bin/python \
  FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_solve.py \
  --checkpoint-dir "<checkpoint_dir>" \
  --factor-solver mkl_pardiso \
  --target-set full9 \
  --output-dir "<solve_dir>"
```

**PASS artifacts (minimum):** `result.json` with solve PASS, `checkpoint_solve_manifest.json`; **no** `rich_modal/` directory required.

### 3.3 Stage B — rich / synthesis

Same as §3.2, plus:

| Argument | Value |
|----------|--------|
| `--B3-export-rich-modal-data` | present (flag, no value) |

**PASS artifacts (additional):** `rich_modal/modes_active.npz`, `rich_modal/modes_catalog.jsonl`, `rich_modal/rich_modal_manifest.json`.

### 3.4 Stage C — rich modal post

**Environment:** production FEM (see §4.1).

**Script:** `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_rich_modal_post.py`

**Required arguments:**

| Argument | Value |
|----------|--------|
| `--checkpoint-dir` | `<checkpoint_dir>` |
| `--rich-modal-dir` | `<solve_dir>/rich_modal` |
| `--output-dir` | `<synthesis_dir>` (typically `<solve_dir>/rich_modal_post`) |

**Example (pattern only):**

```bash
/home/vboxuser/final-project/.venv/bin/python \
  FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_rich_modal_post.py \
  --checkpoint-dir "<checkpoint_dir>" \
  --rich-modal-dir "<solve_dir>/rich_modal" \
  --output-dir "<solve_dir>/rich_modal_post"
```

**PASS artifacts (minimum):** `modes_synthesis.json`, `rich_modal_post_manifest.json` (and optional `.md`).

**Stage C preflight:** Rich bundle from Stage B must exist before launch (see §5).

---

## 4. Environment contract

The orchestrator must **not** rely on the operator to manually `source` the correct venv between stages. It invokes the **explicit Python binary** and sets **explicit env** for subprocesses.

### 4.1 Stage A and Stage C — production FEM env

**Python executable:**

```text
/home/vboxuser/final-project/.venv/bin/python
```

**Required environment variables (VM-validated):**

```bash
export PETSC_DIR=/usr/lib/petscdir/petsc3.15/x86_64-linux-gnu-real
export SLEPC_DIR=/usr/lib/slepcdir/slepc3.15/x86_64-linux-gnu-real
export PYTHONPATH=$PETSC_DIR/lib/python3/dist-packages:$SLEPC_DIR/lib/python3/dist-packages:/usr/lib/python3/dist-packages:$PYTHONPATH
```

**Preflight import expectations:**

| Module | Expected |
|--------|----------|
| `petsc4py` | OK (system PETSc path via `PYTHONPATH`) |
| `dolfinx` | OK |
| `mpi4py` | OK |

**Stage C note:** Default post path must **not** require `petsc4py` inside Stage C script logic (validated); orchestrator still uses production venv for consistency.

**Anti-pattern:** Do not run Stage A/C with `solver-mkl` Python on `PATH` without clearing conflicting `PYTHONPATH`.

### 4.2 Stage B — solver-mkl env

**Python executable:**

```text
/home/vboxuser/solver-mkl/venv/bin/python
```

**Environment:** Use solver-mkl venv isolation (activate equivalent or venv `bin/python` only). Do not inject production FEM `PYTHONPATH` into Stage B.

**Preflight import expectations:**

| Module | Expected |
|--------|----------|
| `petsc4py` | OK (solver-mkl venv) |
| `slepc4py` | OK (solver-mkl venv) |
| `dolfinx` | Not required / must not be required for solve |

**Solver guard:** If `v2_b3_checkpoint_solve.py` performs an MKL/PARDISO environment check, orchestrator preflight should surface the same failure before launch when possible.

### 4.3 Environment recording in manifest

Manifest `environment` block (per M1, extended for M3):

```json
{
  "stage_a": {
    "profile": "production_venv",
    "python": "/home/vboxuser/final-project/.venv/bin/python",
    "PETSC_DIR": "/usr/lib/petscdir/petsc3.15/x86_64-linux-gnu-real",
    "SLEPC_DIR": "/usr/lib/slepcdir/slepc3.15/x86_64-linux-gnu-real"
  },
  "stage_b": {
    "profile": "solver_mkl",
    "python": "/home/vboxuser/solver-mkl/venv/bin/python"
  },
  "stage_c": {
    "profile": "production_venv",
    "python": "/home/vboxuser/final-project/.venv/bin/python"
  }
}
```

Log files (optional M3.3): `pipeline_runs/logs/<run_id>/stage_A.log`, etc.

---

## 5. Preflight checks

Orchestrator runs **preflight** after building the input record and **before** any stage subprocess. Failure → manifest terminal `FAIL`, no stage execution.

### 5.1 Input and config

| Check | Failure code |
|-------|----------------|
| `sample_id`, `run_id`, `mode` present and consistent with policy flags | `PREFLIGHT_INVALID_INPUT` |
| `resolved_core_config` file exists and is readable JSON | `PREFLIGHT_MISSING_CORE_CONFIG` |
| Readiness recommends `solver.clamp_ribs == false` when present | `PREFLIGHT_BC_POLICY` |
| Overlay `readiness_check.json` status PASS (recommended, not hard-fail if missing) | warn only |

### 5.2 Output directory collision

| Check | Failure code |
|-------|----------------|
| `checkpoint_dir` does not exist | `PREFLIGHT_DIR_EXISTS` |
| `solve_dir` does not exist | `PREFLIGHT_DIR_EXISTS` |
| If synthesis: `synthesis_dir` does not exist (or parent solve_dir policy documented) | `PREFLIGHT_DIR_EXISTS` |

### 5.3 Stage Python and imports

| Stage | Check | Failure code |
|-------|-------|----------------|
| A, C | Production Python path exists and is executable | `PREFLIGHT_ENV` |
| A, C | `import petsc4py, dolfinx, mpi4py` smoke (subprocess) | `PREFLIGHT_IMPORT` |
| B | solver-mkl Python path exists | `PREFLIGHT_ENV` |
| B | `import petsc4py, slepc4py` smoke | `PREFLIGHT_IMPORT` |

### 5.4 Stage-specific preflight (before that stage only)

**Before Stage B:**

| Check | Failure code |
|-------|----------------|
| `checkpoint_dir/checkpoint_export_manifest.json` exists, `status == PASS` | `PREFLIGHT_STAGE_A_INCOMPLETE` |
| Required checkpoint matrices/CSR present per pipeline lib | `PREFLIGHT_CHECKPOINT_INCOMPLETE` |

**Before Stage C (synthesis only):**

| Check | Failure code |
|-------|----------------|
| `solve_dir/rich_modal/modes_active.npz` exists | `PREFLIGHT_RICH_MODAL` |
| `solve_dir/rich_modal/modes_catalog.jsonl` exists | `PREFLIGHT_RICH_MODAL` |
| `solve_dir/rich_modal/rich_modal_manifest.json` exists | `PREFLIGHT_RICH_MODAL` |
| Stage B `result.json` indicates PASS | `PREFLIGHT_STAGE_B_INCOMPLETE` |

### 5.5 Mesh reference (informational)

Orchestrator may warn if `resolved_core_config` `solver.mesh_file` does not reference expected `L_prod` / `baseline_coupled_v2.msh` for pilot — warning only unless policy tightens.

---

## 6. Manifest lifecycle

Schema base: `b3_pipeline_run_manifest_v1` ([`B3_M1_PIPELINE_RUN_MANIFEST_SPEC.md`](B3_M1_PIPELINE_RUN_MANIFEST_SPEC.md)). M3 adds **run-level terminal state** and richer provenance.

### 6.1 Lifecycle sequence

1. **Allocate `run_id`** (caller or orchestrator); verify dirs do not exist.
2. **Preflight** (§5). On fail → write manifest with `run_terminal_state: FAIL`, append index, stop.
3. **Create manifest** at `pipeline_runs/manifests/run_<run_id>.json` with:
   - `schema`, `run_id`, `sample_id`, `created_utc`
   - `source`, `policy`, `paths`, `core_config_provenance` (from resolved config)
   - `stages.A/B/C.status` = `PENDING` or `SKIPPED` (C skipped if not synthesis)
   - `run_terminal_state` = `RUNNING` (optional) or omit until terminal
4. **Run Stage A** — record `command`, `started_utc`, log path; on completion update `stages.A` to `PASS` or `FAIL` with `failure_reason`, artifact paths, `core_config_provenance`.
5. If Stage A `FAIL` → set `run_terminal_state: FAIL`, append index, **stop**.
6. **Run Stage B** — same recording pattern.
7. If Stage B `FAIL` → terminal `FAIL`, append index, **stop**.
8. If `stage_c_requested` → **Run Stage C**; else leave C `SKIPPED`.
9. Update Stage C to `PASS`, `FAIL`, or `PASS_WITH_WARNING` (see §8).
10. Compute **run terminal state** (§6.2); write final manifest; **append** `pipeline_runs/index/runs_index.jsonl` **once**.

### 6.2 Run terminal states

| `run_terminal_state` | Meaning |
|----------------------|---------|
| `PASS` | All executed stages passed; no blocking warnings |
| `PASS_WITH_WARNING` | All executed stages completed; non-fatal warnings recorded (e.g. structural region unavailable in Stage C) |
| `FAIL` | Any executed stage failed or preflight failed |
| `SKIPPED` | **Not used** for whole-run terminal state (reserved for individual stages only) |

Stage-level statuses remain: `PENDING`, `PASS`, `FAIL`, `SKIPPED`.

### 6.3 Index append rule

- Append to `runs_index.jsonl` **only after** terminal `run_terminal_state` is known.
- One index line per `run_id` (no duplicate appends on retry — new `run_id` → new line).

### 6.4 Manifest filename convention

M2 practice (informative suffix in filename):

```text
pipeline_runs/manifests/run_<run_id>_<terminal_hint>.json
```

Examples:

- `run_lhs_pilot_002_timing_AB_PASS.json`
- `run_lhs_pilot_003_synthesis_ABC_RICH_PASS_STRUCT_WARN.json`

Orchestrator may write initially as `run_<run_id>.json` and rename on terminal, or write final name once at end — implementation choice in M3.3; contract requires **stable `run_id` inside JSON** regardless of filename suffix.

---

## 7. Failure policy

| Rule | Requirement |
|------|-------------|
| Fail-stop | Stop immediately on first stage `FAIL` |
| No B after A fail | Never launch Stage B if Stage A not `PASS` |
| No C after B fail | Never launch Stage C if Stage B not `PASS` |
| Preserve dirs | Do not delete or move failed output directories |
| Retry identity | Retry requires **new `run_id`**; same `sample_id` and overlay path allowed |
| No overwrite | Never write into existing `checkpoint_dir` or `solve_dir` without future explicit override flag |
| Manifest record | Every `FAIL` records `failure_reason`, stage, optional log tail pointer |
| Provenance on FAIL | `core_config_provenance` (mode, path, sha256, canonical path, material fingerprint) required on FAIL manifests (M2.4.2) |

Failure reason format (recommended):

```text
<STAGE>_<CATEGORY>:<detail>
```

Examples: `A_RUNTIME:RuntimeError:Displacement BC constrains...`, `PREFLIGHT_DIR_EXISTS:checkpoint_dir exists`, `B_ENV:MKL guard failed`.

---

## 8. Warning policy

Warnings **do not** automatically fail the run unless policy explicitly elevates them.

### 8.1 Known M2 warning — structural region indices

When Stage A uses `--B3-synthesis-region-dofs off`, Stage C may complete with:

| Field | Value |
|-------|--------|
| `pressure_region_status` | `available` |
| `structural_region_participation_status` | `unavailable_region_indices` |
| `structural_audio_proxy_status` | `unavailable_region_indices` |
| `region_dof_source` | `built_metadata_pressure_only` |

**Representation:**

- `stages.C.status` = `PASS` (stage script exited success)
- `run_terminal_state` = `PASS_WITH_WARNING`
- `warnings[]` includes structured entry, e.g. `structural_region_indices_unavailable`

**Do not** automatically rerun with `--B3-synthesis-region-dofs best_effort`. That is a **separate quality decision** (DOLFINx subprocess risk vs. structural proxy requirement).

### 8.2 Other warnings

Orchestrator may record non-fatal warnings from:

- Stage A `checkpoint_export_manifest.json` `warnings` array
- Stage C synthesis metadata / post manifest
- Preflight soft checks (e.g. missing readiness_check.json)

Default: accumulate in manifest `warnings[]`; elevate to `PASS_WITH_WARNING` only when documented policy matches (e.g. structural region case).

---

## 9. Output path policy

All templates are repo-relative under `FEM/experiments/active_domain_validation/physics_integrity/`. Substitute `<run_id>` everywhere below.

### 9.1 Canonical templates

| Stage | Path template |
|-------|----------------|
| A — checkpoint | `v2_mesh_convergence/diagnostics/st_worker_scaling_<mesh_level>_<run_id>` |
| B — solve | `v2_mesh_convergence/diagnostics/solver_benchmarks/checkpoint_solve_mkl_pardiso_<target_set>_<run_id>` |
| C — synthesis | `<solve_dir>/rich_modal_post` |

Pilot defaults: `mesh_level=L_prod`, `target_set=full9`.

### 9.2 Overlay path (keyed by `sample_id`)

```text
pipeline_runs/config_overlays/<sample_id>/resolved_core_config.json
pipeline_runs/config_overlays/<sample_id>/overlay_applied.json
pipeline_runs/config_overlays/<sample_id>/readiness_check.json
```

### 9.3 Retry example

| Attempt | `sample_id` | `run_id` | `checkpoint_dir` suffix |
|---------|-------------|----------|-------------------------|
| 1 (failed) | `lhs_pilot_001_timing` | `lhs_pilot_001_timing` | `..._lhs_pilot_001_timing` |
| 2 (PASS) | `lhs_pilot_001_timing` | `lhs_pilot_001_timing_retry1` | `..._lhs_pilot_001_timing_retry1` |
| 3 (hypothetical) | `lhs_pilot_001_timing` | `lhs_pilot_001_timing_retry2` | `..._lhs_pilot_001_timing_retry2` |

M2.2 **preview** may default `run_id == sample_id`; orchestrator and runtime manifests are **authoritative** for executed paths.

---

## 10. Relationship to M2 runtime records

The following VM-validated manifests are **execution evidence**, not planning previews:

| Manifest | Role |
|----------|------|
| `pipeline_runs/manifests/run_lhs_pilot_001_timing_retry1_AB_PASS.json` | Timing A+B PASS; C SKIPPED |
| `pipeline_runs/manifests/run_lhs_pilot_002_timing_AB_PASS.json` | Timing A+B PASS |
| `pipeline_runs/manifests/run_lhs_pilot_003_synthesis_ABC_RICH_PASS_STRUCT_WARN.json` | Synthesis A+B rich + C with structural-region warning |

**Not authoritative for paths:**

- `pipeline_runs/specs/previews/m2_2_pilot_3_sample_command_preview.json` (dry-run template; may omit `retry` suffixes)

**Official reference (unchanged):**

- `official_abc_rich_pass_20260601T203438Z` — canonical validated A+B+C; not superseded by pilot manifests

M3 orchestrator must be able to **read** these manifests as examples when implementing M3.2/M3.3 but must not mutate them.

---

## 11. Non-goals

M3 orchestrator (M3.1–M3.5) explicitly does **not**:

- run full LHS batches or parameter-space exploration,
- execute automatic cleanup, deletion, or archival of failed/obsolete dirs,
- move or rename diagnostics trees as part of orchestration,
- promote checkpoints or meshes into production `FEM/mesh` or GUI configs,
- rerun successful M2 pilot samples for path or preview alignment,
- rerun sample 3 with `--B3-synthesis-region-dofs best_effort` without a separate quality approval,
- modify internals of `v2_b3_checkpoint_export.py`, `v2_b3_checkpoint_solve.py`, or `v2_b3_rich_modal_post.py`,
- replace the official A+B+C reference archive,
- implement ROM/STK ingestion or GUI integration.

---

## 12. M3 implementation sequence

### M3.1 — Orchestrator contract (this document)

Planning-only. Defines inputs, commands, env, preflight, manifest lifecycle, failure/warning policy, paths, and non-goals.

**Exit criterion:** Review approved on VM policy holder.

### M3.2 — Dry-run orchestrator preview

Deliverable: extend or add script (e.g. evolve `v2_b3_lhs_orchestrator_preview.py`) to:

- read input record / JSONL with explicit `run_id`,
- emit exact per-stage commands with correct Python prefixes,
- emit env blocks (print-only),
- run preflight except import subprocesses optional (configurable),
- detect output dir collisions,
- write preview JSON under `pipeline_runs/specs/previews/` only,
- **no** execution, **no** writes to `pipeline_runs/manifests/`.

### M3.3 — Single-run execution orchestrator (timing only)

Deliverable: `v2_b3_lhs_orchestrator_run.py` (name TBD):

- **one** timing sample only,
- requires explicit human approval gate,
- creates manifest PENDING → runs A → B → updates manifest → terminal state,
- fail-stop, no overwrite,
- append index once.

**Not in M3.3:** Stage C, batch loop, synthesis.

### M3.4 — Synthesis execution path

Same runner; add Stage B rich flag + Stage C for **one** synthesis sample after M3.3 timing PASS on VM.

### M3.5 — Small batch mode

Process N rows from JSONL sequentially with fail-stop per sample — only after M3.3 and M3.4 validated.

---

## Appendix A — Minimal input record example (timing)

```json
{
  "sample_id": "lhs_pilot_002_timing",
  "run_id": "lhs_pilot_002_timing",
  "mode": "timing",
  "mesh_level": "L_prod",
  "target_set": "full9",
  "selection_reason": "lhs_timing_baseline",
  "resolved_core_config": "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/config_overlays/lhs_pilot_002_timing/resolved_core_config.json",
  "checkpoint_dir": "FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/st_worker_scaling_L_prod_lhs_pilot_002_timing",
  "solve_dir": "FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/solver_benchmarks/checkpoint_solve_mkl_pardiso_full9_lhs_pilot_002_timing",
  "synthesis_dir": null,
  "rich_requested": false,
  "synthesis_requested": false,
  "stage_c_requested": false
}
```

## Appendix B — Minimal input record example (synthesis)

```json
{
  "sample_id": "lhs_pilot_003_synthesis",
  "run_id": "lhs_pilot_003_synthesis",
  "mode": "synthesis",
  "mesh_level": "L_prod",
  "target_set": "full9",
  "selection_reason": "lhs_synthesis_subset_v0",
  "resolved_core_config": "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/config_overlays/lhs_pilot_003_synthesis/resolved_core_config.json",
  "checkpoint_dir": "FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/st_worker_scaling_L_prod_lhs_pilot_003_synthesis",
  "solve_dir": "FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/solver_benchmarks/checkpoint_solve_mkl_pardiso_full9_lhs_pilot_003_synthesis",
  "synthesis_dir": "FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/solver_benchmarks/checkpoint_solve_mkl_pardiso_full9_lhs_pilot_003_synthesis/rich_modal_post",
  "rich_requested": true,
  "synthesis_requested": true,
  "stage_c_requested": true
}
```

---

**Next step after M3.1 approval:** M3.2 dry-run orchestrator (commands + env + collision checks only). No further manual M2 pilot execution.
