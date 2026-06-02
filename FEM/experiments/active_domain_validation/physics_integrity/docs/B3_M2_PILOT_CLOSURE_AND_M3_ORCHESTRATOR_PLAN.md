# B3 M2 pilot closure and M3 orchestrator plan (planning-only)

**Status:** M2 pilot complete on VM (2026-06). This document closes M2 and defines M3 scope.  
**No Stage A/B/C execution is authorized by this document.**

Related:

- [`B3_M2_LHS_INTEGRATION_PLAN.md`](B3_M2_LHS_INTEGRATION_PLAN.md)
- [`B3_M2_3_PILOT_PERTURBATION_POLICY.md`](B3_M2_3_PILOT_PERTURBATION_POLICY.md)
- [`B3_M2_4_MATERIAL_OVERLAY_EXECUTION_CONTRACT.md`](B3_M2_4_MATERIAL_OVERLAY_EXECUTION_CONTRACT.md)
- [`B3_OFFICIAL_RICH_PIPELINE_COMMANDS.md`](B3_OFFICIAL_RICH_PIPELINE_COMMANDS.md)
- [`B3_MIGRATION_TO_OFFICIAL_PIPELINE_M0.md`](B3_MIGRATION_TO_OFFICIAL_PIPELINE_M0.md)

---

## 1. M2 pilot purpose

M2 was a **controlled 3-sample mini-pilot** to prove that the official A+B+C checkpoint pipeline can run under LHS-style orchestration constraints without full batch LHS or production promotion.

Goals:

- Move from orchestration smoke (empty `material_delta`) to **physically meaningful** near-baseline material perturbations on shared `L_prod` mesh.
- Prove **resolved config overlays** (`--core-config`) and authoritative complete solver policy (including `solver.clamp_ribs=false`).
- Prove **timing** (A+B) and **synthesis** (A+B rich + C) policy on `target_set=full9`.
- Establish **runtime manifests and index** as the record of what actually executed (not M2.2 preview paths alone).

M2 was **not** a final audio/STK quality campaign, full LHS rollout, or cleanup/deprecation execution.

---

## 2. VM-validated sample results

| Record | Mode | Stage A | Stage B | Stage C | Notes |
|--------|------|---------|---------|---------|-------|
| `official_abc_rich_pass_20260601T203438Z` | reference | PASS | PASS | PASS | Official validated A+B+C archive; unchanged |
| `lhs_pilot_001_timing_retry1_AB_PASS` | timing | PASS | PASS | SKIPPED | Material: `top.density=445.5`; retry run id after first BC-policy failure |
| `lhs_pilot_002_timing_AB_PASS` | timing | PASS | PASS | SKIPPED | Material: `back.density=842.45`; canonical preview paths |
| `lhs_pilot_003_synthesis_ABC_RICH_PASS_STRUCT_WARN` | synthesis | PASS | PASS (rich) | PASS (struct warn) | Material: `top.density=456.75`; rich + synthesis; structural region DOFs unavailable |

### Sample 1 — `lhs_pilot_001_timing_retry1`

- **Checkpoint:** `v2_mesh_convergence/diagnostics/st_worker_scaling_L_prod_lhs_pilot_001_timing_retry1`
- **Solve:** `solver_benchmarks/checkpoint_solve_mkl_pardiso_full9_lhs_pilot_001_timing_retry1`
- **Manifest:** `pipeline_runs/manifests/run_lhs_pilot_001_timing_retry1_AB_PASS.json`
- **Provenance:** `core_config_mode=override`, material fingerprint `top=445.5`, `back=830.0`

### Sample 2 — `lhs_pilot_002_timing`

- **Checkpoint / solve:** `..._lhs_pilot_002_timing` (no retry suffix)
- **Manifest:** `pipeline_runs/manifests/run_lhs_pilot_002_timing_AB_PASS.json`

### Sample 3 — `lhs_pilot_003_synthesis`

- **Manifest:** `pipeline_runs/manifests/run_lhs_pilot_003_synthesis_ABC_RICH_PASS_STRUCT_WARN.json`
- **Stage C outputs:** `modes_synthesis.json` present (`mode_count=113`)
- **Region status:** pressure available; structural participation/proxy unavailable (see §4)

### Official reference (unchanged)

`official_abc_rich_pass_20260601T203438Z` remains the **canonical** validated A+B+C PASS reference for governance, docs, and regression anchors. Pilot runs do not replace it.

---

## 3. What M2 proved

| Area | Result |
|------|--------|
| Material overlays | M2.4.1 resolver + readiness checks work; deltas reach assembly |
| `--core-config` | Stage A loads per-sample resolved config; provenance recorded on PASS |
| BC policy | `solver.clamp_ribs=false` required in authoritative resolved config |
| Stage A | Operator export PASS with override config and `L_prod` mesh |
| Stage B timing | `mkl_pardiso`, `full9`, 9/9 targets on timing samples |
| Stage B rich | Rich export PASS on synthesis sample |
| Stage C | Post runs; `modes_synthesis.json` written |
| Manifests / index | Runtime JSON + index append usable as execution audit trail |
| `sample_id` vs `run_id` | Retry suffixes (`retry1`) separate logical sample from physical dirs |

---

## 4. What remains limited or warning-level

### Structural region DOFs (sample 3)

Stage C completed with **structural-region warning**, not pipeline failure:

- `pressure_region_status = available`
- `structural_region_participation_status = unavailable_region_indices`
- `structural_audio_proxy_status = unavailable_region_indices`
- `region_dof_source = built_metadata_pressure_only`

**Cause:** Stage A ran with `--B3-synthesis-region-dofs off` (default); `region_dof_indices.npz` was not produced. Stage C correctly reports null structural participation rather than zero.

**M2 acceptance:** Acceptable for **pipeline integration** proof (A+B rich + C path, manifests, synthesis artifact).

**Not accepted as:** Final rich synthesis quality gate for audio/STK consumption.

**Explicit deferral:** Do **not** rerun sample 3 with `--B3-synthesis-region-dofs best_effort` until a separate **quality decision** is approved (DOLFINx subprocess risk vs. structural proxy requirement).

### Path drift vs M2.2 preview

M2.2 preview templates use `run_id == sample_id`. Sample 1 validated under `..._retry1`. **Runtime manifests are authoritative** for executed paths; preview is pre-run planning only.

### Failed first attempt (sample 1)

Pre-fix run dir `st_worker_scaling_L_prod_lhs_pilot_001_timing` (no retry) is **temporary debug evidence** only. After `retry1` PASS is registered, it is a **delete candidate** (not P0/P1).

---

## 5. Runtime manifest / index status

**Convention (established in M2):**

- `pipeline_runs/manifests/run_<run_id>_<terminal_state>.json` — per-run record
- `pipeline_runs/index/runs_index.jsonl` — append-only after terminal state
- `pipeline_runs/config_overlays/<sample_id>/` — resolved configs (ignored by git)
- `pipeline_runs/logs/` — stage logs when used (ignored)

**M2 pilot manifests (VM):**

- `run_lhs_pilot_001_timing_retry1_AB_PASS.json`
- `run_lhs_pilot_002_timing_AB_PASS.json`
- `run_lhs_pilot_003_synthesis_ABC_RICH_PASS_STRUCT_WARN.json`

**Index:** Appended successfully for completed runs.

**Authoritative path rule:** Read checkpoint/solve paths from the **runtime manifest**, not from regenerated M2.2 preview JSON.

---

## 6. Cleanup / deprecation implications

Project policy (post-M2): prioritize **clean migration and reproducibility** of the official pipeline, not archival completeness of every failed or superseded run.

### Protect / keep

- Official A+B+C PASS archive (`official_abc_rich_pass_20260601T203438Z` and linked artifacts)
- Canonical configs: `coupled_physical_core_v2.json`, `v2_mesh_convergence_manifest.json`, specs under `pipeline_runs/specs/`
- `mesh/L_prod` (and mesh referenced by validated runs)
- Current Stage A/B/C scripts and migration docs (M0–M2.4, official commands)
- **Active** pilot runtime outputs while needed for audit (the three PASS manifests + their checkpoint/solve dirs)

### Delete candidates (after review, not in M3.0)

- Failed sample-1 attempt dir without `retry1` (superseded by `retry1` PASS)
- Any duplicate preview-only or smoke dirs not referenced by manifests
- Old diagnostics/benchmark runs not P0/P1 and not cited in closure docs

### Do not archive heavily

- Failed pilot attempts replaced by PASS
- Obsolete benchmark trees without unique decision evidence

**Keep concise docs** (this closure doc, M2.4 contracts, official commands) as the decision record.

**M3.0:** No cleanup execution. Deletion only via explicit approved dry-run + user sign-off (existing cleanup script policy).

---

## 7. Decision: stop manual sample execution here

**M2 is closed.** No further manual Stage A/B/C runs for the 3-sample pilot.

Rationale:

- All three samples reached intended terminal states on VM.
- Timing and synthesis policy paths are proven.
- Known structural-region limitation is documented and deferred.
- Next value is **orchestration contract and thin runner**, not more ad-hoc commands.

---

## 8. M3 goal: thin env-aware orchestrator

Build a **thin single-sample orchestrator** that:

- Reads pilot/LHS row (`sample_id`, optional `run_id`, policy flags, overlay paths).
- Creates manifest `PENDING` **before** execution.
- Runs Stage A → B → (C if synthesis) with correct **Python executable and environment** per stage.
- Updates manifest after each stage; **stops on FAIL**.
- Never overwrites existing run directories.
- Appends index only after **final** terminal state (PASS or FAIL).

M3 does **not** replace Stage scripts; it wraps them with env handoff, path assignment, and manifest lifecycle.

---

## 9. M3 orchestrator requirements

### Identity

| Field | Role |
|-------|------|
| `sample_id` | Logical LHS identity, overlay dir key (`config_overlays/<sample_id>/`) |
| `run_id` | Physical output key (`st_worker_scaling_<mesh>_<run_id>`, solve dirs, manifest filename) |

Default: `run_id = sample_id`. On retry: `run_id = <sample_id>_retry<N>` or timestamped suffix; never reuse failed dir.

### Per-stage executable / environment

| Stage | Python / env | Notes |
|-------|----------------|-------|
| **A** | Production FEM `.venv` | DOLFINx + PETSc for export |
| **B** | `/home/vboxuser/solver-mkl/venv/bin/python` | MKL/PARDISO; no DOLFINx requirement on solve path |
| **C** | Production FEM `.venv` | Numpy-only default post; optional `best_effort` later |

### Stage A environment variables (VM production)

Orchestrator must set before Stage A (exact values from validated VM):

```bash
export PETSC_DIR=/usr/lib/petscdir/petsc3.15/x86_64-linux-gnu-real
export SLEPC_DIR=/usr/lib/slepcdir/slepc3.15/x86_64-linux-gnu-real
export PYTHONPATH="${PETSC_DIR}/${PETSC_ARCH}/lib/python3.10/site-packages:/usr/lib/python3/dist-packages:${PYTHONPATH}"
```

(Adjust `PETSC_ARCH` / Python version if repo documents a different arch string; orchestrator should read from a small env profile file rather than hardcoding in multiple places.)

Stage B: activate or invoke `solver-mkl` venv explicitly (no Stage A env leakage).

Stage C: production `.venv` only; verify no `solver-mkl` on `PATH` for default post.

### Manifest lifecycle

1. **Pre-run:** Write `pipeline_runs/manifests/run_<run_id>.json` with `stages.*.status = PENDING` (C `SKIPPED` if timing-only).
2. Record: `sample_id`, `run_id`, `core_config_path`, overlay SHA256, predicted dirs, commands (optional dry-run block).
3. **After each stage:** Update status `PASS`/`FAIL`, paths, timestamps, log paths, failure_reason.
4. **On FAIL:** Do not run subsequent stages; set later stages `SKIPPED`; do not append index until terminal.
5. **On terminal PASS/FAIL:** Append one line to `pipeline_runs/index/runs_index.jsonl`.

### Directory safety

- If checkpoint or solve output dir exists → **abort** (unless explicit `--force-new-run-id`).
- Never delete or move existing run dirs from orchestrator.

### Command construction

- Stage A: `--mesh-level L_prod`, `--core-config` → `config_overlays/<sample_id>/resolved_core_config.json`, `--B3-synthesis-region-dofs off` (default pilot), `--output-dir` from `run_id`.
- Stage B: `--checkpoint-dir` from Stage A manifest; add `--B3-export-rich-modal-data` when policy requires.
- Stage C: only when synthesis; `--checkpoint-dir`, `--rich-modal-dir`, `--output-dir` from manifest-linked paths.

### Provenance on all terminal manifests

Include on PASS and FAIL:

- `core_config_provenance` (mode, path, sha256, canonical path, material fingerprint)

---

## 10. Explicit non-goals (M3)

- **No full LHS** batch yet (N-sample driver, parameter pools, MAC dedupe).
- **No cleanup execution** (delete/archive) in M3.0–M3.2.
- **No rerun** of successful M2 pilot samples for path alignment.
- **No automatic production promotion** of checkpoints into `FEM/mesh` or GUI paths.
- **No sample 3 region-DOF quality rerun** until approved.
- **No GUI / ROM / STK ingestion** automation.

---

## 11. Next implementation proposal

### M3.1 — Planning-only orchestrator contract (next doc step)

Create `docs/B3_M3_ORCHESTRATOR_CONTRACT.md` (or extend M1 manifest spec) with:

- JSON schema for orchestrator input row + env profile
- State machine diagram (PENDING → A → B → C → terminal)
- Exact manifest field updates per stage
- Failure codes (`FAIL_ENV`, `FAIL_STAGE_A`, …)
- `run_id` allocation rules

**Deliverable:** Spec only; no runner code.

### M3.2 — Dry-run orchestrator with env/path preview

Extend or replace `v2_b3_lhs_orchestrator_preview.py` with:

- `--run-id` override
- Per-stage env block (print-only, no subprocess)
- Manifest JSON preview (write under `pipeline_runs/specs/previews/` only, not `manifests/`)
- Validation: resolved config exists, `clamp_ribs=false`, dirs do not exist

**Deliverable:** Script + preview artifacts; **no execution**.

### M3.3 — Single-sample execution runner (after explicit approval)

Thin script, e.g. `v2_b3_lhs_orchestrator_run.py`:

- **One timing sample only** for first execution test (suggest fresh `run_id` on unused sample or new smoke `run_id`).
- Implements M3.1 lifecycle + M3.2 command templates.
- Stops on first FAIL; no batch loop.

**Gate before M3.3:** M3.1 approved, M3.2 dry-run reviewed on VM, env profile verified once manually.

---

## Summary

| Item | Status |
|------|--------|
| M2 pilot | **Closed — PASS on VM** |
| Manual sample execution | **Stop** |
| Official reference | **Unchanged** |
| Sample 3 structural regions | **Warning accepted for M2; quality rerun deferred** |
| Next work | **M3.1 contract → M3.2 dry-run → M3.3 single-sample runner** |
| Cleanup | **Plan only; failed sample-1 dir = delete candidate later** |

**Immediate next action (planning):** Author M3.1 orchestrator contract document; do not execute stages or delete artifacts.
