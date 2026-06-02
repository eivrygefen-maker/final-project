# B3 M2 LHS integration plan (planning-only)

## 1. Purpose

Define how the validated official A+B/C pipeline is used for the first controlled LHS runs without major refactor.

This M2 plan is:

- not full production migration,
- not GUI integration work,
- not ROM/STK full ingestion design,
- not cleanup/deprecation execution.

It is a first-batch orchestration contract for A/B (+ optional rich/C) with manifest-driven provenance.

---

## 2. First LHS pilot scope

Recommended pilot size: **5-10 samples** (preferred: 6).

Why this is safe:

- enough variability to test policy and reliability,
- small enough to manually inspect manifests/results,
- bounded disk/runtime risk while stage contracts are new.

Pilot PASS criteria:

- all samples have run manifests with consistent status transitions,
- at least 80% of samples complete Stage A and Stage B PASS,
- rich subset produces valid `rich_modal` bundle when requested,
- synthesis subset produces valid Stage C outputs when requested,
- failures (if any) are classified with explicit reasons, no silent drops.

Not attempted in first pilot:

- broad target planner/zones rollout,
- global rich export for all samples,
- automatic cleanup/retention actions,
- GUI/interactive orchestration.

---

## 3. LHS input contract (v0)

Each sample should minimally include:

- `sample_id`
- geometry/material parameter payload
- `mesh_level` (initially `L_prod`)
- source references:
  - `configs/v2_mesh_convergence_manifest.json`
  - canonical base config reference (`configs/coupled_physical_core_v2.json`)
- policy flags:
  - `timing_only` (A+B)
  - `rich_requested` (A+B rich)
  - `synthesis_requested` (A+B rich + C)
- target policy:
  - initial `target_set=full9`

Input format for first pilot:

- **manual prepared JSONL** (recommended),
- one row per sample with explicit policy flags.

Rationale: keeps first pilot deterministic and reviewable before integrating larger existing pools.

---

## 4. M2 orchestrator contract (no implementation yet)

Future orchestrator responsibilities:

1. Create/update run manifest via `scripts/v2_b3_pipeline_run_manifest.py`.
2. Execute Stage A in production `.venv`.
3. Execute Stage B in `solver-mkl`.
4. Execute optional Stage B rich when policy requires.
5. Execute optional Stage C in production `.venv`.
6. Update stage statuses (`PENDING`/`PASS`/`FAIL`/`SKIPPED`) with reason fields.
7. Write per-stage logs and command provenance.
8. Never overwrite validated official reference artifacts.

M2 shape recommendation:

- start with **one thin orchestrator script** for single-sample execution,
- add batch driver only after single-sample flow is stable.

---

## 5. Stage status transition rules

Initial state:

- `A.status = PENDING`
- `B.status = PENDING`
- `C.status = PENDING` if synthesis requested; else `SKIPPED`

Stage A PASS condition:

- `checkpoint_export_manifest.json` exists and `status == PASS`.

Stage B PASS condition:

- `result.json` exists and solve status indicates PASS.

Stage B rich PASS condition (when rich requested):

- Stage B PASS, plus:
  - `rich_modal/modes_active.npz`
  - `rich_modal/modes_catalog.jsonl`
  - `rich_modal/rich_modal_manifest.json`
  all exist.

Stage C PASS condition (when synthesis requested):

- `modes_synthesis.json`
- `modes_synthesis.md`
- `rich_modal_post_manifest.json`
  all exist.

Stage C SKIPPED condition:

- rich/synthesis not requested by policy.

Failure handling:

- any unmet required condition marks stage `FAIL` with explicit reason.

---

## 6. Environment handoff policy

Stage A environment:

- production `.venv`
- DOLFINx/PETSc available

Stage B environment:

- `solver-mkl`
- MKL/PARDISO/PETSc/SLEPc stack available

Stage C default environment:

- production `.venv`
- numpy-only default post path (no petsc4py/dolfinx requirement)

Stage C `best_effort` note:

- optional path requiring DOLFINx subprocess support,
- **excluded from first LHS pilot unless explicitly approved**.

Logging requirements:

- each stage log should record active env identity (venv and key solver probes),
- wrong-env failures are classified as stage `FAIL_ENV`.

---

## 7. Failure and retry policy v0

Conservative first-pilot rules:

- no infinite retries,
- Stage A `FAIL` stops sample (B/C become `SKIPPED`),
- Stage B may retry **once** only for likely env/setup failures,
- Stage C failure does not invalidate A/B timing result; mark synthesis `FAIL`,
- failed samples remain indexed with reason, no cleanup during pilot.

---

## 8. Target policy v0

First pilot target policy:

- use validated `target_set=full9` for all pilot samples.

Rationale:

- reuses validated baseline behavior,
- isolates orchestration policy risks from target-planner risks.

Target planner/zones:

- planned for later phase,
- not required to start first LHS pilot.

Historical target density/alignment:

- informs later planner design,
- does not block first pilot.

Manifest recording:

- run manifest should include target set (`full9`) in stage command/provenance metadata.

---

## 9. Rich/synthesis subset selection rule v0

Deterministic first-pilot policy:

- all pilot samples: Stage A + Stage B timing/scalar summaries,
- **1-2 samples**: Stage B rich enabled,
- **1 sample** (from rich subset): Stage C synthesis.

Official reference sample:

- remains registered in manifests/index,
- not rerun in pilot unless explicitly requested.

Suggested `selection_reason` values:

- `lhs_timing_baseline`
- `lhs_rich_subset_v0`
- `lhs_synthesis_subset_v0`
- `official_validated_pass_reference`

Storage implication:

- rich export remains opt-in to control volume and runtime.

---

## 10. Scalar summary outputs for LHS

Minimum summaries to capture from Stage B `result.json`:

- solve status,
- accepted frequencies (or equivalent summary frequency list),
- accepted mode count / per-target accepted counts,
- target provenance (`target_set`, target list),
- quality/residual indicators (where present),
- wall/solve timing,
- rich export requested/status/paths (if rich requested).

Policy:

- **required before full LHS rollout**,
- for first pilot, summaries can be extracted post-run from existing `result.json` if no extractor helper exists yet.

---

## 11. Output/run-root policy

Runtime root:

- `FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/`

Runtime subpaths:

- `manifests/` (run JSON)
- `index/` (JSONL)
- `logs/` (stage logs)
- `specs/` (tracked templates/examples)

Policy:

- manifests/index/logs are runtime data (ignored),
- specs/docs are tracked.

Near-term transition:

- stage outputs may remain in current diagnostics directories in first pilot,
- wrapper records links in run manifests,
- full output normalization (checkpoints/solves/synthesis under unified root) is a later phase.

---

## 12. Safety gates before first LHS pilot

Required gates:

- git tracked source clean,
- official A+B+C archive exists and verified,
- M1.5 manifest helper working,
- this M2 plan reviewed/approved,
- env checks for Stage A/B/C verified,
- no cleanup/deletion coupled to pilot execution,
- disk space check completed,
- dry-run command preview reviewed before execute.

---

## 13. What explicitly does not block first LHS

- GUI integration updates,
- full ROM/STK ingestion pipeline,
- MAC-based dedupe,
- advanced target planner/zones,
- deep legacy cleanup/deprecation,
- migration out of `experiments/`.

---

## 14. M2 implementation options (for later execution)

### Option A: Document + manual commands + manifest registration

Pros:

- zero new runtime code risk,
- easiest to audit manually.

Risks:

- operator burden and inconsistency,
- weaker repeatability for larger batches.

Expected files:

- pilot sample list,
- run manifests + logs,
- existing Stage outputs.

### Option B: Thin single-sample orchestrator (A/B first)

Pros:

- controlled automation with low complexity,
- enforces status transitions consistently.

Risks:

- still requires external batching,
- env handoff logic must be correct.

Expected files:

- one orchestrator script,
- run manifests/index/logs,
- linked Stage outputs.

### Option C: Batch driver with policy gates (A/B/rich/C)

Pros:

- scalable and policy-driven from the start.

Risks:

- higher initial complexity,
- harder to debug before policy settles.

Expected files:

- batch driver + policy config,
- richer index/provenance,
- more runtime artifact volume.

Recommendation for first pilot:

- **Option B** (thin single-sample orchestrator), then add a simple batch loop after 1-2 stable samples.

---

## 15. Final recommendation (next step)

Create a **3-sample pilot manifest plan** (no execution yet):

- 2 timing-only samples (`lhs_timing_baseline`)
- 1 rich+synthesis sample (`lhs_synthesis_subset_v0`)

Use `v2_b3_pipeline_run_manifest.py` to pre-register these runs with `PENDING` stage statuses and explicit selection reasons, then review before any Stage execution.
