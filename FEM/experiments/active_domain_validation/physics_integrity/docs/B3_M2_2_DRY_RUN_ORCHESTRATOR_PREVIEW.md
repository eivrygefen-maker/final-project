# B3 M2.2 — dry-run orchestrator preview spec (planning-only)

## 1. Purpose

Define a **non-executing command-preview** step that reads the M2.1 pilot JSONL and produces the exact intended Stage A/B/C command plan, environment handoff, expected output paths, and manifest status transitions **before any real execution**.

This is a planning/specification artifact only.

## 2. Pilot type clarification

The current M2.1 sample file:

- `pipeline_runs/specs/m2_1_pilot_3_samples.jsonl`

is an **orchestration smoke plan**, not a physical mini-LHS, because parameter deltas are placeholders:

- `geometry_delta = {}`
- `material_delta = {}`

Interpretation rule:

- Valid for orchestration/manifest/command preview.
- Not valid for physical LHS coverage claims.

Follow-up requirement:

- M2.3 must define real near-baseline geometry/material perturbations before physical LHS claims.

## 3. Input contract

Primary input:

- `FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/specs/m2_1_pilot_3_samples.jsonl`

Each row should provide:

- `sample_id`
- `mesh_level`
- `target_set`
- `selection_reason`
- policy flags (`timing_only`, `rich_requested`, `synthesis_requested`)
- source refs and parameter payload
- initial stage statuses

## 4. Expected dry-run output (per sample)

For each sample, preview must emit:

- `run_id` (proposed)
- `mode` (`timing` / `rich` / `synthesis`)
- `selection_reason`
- `target_set`
- predicted Stage A command (not executed)
- predicted Stage B command (not executed)
- predicted Stage C command or `SKIPPED`
- expected env per stage
- expected output path links:
  - Stage A checkpoint dir
  - Stage B solve dir
  - Stage B rich dir (if applicable)
  - Stage C synthesis dir (if applicable)
- initial manifest statuses

No subprocess calls are permitted in this step.

## 5. Manifest behavior policy for dry-run preview

Decision for first version:

- **Do not** create runtime manifests under `pipeline_runs/manifests/`.
- Only generate preview artifacts under non-runtime planning locations, for example:
  - `pipeline_runs/specs/` (tracked planning artifact), or
  - `pipeline_runs/logs/preview/` (runtime preview output, ignored).

Reason:

- avoids polluting runtime state with non-executed manifests,
- keeps dry-run preview clearly separate from actual run registration.

## 6. Safety requirements

Preview step must:

- execute **no** Stage A/B/C commands,
- execute **no** subprocesses,
- perform no deletions/moves,
- avoid overwriting validated reference artifacts,
- emit explicit warning if `parameter_payload` is placeholder/empty,
- emit explicit warning that this is not physical LHS validity.

## 7. Environment handling in preview

Preview should:

- print/document expected env commands only, e.g.:
  - Stage A/C: production `.venv`
  - Stage B: `solver-mkl`

Preview should **not**:

- activate environments,
- validate package imports by execution,
- mutate shell state.

## 8. Relationship to future orchestrator

M2.2 preview is the blueprint for a later thin orchestrator:

1. Parse sample JSONL row.
2. Resolve mode/policy and command templates.
3. Resolve expected output path templates.
4. Produce deterministic stage transition intent.

Later orchestrator versions can reuse the same mapping logic and simply swap:

- `preview only` -> `execute + update runtime manifests`.

## 9. Open question (M2.3 dependency)

Should M2.3 define real perturbation deltas before any Stage A/B execution?

Recommendation:

- **Yes** for physical LHS pilot claims.
- For pure orchestration smoke execution only, placeholders are acceptable if explicitly labeled as non-physical.

## 10. Final recommendation (next step)

After this spec is approved, implement a **dry-run-only helper** that:

- reads `m2_1_pilot_3_samples.jsonl`,
- computes command/environment/output previews per sample,
- writes a preview report file (JSON + optional MD),
- does not call Stage scripts,
- does not create runtime manifests,
- does not execute any subprocesses.

Suggested helper target:

- `scripts/v2_b3_lhs_orchestrator_preview.py` (preview-only)

Suggested output target:

- `pipeline_runs/specs/m2_2_pilot_3_sample_command_preview.json`
