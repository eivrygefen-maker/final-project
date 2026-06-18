# AGENTS.md — Final Project Agent Instructions

## Project role

This repository is the final project FEM/ROM guitar simulation project.

Codex is allowed to inspect and edit code, but the VM is the only authoritative runtime environment.

Do not run heavy simulations or production validation inside Codex.

## Current project status

CLASSIC guitar pipeline:

* Frozen and protected.
* Valid historical baseline: 67 completed CLASSIC simulations.
* CLASSIC previously produced 500+ modes per completed simulation.
* Do not modify CLASSIC behavior, thresholds, validation profiles, outputs, or expected results unless explicitly requested.

BOX pipeline:

* Experimental shape-aware extension.
* BOX runs complete technically, but currently produce far too few modes, around 9–13 deduped modes.
* This is suspicious and under investigation.
* BOX changes must be shape-gated and must not affect CLASSIC.

ACOUSTIC pipeline:

* Future shape-aware extension.
* Do not modify unless explicitly requested.

## Hard rules

1. Do not change CLASSIC behavior.
2. Do not change CLASSIC thresholds.
3. Do not regenerate CLASSIC LHS.
4. Do not delete CLASSIC outputs.
5. Do not run full FEM, ROM, worker, STK, WAV, audio, or production simulations.
6. Do not run `tools/run_shape_fom_overnight_batch.sh`.
7. Do not run full test suites.
8. Do not make broad refactors.
9. Prefer small, focused patches.
10. If a change may affect CLASSIC, stop and report the risk before editing.

## Allowed actions

You may:

* read relevant files
* edit code
* add small targeted tests
* run lightweight syntax checks
* run small unit tests only when clearly lightweight
* provide exact VM commands for the user to run

You may not:

* execute heavy FEM/FOM/ROM simulations
* execute worker batches
* generate audio/WAV/STK outputs
* run long validation
* run production pipelines
* modify unrelated files

## Runtime validation

The user runs real validation in the VM.

When a task requires validation, provide VM commands only.

Do not claim runtime success unless the user provides VM output.

## Current priority

Immediate investigation:
`BOX_RAW_MODAL_DISCOVERY=1` must propagate into worker subprocesses.

Suspected issue:
The parent batch sees `BOX_RAW_MODAL_DISCOVERY=1`, but worker subprocesses may not inherit it.

Relevant expected behavior:

* Worker subprocess receives `BOX_RAW_MODAL_DISCOVERY`
* Worker subprocess receives `SHAPE=box`
* Worker logs:
  `BOX_RAW_MODAL_DISCOVERY_WORKER_ENV shape=box enabled=1`

Expected artifacts after VM rerun:

* `worker_results/<chunk_id>/raw_modal_diagnostic.jsonl`
* `aggregation/raw_solver_candidate_catalog.json`
* `aggregation/unfiltered_mode_catalog.json`
* `validation/raw_solver_candidate_catalog.json`
* `validation/unfiltered_mode_catalog.json`
* `validation/target_candidate_audit_merged.jsonl`

## Response style

Keep responses short and practical.

For each task, report:

* changed files
* what changed
* what did not change
* whether CLASSIC is affected
* lightweight tests run
* VM commands for validation

## Git / branch safety

Work in small patches.

Do not reformat unrelated files.

Do not commit generated runtime outputs.

Do not edit `.cursor/AGENT_CONTEXT.md` unless the task specifically concerns local Cursor handoff.

## If unsure

Stop and ask before changing code.
