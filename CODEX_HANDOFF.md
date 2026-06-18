# CODEX_HANDOFF.md

## Files changed
- `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_st_sinvert_solver_lib.py`
- `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_solve_target_list.py`
- `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_box_raw_modal_discovery_test.py`
- `CODEX_HANDOFF.md`

## Exact logic changed
- Added `box_bypass_target_window_acceptance` to the accepted-mode path.
- `shape_name == "box"` in `v2_b3_checkpoint_solve_target_list.py` enables the bypass for BOX workers only.
- In `collect_accepted_st_modes()`, BOX candidates outside the per-target window may now pass only if all other existing checks pass:
  - finite positive frequency
  - residual `eps_ok`
  - inactive DOF check
  - boundary DOF check
  - not lambda-near-unity
  - support/physical participation check
- Raw diagnostics still preserve `inside_target_window` and `normal_filter_rejection_reasons`.
- Added `box_target_window_bypass_applied=true` on accepted/raw diagnostic rows when the BOX-only bypass is actually used.
- Added solver/worker result field `box_bypass_target_window_acceptance`.

## What stayed unchanged
- CLASSIC does not enable the bypass.
- Discovery frequency band is unchanged.
- Per-target windows are still reported and still used for CLASSIC.
- Solver algorithm, assembly, boundary conditions, residual checks, physical filters, and dedupe are unchanged.
- Non-BOX callers use the same default behavior as before.

## CLASSIC risk
- PASS for intended behavior: bypass defaults false and is only set when resolved `shape_name == "box"`.
- Residual risk is low/medium because the edited acceptance helper is shared; lightweight tests cover the CLASSIC outside-window rejection case.

## Lightweight tests run
- Passed:
  - `python -m py_compile FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_st_sinvert_solver_lib.py`
  - `python -m py_compile FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_solve_target_list.py`
  - `python -m py_compile FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_box_raw_modal_discovery_test.py`
- Attempted but not completed in Codex Windows:
  - `python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_box_raw_modal_discovery_test.py`
  - Blocked at import: `ModuleNotFoundError: No module named 'resource'`.

## VM validation commands
```bash
git pull

python3 FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_box_raw_modal_discovery_test.py

RUN="FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/box_sample_000/runs/box_sample_000_box_fom_v1"
rm -rf "$RUN"

SHAPE=box START=0 COUNT=1 WORKERS=3 BOX_RAW_MODAL_DISCOVERY=1 \
  bash tools/run_shape_fom_overnight_batch.sh

RUN="FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/box_sample_000/runs/box_sample_000_box_fom_v1"
jq '.box_bypass_target_window_acceptance' "$RUN/worker_results/"*/solver_result.json
jq '[.rows[] | select(.box_target_window_bypass_applied==true)] | length' \
  "$RUN/validation/raw_solver_candidate_catalog.json"
jq '.raw_vs_filtered_analysis' "$RUN/validation/modal_discovery_audit.json"
python3 FEM/experiments/active_domain_validation/physics_integrity/scripts/evaluate_modal_discovery_audit.py \
  --run-root "$RUN" --shape box
```
