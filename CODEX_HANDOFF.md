# CODEX_HANDOFF.md

## Files inspected
- `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_box_raw_modal_discovery_lib.py`
- `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_st_sinvert_solver_lib.py`
- `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_solve_target_list.py`
- `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_minimal_rom_compaction.py`
- `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_aggregate_worker_results.py`

## Root cause / strongest hypothesis
- Strongest hypothesis: reporting blind spot, not dedupe.
- Prior raw catalog rows did not carry `box_target_window_bypass_applied`, so VM saw `None` even when BOX bypass could have made `would_pass_normal_filters=True`.
- `would_pass_normal_filters=True` with large target distance can occur when BOX target-window bypass is active and other checks pass.
- Also fixed a safety issue: BOX bypass now preserves the broad discovery band and bypasses only the per-target window.

## Reporting bug or acceptance bug
- Reporting bug for `box_target_window_bypass_applied=None`.
- Acceptance behavior is intentional from the BOX-only target-window bypass, but needed clearer diagnostics.
- No evidence found that raw catalog `target_hz` is rewritten by aggregation.

## Raw row target metadata path
- `run_checkpoint_st_target()` creates per-target `diagnostic_candidates`.
- `write_worker_diagnostic_from_solver_targets()` iterates each solver `target_row`.
- `build_catalog_row()` attaches `target_hz` from that same `target_row`.
- `merge_box_raw_catalogs_for_run()` concatenates `worker_results/<chunk>/raw_modal_diagnostic.jsonl`; it does not copy rows into other target contexts.

## Compaction
- Yes, compaction can remove `worker_results`.
- `v2_b3_m4_minimal_rom_compaction.py` retains validation/aggregation raw catalogs but treats top-level `worker_results` as deletable.
- That explains `solver_result files: 0` after compaction.

## Diagnostic-only changes made
- Added raw catalog fields:
  - `distance_to_target_hz`
  - `acceptance_target_hz`
  - `acceptance_window_hz`
  - `inside_acceptance_window_at_decision`
  - `box_target_window_bypass_applied`
  - `source_target_index`
  - `source_chunk_id`
  - `source_sigma_lambda`
- Added these fields to CSV where relevant.
- Added lightweight test that target/window/sigma metadata stays separate between two target rows.

## CLASSIC risk
- PASS.
- Changes are BOX raw diagnostic/reporting scoped plus a BOX bypass safety guard.
- No CLASSIC solver, assembly, boundary, dedupe, or frequency-range change.

## Lightweight tests
- Passed:
  - `python -m py_compile FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_st_sinvert_solver_lib.py`
  - `python -m py_compile FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_box_raw_modal_discovery_lib.py`
  - `python -m py_compile FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_box_raw_modal_discovery_test.py`

## VM validation commands
```bash
git pull
python3 FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_box_raw_modal_discovery_test.py

RUN="FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/box_sample_000/runs/box_sample_000_box_fom_v1"
rm -rf "$RUN"
SHAPE=box START=0 COUNT=1 WORKERS=3 BOX_RAW_MODAL_DISCOVERY=1 bash tools/run_shape_fom_overnight_batch.sh

RUN="FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/box_sample_000/runs/box_sample_000_box_fom_v1"
jq '.rows[0] | keys' "$RUN/validation/raw_solver_candidate_catalog.json"
jq -r '.rows[] | select(.would_pass_normal_filters==true) | [.frequency_hz,.target_hz,.distance_to_target_hz,.inside_acceptance_window_at_decision,.box_target_window_bypass_applied,.source_target_index,.source_sigma_lambda] | @tsv' "$RUN/validation/raw_solver_candidate_catalog.json" | sort -n | head -80
jq -r '.rows[] | select(.would_pass_normal_filters==true and .inside_acceptance_window_at_decision==false) | [.frequency_hz,.target_hz,.box_target_window_bypass_applied] | @tsv' "$RUN/validation/raw_solver_candidate_catalog.json" | head -40
find "$RUN/worker_results" -name solver_result.json | wc -l
```
