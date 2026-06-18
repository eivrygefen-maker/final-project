# CODEX_HANDOFF.md

## Files inspected
- `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_st_sinvert_solver_lib.py`
- `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_solve_target_list.py`
- `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_aggregate_worker_results.py`
- `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_box_raw_modal_discovery_lib.py`

## Tiny safety/reporting changes made
- Preserved the broad discovery band when BOX bypasses only the per-target window.
- Added `box_target_window_bypass_applied` to BOX raw catalog rows/CSV.
- Added a lightweight test that BOX bypass does not accept out-of-band candidates.

## Exact reason modes repeat
- Sigma is passed per target, but every target accepts all converged Ritz pairs returned by SLEPc.
- `collect_accepted_st_modes()` loops `eps.getEigenpair(i)` for all converged slots and appends every passing mode.
- There is no post-solve ranking by `abs(frequency_hz - target_hz)`.
- There is no per-target novelty rule before aggregation.
- Therefore if many target solves return the same stable BOX Ritz pairs, those same frequencies are accepted repeatedly and dedupe later collapses them to 15 true clusters.

## Sigma / shift usage
- `target_lambda = (2*pi*target_hz)^2` in `run_checkpoint_st_target()`.
- `configure_eps_krylovschur_sinvert()` sets:
  - `eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)`
  - `eps.setTarget(target_lambda)`
  - `st.setType(SINVERT)`
  - `st.setShift(target_lambda)`
- Static code review says sigma is wired correctly.
- VM should confirm `configure_meta.target_lambda` changes with `target_frequency_hz`.

## Candidate sorting
- Solver candidates are consumed in SLEPc EPS slot order.
- `accepted_frequencies_hz` is sorted by absolute frequency only after acceptance.
- Aggregation sorts by absolute frequency for catalog/dedupe.
- No code sorts candidates by distance to the requested target.

## nev / ncv / which / sigma suitability
- Current worker defaults are `nev=12`, `ncv=24`, `which=TARGET_MAGNITUDE`.
- This is CLASSIC-era policy reused by BOX.
- It can repeatedly return the same nearby/easy Ritz set across many BOX targets.
- For BOX discovery, the missing piece is target-aware candidate selection/novelty, not dedupe and not wider frequency bands.

## Dedupe
- Dedupe is behaving correctly.
- It groups repeated accepted rows by frequency tolerance and reports 15 true frequency clusters.
- Do not disable or widen dedupe.

## Minimal BOX-only fix proposal
- Do not change CLASSIC.
- Add BOX-only post-solve diagnostics first: per target, record candidate distance to target and rank by distance.
- Then add BOX-only acceptance policy that keeps only the nearest in-band candidates per target, or only candidates inside a local target neighborhood, while preserving residual/sanity/physical checks.
- If repeats persist, test a BOX-only solver policy variant: larger `nev/ncv` or `TARGET_REAL` vs `TARGET_MAGNITUDE`, validated on VM only.
- Keep dedupe unchanged.

## CLASSIC risk
- Investigation: PASS.
- Tiny safety/reporting changes are BOX diagnostic / BOX bypass scoped.
- Proposed solver policy changes would be MEDIUM unless strictly shape-gated and backed by CLASSIC unchanged tests.

## Lightweight tests
- Passed:
  - `python -m py_compile FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_st_sinvert_solver_lib.py`
  - `python -m py_compile FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_box_raw_modal_discovery_lib.py`
  - `python -m py_compile FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_box_raw_modal_discovery_test.py`
  - `python -m py_compile FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_solve_target_list.py`

## VM validation commands
```bash
git pull
python3 FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_box_raw_modal_discovery_test.py
RUN="FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/box_sample_000/runs/box_sample_000_box_fom_v1"
jq '.targets[] | {target:.target_frequency_hz, lambda:.target_lambda, cfg:.configure_meta}' "$RUN/worker_results/"*/solver_result.json | head -80
jq -r '.targets[] as $t | ($t.accepted_modes[]? | (.frequency_hz-$t.target_frequency_hz) as $d | [.frequency_hz,$t.target_frequency_hz,(if $d < 0 then -$d else $d end)] | @tsv)' "$RUN/worker_results/"*/solver_result.json | sort -n | head -80
jq -r '.rows[] | (.frequency_hz-.target_hz) as $d | [.frequency_hz,.target_hz,(if $d < 0 then -$d else $d end),.box_target_window_bypass_applied] | @tsv' "$RUN/validation/raw_solver_candidate_catalog.json" | sort -n | head -80
jq '.dedupe_merge_groups | length' "$RUN/aggregation/aggregation_result.json"
```
