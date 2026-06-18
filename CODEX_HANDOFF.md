# CODEX_HANDOFF.md

## Latest commit
- Commit: 0ba16c0d747e35adf5239b988d45b39aa58fc928
- Title: Fix BOX raw modal discovery worker env propagation

## What changed
- Added `WORKER_CONTEXT_ENV_KEYS = ("SHAPE", "BOX_RAW_MODAL_DISCOVERY")`.
- `production_worker_subprocess_env()` now copies those two parent env vars into the solver-mkl worker subprocess env when set.
- Added a lightweight test asserting `SHAPE=box` and `BOX_RAW_MODAL_DISCOVERY=1` propagate into the worker env builder.
- Added/committed `AGENTS.md` project instructions.

## Files changed
- `AGENTS.md`
- `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_worker_run_lib.py`
- `FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_box_raw_modal_discovery_test.py`

## What did not change
- No solver behavior changes.
- No filter behavior changes.
- No assembly changes.
- No frequency-band changes.
- CLASSIC behavior unchanged.

## CLASSIC risk
- PASS
- The change only propagates two env vars into production worker subprocesses. The diagnostic remains gated by `shape=box` plus `BOX_RAW_MODAL_DISCOVERY=1`, so CLASSIC does not enable the BOX raw diagnostic.

## Tests run in Codex
- No tests completed in Codex.
- A targeted pytest command was started but aborted before producing results.

## Expected VM validation
```bash
git pull

bash tools/run_shape_fom_smoke.sh

# If needed, reset only BOX sample 000 before rerun.
RUN="FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/box_sample_000/runs/box_sample_000_box_fom_v1"
rm -rf "$RUN"

SHAPE=box START=0 COUNT=1 WORKERS=3 BOX_RAW_MODAL_DISCOVERY=1 \
  bash tools/run_shape_fom_overnight_batch.sh

RUN="FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/box_sample_000/runs/box_sample_000_box_fom_v1"
find "$RUN/worker_results" -name raw_modal_diagnostic.jsonl -exec wc -l {} +
find "$RUN" \( -path '*/raw_solver_candidate_catalog.json' -o -path '*/unfiltered_mode_catalog.json' \) -print

python3 FEM/experiments/active_domain_validation/physics_integrity/scripts/evaluate_modal_discovery_audit.py \
  --run-root "$RUN" \
  --shape box
```

## Expected success signs
- `BOX_RAW_MODAL_DISCOVERY_WORKER_ENV shape=box enabled=1`
- Raw diagnostic files exist.
- Raw/unfiltered catalogs exist.
- Modal audit reports `candidate_level_diagnostics_available=True`.
- Raw vs filtered section exists.

## If validation fails
- Check worker logs for `BOX_RAW_MODAL_DISCOVERY_WORKER_ENV shape=box enabled=1`.
- Check `worker_results/*/solver_result.json` for `box_raw_modal_discovery`.
- Confirm `SHAPE=box` and `BOX_RAW_MODAL_DISCOVERY=1` are present in the batch environment.
- Confirm aggregation ran after workers and copied raw/unfiltered catalogs into `validation/`.
