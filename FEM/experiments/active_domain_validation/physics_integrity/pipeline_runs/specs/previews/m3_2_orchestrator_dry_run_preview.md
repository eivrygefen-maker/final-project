# M3.2 orchestrator dry-run preview

- generated_utc: `2026-06-02T18:52:21Z`
- samples_jsonl: `FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/specs/m3_2_dry_run_samples.jsonl`
- will_execute: `False`

## Summary

- sample_count: `3`
- timing_count: `2`
- synthesis_count: `1`
- ready_count: `3`
- blocker_count: `0`

## Per sample

| sample_id | run_id | mode | A | B | C | blockers | ready |
|---|---|---|---|---|---|---|---|
| lhs_pilot_001_timing | lhs_pilot_001_timing_m3dry | timing | PENDING | PENDING | SKIPPED | 0 | True |
| lhs_pilot_002_timing | lhs_pilot_002_timing_m3dry | timing | PENDING | PENDING | SKIPPED | 0 | True |
| lhs_pilot_003_synthesis | lhs_pilot_003_synthesis_m3dry | synthesis | PENDING | PENDING | PENDING | 0 | True |

## Notes

- Dry-run only: no Stage A/B/C execution, no runtime manifests, no index append.
- Python paths target VM production/solver-mkl environments.
- `run_id` keys output directories; `sample_id` keys config overlays.
