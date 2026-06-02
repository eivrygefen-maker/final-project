# M2.2 dry-run orchestrator preview

- generated_utc: `2026-06-02T11:02:40Z`
- samples_jsonl: `C:\projects\final-project\final-project\FEM\experiments\active_domain_validation\physics_integrity\pipeline_runs\specs\m2_1_pilot_3_samples.jsonl`
- will_execute: `False`

## Summary

- sample_count: `3`
- timing_count: `2`
- rich_count: `1`
- synthesis_count: `1`
- placeholder_warning_count: `3`

## Per sample

| sample_id | run_id | mode | A | B | C | placeholder_warning |
|---|---|---|---|---|---|---|
| lhs_pilot_001_timing | lhs_pilot_001_timing | timing | PENDING | PENDING | SKIPPED | True |
| lhs_pilot_002_timing | lhs_pilot_002_timing | timing | PENDING | PENDING | SKIPPED | True |
| lhs_pilot_003_synthesis | lhs_pilot_003_synthesis | synthesis | PENDING | PENDING | PENDING | True |

## Warnings

- `placeholder_parameter_payload` indicates this is orchestration smoke preview only.
- `physical_lhs_ready=false` means no physical LHS interpretation should be made.
- No commands were executed; this file is preview metadata only.
