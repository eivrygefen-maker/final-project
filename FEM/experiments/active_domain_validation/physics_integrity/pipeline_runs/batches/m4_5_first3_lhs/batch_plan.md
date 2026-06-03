# M4.5.1 batch dry-run — m4_5_first3_lhs

**will_execute=false** — planning only; no solvers.

- Reference (frozen, not in batch): `sample_001` / `sample_001_m4dry1`
- Samples: sample_002, sample_003, sample_004

## Reuse policy

| Status | Meaning | Write run tree on dry-run |
|--------|---------|---------------------------|
| `planned_new_run` | No run dir | yes |
| `already_complete_reuse` | E2E PASS | no |
| `resume_possible` | Partial tree | only with `--force` |
| `requires_review` | FAIL artifacts | only with `--force` |

## Per-sample

- **sample_002** (`sample_002_m45dry1`): planned_new_run — tree written
- **sample_003** (`sample_003_m45dry1`): planned_new_run — tree written
- **sample_004** (`sample_004_m45dry1`): planned_new_run — tree written

## Non-goals

- No batch execution (M4.5)
- No Stage C / rich modal / cleanup / promotion

Generated: 2026-06-03T21:40:19Z
