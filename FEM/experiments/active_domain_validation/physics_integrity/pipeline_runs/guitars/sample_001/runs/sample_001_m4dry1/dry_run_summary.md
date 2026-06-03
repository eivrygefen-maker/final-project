# M4.2 pipeline dry-run summary

- **will_execute:** false
- **sample_id:** sample_001
- **run_id:** sample_001_m4dry1
- **run_root:** `FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1/`

## Policy

- Scout: discovery on `L_scout_coarse`, sample-specific mesh path.
- Zones / targets: `pending_scout` until Stage 2 completes (M4.3).
- Workers: placeholder chunks only; status `pending_target_plan`.

## Planned counts

- Density bins (placeholder): 20
- Worker chunks (placeholder): 13
- Planned FCFS workers: 3

## Artifacts

| Path | Role |
|------|------|
| `sample/sample_input.json` | Copied LHS input |
| `sample/sample_resolved_config_manifest.json` | Stage 0 manifest stub |
| `scout/scout_plan.json` | Stage 1–2 command plan |
| `lprod/lprod_plan.json` | Stage 4–5 command plan |
| `pipeline_run_manifest.json` | Terminal manifest stub |

## Safety

- No mesh build, Stage A/B/C, or worker execution.
- Does not modify `v2_mesh_convergence/` outputs or legacy M2/M3 trees.

Generated: 2026-06-03T07:25:29Z
