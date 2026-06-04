# B3 pipeline runtime hygiene

**Operational requirement:** FEM pipeline execution produces hundreds of JSON, mesh, modal, and log files. These must not pollute Git history or IDE indexing (VS Code / Cursor file watchers caused VM overheating when scanning `pipeline_runs/guitars/**`).

## Default rule

1. **Do not commit** generated runtime artifacts unless explicitly requested for a milestone snapshot.
2. **Write heavy outputs** only under ignored directories (see table below).
3. **Keep in Git** only specs, schemas, scripts, and documentation.

## Ignored paths (`.gitignore` + `.cursorignore`)

| Directory | Contents |
|-----------|----------|
| `pipeline_runs/guitars/` | Full per-run trees: scout, lprod, mesh/checkpoints, worker_results, aggregation, freeze, logs |
| `pipeline_runs/batches/` | Batch manifests, plans, per-sample command exports |
| `pipeline_runs/scout_density_reports/` | M3.4 scout density reports and summaries |
| `pipeline_runs/config_overlays/` | Regenerable resolved overlay JSON |
| `pipeline_runs/logs/` | Pipeline-wide runtime logs |

Additional patterns under `pipeline_runs/**`: `worker_results/`, `aggregation/`, `freeze/`, `lprod/checkpoint/`, `modes_catalog.jsonl`, meshes (`.msh`), arrays (`.npz`).

## Tracked (source of truth)

- `pipeline_runs/specs/` — batch specs (e.g. `m4_5_small_lhs_batch_first3.json`)
- `pipeline_runs/schemas/` — JSON schemas and examples
- `pipeline_runs/README.md` — layout pointer (this policy)

## Editor excludes

- **VS Code:** repo-root `.vscode/settings.json` — `files.watcherExclude`, `files.exclude`, `search.exclude` for pipeline runtime trees.
- **Cursor:** repo-root `.cursorignore` mirrors Git ignores.

Reload the window after changing excludes if the explorer still shows large trees.

## Untracking files already in Git

If runtime files were committed earlier, remove from the index without deleting disk data:

```bash
git rm -r --cached \
  FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars \
  FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/batches \
  FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/scout_density_reports \
  FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/config_overlays \
  FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/logs
```

Then commit the `.gitignore` / hygiene update when ready.

## M4 execution

**Production batch (default):** `v2_b3_m4_lhs_production_batch.py --execute`  
**Single sample:** `v2_b3_m4_run_one_sample.py --production-mode` or `--m45-batch-mode` (validation spec)

Outputs land under `guitars/<sample_id>/runs/<run_id>/` and `batches/<batch_id>/` — local to the VM, not committed unless explicitly requested.

See `docs/B3_M4_PRODUCTION_PROMOTION_AUDIT.md` for promotion mapping and legacy deprecation.
