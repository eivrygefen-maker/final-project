# B3 M4.5-pre — First successful M4 end-to-end run

## 1. Milestone statement

**This is the first successful M4 end-to-end run for one guitar sample.**

Frozen run: `sample_001_m4dry1` (`sample_001`) on VM. Documentation and metadata only; no solver re-execution.

Pipeline proven:

```text
sample input → scout → density zones → adaptive L_prod target plan
→ L_prod mesh → L_prod checkpoint → all worker chunks → full aggregation
```

Regenerate live tables and `freeze/` artifacts on the VM with:

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_freeze_first_e2e_run.py \
  --run-dir FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1 \
  --force
```

## 2. Pipeline stages and status

| Stage | Status | Primary artifacts |
|-------|--------|-------------------|
| Stage 0 — sample/config | **PASS** | `sample/sample_input.json`, `sample/resolved_core_config.json` |
| Stage 1 — scout mesh/checkpoint | **PASS** | `scout/scout_plan.json`, `scout/checkpoint/` |
| Stage 2 — scout discovery | **PASS** | `scout/scout_result.json`, `scout/density_zones.json` |
| Stage 3 — zones + adaptive L_prod target plan | **PASS** | `lprod/lprod_target_plan.json` |
| Stage 4 — L_prod mesh/checkpoint | **PASS** | `lprod/checkpoint/checkpoint_export_manifest.json`, `lprod/mesh/` |
| Stage 5 — L_prod workers | **PASS** | `worker_results/*/worker_result.json` (11/11) |
| Stage 6 — aggregation | **PASS** | `aggregation/aggregation_result.json`, `aggregation/modes_summary.json` |

## 3. Key run metrics

| Field | Value |
|-------|-------|
| sample_id | **sample_001** |
| run_id | **sample_001_m4dry1** |
| target_count | **56** |
| worker_chunk_count | **11** |
| completed_chunk_count | **11** |
| failed_chunk_count | **0** |
| raw_mode_count | **733** |
| deduped_mode_count | **568** |
| dedupe_tolerance_hz | **0.05** |
| final_aggregation_ready | **true** |
| terminal_status | **LPROD_WORKERS_AND_AGGREGATION_PASS** |
| aggregation_status | **AGGREGATION_PASS** |

## 4. Worker summary

All 11 planned chunks completed with real solver-mkl worker results (`PASS` or `PASS_WITH_WARNING`).

Per-chunk `targets_attempted`, `targets_passed`, `unique_modes`, warnings, and errors are recorded in each `worker_results/<chunk_id>/worker_result.json`. After running the freeze script on the VM, see `freeze/first_end_to_end_run_manifest.json` for the collected worker summary table.

| chunk_id | status (expected) |
|----------|-------------------|
| sample_001_chunk_01 … sample_001_chunk_11 | PASS |

## 5. Adaptive planning summary

- Scout spacing: **7.5 Hz**
- L_prod zone policy: **6.0 / 9.0 / 12.5 Hz** (`ZONE_1_dense` / `ZONE_2_medium` / `ZONE_3_sparse`)
- Target generation policy: `gapless_grid_v2_segment_endpoint_plus_coverage_repair`
- Chunk policy version: `v1_1`
- Target coverage pass: **true**
- Coverage max gap: **0 Hz**
- Target count: **56**

## 6. Artifact index

See run-tree `freeze/artifact_index.md` (written by freeze script on VM). Essential artifacts:

| Artifact | Essential |
|----------|-----------|
| `aggregation/aggregation_result.json` | yes |
| `lprod/lprod_target_plan.json` | yes |
| `lprod/checkpoint/checkpoint_export_manifest.json` | yes |
| `worker_results/*/worker_result.json` (×11) | yes |
| `aggregation/modes_catalog.jsonl` | no |
| `aggregation/mode_frequency_plot.png` | no |
| `scout/scout_result.json` | no (optional) |

## 7. Explicit non-goals / not yet done

- Stage C / rich modal export not run
- No audio / STK export
- No production promotion
- No cleanup or archival of legacy runs
- Multi-guitar LHS batch not yet run
- Only `sample_001` validated end-to-end on VM

## 8. Next steps

- **M4.5** — small multi-guitar batch, 2–3 real LHS samples
- **M4.6** — validation/comparison against reference/legacy expectations
- **M4.7** — promote new pipeline as main path
- Cleanup/archive only after small batch passes

---

*Milestone record for M4.5-pre. Live freeze metadata: `pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1/freeze/`.*
