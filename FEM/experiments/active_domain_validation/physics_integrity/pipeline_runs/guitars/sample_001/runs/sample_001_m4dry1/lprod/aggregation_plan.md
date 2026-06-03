# Aggregation plan (dry-run) — sample_001

- chunks: **11**
- dedupe tolerance: **0.5 Hz**

## Outputs

- `aggregation_result_json`: `FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1/aggregation/aggregation_result.json`
- `modes_catalog_jsonl`: `FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1/aggregation/modes_catalog.jsonl`
- `modes_summary_json`: `FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1/aggregation/modes_summary.json`
- `modal_data_npz`: `FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1/aggregation/modal_data.npz`
- `mode_frequency_plot_png`: `FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1/aggregation/mode_frequency_plot.png`
- `runtime_summary_json`: `FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1/aggregation/runtime_summary.json`
- `warnings_and_failures_json`: `FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1/aggregation/warnings_and_failures.json`

## Rules
- **collect:** all accepted modes from worker_result.json per chunk
- **dedupe_tolerance_hz:** 0.5
- **sort:** by frequency_hz ascending
- **provenance_fields:** ['chunk_id', 'worker_id', 'target_hz', 'source_result_path']
- **validate:** every chunk terminal PASS/PARTIAL; all targets attempted; report missing chunks
