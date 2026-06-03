# Worker commands (dry-run) — sample_001

- chunks: **11**
- FCFS: assign first N chunks; on finish assign next queued

| chunk | targets | est. s |
|-------|---------|--------|
| sample_001_chunk_01 | 7 | 665.0 |
| sample_001_chunk_02 | 7 | 665.0 |
| sample_001_chunk_03 | 6 | 570.0 |
| sample_001_chunk_04 | 5 | 475.0 |
| sample_001_chunk_05 | 5 | 475.0 |
| sample_001_chunk_06 | 5 | 475.0 |
| sample_001_chunk_07 | 5 | 475.0 |
| sample_001_chunk_08 | 4 | 380.0 |
| sample_001_chunk_09 | 4 | 380.0 |
| sample_001_chunk_10 | 4 | 380.0 |
| sample_001_chunk_11 | 4 | 380.0 |

## Example planned command (M4.4)

```bash
/home/vboxuser/solver-mkl/venv/bin/python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_solve_target_list.py --checkpoint-dir "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1/lprod/checkpoint" --targets-json "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1/worker_results/sample_001_chunk_01/chunk_targets.json" --factor-solver mkl_pardiso --output-dir "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_001/runs/sample_001_m4dry1/worker_results/sample_001_chunk_01" --dry-run
```
