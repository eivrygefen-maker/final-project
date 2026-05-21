# Production pipeline: 60–550 Hz (staged harvest)

## Flow

1. **`run_pipeline.py`** (optional LHS merge) → writes `FEM/SORTING/pipeline_merged_configs/sample_XXX.json`
2. **`fem_master_dynamic.py`** `--clean-start` → resets `temp_results/`, `temp_modes/`, fresh `candidates_log.json` with `pipeline_meta`
3. **Workers** (`fem_worker_single.py`) → SLEPc harvest window per spectral band; staged gate in `fem_harvest_filter`; `source_target_hz` on each mode
4. **Master merge** → same staged classifier; stamps `harvest_filter_policy`; flush pending results **after** harvest config is loaded
5. **`dynamic_filter_tuner.py`** → MMR split-quota 150 modes; uniqueness floor **0.04** (aligned with harvest)
6. **`package_rom.py`** → NPZ with `frequencies`, CSR `ev_*`, `source_target_hz`, `pipeline_harvest_filter_policy`, sweep bounds

## Hardcoded production constants

| Constant | Value | Location |
|----------|-------|----------|
| Coupled worker `num_modes` cap | **40** | `fem_master_dynamic.COUPLED_WORKER_NUM_MODES_CAP`, `fem_worker_single._DEFAULT_WORKER_NUM_MODES_CAP` |
| Harvest policy ID | `staged_v1_60_550` | `fem_harvest_filter.HARVEST_FILTER_POLICY_VERSION` |
| Staged crossover | 350 Hz | `solver.harvest_filter` in JSON |
| `p_frac` low / high | 0.10 / 0.03 | same |

## Clean start

`--clean-start` on master or `run_pipeline` removes scratch and reinitializes the log. It does **not** delete `selected_modes.csv` or existing snapshot NPZ files.

## Recommended command

See project README or run:

```powershell
python FEM/scripts/run_pipeline.py --sample-id sample_001 --max-workers 2 --clean-start `
  --config FEM/configs/guitar_3d.json
```

Or master only:

```powershell
python FEM/scripts/fem_master_dynamic.py --clean-start `
  --config FEM/SORTING/pipeline_merged_configs/sample_001.json `
  --sorting-root FEM/SORTING --hz-min 60 --hz-max 550 `
  --max-workers 2 --use-mpiexec --schedule spectral-bands+fill
```
