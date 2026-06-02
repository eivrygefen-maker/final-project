# Coarse frequency plan (M3.4-pre)

- generated_utc: `2026-06-02T20:55:03Z`
- schema: `b3_coarse_frequency_plan_v1`
- mode: `dry-run`
- calibration_status: `not_calibrated_yet`
- checkpoint_dir: `FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/st_worker_scaling_L_prod_lhs_pilot_001_timing_m3exec2`
- frequency_range_hz: `220.0` – `265.0`
- coarse_step_hz: `5.0`
- coarse_target_count: `10`

## Validated full9 evidence

- targets_hz: `[221.5, 227.0, 232.5, 238.0, 243.5, 249.0, 254.5, 260.0, 264.0]`
- note: `Validated timing pilot band; not the whole physical range`

## Proposed regions (thresholds not calibrated)

- **R_below_full9** `[220.0, 221.5]` — `not_calibrated_yet`: Inside planner band but below validated full9 targets
- **R_full9_validated** `[221.5, 264.0]` — `validated_by_m3_pilot_full9`: M3 pilot timing 9/9 PASS on m3exec2; historical ~5.5 Hz spacing — not proven optimal
- **R_above_full9** `[264.0, 265.0]` — `not_calibrated_yet`: Inside planner band but above validated full9 targets

## Frequency windows (mode counts pending coarse scan)

| window | range_hz | mode_count | modes_per_hz | status |
|--------|----------|------------|--------------|--------|
| W00 | [220.0, 225.0] | None | None | not_calibrated_yet |
| W01 | [225.0, 230.0] | None | None | not_calibrated_yet |
| W02 | [230.0, 235.0] | None | None | not_calibrated_yet |
| W03 | [235.0, 240.0] | None | None | not_calibrated_yet |
| W04 | [240.0, 245.0] | None | None | not_calibrated_yet |
| W05 | [245.0, 250.0] | None | None | not_calibrated_yet |
| W06 | [250.0, 255.0] | None | None | not_calibrated_yet |
| W07 | [255.0, 260.0] | None | None | not_calibrated_yet |
| W08 | [260.0, 265.0] | None | None | not_calibrated_yet |

## Diagnostic notes

- Zone density thresholds are not calibrated yet; regions are hypotheses only.
- full9 band is validated M3 pilot evidence, not proof of global spectral coverage.
- Solver acceptance band is 220.0-265.0 Hz in v2_b3_st_sinvert_solver_lib.py.
- First coarse solve should use existing Stage B / target_density_experiment on a PASS checkpoint.
- Do not overwrite m3exec1/m3exec2 runtime diagnostics.
- WARN: checkpoint_dir not found on this host: C:\projects\final-project\final-project\FEM\experiments\active_domain_validation\physics_integrity\v2_mesh_convergence\diagnostics\st_worker_scaling_L_prod_lhs_pilot_001_timing_m3exec2

## Next step if approved (not executed by this tool)

```bash
# After explicit approval — solver-only, new output dir, isolated solver-mkl env:
# checkpoint: FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/st_worker_scaling_L_prod_lhs_pilot_001_timing_m3exec2
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_target_density_experiment.py \
  --checkpoint-dir FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/st_worker_scaling_L_prod_lhs_pilot_001_timing_m3exec2 \
  --reference-json FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/solver_benchmarks/checkpoint_solve_mkl_pardiso_full9_lhs_pilot_001_timing_m3exec2/result.json \
  --start-hz 220.0 --stop-hz 265.0 --spacings-hz 5.0
```
