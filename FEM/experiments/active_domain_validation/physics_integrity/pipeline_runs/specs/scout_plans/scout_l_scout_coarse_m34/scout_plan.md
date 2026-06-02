# Coarse-mesh modal-density scout plan (dry-run)

- **Generated:** `2026-06-02T22:13:56Z`
- **will_execute:** `False`
- **Strategy:** coarse_fem_mesh_modal_density_scout

## Compact summary

- **will_execute:** `False`
- **run_id:** `scout_l_scout_coarse_m34`
- **mesh_level:** `L_scout_coarse`
- **mesh_exists:** `False`
- **checkpoint_dir_exists:** `False`
- **core_config_path:** `FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/config_overlays/scout_l_scout_coarse_m34/resolved_core_config.json`
- **core_config_mesh_file:** `FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/mesh/L_scout_coarse/baseline_coupled_v2.msh`
- **top_density:** `450.0`
- **back_density:** `830.0`
- **clamp_ribs:** `False`
- **readiness_status:** `PENDING_MESH`
- **baseline_material_fingerprint:** `True`
- **lhs_perturbation_applied:** `False`
- **mesh_controls_mm_plate_wood_air_min:** `3 / 8.5 / 11`

## L_prod verified sizing (mm)

- `wood_surface_size_m`: **7.0**
- `wood_thickness_size_m`: **1.0**
- `air_threshold_size_min_m`: **4.0**
- `air_threshold_size_max_m`: **50.0**
- `air_threshold_dist_min_m`: **15.0**
- `air_threshold_dist_max_m`: **250.0**
- **active_dim (m3exec2 ref):** 316017

## Proposed L_scout_coarse (mm)

- **reuse L_dev_coarse:** False

## Active dimension estimate

- Point estimate: **47941**
- Rough band: **[31522, 78064]**

## Warnings

- L_prod mesh file not found on this host: C:\projects\final-project\final-project\FEM\experiments\active_domain_validation\physics_integrity\v2_mesh_convergence\mesh\L_prod\baseline_coupled_v2.msh
- Scout mesh not built yet: C:\projects\final-project\final-project\FEM\experiments\active_domain_validation\physics_integrity\v2_mesh_convergence\mesh\L_scout_coarse\baseline_coupled_v2.msh

## Command previews (do not run without approval)

### Mesh build
```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_B3_scout_coarse_mesh_build.py
```

### Stage A
```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_export.py --mesh-level L_scout_coarse --B3-block-compose-backend csr_bulk --B3-synthesis-region-dofs off --core-config "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/config_overlays/scout_l_scout_coarse_m34/resolved_core_config.json" --output-dir "C:/projects/final-project/final-project/FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/st_worker_scaling_L_scout_coarse_scout_l_scout_coarse_m34"
```

### Stage B discovery
```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_target_density_experiment.py --checkpoint-dir "C:/projects/final-project/final-project/FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/st_worker_scaling_L_scout_coarse_scout_l_scout_coarse_m34" --start-hz 60.0 --stop-hz 550.0 --spacings-hz 15.0 --B3-discovery-mode --discovery-band-hz 60.0 550.0 --target-window-half-width-hz 7.5 --output-dir "C:/projects/final-project/final-project/FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/solver_benchmarks/target_density_discovery_60_550_step15_L_scout_coarse_m34"
```
