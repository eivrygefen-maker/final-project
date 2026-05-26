# v2 validation status and roadmap

Frozen formulation: **`coupled_physical_core_v2`** (no Nitsche, `fsi_coupling_gain=1`, physical-energy branch selection).

## Promotion status (staged)

| Field | Status |
|-------|--------|
| `coupled_physical_core_v2_baseline_validation` | **PASS** |
| `acoustic_geometric_validation_pass` | **True** (phase-1 controlled samples) |
| `material_species_validation_pass` | **Pending** (phase-2 wood subset) |
| `production_parameter_coverage_pass` | **Pending** (length/width + remaining LHS axes) |
| `mesh_convergence_pass` | **Pending** |
| `lhs_promotion_blocked` | **True** |

Do **not** use `full_nonrandom_suite_pass=True` as a single promotion gate. Stiffness scaling (`E_L×0.9/×1.1`) is **exploratory only** — not evidence for production wood-material behavior.

## Confirmed phase-1 results (preserved)

**Coupled baseline:** `f = 244.394153 Hz`, `p_frac_energy_phys = 0.9998`

| Sample | f (Hz) | Δf (Hz) | p_frac | Gate |
|--------|--------|---------|--------|------|
| hole_radius_small (0.041 m) | 224.718 | −19.676 | 0.9979 | PASS |
| hole_radius_large (0.053 m) | 265.305 | +20.911 | 0.9976 | PASS |
| depth_small | 255.156 | +10.761 | 0.9995 | PASS |
| depth_large | 236.726 | −7.668 | 0.9967 | PASS |
| top_thickness_small / large | ~245.2 / ~245.7 | ~+0.8 / +1.3 | ~0.999 | acoustic-dominated, stable |

**Radius trend:** 224.7 < 244.4 < 265.3 Hz — monotonic with hole radius.

**Depth trend:** smaller cavity → higher acoustic frequency (depth_small > depth_large).

**Exploratory (not production material gate):** `top_stiffness_soft` / `top_stiffness_stiff` — artificial `E_L` scaling on baseline spruce top only.

## Actual production / LHS parameter space

See `configs/v2_lhs_parameter_schema.json`.

**Geometry (Classical LHS):**

- `geometry.length` — varies (validation baseline 0.48 m; LHS 0.35–0.60 m)
- `geometry.width` — **single scalar**; bouts/waist derived in `build_3d_guitar.py` unless explicit bout keys set
- `geometry.depth` — validated phase-1
- `geometry.top_thickness` — validated phase-1
- `geometry.hole_radius` — validated phase-1

**Materials:**

- `top_wood_id`, `back_wood_id` — any of five species on either plate (**25** valid combinations)
- Solver consumes full orthotropic records: `density`, `E_L`, `E_T`, `E_R`, `nu_*`, `G_*` (via `wood_library`)

**Baseline materials:** top = spruce, back = rosewood.

## Phase-2 controlled validation (prepared, not yet run on VM)

Manifest: `configs/v2_production_parameter_manifest.json`

| Sample | Varies | Remesh |
|--------|--------|--------|
| `length_small` / `length_large` | length 0.44 / 0.52 m | yes |
| `width_small` / `width_large` | width 0.30 / 0.35 m | yes |
| `material_top_cedar` / `material_top_maple` | top species; back fixed rosewood | no (baseline mesh) |
| `material_back_cedar` / `material_back_maple` | back species; top fixed spruce | no |
| `material_pair_cedar_top_maple_back` | combined extreme pair (optional) | no |

**Branch capture (geometry samples):** acoustic-cavity-only locator → targeted coupled harvest band → widen only if needed. Records: `locator_frequency_hz`, `coupled_target_hz`, `harvest_band_hz`, `branch_captured`, `targeted_retry_required`.

**Material samples:** same mesh; structural displacement MAC vs baseline structural branch where mode vectors exist.

## Deferred (explicitly out of scope here)

- Mesh convergence study
- Full **25** top×back material combination sweep / mini-LHS
- LHS sampling promotion

## VM command (phase-2 subset only)

From repo root on the validation VM:

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_sensitivity_production_stage.sh --resume
```

If all phase-2 solves already exist and only row/MAC reporting failed:

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_sensitivity_production_stage.sh --resume --report-only
```

Runs **only** phase-2 samples: `length_small`, `length_large`, `width_small`, `width_large`, `material_top_cedar`, `material_top_maple`, `material_back_cedar`, `material_back_maple`, `material_pair_cedar_top_maple_back`. Does **not** rerun radius, depth, or thickness phase-1 cases.

Optional single sample: append `--sample-id length_small`. Skip optional pair: `--skip-optional`.

After solves: refresh status flags with `run_v2_sensitivity_report_post.sh` (reads preserved phase-1 + new phase-2 artifacts).

## Material structural harvest extension (experiment-only)

**Purpose:** Uniform expanded structural spectrum on the **same validation mesh** and frozen `coupled_physical_core_v2` so baseline/material MAC and subspace comparisons use comparable in-band structural mode sets. Does **not** rerun geometry samples or acoustic-only material stability logic.

| Item | Value |
|------|--------|
| Suite | `v2_material_structural_harvest_extension` |
| Cases | `baseline_coupled_v2_material_reference` + five phase-2 wood samples |
| Harvest band | **200–320 Hz** (identical all cases) |
| `num_modes` | **30** (shift-invert target **260 Hz**) |
| Outputs | `v2_sensitivity_validation/material_structural_harvest_extension/` |
| Manifest | `configs/v2_material_structural_harvest_extension_manifest.json` |

**Validation criterion (post-solve, not per-mode MAC≥0.85 for every mode):** adequate structural coverage (≥8 modes per bank), ≥2 high-confidence Hungarian-assigned pairs (MAC≥0.85) and/or subspace preservation (min cosine≥0.75 or mean≥0.85), no solver failure. Large Δf with high MAC is documented as shape-preserved branch shift.

**Gates until explicit `--apply-promotion`:** `material_structural_branch_validation_pass` = **Pending**; `lhs_promotion_blocked` = **True**.

### VM command (six solves + report)

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_material_structural_harvest_extension.sh
```

Post-only (artifacts already present): `bash …/run_v2_material_structural_harvest_extension.sh --skip-solve`
