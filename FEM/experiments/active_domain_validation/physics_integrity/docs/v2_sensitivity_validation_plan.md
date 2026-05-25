# v2 sensitivity validation plan

Experiment-only suite around the **frozen** `coupled_physical_core_v2` baseline on the repaired validation guitar.
v1 scaled/Nitsche paths remain archived diagnostic history only.

## Frozen baseline (do not modify)

| Item | Value |
|------|-------|
| Formulation | `coupled_physical_core_v2` (physical blocks, `fsi_coupling_gain=1`, no Nitsche) |
| Reference mesh | `mesh/validation_tiny_guitar_3d.msh` (`hole_radius=0.047 m`) |
| Acoustic reference (disabled) | 244.39159990162557 Hz |
| Coupled match (enabled) | 244.394153389752 Hz, `p_frac_energy_phys≈1`, pressure MAC=1.0 |

## Suite scope

| Phase | Samples | Remesh? |
|-------|---------|---------|
| **Pilot** | `hole_radius_small` (0.041 m), `hole_radius_large` (0.053 m) | Yes |
| **Phase-1 controlled** | + `depth_small/large`, `top_thickness_small/large` | Mixed |
| **Exploratory** | `top_stiffness_soft/stiff` (`E_L×0.9/×1.1`) — **not** production material gate | No remesh |
| **Phase-2 production** | `length_*`, `width_*`, wood species subset | See `v2_production_parameter_manifest.json` |

Nominal baseline metrics are **ingested** from `coupled_physical_core_v2/` (no re-solve).

## Mesh / operator gates (every perturbed sample)

1. **Soundhole aperture** — tag-2 area within ±15% of πr² for sample `hole_radius`; radial/z/horizontal checks (`audit_soundhole_aperture_geometry.py --mesh --hole-radius`).
2. **Air connectivity** — tag-2 adjacent to air tag 10; soundhole pressure DOFs in air subgraph (`audit_soundhole_air_adjacency.py --mesh`).
3. **Active pressure DOFs** — `n_p_active` / restriction metadata from v2 solve (`coupled_air_pressure_restriction_diagnosis`).
4. **v2 convergence** — `physical_coupling_enabled` subcase only; `nconv > 0`, in-band harvest 220–265 Hz.

## Per-sample report fields

- Varied parameter(s) and exact values
- Mesh gate status (aperture, adjacency, combined)
- Nearest acoustic-branch frequency (physical-energy classification)
- Δf from frozen baseline acoustic reference
- `p_frac_energy_phys`, structural/acoustic energies, cross term
- Pressure MAC vs baseline branch **only when** `n_p_active` matches baseline (otherwise `null` + reason)

## Expected-direction checks (defined before run)

| Perturbation | Expectation (recorded, soft in pilot) |
|--------------|----------------------------------------|
| **hole_radius** ↑ / ↓ | Interpretable monotonic trend of tracked acoustic frequency vs radius; \|Δf\| between small/large > ~0.01 Hz noise |
| **depth** ↑ / ↓ | Cavity volume change shifts acoustic branch (inner_depth = depth − 2×top_thickness) |
| **top_thickness** ↑ / ↓ | Structural branch / coupling participation change; v2 still converges |
| **top E_L** ×0.9 / ×1.1 | Exploratory only — not production wood validation |
| **length / width** | LHS scalars; locator-guided coupled branch capture |
| **top_wood_id / back_wood_id** | Full `wood_library` records; 25 combos deferred |

Failed direction checks **do not** auto-fail the pilot; they are logged in `expected_direction_evaluation` for human review.

## Radius pilot (passed — recorded)

First parametric validation of frozen `coupled_physical_core_v2`:

```text
224.718 < 244.394 < 265.305 Hz  (hole_radius 0.041 / 0.047 / 0.053 m)
pilot_radius_trend_pass = True
```

Artifacts preserved under `v2_sensitivity_validation/samples/hole_radius_{small,large}/`.

## Controlled suite (depth, top thickness, E_L)

After radius pilot passes:

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_sensitivity_controlled_suite.sh
```

- Does **not** re-solve hole-radius samples
- Branch selection: physical energy (`p_frac_energy_phys` ≥ 0.85), not nearest frequency
- Auto widen-band retry if acoustic branch missing in 220–265 Hz
- Structural-property samples also report `structural_branches_in_band`

## Report completion (no solves)

After controlled suite artifacts exist on the VM:

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_sensitivity_report_post.sh
```

Reads `structural_branches_in_band` from saved `samples/*/results/*.json`, evaluates thickness/stiffness structural trends via frequency matching (±8 Hz), fills baseline/large-radius energy fields from v2 artifacts, and writes separate validation flags.

## Staged promotion gates

Do **not** promote to full LHS until all of:

- `acoustic_geometric_validation_pass` = True (phase-1 radius/depth/thickness)
- `material_species_validation_pass` = PASS (phase-2 wood subset on baseline mesh)
- `production_parameter_coverage_pass` = PASS (phase-2 length/width)
- `mesh_convergence_pass` = PASS (not started)

`top_stiffness_soft/stiff` are **exploratory** (`E_L` scaling only); they are not the production wood-material gate.
See `docs/v2_validation_status_and_roadmap.md` and `diagnostics/v2_validation_status.json`.

## Phase-2 production parameters (prepared)

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_sensitivity_production_stage.sh
```

Runs only: `length_small`, `length_large`, `width_small`, `width_large`, material species subset.
Does **not** rerun phase-1 radius/depth samples. Geometry samples use acoustic locator → targeted coupled harvest.

## Artifacts

```
physics_integrity/v2_sensitivity_validation/
  mesh/                    # per-sample .msh
  samples/<id>/            # logs, results, modes, diagnostics/gates/
  diagnostics/
    v2_sensitivity_validation_summary.{json,md}
```

## Commands

**Pilot (soundhole radius only):**

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_sensitivity_pilot.sh
```

**Large-radius acoustic branch capture** (if 220–265 Hz harvest missed the shifted branch):

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_sensitivity_pilot_large_capture.sh
```

Uses existing `hole_radius_large.msh` and gates; solves 255–300 Hz with energy-based branch selection only.

**Full suite (after pilot passes):**

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_sensitivity_validation.sh
```
