# Coupled physics integrity gate

Isolated experiment to decide whether the **validation coupled FOM** is trustworthy
before any active-domain DOF reduction. All artifacts stay under this directory.

## Layout

```
physics_integrity/
  configs/           experiment JSON (physics_integrity_capture=true)
  scripts/           runners, diagnostics, report builder
  coupled_nominal/   TEST 1 — 202 Hz band [156, 248] Hz, 8 modes
  structural_only/   TEST 2 — shell EVP, no air/FSI
  acoustic_only/     TEST 3 — pressure-only cavity EVP (60–250 Hz)
  coupled_low_frequency/  TEST 4 — coupled near first air mode (default 120 Hz)
  coupled_near_acoustic/  TEST 5 — coupled near validated acoustic mode (~244 Hz)
  comparison/        report, CSVs, coupling_audit.json
```

## Production impact

- **No** changes to production LHS, `FEM/SORTING`, or default `guitar_3d.json`.
- Solver hooks are **opt-in** via `solver.physics_integrity_capture` and
  `solver.acoustic_cavity_only_diagnosis` (only set in configs here).
- Production `p_frac` / harvest logic is unchanged; this experiment adds parallel
  **p_frac_phys_gnhep** metrics.

## TEST 1 — nominal coupled baseline

If `../baseline/` already has a successful 202 Hz run, ingest without re-solving:

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_physics_case.py \
  --case coupled_nominal \
  --config FEM/experiments/active_domain_validation/physics_integrity/configs/coupled_nominal_202hz.json \
  --ingest-baseline
python FEM/experiments/active_domain_validation/physics_integrity/scripts/analyze_modes.py \
  --case-dir FEM/experiments/active_domain_validation/physics_integrity/coupled_nominal \
  --config FEM/experiments/active_domain_validation/physics_integrity/configs/coupled_nominal_202hz.json \
  --target-hz 202
```

Otherwise run a full coupled solve into `coupled_nominal/`.

## TEST 5 — coupled near acoustic reference (244.39 Hz)

After **TEST 3** (acoustic-only) and **TEST 2** (structural-only) pass on the repaired validation mesh,
run a narrow-band coupled harvest to check FSI energy exchange near the cavity mode:

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_coupled_near_acoustic.sh
```

- Config: `configs/coupled_near_acoustic_244hz.json`
- Band: 220–265 Hz, 16 modes, `pressure_release` + `pressure_gauge=none`
- Post-run summary: `coupled_near_acoustic/diagnostics/coupled_near_acoustic_summary.json`

## Participation / scaling audit (post-TEST-5)

Replays saved modes against assembled `A`/`M` (no SLEPc). Compares production `p_frac`,
GNHEP-undone and fully-unscaled L2 ratios, and mass-matrix acoustic energy participation.
Does not reject modes by `p_frac`.

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_coupled_participation_audit.sh
```

Outputs: `coupled_near_acoustic/diagnostics/coupled_participation_scaling_audit.{json,md}`

## Coupled pressure-domain audit (pre-restricted solve)

Audits whether coupled operators retain full-mesh pressure DOFs vs acoustic-only
`active_p=9998`, and whether algebraic restriction to air-supported pressure is feasible.

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_coupled_pressure_domain_audit.sh
```

Outputs: `diagnostics/coupled_pressure_domain/coupled_pressure_domain_audit.{json,md}`

## Diagnostic coupled solve — air-supported pressure only

After the domain audit, optional solve with `coupled_air_pressure_restriction_diagnosis`
(same weak forms; drops wood-only pressure DOFs algebraically):

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_coupled_near_acoustic_air_p_restricted.sh
```

Then rerun `run_coupled_participation_audit.sh` on `coupled_near_acoustic_air_p/` if needed.

## Decoupled-union diagnostic (block-diagonal, reduced domain)

Zeros all FSI/interface blocks (`A_up`, `A_pu`, `M_pu`, Nitsche) on the same
`n_u=102102`, `n_p_air=9998` operator. Harvest does not rank by wood. Verdict:
`DECOUPLED_UNION_PASS` if an acoustic-dominated mode appears within ±1 Hz of 244.39 Hz.

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_coupled_decoupled_union.sh
```

Outputs: `coupled_decoupled_union/diagnostics/decoupled_union_summary.json`

## Physical-FSI-only isolation (Nitsche disabled, reduced domain)

Same reduced operator (`n_u=102102`, `n_p_air=9998`, `n_reduced_W=112100`) with
physical cross-blocks `A_up`, `A_pu`, `M_pu` retained and all Nitsche blocks omitted.
Harvest does not rank by wood or p_frac. Verdict in summary JSON:
`PHYSICAL_FSI_ACOUSTIC_SURVIVES` if an acoustic-dominated mode is within ±1 Hz of 244.39 Hz.

Run **after** a clean decoupled-union PASS (report-only rerun is fine if solve already completed):

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_coupled_physical_fsi_only.sh
```

Outputs: `coupled_physical_fsi_only/diagnostics/physical_fsi_only_summary.json`

Post-solve participation / pressure-overlap audit (no eigen solve; refreshes verdict):

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_physical_fsi_participation_audit.sh
```

Outputs: `physical_fsi_participation_audit.json` / `.md`; updates `physical_fsi_only_summary.json`
when MAC and energy metrics support branch survival.

### Next Nitsche isolation cases (prepared, not auto-run)

Reduced domain + physical FSI + one Nitsche group per case:

| Case | Script (manual) | Config flag |
|------|-----------------|-------------|
| `nit_uu` only | `run_coupled_physical_fsi_nit_uu.sh` | `coupled_physical_fsi_nitsche_isolation_diagnosis: nit_uu` |
| `nit_up` only | `run_coupled_physical_fsi_nit_up.sh` | `nit_up` |
| `nit_pu` only | `run_coupled_physical_fsi_nit_pu.sh` | `nit_pu` |
| `nit_up` + `nit_pu` | `run_coupled_physical_fsi_nit_up_pu.sh` | `nit_up_pu` |

Run only after physical-FSI participation audit confirms branch survival.

## Physical-FSI continuation pilot (alpha_fsi on A_up/A_pu/M_pu only)

Low MAC at `alpha=1` shows branch mixing is already physical-FSI-driven, not Nitsche.
Pilot reuses saved endpoints:

| alpha | Source case |
|-------|-------------|
| 0.00 | `coupled_decoupled_union` (244.3916 Hz acoustic) |
| 0.01 | **new solve** `coupled_physical_fsi_alpha_pilot` |
| 1.00 | `coupled_physical_fsi_only` (245.2998 Hz) |

Full sweep sequence prepared (not auto-run): `0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00`.

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_physical_fsi_continuation_pilot.sh
```

Report: `coupled_physical_fsi_alpha_pilot/diagnostics/physical_fsi_continuation_report.json`

Outcomes: `PHYSICAL_FSI_BRANCH_CONTINUOUS`, `PHYSICAL_FSI_BRANCH_BREAKS_AT_ALPHA`, or
`PHYSICAL_FSI_ASSEMBLY_SUSPECT_AT_SMALL_ALPHA`. Do not run Nitsche isolation until pilot is interpreted.

## Scaling audit

`p_frac_raw` matches production (eigenvector in GNHEP block-scaled assembled basis).
`p_frac_phys_gnhep` multiplies **u** coeffs by `s_uu` and **p** coeffs by `s_pp` to undo
block Frobenius form scaling. `pressure_dof_scale` remains part of the physical model.

Parse scales from logs if capture JSON is missing: `GNHEP block Frobenius scales:` line.

## Soundhole ↔ air adjacency audit

Validation mesh rebuild uses **air-cavity opening** tag-2 selection (`FEM_VALIDATION_MESH=1`
in `build_3d_guitar.py`). Production FOM mesh is unchanged unless `FEM_SOUNDHOLE_TAG_AIR_OPENING=1`.

Rebuild mesh + audit (no eigen solve):

```bash
bash FEM/experiments/active_domain_validation/scripts/rebuild_validation_mesh_and_audit.sh
```

Reports under `diagnostics/soundhole_air_audit/` (JSON, Markdown, XDMF, CSV).

## coupled_physical_core_v2 — frozen baseline

v2 is the trusted coupled baseline on the repaired validation guitar (physical coupling,
`fsi_coupling_gain=1`, no Nitsche). v1 scaled/Nitsche paths are diagnostic history only.

Post-process validation (no eigensolve):

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_physical_core_v2_post.sh
```

## v2 sensitivity validation (experiment-only, pre-LHS)

Small controlled perturbations around the frozen v2 baseline. Plan:
`docs/v2_sensitivity_validation_plan.md`, manifest: `configs/v2_sensitivity_manifest.json`.

**Pilot (soundhole radius only — run this first):**

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_sensitivity_pilot.sh
```

**Radius pilot passed** (224.7 &lt; 244.4 &lt; 265.3 Hz). Controlled suite (depth, top thickness, E_L):

```bash
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_sensitivity_controlled_suite.sh
```

Preserves radius-pilot artifacts; does not re-solve `hole_radius_*` samples.

Summary: `v2_sensitivity_validation/diagnostics/v2_sensitivity_validation_summary.{json,md}`

## Decision outputs

`comparison/physics_integrity_report.md` ends with one of:

- `PASS_REFERENCE_MODEL`
- `FAIL_SCALING_METRIC`
- `FAIL_FSI_FORMULATION`
- `INCONCLUSIVE`

See `scripts/build_physics_integrity_report.py` for metric thresholds.
