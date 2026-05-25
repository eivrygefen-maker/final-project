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

## Decision outputs

`comparison/physics_integrity_report.md` ends with one of:

- `PASS_REFERENCE_MODEL`
- `FAIL_SCALING_METRIC`
- `FAIL_FSI_FORMULATION`
- `INCONCLUSIVE`

See `scripts/build_physics_integrity_report.py` for metric thresholds.
