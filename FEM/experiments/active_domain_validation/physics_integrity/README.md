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

## Scaling audit

`p_frac_raw` matches production (eigenvector in GNHEP block-scaled assembled basis).
`p_frac_phys_gnhep` multiplies **u** coeffs by `s_uu` and **p** coeffs by `s_pp` to undo
block Frobenius form scaling. `pressure_dof_scale` remains part of the physical model.

Parse scales from logs if capture JSON is missing: `GNHEP block Frobenius scales:` line.

## Decision outputs

`comparison/physics_integrity_report.md` ends with one of:

- `PASS_REFERENCE_MODEL`
- `FAIL_SCALING_METRIC`
- `FAIL_FSI_FORMULATION`
- `INCONCLUSIVE`

See `scripts/build_physics_integrity_report.py` for metric thresholds.
