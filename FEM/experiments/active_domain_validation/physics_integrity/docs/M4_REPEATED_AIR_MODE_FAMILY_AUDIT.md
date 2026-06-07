# M4 repeated air mode family audit

**Verdict:** `INCONCLUSIVE`

Read-only audit of ~281 Hz and ~390 Hz `mic_output_proxy` peak families across completed samples.
No FOM, ROM, physics, solver, or production data were modified.

## Executive summary

## Distinction (validity layers)

| Layer | This audit scope |
|-------|------------------|
| Frequency validity | Eigenvalue locations vs targets/chunks |
| Mode classification | `coupling_class`, `dominant_region`, shares |
| Mic proxy validity | `mic_output_proxy` / soundhole mask sensitivity |
| Duplicate/plot validity | Raw vs deduped catalog, chunk provenance |
| ROM training impact | Deduped catalog used by ROM compare — separate regen if dedupe-only fix |

## Task 1 — Catalog audit

- Samples audited: `sample_000, sample_001, sample_002`
- Dedupe tolerance: **0.05 Hz** (production semantics via `load_fom_modes_catalog_deduped`)

### Data-driven repeated families


### Correlations / regressions

```json
{
  "band_281": {
    "n": 0
  },
  "band_390": {
    "n": 0
  }
}
```

## Task 2 — Full-artifact audit (retained heavy samples)

## Task 3 — Solver target/chunk audit

```json
{
  "281": {
    "modes_with_target_within_0p1_hz": 0,
    "fraction_of_chunk_rows": 0.0,
    "distinct_target_values": []
  },
  "390": {
    "modes_with_target_within_0p1_hz": 0,
    "fraction_of_chunk_rows": 0.0,
    "distinct_target_values": []
  }
}
```

## Task 4 — Minimal-fix decision tree

### If `PLOT_OR_DEDUPE_ARTIFACT`
- Change aggregation dedupe/plotting only; regenerate catalogs/plots from retained `worker_results` where present.
- No FOM rerun unless raw worker results were deleted.
- ROM catalogs: re-derive deduped view from raw `modes_catalog.jsonl` (ROM compare already dedupes at load).

### If `PROXY_INSENSITIVITY_SUSPECTED`
- Fix `mic_output_proxy` / soundhole mask weighting in `v2_b3_mode_audio_coupling.py`.
- Frequency and share fields may remain valid; recompute proxy from checkpoints where `region_dof_indices.npz` exists.
- Estimate rerun: only samples needing proxy re-aggregation (not full eigen solve) if mode vectors retained.

### If `FIXED_AIR_DOMAIN_SUSPECTED`
- Inspect exterior air domain BC/mesh tags; validate cavity vs exterior participation on full-artifact samples.
- Targeted rerun of affected frequency bands after boundary/mesh fix.

### If `GEOMETRY_PROPAGATION_SUSPECTED`
- Identify mesh/checkpoint reuse bug; invalidate only samples sharing duplicate mesh/checkpoint hashes.
- Run extreme-case validation (Task 5) before mass rerun.

### If `PHYSICALLY_PLAUSIBLE`
- Document expected sensitivity from LHS geometry range; set acceptance threshold for future samples (e.g. freq should track Helmholtz estimate with |r|>0.4).
- STK/ROM may still need deduped catalog and mic-proxy caveats.

## Task 5 — Extreme validation specs (do not run automatically)

Cheapest pre-L_prod scout/coarse commands to confirm air family shifts with geometry:

```bash
# 1) small body + small soundhole (near LHS minima)
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py \
  --lhs-json ROM/classic/lhs_pool.json --sample-ids sample_extreme_small \
  --execute --workers 1 --max-samples 1 \
  --geometry-override '{"length":0.36,"width":0.23,"depth":0.085,"hole_radius":0.040,"top_thickness":0.003}' \
  --run-scout-only

# 2) large body + large soundhole (near LHS maxima)
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py \
  --lhs-json ROM/classic/lhs_pool.json --sample-ids sample_extreme_large \
  --execute --workers 1 --max-samples 1 \
  --geometry-override '{"length":0.57,"width":0.42,"depth":0.14,"hole_radius":0.050,"top_thickness":0.003}' \
  --run-scout-only

# 3) same body, extreme hole_area/volume ratio
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py \
  --lhs-json ROM/classic/lhs_pool.json --sample-ids sample_extreme_hole_ratio \
  --execute --workers 1 --max-samples 1 \
  --geometry-override '{"length":0.48,"width":0.325,"depth":0.10,"hole_radius":0.050,"top_thickness":0.003}' \
  --run-scout-only
```

**Acceptance:** if ~281 Hz family stays within ~0.01 Hz across these extremes, treat as strong artifact indication.

## New FOM computation required?

Run Task 2 full-artifact audit on VM; then Task 5 extreme scouts if still inconclusive.

