# M4 ROM Intensity v2.1 vs v2.2 Comparison

**Status:** Experimental — run `compare_rom_intensity_v21_v22.py` on VM to populate aggregate table below.  
**Validation:** 30-sample leave-one-out (`training_pool_size=29`, `leakage_count=0`).

---

## Diagnostic baseline (v2.1, pre-v2.2)

From `M4_ROM_INTENSITY_V22_DIAGNOSTICS.json` holdout LOO:

| Metric | v2.1 median |
|--------|-------------|
| Frequency median relative error | 0.0142 |
| Mic p95 norm MAE | 0.2334 |
| Radiation p95 norm MAE | 0.2534 |
| Bridge p95 norm MAE | 0.2778 |
| Mic rank correlation | 0.7816 |
| Radiation rank correlation | 0.1635 |
| Top-20% mic overlap | 0.4709 |
| Top-20% radiation overlap | 0.2351 |

---

## Aggregate comparison (filled by compare script)

| Metric | v2.1 | v2.2 B | v2.2 C | v2.2 D | Best | Δ vs v2.1 |
|--------|------|--------|--------|--------|------|-----------|
| *Run compare script on VM* | | | | | | |

**Methods:**

| ID | Description |
|----|-------------|
| **A / v2.1** | Nearest-frequency k-NN IDW; blends neighbor-normalized values |
| **B** | Physics-aware class/region/band matching + raw IDW + post-hoc p95 norm |
| **C** | B + geometry-derived neighbor penalty |
| **D** | Physics-aware matching + per-field ridge regression with class/band fallback |

---

## Alignment `1.0` cases — not leakage

Diagnostics reported `overall_class_match_rate = 1.0` for some samples. Investigation:

1. **Holdout LOO excludes the target** from `build_holdout_surrogate_model(exclude_sample_ids=[target])`.
2. **Neighbor IDs** come only from the 29 training guitars; compare script asserts `target ∉ neighbor_sample_ids`.
3. **Why 1.0?** When most modes on a guitar share one `coupling_class` (often `top_back_mixed`), and Hz-nearest neighbor modes from training guitars agree on class, every mode scores a match. This is **high alignment**, not self-comparison.
4. **Air/back-dominant** subsets more often show low match rates (0.3–0.6) per diagnostics `by_coupling_class`.

---

## Adoption criteria

| Criterion | Threshold |
|-----------|-----------|
| Frequency | No material regression (stay ~≤ 0.018 median) |
| Mic p95 norm MAE | Clear improvement toward ≤ 0.18–0.20 |
| Radiation / bridge MAE | Material improvement below 0.25 / 0.28 |
| Radiation rank | ≥ 0.30 (stretch 0.45+) |
| Top-k radiation | ≥ 0.35 |
| Leakage | `training_includes_target=false` all 30 |
| Runtime | Practical (~≤ 20 s median) |

Long-term target 0.10–0.15 normalized MAE — **not claimed** unless validation supports it.

---

## VM commands

### 1. Build experimental artifact (optional; does not touch production v2.1)

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/build_m4_rom_v22_experimental.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --max-samples 30
```

Outputs: `ROM/classic/experimental_v22/m4_modal_surrogate_v22_experimental.{json,npz}`

### 2. Run 30-sample LOO comparison (required)

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/compare_rom_intensity_v21_v22.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --samples 0-29 \
  --json-out ROM/classic/experimental_v22/comparison_summary.json \
  --md-out FEM/experiments/active_domain_validation/physics_integrity/docs/M4_ROM_INTENSITY_V21_V22_COMPARISON.md
```

### 3. Re-run diagnostics (optional reference)

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/audit_rom_intensity_v22_diagnostics.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --samples 0-29 \
  --json-out FEM/experiments/active_domain_validation/physics_integrity/docs/M4_ROM_INTENSITY_V22_DIAGNOSTICS.json \
  --csv-out FEM/experiments/active_domain_validation/physics_integrity/docs/M4_ROM_INTENSITY_V22_HOLDOUT_PER_SAMPLE.csv
```

---

## Recommendation (pending VM results)

| Outcome | Action |
|---------|--------|
| B/C/D beat v2.1 on radiation rank + MAE, frequency stable | **Continue experimentation** → promote best method |
| Only marginal gains | **Collect more FOM** + refine proxies |
| Radiation unchanged | **Improve FOM proxy definitions** (air/structural separation) |
| Mic regresses | Do **not** promote; keep v2.1 production |

*Populate after running compare script.*

---

## Artifacts

```text
ROM/classic/experimental_v22/
  model_manifest.json
  build_report.json
  comparison_summary.json
  per_sample_comparisons/sample_XXX_v21_v22.json
  m4_modal_surrogate_v22_experimental.json
  m4_modal_surrogate_v22_experimental.npz
```

Production unchanged:

```text
ROM/classic/m4_modal_surrogate.json
ROM/classic/m4_modal_surrogate.npz
ROM/classic/rom_model_manifest.json
```
