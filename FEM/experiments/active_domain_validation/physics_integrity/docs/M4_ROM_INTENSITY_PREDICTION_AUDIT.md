# M4 ROM intensity prediction audit

**Date:** 2026-06-05  
**Scope:** Phase-2 ROM output-proxy prediction (`radiation_proxy`, `mic_output_proxy`, `bridge_excitation_*`) vs FOM.  
**Status:** FOM unchanged; ROM Phase-2 holdout validation working on `sample_000`–`sample_022`.

---

## Executive summary

| Category | Median / typical | Assessment |
|----------|------------------|------------|
| Frequency (`median_relative_error`) | ~1.2% | **Strong** — meets 5% target |
| Classification (`coupling_class`, `dominant_region`) | ~87–91% | **Strong** |
| Intensity proxies (`radiation_proxy`, `mic_output_proxy`) | ~38–45% rel. error | **Weak** — not STK-ready |

**Primary findings:**

1. ROM predicts **raw absolute normalized proxy values** via **inverse-distance weighted (IDW) blending** of neighbor FOM scalars — not log-space, not per-guitar normalized amplitudes.
2. **ROM trains and compares on `modes_catalog.jsonl` as written** — which is the **raw** per-chunk catalog, **not** the deduped set used for plots and `modes_summary.json`.
3. Duplicate/near-duplicate rows in `modes_catalog.jsonl` are expected (raw targets from overlapping chunks). They **inflate training mode count**, skew sorted-index alignment, and can make inspection look like repeated ~281 Hz peaks.
4. The ~281 Hz `mic_output_proxy` cluster is **mostly physically plausible** (air-dominant cavity/soundhole family on fixed classical topology), but **partially amplified by raw-catalog duplicates** and **ROM alignment that ignores `coupling_class`**.
5. **Recommended minimal fix (Intensity ROM v2.1):** keep frequency + shares + classification unchanged; train/compare intensity on **deduped** modes; predict **log₁₀(proxy + ε)** and track **per-sample normalized** proxy metrics.

---

## 1. How Phase-2 ROM predicts intensity fields

### Prediction pipeline (two-step hybrid)

| Step | What | Method |
|------|------|--------|
| **Frequencies** | `frequency_hz` per `mode_index` | **Sorted-index IDW** across k nearest LHS guitars: `pred_freq[m] = weighted_avg(neighbor_freq[m])` |
| **Scalars** (incl. intensity) | all Phase-2 numeric fields | **Nearest-frequency IDW** per neighbor: find closest catalog mode to `pred_freq[m]`, then blend values |

Implementation: `predict_modal_catalog()` in `v2_b3_m4_modal_surrogate_lib.py`, scalar blend in `predict_mode_scalars_at_frequency()` in `v2_b3_m4_rom_scalar_fields.py`.

### Per-field behavior

| Field | Prediction type | Notes |
|-------|-----------------|-------|
| `radiation_proxy` | **Absolute** weighted blend of neighbor absolute values | FOM: `0.45·top + 0.15·back + 0.40·air` output proxies, each ÷ `modal_norm` |
| `mic_output_proxy` | **Absolute** weighted blend | FOM priority: soundhole RMS → cavity pressure → radiation blend (see `v2_b3_mode_audio_coupling.py`) |
| `bridge_excitation_abs` | **Absolute** weighted blend | RMS bridge/top displacement ÷ `modal_norm` |
| `bridge_excitation_coupling` | **Absolute** weighted blend (signed mean ÷ `modal_norm`) | Same alignment as above |
| `top_share` / `back_share` / `air_share` | Absolute blend | Bounded [0,1]; errors stay moderate |
| `coupling_class` / `dominant_region` | **Categorical vote** (not regression) | Coarse bins → high accuracy despite scalar error |

**Not used today:** log targets, per-sample max/p95 normalization, class-filtered nearest-frequency, separate air-mode models.

---

## 2. Absolute vs relative vs normalized

| Question | Answer |
|----------|--------|
| Does ROM predict absolute values? | **Yes** — direct IDW of FOM scalar magnitudes |
| Relative values? | **No** — no ratio-to-peak or ratio-to-band-mean |
| Normalized per guitar? | **No** — cross-guitar absolute scale is learned implicitly via k-NN |
| FOM proxy definition | Each proxy is already **modal_norm-normalized** (‖x‖₂ on full W layout), but **not normalized across modes within one guitar** |

ROM therefore mixes **physics-normalized** FOM scalars (per mode) as if they were **geometry-transferable amplitudes**. That is reasonable for classification (thresholded shares) but fragile for continuous intensity when mode identity drifts.

---

## 3. Raw vs deduped training data

### What FOM aggregation writes

In `v2_b3_m4_aggregate_worker_results.py`:

```python
# modes_catalog.jsonl — ALL raw accepted records (per chunk target)
for rec in sorted(all_records, ...):
    fh.write(json.dumps(rec) + "\n")

# modes_summary.json, plots — DEDUPED catalog
deduped, merge_groups = _dedupe_catalog(all_records, tol_hz=0.05)
```

| Artifact | Content |
|----------|---------|
| `modes_catalog.jsonl` | **Raw** — one row per accepted chunk target (overlapping frequency windows) |
| `modes_summary.json` | Stats from **deduped** set (`deduped_mode_count`, summaries) |
| `aggregation_result.json` | `raw_mode_count` vs `deduped_mode_count` |
| M4 plots (`mode_frequency_vs_mic_output_proxy.png`, etc.) | **Deduped** only |
| **ROM train** (`collect_completed_fom_training_rows`) | Reads **`modes_catalog.jsonl` → raw** |
| **ROM compare** (`load_fom_modes_catalog`) | Reads **`modes_catalog.jsonl` → raw** |

**Documentation drift:** several docs say `modes_catalog.jsonl` is deduped; **code currently writes raw**. Treat `modes_summary.deduped_mode_count` as authoritative mode count.

---

## 4. Duplicate / near-duplicate modes

### Are duplicates present?

**Yes, in raw `modes_catalog.jsonl`:**

- Chunked frequency scouting revisits overlapping Hz windows.
- Multiple chunk targets can land within **0.05 Hz** (dedupe tolerance) or at **identical** `frequency_hz` with identical proxies (same mode accepted in adjacent chunks).
- Dedupe merges these for summaries/plots; **raw file keeps all rows**.

### Exact duplicate rows (same freq + same proxies)

Possible when:

- Two chunk records reference the same accepted mode (provenance differs; scalar fields copied).
- User inspection of “duplicate rows” in `modes_catalog.jsonl` is **expected**, not necessarily a solver bug.

### Near-duplicates (>0.05 Hz apart)

Also present — dedupe tolerance is **0.05 Hz**. ROM nearest-frequency alignment can pick different duplicates per neighbor.

---

## 5. Double-counting impact

| Consumer | Uses raw or deduped? | Double-count risk |
|----------|----------------------|-------------------|
| **ROM training** | **Raw** | **Yes** — extra rows inflate `mode_count`, shift sorted-index alignment, duplicate Hz in neighbor catalogs |
| **ROM comparison** | **Raw** FOM side | **Yes** — greedy Hz match can match multiple FOM rows near same ROM frequency; intensity metrics averaged over redundant pairs |
| **M4 plots** | Deduped | **No** |
| **`modes_summary.json`** | Deduped | **No** |
| **STK export (future)** | Undefined — must choose explicitly | **Risk** if raw catalog used |

**ROM does not deduplicate** in `load_fom_modes_catalog()` — it only sorts by frequency.

---

## 6. Nearest-frequency alignment and class awareness

Scalar alignment (`predict_mode_scalars_at_frequency`):

1. For each k-NN training guitar, find **single** catalog index minimizing `|f - target_hz|`.
2. IDW-blend numeric fields; majority vote categoricals.

**Does not consider:**

- `coupling_class`
- `dominant_region`
- `mic_output_method`
- `provenance_count`

**Consequence:** At ~281 Hz, neighbor A’s nearest mode may be **air_dominant** while neighbor B’s nearest at the same Hz might be **top_back_mixed** from a different branch. IDW blends incompatible intensities → large proxy error even when frequency and coarse class labels look good **after** separate categorical voting on the blended prediction.

Frequency uses **sorted index** (ordinal); intensity uses **Hz-only nearest** — **inconsistent mode identity assumptions**.

---

## 7. Why frequency/classification are good but intensity is weak

| Factor | Frequency / class | Intensity |
|--------|-------------------|-----------|
| Target smoothness | Hz varies smoothly with geometry (k-NN works) | Proxies nonlinear; air vs top modes swap |
| Output type | Continuous / coarse bins | Continuous, wide dynamic range |
| Alignment | Sorted index + Hz match | Hz-only nearest → wrong mode branch |
| Training data | Raw duplicates add noise mainly to ordinal tail | Duplicates **directly distort** local Hz→proxy mapping |
| Metric | Relative error on Hz (stable scale) | Relative error on small proxies (harsh) |
| FOM definition | Single scalar `frequency_hz` | `mic_output_proxy` uses **priority chain** (soundhole vs pressure vs radiation) — method can flip with small geometry changes |

Observed ~45% radiation / ~38% mic median relative error is consistent with **mode-misassignment + raw catalog noise + relative-error metric on small values**, not with broken FOM physics.

---

## 8. Current intensity metrics and appropriateness

### Primary tracked metrics (`compute_phase2_scalar_metrics`)

| Metric | Definition |
|--------|------------|
| `radiation_proxy_relative_error_median` | `median(|rom-fom| / |fom|)` — skips `|fom| < 1e-12` |
| `mic_output_proxy_relative_error_median` | Same |
| `bridge_excitation_abs_relative_error_median` | Same |
| `audio_weighted_output_proxy_error` | Radiation rel-error weighted by **FOM** `radiation_proxy` |

### Issues with relative error for proxies

1. **Near-zero denominators** — many bridge modes have tiny `bridge_excitation_abs`; one small mismatch → huge relative error (partially masked by skip threshold `1e-12`, still harsh).
2. **Not STK-audible** — STK cares about **relative loudness across modes within a guitar**, not cross-guitar absolute proxy match.
3. **Outliers dominate mean** — median ~45% hides that a subset may be >200% error.
4. **Does not use log domain** — proxies often span orders of magnitude across 60–550 Hz.

**Share MAEs** (`top_share_mae` etc.) are more appropriate for bounded fields and typically look better than proxy relative errors.

---

## 9. Would log-space targets help?

**Likely yes, as a partial fix.**

Proxies are positive (after norm division). Predicting `log10(proxy + ε)`:

- Compresses dynamic range
- Makes k-NN averaging behave more like multiplicative error
- Reduces penalty on small absolute values when back-transformed

**Caveats:**

- Must use consistent `ε` (e.g. `1e-8` or percentile floor)
- Modes with `mic_output_proxy = None` need exclusion
- Does not fix mode misalignment by itself — combine with deduped training + optional class filter

---

## 10. Would per-guitar normalization help STK?

**Likely yes for STK/website loudness curves.**

Within each guitar, define:

```text
radiation_proxy_norm = radiation_proxy / p95(radiation_proxy)
mic_output_proxy_norm  = mic_output_proxy  / p95(mic_output_proxy)
```

STK synthesis weights modes **relative to that guitar’s peak modes**, not absolute cross-sample scale.

ROM could:

- Predict normalized targets (per training sample, computed at train time)
- Compare in normalized space
- Denormalize at inference using predicted p95 or stored training prior

This is **orthogonal to log-space**; both can be tracked.

---

## Task B — ~281 Hz mic peak sanity check

### Physical context

For **fixed classical guitar topology** (same bracing template, same mesh family):

- Low air / Helmholtz-related modes often appear ~100–200 Hz.
- Higher **monopole / breathing / cavity** air modes commonly appear **250–400 Hz**.
- Modes with `coupling_class = air_dominant`, `air_share ≈ 0.84–1.0`, and high `mic_output_proxy` are **expected** when `mic_output_method` is `soundhole_displacement_rms_proxy_v1` or cavity-pressure proxy — air motion drives soundhole/cavity scalars.

**Conclusion on plausibility:** A **strong mic peak near ~281 Hz** on multiple classical samples is **physically plausible (A)** as an air/cavity mode family, **not** automatically a bug.

### When it becomes suspicious

| Signal | Interpretation |
|--------|----------------|
| Same `frequency_hz` **and** same proxies repeated many times **in one sample** | **(B) raw-catalog duplicates** |
| Identical 281.000 Hz across samples with **very different** `hole_radius` / `depth` / volume | **(C) mesh/template artifact** or over-constrained cavity |
| Peak only in ROM prediction, absent in FOM deduped catalog | **(D) ROM/plotting issue** |
| Peak in FOM deduped catalog, shifts **±2–10 Hz** with geometry | **(A) plausible** |

M4 aggregation **plots use deduped** data — if the ~281 Hz peak appears in `mode_frequency_vs_mic_output_proxy.png`, it is probably real FOM physics, not a plot bug.

### VM inspection commands (samples 018–022)

Run on the VM where FOM outputs exist:

```bash
REPO_ROOT=<repo>
GUITARS="$REPO_ROOT/FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
RUN_SUFFIX=m4prod1   # adjust if needed

for SID in sample_018 sample_019 sample_020 sample_021 sample_022; do
  CATALOG="$GUITARS/$SID/runs/${SID}_${RUN_SUFFIX}/aggregation/modes_catalog.jsonl"
  echo "=== $SID top mic_output_proxy (raw catalog) ==="
  python3 - <<'PY' "$CATALOG" "$SID"
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
sid = sys.argv[2]
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
rows.sort(key=lambda r: -(r.get("mic_output_proxy") or 0))
for r in rows[:8]:
    print(
        f"{sid}",
        f"hz={r.get('frequency_hz')}",
        f"mic={r.get('mic_output_proxy')}",
        f"rad={r.get('radiation_proxy')}",
        f"class={r.get('coupling_class')}",
        f"dom={r.get('dominant_region')}",
        f"shares={r.get('top_share')}/{r.get('back_share')}/{r.get('air_share')}",
        f"method={r.get('mic_output_method')}",
        f"prov={r.get('provenance_count', 1)}",
    )
PY
done
```

**Deduped view** (matches plots):

```bash
python3 FEM/experiments/active_domain_validation/physics_integrity/scripts/audit_catalog_dedupe.py \
  --sample sample_018 --run-id sample_018_m4prod1
```

*(Script does not exist yet — proposed in Task D below.)*

### Expected table columns for report section

| sample | frequency_hz | mic_output_proxy | radiation_proxy | coupling_class | dominant_region | top/back/air share |
|--------|-------------|------------------|-----------------|----------------|-----------------|--------------------|
| 018–022 | ~275–290 Hz (if plausible) | high on air modes | moderate | `air_dominant` | `air` | low/low/high |

Fill from VM run above. If frequencies are **nearly identical** across 018–022 despite LHS depth/hole_radius spread <5 Hz, note **(C)**. If duplicates share exact Hz+mic within one file, note **(B)**.

### Geometry sensitivity check

```bash
# Compare hole_radius / depth vs top mic-peak Hz from lhs_pool + catalog
python3 -c "
import json
from pathlib import Path
pool = json.loads(Path('ROM/classic/lhs_pool.json').read_text())
# join with per-sample peak mic Hz from catalogs
"
```

Peaks should shift **somewhat** with cavity volume and hole radius. Stiff ~281.0 Hz across all samples → investigate template; gradual drift → supports (A).

---

## Task C — Safest intensity improvements (ranked)

| Priority | Change | Risk | Expected impact |
|----------|--------|------|-----------------|
| **P0** | Train + compare on **deduped** catalog (reuse `_dedupe_catalog`, tol=0.05 Hz) | **Low** — ROM-only read path | Removes duplicate Hz rows; stabilizes sorted index; aligns ROM with plots |
| **P1** | Predict intensity in **log₁₀(proxy + ε)**; invert at output | **Low** — intensity fields only | Better k-NN for wide dynamic range |
| **P1** | Add **per-sample normalized** proxy targets + metrics | **Low** | STK-relevant amplitude tracking |
| **P2** | **Class-aware** nearest-frequency: prefer same `coupling_class` within Δf tie-break | **Medium** | Helps ~281 Hz air family vs mixed modes |
| **P2** | Audio-weighted **mic** error (weight by FOM `mic_output_proxy`) | **Low** — metrics only | Better STK relevance tracking |
| **P3** | Separate predictors per frequency band | **Higher** | Defer until 40–100+ samples |
| **Avoid now** | Full mode shapes, Stage C, FOM aggregation changes | — | Out of scope |

### Proposed new metrics (tracking only, no fail gate)

```text
radiation_proxy_log_mae
mic_output_proxy_log_mae
bridge_excitation_abs_log_mae
radiation_proxy_norm_mae
mic_output_proxy_norm_mae
bridge_excitation_abs_norm_mae
mic_output_proxy_audio_weighted_rel_error
```

Normalization: per-sample `p95` of each proxy within 60–550 Hz band (deduped).

---

## Task D — Minimal safe implementation plan (Intensity ROM v2.1)

**Do not change FOM physics, solver, aggregation write path, or production pipeline flags.**

### v2.1 scope (~3 files, ROM-only)

| File | Change |
|------|--------|
| `v2_b3_m4_rom_fom_compare_lib.py` | Add `load_fom_modes_catalog_deduped()` wrapping `_dedupe_catalog` (import from aggregate module or copy 30-line helper) |
| `v2_b3_m4_modal_surrogate_lib.py` | Training uses deduped loader; store optional `log10` arrays for intensity fields |
| `v2_b3_m4_rom_scalar_fields.py` | Log-blend for `radiation_proxy`, `mic_output_proxy`, `bridge_excitation_abs`; add log/norm metrics |

### Unchanged in v2.1

- Sorted-index frequency IDW
- Share prediction (raw blend)
- Categorical vote for `coupling_class` / `dominant_region`
- Production surrogate schema version bump: `m4_modal_surrogate_v2_1`
- Holdout validation path

### Optional tiny audit helper (ROM-only)

`scripts/audit_catalog_dedupe.py` — print raw vs deduped counts, duplicate groups near 281 Hz, top mic modes. No FOM changes.

### Validation workflow (no-leakage, samples 000–022)

```bash
# 1) Rebuild production surrogate from deduped training (after v2.1 code)
python FEM/experiments/active_domain_validation/physics_integrity/scripts/build_m4_rom_from_completed_fom.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --completed-only \
  --max-samples 23

# 2) Holdout compare each sample (or batch)
for i in $(seq 0 22); do
  SID=$(printf "sample_%03d" "$i")
  python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_rom_compare.py \
    --lhs-json ROM/classic/lhs_pool.json \
    --force-sample "$SID" \
    --exclude-target-from-training
done

# 3) Review rolling summary
cat ROM/classic/comparisons/rom_accuracy_summary.json
```

**Acceptance checks:**

| Check | Expect |
|-------|--------|
| `median_relative_error` | Still ≤ 5% (unchanged frequency path) |
| `coupling_class_accuracy` | Still ~85%+ |
| `radiation_proxy_relative_error_median` | **Lower** than ~45% (or flat — then try class-aware) |
| `radiation_proxy_log_mae` | New baseline; should improve vs raw rel error interpretability |
| `mic_output_proxy_norm_mae` | New STK-relevant tracker |

Single-sample quick check:

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_rom_compare.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --force-sample sample_016 \
  --exclude-target-from-training
```

---

## Acceptance criteria answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Is ~281 Hz mic peak physically plausible? | **Yes, mostly (A)** — air_dominant cavity/soundhole family on shared classical topology; verify geometry sensitivity on VM |
| 2 | Are duplicates present and do they affect training/plots? | **Yes in raw catalog**; **plots use deduped**; **ROM train/compare use raw → affects ROM** |
| 3 | Current ROM intensity prediction method? | **Raw absolute proxy values**, IDW of nearest-frequency neighbor scalars; **not** log or per-guitar norm |
| 4 | Recommended minimal improvement? | **Intensity ROM v2.1:** deduped train/compare + log-space intensity + norm metrics; keep frequency/shares/class unchanged |
| 5 | Validation command? | Holdout loop on `sample_000`–`sample_022` above; inspect `ROM/classic/comparisons/rom_accuracy_summary.json` |

---

## Appendix — FOM proxy definitions (reference)

From `v2_b3_mode_audio_coupling.py`:

| Proxy | Formula (simplified) |
|-------|----------------------|
| `radiation_proxy` | Weighted blend of top/back/air output proxies (45/15/40) |
| `mic_output_proxy` | `soundhole_rms/modal_norm` → else `cavity_pressure/modal_norm` → else `radiation_proxy` |
| `bridge_excitation_abs` | `RMS(u_bridge) / modal_norm` |
| `bridge_excitation_coupling` | `mean(u_bridge) / modal_norm` |

All are **scalar solve-time** computations — no exterior acoustic solve.

---

## Related docs

- [M4_ROM_OUTPUT_GAP_ANALYSIS.md](M4_ROM_OUTPUT_GAP_ANALYSIS.md) — field parity (note: catalog dedupe wording should match this audit)
- [B3_M4_MODE_AUDIO_COUPLING_METADATA.md](B3_M4_MODE_AUDIO_COUPLING_METADATA.md) — FOM proxy semantics
- [M4_ROM_BUILD_AND_SURROGATE.md](M4_ROM_BUILD_AND_SURROGATE.md) — surrogate build workflow
