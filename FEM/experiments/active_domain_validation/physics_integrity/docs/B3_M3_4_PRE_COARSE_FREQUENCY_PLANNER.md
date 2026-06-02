# B3 M3.4-pre — Coarse frequency planner (planning / consultation)

**Status:** Planning-only — no LHS, no orchestrator execution, no Stage A/B/C runs authorized by this document.  
**Follows:** M3.3 orchestrator PASS (`lhs_pilot_001_timing_m3exec2`), [`B3_M3_ORCHESTRATOR_CONTRACT.md`](B3_M3_ORCHESTRATOR_CONTRACT.md)  
**Tool:** `scripts/v2_b3_frequency_coarse_planner.py` (dry-run only, schema `b3_coarse_frequency_plan_v2`)

---

## 0. Purpose

Before M3.4 synthesis/batch LHS, perform a **data-driven calibration** of modal-density zones across the **guitar/modal planning band (60–550 Hz)**.

The validated `full9` slice (221.5–264.0 Hz) is **reference evidence only**, not the full exploration range.

Zone thresholds (dense/sparse, spacing, targets per region) are **`not_calibrated_yet`** until a controlled coarse scan produces mode-count data.

---

## 1. Frequency ranges (two concepts)

| Concept | Range (Hz) | Role |
|---------|------------|------|
| **Planning / exploration band** | **60–550** | Target for coarse modal-density discovery |
| **Validated reference slice (`full9`)** | **221.5–264.0** | M3 m3exec2 timing 9/9 PASS; historical pilot spacing ~5.5 Hz — **not proven optimal** |
| **Solver acceptance filter (code today)** | **220–265** | Hard filter on **accepted** modes in `v2_b3_st_sinvert_solver_lib.py` |

Do not conflate the planning band with the validated slice or the acceptance filter.

---

## 2. Acceptance-band blocker (critical)

### 2.1 What the code does today

In `collect_accepted_st_modes()` / `collect_converged_modes()`:

- Default `freq_lo=ACCEPTANCE_FREQ_LO_HZ` (**220.0**)
- Default `freq_hi=ACCEPTANCE_FREQ_HI_HZ` (**265.0**)
- A mode is **accepted** only if its computed frequency lies **inside** that interval (plus other physics checks).

`run_checkpoint_st_target()` can shift the ST solve to **any** `target_hz`, but **accepted** mode lists exclude frequencies outside 220–265 Hz.

Stage B (`v2_b3_checkpoint_solve.py`) and `v2_b3_checkpoint_target_density_experiment.py` record `acceptance_interval_hz: [220, 265]` in outputs but **do not** expose CLI to widen acceptance for discovery scans.

### 2.2 Can we scan 60–550 Hz without code changes?

| Aspect | 60–220 Hz | 220–265 Hz | 265–550 Hz |
|--------|-----------|------------|------------|
| ST solve at target | Can run | Can run | Can run |
| Modes **accepted** near target | **No** (outside band) | **Yes** | **No** |
| Useful for modal-density discovery | **No** (with current acceptance) | **Yes** | **No** |

**Verdict:** A **60–550 Hz coarse scan is NOT executable for discovery** under current acceptance limits.

Mark wide-band execution as:

```text
requires acceptance-band / general target-set support before execution
```

**Narrow slice executable now:** ~220–265 Hz only (matches acceptance; overlaps `full9`).

### 2.3 Required before wide-band scan

Reviewed change (separate from M3.4-pre planning), e.g.:

- Parameterize `freq_lo` / `freq_hi` per scan or per target window in `run_checkpoint_st_target` + CLI, **or**
- Discovery mode that records **converged** mode frequencies without the 220–265 acceptance gate (with explicit policy for spurious modes).

Until then, planner dry-runs may show **proposed** 60–550 targets but must label them **blocked**.

---

## 3. Coarse grid intent and spacing recommendation

### 3.1 Goal

Discover modal-density structure across **60–550 Hz**, not only around `full9`.

### 3.2 Do not assume 5 Hz band-wide

5 Hz over 60–550 → **~99 targets** — too dense and unjustified before calibration.

### 3.3 Recommended first-pass spacing (planning)

| Policy | Step | ~Target count (60–550) | Notes |
|--------|------|------------------------|--------|
| **uniform_15hz** | **15 Hz** | **~34** | **Recommended first pass** after acceptance fix |
| uniform_10hz | 10 Hz | ~50 | Finer; use if 15 Hz leaves gaps |
| uniform_20hz | 20 Hz | ~26 | Coarser; cheaper discovery |
| adaptive_v0 (planning) | 20 / 10 / 20 Hz segments | ~30 | 20 Hz wings + 10 Hz in 200–280 Hz; hypothesis only |

**Recommendation:** **15 Hz uniform** for first approved wide-band scan — balance of cost vs coverage.

**Alternative:** `adaptive_v0` if we want finer sampling near the known `full9` neighborhood without committing to 5 Hz globally.

All spacing values remain **`not_calibrated_yet`** until post-scan review.

---

## 4. JSON schema (`b3_coarse_frequency_plan_v2`)

Top-level fields for easy inspection:

| Field | Description |
|-------|-------------|
| `mode` | `dry-run` (only mode implemented) |
| `will_execute` | Always `false` in M3.4-pre |
| `calibration_status` | `not_calibrated_yet` |
| `zone_policy_status` | `not_calibrated_yet` |
| `input_summary.freq_range_hz` | e.g. `[60, 550]` |
| `input_summary.coarse_step_hz` | e.g. `15` (null if adaptive) |
| `input_summary.target_window_half_width_hz` | Applied half-width (default: `coarse_step_hz / 2`) |
| `input_summary.recommended_target_window_half_width_hz` | `spacing / 2` for touching adjacent windows |
| `discovery_window_policy` | How half-width was chosen (`spacing_over_2` vs `explicit_override`) |
| `discovery_coverage_analysis` | Gap warnings when `2 * half_width < spacing` |
| `coarse_targets_hz` | Proposed target list |
| `coarse_target_count` | Integer |
| `regions` | Placeholder region table |
| `executable_feasibility.executable_now_count` | Targets inside current [220, 265] acceptance |
| `executable_feasibility.blocked_count` | Targets outside current acceptance |
| `cost_estimate` | Target count × per-target time range |
| `recommended_next_step` | Human-readable gate + command template |

### 4.1 Placeholder regions (always present for 60–550 planning)

| region_id | range_hz | calibration_status |
|-----------|----------|-------------------|
| `R_low_60_220` | [60, 221.5] | `not_calibrated_yet` |
| `R_full9_validated_220_265` | [221.5, 264.0] | `validated_by_m3_pilot_full9` |
| `R_mid_high_265_550` | [264.0, 550] | `not_calibrated_yet` |

---

## 5. Parallel execution caution

| Phase | Parallel OK? |
|-------|----------------|
| **Dry-run planner** | **Yes** — no solver, no manifests |
| **Actual coarse ST scan on L_prod** | **No — run alone** |

**Rationale:**

- `m3exec2` checkpoint: **~316k active DOF**, Stage A build **~273 s**; Stage B `full9` is **9 sequential ST solves** on MKL/PARDISO.
- Wide-band scan at 15 Hz → **~34 targets** → rough wall-time **~1–4 hours** (2–8 min/target conservative; **measure on first approved slice**).
- Concurrent solver jobs contend for CPU/RAM; do **not** assume “a few minutes” for L_prod wide-band scans.

Planner output includes `cost_estimate.parallel_execution_guidance.recommended_concurrency: exclusive_solver_slot`.

---

## 6. Data-driven calibration process

1. **Fix acceptance / discovery policy** (reviewed code change).
2. **Approve** checkpoint (m3exec2 L_prod and/or future L_dev_refined export).
3. **Run** solver-only coarse scan → new output dir, isolated solver-mkl env.
4. **Collect** per-target `accepted_frequencies_hz` (or converged modes if policy extended).
5. **Dedupe** with `deduplicate_frequencies_hz` (`FREQ_PARITY_TOL_HZ=0.05`).
6. **Bin** into windows; compute `mode_count`, `modes_per_hz`.
7. **Propose** zones from observed distribution — thresholds **`estimated_from_first_coarse_scan`**, not hard-coded constants.
8. **Feed** orchestrator/LHS via approved `targets_hz` / target-set specs.

---

## 7. Open questions — answers

### Q1. Suitable coarse checkpoint?

| Asset | Status |
|-------|--------|
| `m3exec2` L_prod | **PASS** — production-representative; expensive per target |
| L_dev_coarse / L_dev_refined | Manifest-defined smoke meshes; **no M3-validated checkpoint yet** |

For **discovery cost**, prefer future **L_dev_refined** Stage A; for **production-aligned** density, use **m3exec2**.

### Q2. Stage B vs separate diagnostic?

**Reuse Stage B stack** (`v2_b3_checkpoint_solve`, `v2_b3_checkpoint_target_density_experiment`). No new eigen solver.

### Q3. First scan frequency range?

**Plan:** 60–550 Hz. **Execute (today):** only 220–265 Hz until acceptance extended.

### Q4. First spacing?

**15 Hz uniform** recommended; compare 10 / 20 Hz in planner `spacing_alternatives`.

### Q5. Output to inspect?

`result.json` → `targets[]`, `aggregate.unique_accepted_frequencies_hz`; planner post-process → per-window counts.

### Q6. Density metric?

**Primary:** deduped unique **accepted** frequencies. Raw per-target lists are diagnostic only.

### Q7. Duplicate modes?

Use existing dedupe; assign each mode to one window; do not sum per-target counts without dedup.

### Q8. Orchestrator feed?

Planner → approved plan JSON → JSONL run specs with `targets_hz` → orchestrator (future schema beyond hardcoded `full9`).

---

## 8. Discovery window half-width (coarse scan)

Per-target discovery acceptance uses `± target_window_half_width_hz` around each shift center. For a uniform coarse grid, adjacent windows should **touch** at the nominal step:

```text
target_window_half_width_hz = coarse_step_hz / 2
```

| `coarse_step_hz` | Recommended `--target-window-half-width-hz` |
|------------------|---------------------------------------------|
| 15 | **7.5** |
| 10 | 5.0 |
| 20 | 10.0 |

The planner applies this when `--target-window-half-width-hz` is **omitted** (`discovery_half_width_source: spacing_over_2`).

If a **smaller** value is set (e.g. 1.5 Hz with 15 Hz spacing), `discovery_coverage_analysis.has_coverage_gaps` is true and diagnostic notes warn that up to **12 Hz** can fall between adjacent target windows. Do not use 1.5 Hz for 15 Hz coarse discovery; that width is only appropriate for narrow full9 reference context.

---

## 9. Relationship to orchestrator

Frequency planner **feeds** target policy; orchestrator **executes** Stage A/B with env isolation, manifests, fail-stop.

No runtime manifests or index from the planner itself.

---

## 10. Safest next non-destructive steps

1. **VM dry-run** 60–550 Hz plan (see §10).
2. **Review** acceptance-band extension proposal (code + policy).
3. **Optional interim scan:** 220–265 Hz only on m3exec2 (executable now) — calibrates tooling, **not** full guitar band.
4. **After acceptance fix:** approve 15 Hz / 60–550 scan on exclusive VM slot.
5. Post-process → update regions with `estimated_from_first_coarse_scan`.

**Do not:** delete m3exec1 evidence; run wide-band scan pretending acceptance is already wide; parallel L_prod solves with other benchmarks.

**Next:** [`B3_M3_4_GATE_A_ACCEPTANCE_DISCOVERY_MODE.md`](B3_M3_4_GATE_A_ACCEPTANCE_DISCOVERY_MODE.md) — acceptance / discovery-mode design (planning-only).

---

## 11. Dry-run command (60–550 Hz)

```bash
python -m py_compile FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_frequency_coarse_planner.py

python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_frequency_coarse_planner.py \
  --checkpoint-dir FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/st_worker_scaling_L_prod_lhs_pilot_001_timing_m3exec2 \
  --reference-result-json FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/solver_benchmarks/checkpoint_solve_mkl_pardiso_full9_lhs_pilot_001_timing_m3exec2/result.json \
  --freq-min-hz 60 \
  --freq-max-hz 550 \
  --coarse-step-hz 15 \
  --mode dry-run \
  --output-dir FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/specs/frequency_plans/m3_4_pre_coarse_demo \
  --force
```

Adaptive alternative:

```bash
python .../v2_b3_frequency_coarse_planner.py \
  ... \
  --spacing-policy adaptive_v0 \
  --freq-min-hz 60 --freq-max-hz 550 \
  --mode dry-run --force
```

---

## 12. First actual scan recommendation (after gates)

| Step | Action |
|------|--------|
| **Gate A** | Acceptance-band / discovery-mode code review |
| **Gate B** | Approve spacing (15 Hz uniform) + checkpoint + exclusive VM |
| **Scan** | `v2_b3_checkpoint_target_density_experiment.py` with `--B3-discovery-mode`, band 60–550, spacing 15, **`--target-window-half-width-hz 7.5`** (or omit half-width on planner-driven command) on **new** output dir |
| **Measure** | Wall time per target; update `cost_estimate` assumptions |
| **Calibrate** | Regions + spacing from data; set `zone_policy_status` → `estimated_from_first_coarse_scan` |

**Interim (no Gate A):** Re-run density experiment on **220–265 only** — validates pipeline, does **not** answer 60–550 structure.

---

*Document version: M3.4-pre v2 — 60–550 Hz planning band; acceptance blocker explicit; zone thresholds not calibrated.*
