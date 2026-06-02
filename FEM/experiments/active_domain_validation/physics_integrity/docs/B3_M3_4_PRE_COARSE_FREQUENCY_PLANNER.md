# B3 M3.4-pre — Coarse frequency planner (planning / consultation)

**Status:** Planning-only — no LHS, no orchestrator execution, no Stage A/B/C runs authorized by this document.  
**Follows:** M3.3 orchestrator PASS (`lhs_pilot_001_timing_m3exec2`), [`B3_M3_ORCHESTRATOR_CONTRACT.md`](B3_M3_ORCHESTRATOR_CONTRACT.md)  
**Related code (existing, not modified here):** `v2_b3_st_sinvert_solver_lib.py` (`L_PROD_ST_FULL9_TARGETS_HZ`), `v2_b3_checkpoint_target_density_experiment.py`, `v2_b3_checkpoint_solver_multi_benchmark.py`

---

## 0. Purpose

Before M3.4 synthesis/batch LHS and before committing to fixed target-spacing policy, we need a **data-driven calibration step** that:

1. Explores frequency space on a **controlled coarse setup** (or a deliberate production checkpoint with bounded cost).
2. Collects **observed modal structure** (accepted mode frequencies, deduped).
3. **Only then** proposes dense/sparse regions, target spacing, and future Stage B window plans.

This document defines that process. It does **not** hard-code zone thresholds (no fixed “10 modes / 25 Hz = dense”, no fixed 5.5 Hz spacing rules). Any numeric examples below are **illustrative** unless explicitly marked as **validated evidence**.

---

## 1. What frequency range are we interested in?

### 1.1 Validated evidence today (narrow band)

The current production-validated timing band (`target_set=full9`) is:

```text
221.5, 227.0, 232.5, 238.0, 243.5, 249.0, 254.5, 260.0, 264.0 Hz
```

(Source: `L_PROD_ST_FULL9_TARGETS_HZ` in `v2_b3_st_sinvert_solver_lib.py`.)

This is a **high-confidence pilot slice**, not a claim about the full physical spectrum.

Approximate span: **221.5–264.0 Hz** (~42.5 Hz) with ~5.5 Hz spacing (historical pilot choice — **not calibrated** as optimal).

### 1.2 Solver acceptance window (code constraint)

Mode acceptance in the ST pipeline currently filters to:

| Constant | Value |
|----------|--------|
| `ACCEPTANCE_FREQ_LO_HZ` | 220.0 |
| `ACCEPTANCE_FREQ_HI_HZ` | 265.0 |

Any exploratory scan **outside 220–265 Hz** will not produce “accepted” modes under today’s solver library unless acceptance bounds are extended in a **separate reviewed change** (out of scope for M3.4-pre planning).

### 1.3 Recommended planning range (first calibration pass)

| Phase | Range (Hz) | Rationale |
|-------|------------|-----------|
| **Pass 1 (recommended)** | **220–265** | Aligns with existing acceptance filter; directly comparable to `full9`; minimizes policy churn. |
| **Pass 2 (optional, after Pass 1)** | **200–280** or user band e.g. **180–320** | Requires acceptance-band extension + review; needed only if physics/product requires modes outside 220–265. |

**M3.4-pre default assumption:** first approved coarse scan uses **220–265 Hz** unless product explicitly needs wider band.

---

## 2. What is the initial coarse grid?

### 2.1 Status: **not calibrated yet**

No grid step is validated as “correct” for modal-density discovery. Candidates for the **first exploratory scan only**:

| Candidate step (Hz) | Targets in 220–265 | Notes |
|---------------------|-------------------|--------|
| 10 | ~6 | Coarse; cheap; may miss closely spaced modes |
| 5 | ~10 | Matches illustrative planner example; moderate cost |
| 2.5 | ~19 | Finer; approaches `full9` density in band |

**Recommendation for first approved solve:** start with **uniform 5 Hz** from 220 to 265 (inclusive endpoints), i.e. ~10 ST targets — **exploratory**, not policy.

After Pass 1, compare:

- deduped unique accepted mode list vs `full9` reference coverage (pattern already in `v2_b3_checkpoint_target_density_experiment.py`).

### 2.2 Dry-run planning grid (no solve)

The dry-run planner tool may emit a **proposed** grid from CLI, e.g.:

- `--freq-min-hz 220 --freq-max-hz 265 --coarse-step-hz 5`

with metadata:

```json
"calibration_status": "not_calibrated_yet"
```

---

## 3. What solver / checkpoint input does it use?

### 3.1 Inputs

| Input | Role |
|-------|------|
| **Stage A checkpoint directory** | Frozen `A_active` / `M_active` + `built_metadata.json` (mesh level, active dimension) |
| **Factor solver** | `mkl_pardiso` (validated on VM) |
| **Target list** | Coarse grid and/or region-derived windows (execution phase only) |
| **Reference result (optional)** | e.g. `m3exec2` `result.json` for coverage comparison |

### 3.2 Do we already have a suitable coarse checkpoint?

| Asset | Status | Recommendation |
|-------|--------|----------------|
| **`L_prod` checkpoint — `lhs_pilot_001_timing_m3exec2`** | **Exists, PASS** (`active_dim≈316017`, Stage A build ~273 s) | Usable for **solver-only** coarse scan; expensive per target but no new Stage A. |
| **`L_prod` — other pilot / official refs** | Several PASS checkpoints in docs | Alternative references for A/B parity. |
| **`L_dev_coarse` / `L_dev_refined`** | Defined in `v2_mesh_convergence_manifest.json` as **solver smoke / not final validation** | **No M3-validated checkpoint export yet** in pipeline manifests. Cheaper discovery mesh, but requires **new Stage A run** (orchestrator + new `run_id`) before scan. |
| **`L_mid` / `L0`** | Validation meshes | Possible for methodology experiments; not current LHS pilot path. |

**Answer:** We have a **production-representative** checkpoint (`m3exec2`). We do **not** yet have a **validated cheap coarse-mesh checkpoint** in the official pipeline. For modal-density **discovery**, prefer creating **`L_dev_refined` or `L_dev_coarse` checkpoint** once; for **production-aligned density**, reuse `m3exec2`.

### 3.3 Should the first coarse scan use Stage B flow or a separate diagnostic?

**Use the existing Stage B solver stack** (`run_checkpoint_st_target` via `v2_b3_checkpoint_solve.py` or `v2_b3_checkpoint_solver_multi_benchmark.py` / `v2_b3_checkpoint_target_density_experiment.py`).

| Approach | Verdict |
|----------|---------|
| Reuse Stage B / checkpoint multi-target solver | **Yes** — same acceptance rules, same `result.json` shape, comparable to `full9`. |
| New cheaper eigen solver | **No** for M3.4-pre — would not calibrate production target policy. |
| FEM re-solve | **No** — checkpoint already freezes operators. |

The **frequency planner** is a **meta-layer**: it plans targets and interprets results; it does not replace Stage B.

---

## 4. What does it output?

Artifacts under planning paths (not runtime diagnostics), e.g.:

`pipeline_runs/specs/frequency_plans/<plan_id>/`

| File | Contents |
|------|----------|
| `coarse_frequency_plan.json` | Machine-readable plan + calibration status |
| `coarse_frequency_plan.md` | Human-readable interpretation |

### 4.1 JSON schema (conceptual)

| Section | Description |
|---------|-------------|
| `input_summary` | Checkpoint, mesh level, freq range, step, window half-width, mode (`dry-run` \| `execute`), policy flags |
| `known_evidence` | Validated `full9` targets, optional reference `result.json` path |
| `coarse_targets_hz` | Uniform or region-derived target list for next solve |
| `proposed_regions` | Labeled bands with **`calibration_status`** per region |
| `stage_b_target_windows` | `{target_hz, window_hz, reason}` for future orchestrator runs |
| `diagnostics` | Notes, open questions, cost estimate |
| `next_approved_command` | Literal command template if execution approved (not run automatically) |

### 4.2 Region object (thresholds not fixed)

```json
{
  "region_id": "R_full9_validated",
  "range_hz": [221.5, 264.0],
  "density_policy": "known_validated_band",
  "calibration_status": "validated_by_m3_pilot_full9",
  "recommended_step_hz": null,
  "recommended_window_half_width_hz": 1.5,
  "reason": "M3 timing pilot 9/9 PASS; not proof of global optimum spacing"
}
```

```json
{
  "region_id": "R_below_full9",
  "range_hz": [220.0, 221.5],
  "density_policy": "unknown_until_coarse_scan",
  "calibration_status": "not_calibrated_yet",
  "recommended_step_hz": null,
  "reason": "Inside acceptance band but outside validated full9 targets"
}
```

After Pass 1 coarse scan, regions may be updated to:

```text
estimated_from_first_coarse_scan
```

with **data-backed** `modes_per_hz` or `modes_per_window` fields — still not arbitrary hard-coded thresholds until a second review step approves them as policy.

---

## 5. How does it divide the range into sub-regions / windows?

### 5.1 Process (data-driven)

```mermaid
flowchart LR
  A[Define scan band] --> B[Choose coarse target grid]
  B --> C[Approved Stage B coarse solve]
  C --> D[Collect per-target accepted modes]
  D --> E[Dedupe unique frequencies]
  E --> F[Bin into frequency windows]
  F --> G[Compute density metrics per window]
  G --> H[Propose regions + spacing]
  H --> I[Human review / freeze policy]
```

### 5.2 Windowing for density (Pass 1 analysis)

1. Partition `[f_min, f_max]` into windows of width `W` Hz (e.g. **5 Hz** or **10 Hz**; **not calibrated**).
2. Assign each **deduped accepted mode** to exactly one window (by mode frequency).
3. Compute per window:
   - `mode_count`
   - `modes_per_hz` = `mode_count / W`
   - optional: `targets_contributing` count

### 5.3 Region labels (after calibration only)

Example **post-scan** labels (illustrative names only):

| Label | Meaning (to be defined from data) |
|-------|-----------------------------------|
| `sparse` | Low `modes_per_hz` vs band median |
| `medium` | Near median |
| `dense` | High `modes_per_hz` vs band median |

**Classification method (recommended):** rank windows by `modes_per_hz`; use tertiles or median ± MAD on **first scan** — thresholds stored in plan JSON as **derived**, not hard-coded in repo constants.

### 5.4 Target windows for Stage B

Each future solve target can be documented as:

```json
{
  "target_hz": 243.5,
  "window_hz": [242.0, 245.0],
  "half_width_hz": 1.5,
  "reason": "full9 validated target"
}
```

`window_hz` is for **interpretation and dedup attribution**, not a separate eigen solve unless policy later requires band solves.

---

## 6. How are dense / sparse regions detected?

| Status | Approach |
|--------|----------|
| **Now** | **Cannot detect reliably** without coarse-scan data. Dry-run marks all non-`full9` bands `not_calibrated_yet`. |
| **After Pass 1** | Per-window **mode_count** / **modes_per_hz** from deduped accepted modes. |
| **Not allowed yet** | Fixed rules like “≥ N modes per 25 Hz” or “spacing ≤ 5.5 Hz means dense” as repo policy. |

**Duplicate handling:** use `deduplicate_frequencies_hz` with `FREQ_PARITY_TOL_HZ = 0.05` (existing). Attribute each deduped mode to one window; do not double-count the same mode across overlapping target contributions.

---

## 7. How do we decide future target windows for Stage B?

| Stage | Decision basis |
|-------|----------------|
| **Immediate (pre-scan)** | Keep `full9` for production timing/LHS until calibration says otherwise. |
| **After coarse scan** | Increase target density in windows with high `modes_per_hz`; decrease in sparse windows. |
| **LHS integration** | Export `targets_hz` list + `target_set` name + region metadata as JSONL row fields for orchestrator. |

**Coverage criterion (reuse existing tooling):** compare coarse scan deduped modes to `full9` reference frequencies with tolerance `~0.1 Hz` (as in target density experiment) to ensure coarser spacing does not miss validated modes.

---

## 8. How does this connect to the orchestrator?

```text
frequency planner (plan / analyze)
        │
        ├─► pipeline_runs/specs/frequency_plans/<plan_id>/
        │         coarse_frequency_plan.json
        │
        └─► future: pipeline_runs/specs/*.jsonl run rows
                  targets_hz / target_set / region_id
                        │
                        ▼
              v2_b3_m3_orchestrator_run_one.py (or batch successor)
                        │
                        ├─ Stage A (checkpoint export)
                        ├─ Stage B (checkpoint solve)
                        └─ Stage C (only if synthesis policy)
```

| Responsibility | Owner |
|----------------|--------|
| Target sets, regions, spacing proposals | Frequency planner |
| Env isolation, manifests, fail-stop, no overwrite | Orchestrator |
| ST solve physics / acceptance | Stage B scripts (unchanged) |

The planner **never** replaces manifests or index; it **feeds** run specs.

---

## 9. Dry-run / planning-only vs actual solve

| Mode | Subprocess solves | Checkpoint required | Output |
|------|-------------------|---------------------|--------|
| **`dry-run`** | **No** | Optional (warn if missing) | Proposed grid, hypothetical regions, `calibration_status: not_calibrated_yet`, cost estimate |
| **`execute` (later, explicit approval)** | **Yes** — Stage B only on existing checkpoint | **Yes** — PASS checkpoint | `result.json` or density-experiment aggregate + updated plan with `estimated_from_first_coarse_scan` |

**M3.4-pre authorizes only `dry-run`** until a separate approval record exists.

Suggested future execute path (not approved here):

```bash
# Option A: orchestrator timing run with custom targets (after planner + spec update)
# Option B: existing density experiment extended to coarse grid
/home/vboxuser/solver-mkl/venv/bin/python \
  FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_target_density_experiment.py \
  --checkpoint-dir <checkpoint> \
  --reference-json <m3exec2_result.json> \
  --start-hz 220 --stop-hz 265 --spacings-hz 5 \
  --output-dir FEM/.../solver_benchmarks/target_density_<plan_id>
```

---

## 10. What to inspect before approving any real coarse solve

### 10.1 Preconditions

- [ ] Checkpoint `checkpoint_export_manifest.json`: `status=PASS`, `export_pass`, `matrix_verify_pass`
- [ ] `built_metadata.json`: mesh level and `active_dimension` documented in plan
- [ ] Solver-mkl env isolation validated (M3.3 `m3exec2`)
- [ ] New output dirs / plan ids — **no overwrite** of `m3exec1` / `m3exec2` diagnostics
- [ ] Target count and wall-time estimate recorded in plan MD
- [ ] Acceptance band (220–265) understood; wider band requires separate code review

### 10.2 After solve (calibration review)

- [ ] `result.json` / density experiment JSON: per-target `accepted_frequencies_hz`
- [ ] `aggregate.unique_accepted_frequencies_hz` and deduped count
- [ ] Coverage vs `full9` reference (matched / missed / extra)
- [ ] Per-window mode table in plan MD
- [ ] Proposed regions updated with `estimated_from_first_coarse_scan`
- [ ] Explicit human sign-off before changing `target_set` policy or LHS specs

---

## 11. Open questions — recommended answers

### Q1. Suitable coarse mesh/checkpoint?

**Partially.** `m3exec2` `L_prod` checkpoint is suitable for a **production-aligned** first scan. A **cheaper** scan needs a new **`L_dev_coarse` or `L_dev_refined` Stage A** (new orchestrator `run_id`) — not yet done in M3.

### Q2. Stage B vs separate diagnostic?

**Stage B stack** (or `v2_b3_checkpoint_target_density_experiment.py` wrapper). No new diagnostic solver.

### Q3. Frequency range for first scan?

**220–265 Hz** (matches acceptance). Extend later only with reviewed acceptance-band change.

### Q4. Coarse spacing before knowing density?

**5 Hz uniform** exploratory (~10 targets) for Pass 1; optionally add **10 Hz** arm for cost comparison. Reuse density experiment to sweep spacings {5, 8, 10} against `full9` reference on same checkpoint.

### Q5. Output to inspect for modes per window?

| Artifact | Fields |
|----------|--------|
| `result.json` | `targets[].accepted_frequencies_hz`, `aggregate.unique_accepted_frequencies_hz` |
| Target density JSON | `spacings[].unique_accepted_count`, coverage vs reference |
| Planner post-process | `windows[].mode_count`, `modes_per_hz` |

### Q6. Density metric?

| Metric | Use |
|--------|-----|
| **Deduped unique accepted frequencies** | **Primary** (production-consistent) |
| Per-target accepted lists | Diagnostic per shift center |
| All raw ST modes before acceptance | Optional debug only — not policy |
| Rich eigenvectors | Not needed for spacing calibration |

### Q7. Avoid duplicate-mode misleading?

1. Run `deduplicate_frequencies_hz` with `FREQ_PARITY_TOL_HZ` (0.05).  
2. Assign each deduped mode to **one** frequency window (by `floor(f/W)*W` or nearest window center).  
3. Do not sum `accepted_mode_count` across targets without dedup.  
4. When comparing spacings, use existing `compare_reference_coverage` logic.

### Q8. Feed orchestrator / LHS?

1. Planner writes `coarse_frequency_plan.json`.  
2. Human approves `targets_hz` + `target_set` name.  
3. Resolver/orchestrator spec rows gain optional `targets_hz` override (future schema extension) or new named sets registered beside `full9`.  
4. Each LHS sample still uses `sample_id` / `run_id` / overlay; only Stage B target list changes.

---

## 12. Future tool: `v2_b3_frequency_coarse_planner.py`

**M3.4-pre:** dry-run mode only (no subprocess, no manifests, no index).

Suggested CLI:

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_frequency_coarse_planner.py \
  --checkpoint-dir FEM/.../st_worker_scaling_L_prod_lhs_pilot_001_timing_m3exec2 \
  --freq-min-hz 220 \
  --freq-max-hz 265 \
  --coarse-step-hz 5 \
  --target-window-half-width-hz 1.5 \
  --reference-result-json FEM/.../checkpoint_solve_mkl_pardiso_full9_lhs_pilot_001_timing_m3exec2/result.json \
  --mode dry-run \
  --output-dir FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/specs/frequency_plans/m3_4_pre_coarse_demo
```

| Flag | Purpose |
|------|---------|
| `--mode dry-run` | Plan only (default) |
| `--mode execute` | Reserved; **disabled** until explicit approval |
| `--plan-id` | Subfolder name under `frequency_plans/` |

---

## 13. Relationship to existing experiments

| Script | Relationship |
|--------|----------------|
| `v2_b3_checkpoint_target_density_experiment.py` | **Execution engine candidate** for Pass 1 spacing sweeps on one checkpoint |
| `v2_b3_checkpoint_target_alignment_experiment.py` | Shift-center grid studies; complementary, not duplicate |
| `v2_b3_m3_orchestrator_run_one.py` | Runs full Stage A+B; use when checkpoint does not exist |

---

## 14. Safest next non-destructive step

1. **Review** this document + dry-run plan artifacts under `pipeline_runs/specs/frequency_plans/m3_4_pre_coarse_demo/` (if generated).  
2. **Approve** Pass 1 parameters: checkpoint=`m3exec2`, band=220–265 Hz, step=5 Hz, solver-only, new output dir e.g. `target_density_m3_4_pre_pass1_<utc>`.  
3. **Run** (VM, manual): `v2_b3_checkpoint_target_density_experiment.py` or bounded `v2_b3_checkpoint_solve.py` with explicit `--targets-hz` list — **not** via orchestrator (no Stage A).  
4. **Post-process** into updated `coarse_frequency_plan.json` with `estimated_from_first_coarse_scan`.  
5. **Only then** discuss M3.4 synthesis batch and LHS target policy changes.

**Do not:** delete `m3exec1` evidence; promote to production; run full LHS; extend acceptance band without review.

---

## 15. M3.3 validated state (reference)

| Run | Role |
|-----|------|
| `lhs_pilot_001_timing_m3exec1` | Failed Stage B env contamination — keep as debug |
| `lhs_pilot_001_timing_m3exec2` | PASS A+B; use checkpoint/solve as calibration reference |

---

## 16. Glossary

| Term | Meaning |
|------|---------|
| **Coarse scan** | Many shift centers on one frozen checkpoint |
| **Window** | Fixed Hz interval for density counting |
| **Region** | Labeled band with recommended spacing (after calibration) |
| **full9** | Nine validated production targets (221.5–264 Hz) |
| **not_calibrated_yet** | No data-backed density policy |

---

*Document version: M3.4-pre consultation draft. Zone thresholds intentionally unset.*
