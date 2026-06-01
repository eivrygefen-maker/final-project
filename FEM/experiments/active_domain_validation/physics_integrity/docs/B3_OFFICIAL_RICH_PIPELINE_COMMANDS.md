# B3 official A+B+C rich pipeline (commands)

**Status:** validated PASS (2026-06-01)  
**PASS marker:** [`A_B_C_RICH_PIPELINE_PASS.md`](../v2_mesh_convergence/diagnostics/solver_benchmarks/checkpoint_solve_mkl_pardiso_full9_20260601T203438Z/A_B_C_RICH_PIPELINE_PASS.md)

See also: [`B3_RICH_MODAL_EXPORT_TODO.md`](B3_RICH_MODAL_EXPORT_TODO.md) (schemas, v1 scope).

---

## 1. Purpose

Three-stage pipeline for **audio/STK-ready** modal data from the L_prod B3 checkpoint path:

| Stage | Environment | Role |
|-------|-------------|------|
| **A** | production `.venv` + DOLFINx | Build/export active A/M, CSR, synthesis metadata |
| **B** | `solver-mkl` | ST/EPS solve; optional active eigenvectors (`rich_modal/`) |
| **C** | production `.venv` (numpy only) | Region participation + audio output **proxies** (not microphone pressure) |

FOM eigensolve is **undamped**. Damping/Q belongs in STK/audio later.

**Default:** rich export is **off** on Stage B (timing benchmarks unchanged).

---

## 2. Stage A — official checkpoint requirements

**Script:** `scripts/v2_b3_checkpoint_export.py`

**Must have on PASS:**

- `checkpoint_export_manifest.json` with `status: PASS`
- `A_active.petsc.bin`, `M_active.petsc.bin`
- `A_active_csr.npz`, `M_active_csr.npz`
- `csr_metadata.json`, `built_metadata.json`
- `synthesis_metadata.json` (`schema: b3_synthesis_metadata_v1`)

**Region DOFs:** default `--B3-synthesis-region-dofs off` (no DOLFINx facet locate in Stage A). `region_dof_indices.npz` is optional; may be deferred to Stage C.

**Validated checkpoint:**

`v2_mesh_convergence/diagnostics/st_worker_scaling_L_prod_rich_safe_20260601T164739Z`

```bash
source ~/final-project/.venv/bin/activate
cd ~/final-project

export CKPT="FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/st_worker_scaling_L_prod_$(date -u +%Y%m%dT%H%M%SZ)"

python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_export.py \
  --mesh-level L_prod \
  --B3-block-compose-backend csr_bulk \
  --B3-synthesis-region-dofs off \
  --output-dir "$CKPT"
```

Stage B **rejects** checkpoints without `checkpoint_export_manifest.json` (`status: PASS`).

---

## 3. Stage B — timing only (no rich vectors)

**Script:** `scripts/v2_b3_checkpoint_solve.py`  
**Environment:** `source ~/solver-mkl/activate_solver_mkl.sh`

```bash
source ~/solver-mkl/activate_solver_mkl.sh
cd ~/final-project

export CKPT="FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/st_worker_scaling_L_prod_rich_safe_20260601T164739Z"

python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_solve.py \
  --checkpoint-dir "$CKPT" \
  --factor-solver mkl_pardiso \
  --target-set full9
```

**Outputs:** `solver_benchmarks/checkpoint_solve_mkl_pardiso_full9_<utc>/` — `result.json`, `result.md`, `checkpoint_solve_manifest.json`. **No** `rich_modal/`.

---

## 4. Stage B — rich (synthesis)

Same as §3, plus:

```bash
  --B3-export-rich-modal-data
```

**Startup must show:**

```text
[B3_checkpoint_solve] rich_modal_export_requested=True
[B3_checkpoint_solver_multi] rich_modal_export_requested=True
```

**Validated solve:**

`v2_mesh_convergence/diagnostics/solver_benchmarks/checkpoint_solve_mkl_pardiso_full9_20260601T203438Z`

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_solve.py \
  --checkpoint-dir "$CKPT" \
  --factor-solver mkl_pardiso \
  --target-set full9 \
  --B3-export-rich-modal-data
```

---

## 5. Stage C — default post

**Script:** `scripts/v2_b3_rich_modal_post.py`  
**Environment:** production `.venv` only (not `solver-mkl`). No `petsc4py` / DOLFINx required in default mode.

```bash
source ~/final-project/.venv/bin/activate
cd ~/final-project

export CKPT="FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/st_worker_scaling_L_prod_rich_safe_20260601T164739Z"
export SOLVE_OUT="FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/solver_benchmarks/checkpoint_solve_mkl_pardiso_full9_20260601T203438Z"

python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_rich_modal_post.py \
  --checkpoint-dir "$CKPT" \
  --rich-modal-dir "$SOLVE_OUT/rich_modal"
```

---

## 6. Stage C — optional region DOF locate

**Optional.** Requires DOLFINx in production `.venv`. Runs facet locate in an **isolated subprocess** (segfault boundary). Use only when you need structural region indices and Stage A did not write `region_dof_indices.npz`.

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_rich_modal_post.py \
  --checkpoint-dir "$CKPT" \
  --rich-modal-dir "$SOLVE_OUT/rich_modal" \
  --B3-synthesis-region-dofs best_effort
```

---

## 7. Expected output files

### Stage A (`$CKPT/`)

| File | Required |
|------|----------|
| `checkpoint_export_manifest.json` | PASS |
| `synthesis_metadata.json` | yes |
| `A_active.petsc.bin`, `M_active.petsc.bin` | yes |
| `A_active_csr.npz`, `M_active_csr.npz` | yes |
| `built_metadata.json`, `csr_metadata.json` | yes |
| `region_dof_indices.npz` | optional (default deferred) |

### Stage B rich (`$SOLVE_OUT/`)

| File | Required |
|------|----------|
| `result.json`, `result.md`, `checkpoint_solve_manifest.json` | yes |
| `rich_modal/modes_active.npz` | yes (rich flag) |
| `rich_modal/rich_modal_manifest.json` | yes |
| `rich_modal/modes_catalog.jsonl` | yes |

### Stage C (`$SOLVE_OUT/rich_modal_post/`)

| File | Required |
|------|----------|
| `modes_synthesis.json` | yes |
| `modes_synthesis.md` | yes |
| `rich_modal_post_manifest.json` | yes |

---

## 8. PASS / FAIL checklist

### Stage A

- [ ] `checkpoint_export_manifest.json` → `status: PASS`
- [ ] `synthesis_metadata.json` present
- [ ] CSR + PETSc binaries verify in manifest

### Stage B rich

- [ ] Logs show `rich_modal_export_requested=True` (solve + multi)
- [ ] `result.json` → `rich_modal_export.requested: true`, `status: v1_enabled`
- [ ] `rich_modal/modes_active.npz` exists; `mode_count` > 0
- [ ] `targets` 9/9 PASS (full9)

### Stage B timing (no flag)

- [ ] No `rich_modal/` directory
- [ ] `rich_modal_export.requested: false`

### Stage C

- [ ] `modes_synthesis.json` → `schema: b3_rich_modal_post_v1`
- [ ] `mode_count` matches Stage B rich `mode_count`
- [ ] If no `region_dof_indices.npz`: `structural_region_participation_status: unavailable_region_indices`
- [ ] `participation_top` (and back/ribs) are **null**, not `0`
- [ ] `participation_air_p` and `cavity_pressure_max_proxy_v1` numeric when `p_idx` valid
- [ ] `warnings` list documents deferred structural regions
- [ ] `frequency_dedupe.duplicate_groups` reported (duplicates not silently dropped)

**FAIL:** Stage B without export manifest; rich flag but no `rich_modal/`; Stage C structural participations shown as `0` when indices unavailable.

---

## 9. Deferred region indices (important)

When `region_dof_indices.npz` is **missing** or Stage A used `--B3-synthesis-region-dofs off`:

- Structural participation (`participation_top`, `back`, `ribs`, `soundhole`) → **`null`**
- `structural_region_participation_status` → **`unavailable_region_indices`**
- Structural displacement RMS proxies → **`null`**
- **Not** physical zeros — do not interpret `0` as “no plate motion”

Pressure/cavity proxies may still be valid from **`p_idx_air`** in `built_metadata.json` (`participation_air_p`, `cavity_pressure_max_proxy_v1`).

---

## Quick inspect

```bash
python3 -c "import json; m=json.load(open('$CKPT/checkpoint_export_manifest.json')); print(m['status'], m.get('synthesis_export'))"
python3 -c "import json; r=json.load(open('$SOLVE_OUT/result.json')); print(r['status'], r.get('rich_modal_export'))"
python3 -c "import json; b=json.load(open('$SOLVE_OUT/rich_modal_post/modes_synthesis.json')); print(b['schema'], b['mode_count'], b['structural_region_participation_status'])"
```
