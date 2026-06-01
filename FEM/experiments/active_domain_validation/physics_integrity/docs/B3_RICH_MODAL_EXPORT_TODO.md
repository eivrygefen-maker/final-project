# Rich modal export v1 (audio / STK readiness)

**Status:** implemented (v1)  
**Default:** `--B3-export-rich-modal-data` is **opt-in** on Stage B only.

## Pipeline

| Stage | Script | Always / opt-in | Outputs |
|-------|--------|-----------------|---------|
| A | `v2_b3_checkpoint_export.py` | **Always** on PASS | `synthesis_metadata.json`, `region_dof_indices.npz` (best effort) |
| B | `v2_b3_checkpoint_solve.py` | **Opt-in** flag | `rich_modal/modes_active.npz`, `rich_modal_manifest.json`, `modes_catalog.jsonl` |
| C | `v2_b3_rich_modal_post.py` | Manual / designated runs | `rich_modal_post/modes_synthesis.json`, `modes_synthesis.md` |

FOM eigensolve remains **undamped**. Material Q / damping belongs in **STK/audio**, not in the eigen solver.

Do **not** label outputs as microphone pressure. Use **audio output proxy** or **radiation proxy** wording only.

## Stage B schema (`rich_modal/modes_active.npz`)

| Array | Description |
|-------|-------------|
| `eigenvectors_active` | `(n_active, n_modes)` float64 |
| `frequency_hz`, `lambda_real`, `lambda_imag` | Per mode |
| `st_shift_target_hz`, `target_index` | ST provenance |
| `eps_slot_index`, `eps_compute_error_relative` | SLEPc slot + residual |
| `u_norm_W`, `p_norm_W`, `p_support`, `x_norm_W` | Scalar participation |

Duplicates across shifts are **retained** in v1; see `modes_catalog.jsonl` and Stage C `frequency_dedupe` report.

## Stage A schema (`synthesis_metadata.json`)

`schema: b3_synthesis_metadata_v1` — mesh path, tag protocol, GNHEP scales (when captured), `pressure_dof_scale`, `fsi_coupling_gain`, dimensions, `region_dof_indices_status`.

## Stage C schema (`modes_synthesis.json`)

`schema: b3_rich_modal_post_v1` — per-mode region participation + `audio_output_proxies` (e.g. `top_plate_displacement_rms_proxy_v1`, `cavity_pressure_max_proxy_v1`).

## Commands

```bash
# Stage A (production .venv)
python .../v2_b3_checkpoint_export.py --mesh-level L_prod --B3-block-compose-backend csr_bulk --output-dir "$CKPT"

# Stage B (solver-mkl, synthesis run only)
python .../v2_b3_checkpoint_solve.py --checkpoint-dir "$CKPT" --factor-solver mkl_pardiso --target-set full9 --B3-export-rich-modal-data

# Stage C (production .venv)
python .../v2_b3_rich_modal_post.py --checkpoint-dir "$CKPT" --rich-modal-dir "$SOLVE_OUT/rich_modal"
```

Timing benchmarks: **omit** `--B3-export-rich-modal-data`.

## Future (v1.1+)

- MAC-based dedupe across shifts
- float32 option for large LHS
- Bridge/string pickoff when region is defined
- External radiation transfer layer (not FEM cavity pressure)
