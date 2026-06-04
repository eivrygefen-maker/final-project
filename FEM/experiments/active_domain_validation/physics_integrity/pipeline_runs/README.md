# B3 pipeline runs layout

**Runtime policy:** Generated outputs under this tree must not be committed or indexed by default. See [B3_PIPELINE_RUNTIME_HYGIENE.md](../docs/B3_PIPELINE_RUNTIME_HYGIENE.md).

| Path | Role | Git |
|------|------|-----|
| `specs/` | Batch specs, frequency plans, dry-run inputs | Tracked (source) |
| `schemas/` | JSON schemas and examples | Tracked |
| `guitars/` | Per-sample run trees (`runs/<run_id>/`) | **Ignored** |
| `batches/` | Multi-guitar batch manifests and plans | **Ignored** |
| `scout_density_reports/` | M3.4 coarse scout reports | **Ignored** |
| `config_overlays/` | Resolved pilot overlays | **Ignored** |
| `logs/` | Pipeline-wide logs | **Ignored** |

Write all heavy execution output under `guitars/<sample_id>/runs/<run_id>/` (scout, lprod/checkpoint, worker_results, aggregation, freeze, logs).

**Production LHS:** `scripts/v2_b3_m4_lhs_production_batch.py` — see `docs/B3_M4_PRODUCTION_PROMOTION_AUDIT.md`.
