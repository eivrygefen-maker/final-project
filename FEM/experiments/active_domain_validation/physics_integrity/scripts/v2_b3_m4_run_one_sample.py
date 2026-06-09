#!/usr/bin/env python3
"""M4.5.2+ — run one guitar through the full M4 pipeline (orchestrated stages)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_freeze_first_e2e_run import (  # noqa: E402
    AGG_PASS_FREEZE_WARNING,
    AGG_STATUS_PASS,
    CHECKPOINT_TERMINAL_READY,
    SCOUT_TERMINAL_READY,
    TERMINAL_E2E,
    build_freeze_payload,
    freeze_outputs_present,
    mark_freeze_stage_failed,
    promote_pipeline_terminal_status,
    resolve_freeze_config,
    write_freeze_outputs,
    _validate_milestone,
)
from v2_b3_m4_run_status_repair import (  # noqa: E402
    STALE_RUNNING_REPAIR_REASON,
    maybe_promote_checkpoint_ready_terminal,
)
from v2_b3_m4_worker_run_lib import (  # noqa: E402
    chunk_ids_from_worker_plan,
    chunk_worker_pass_status,
    detect_repo_root,
    load_json,
    rel,
    utc_now,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

# Deferred imports of stage mains (after path setup).
import v2_b3_m4_aggregate_worker_results as agg_mod  # noqa: E402
import v2_b3_m4_lprod_checkpoint_run as ckpt_mod  # noqa: E402
import v2_b3_m4_lprod_worker_dry_run as worker_plan_mod  # noqa: E402
import v2_b3_m4_pipeline_run_scout as scout_mod  # noqa: E402
import v2_b3_m4_worker_run_remaining as workers_mod  # noqa: E402

REFERENCE_SAMPLE_ID = "sample_001"
DEFAULT_M45_BATCH_SPEC_REL = (
    "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/specs/"
    "m4_5_small_lhs_batch_first3.json"
)
DEFAULT_WORKERS = 3
STAGE_ORDER = ("scout", "worker_plan", "checkpoint", "workers", "aggregate", "freeze")

STOP_AFTER_RANK = {
    "scout": 0,
    "checkpoint": 2,
    "workers": 4,
}


def _append_log(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")


def _sample_id_from_run(run_root: Path) -> str:
    try:
        return run_root.parent.parent.name
    except IndexError:
        return ""


def load_m45_batch_allowlist(spec_path: Path) -> Dict[str, str]:
    """Return {sample_id: expected_run_id} for batch samples (honors exclude_from_batch only)."""
    spec = load_json(spec_path)
    exclude = set(spec.get("exclude_from_batch") or [])
    out: Dict[str, str] = {}
    for row in spec.get("samples") or []:
        sid = str(row.get("sample_id") or "").strip()
        rid = str(row.get("run_id") or "").strip()
        if sid and sid not in exclude:
            out[sid] = rid
    return out


def _sample_in_batch_spec(spec_path: Path, sample_id: str, run_root: Path) -> Optional[str]:
    """Return error if sample/run not in spec, else None."""
    if not spec_path.is_file():
        return f"missing batch spec: {spec_path}"
    allowlist = load_m45_batch_allowlist(spec_path)
    if sample_id not in allowlist:
        allowed = ", ".join(sorted(allowlist))
        return f"{sample_id} not in batch spec ({spec_path.name}); allowed: {allowed}"
    expected_run_id = allowlist[sample_id]
    if run_root.name != expected_run_id:
        return (
            f"run_id={run_root.name!r} does not match spec "
            f"expected {expected_run_id!r} for {sample_id}"
        )
    return None


def validate_execution_scope(
    *,
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    execute: bool,
    m45_batch_mode: bool,
    m45_batch_spec: Path,
    production_mode: bool,
    production_samples_json: Path,
    allow_unlisted_sample: bool,
    allow_reference_mutation: bool,
) -> Optional[str]:
    """Return error message if scope check fails, else None."""
    if sample_id == REFERENCE_SAMPLE_ID:
        if allow_reference_mutation:
            return None
        return (
            f"{sample_id} is the frozen M4 reference run; "
            "use --allow-reference-mutation only if you intend to modify it"
        )

    if m45_batch_mode:
        spec_path = m45_batch_spec if m45_batch_spec.is_absolute() else repo_root / m45_batch_spec
        err = _sample_in_batch_spec(spec_path, sample_id, run_root)
        return err

    if production_mode:
        spec_path = (
            production_samples_json
            if production_samples_json.is_absolute()
            else repo_root / production_samples_json
        )
        err = _sample_in_batch_spec(spec_path, sample_id, run_root)
        if err:
            return err.replace("batch spec", "production samples spec")
        return None

    if allow_unlisted_sample:
        if execute:
            print(
                f"warning: --allow-unlisted-sample permits {sample_id} outside M4.5 batch spec",
                flush=True,
            )
        return None

    if execute:
        return (
            "execute requires --production-mode, --m45-batch-mode (listed batch spec), "
            "or explicit --allow-unlisted-sample override"
        )
    return None


def _manifest(run_root: Path) -> Dict[str, Any]:
    path = run_root / "pipeline_run_manifest.json"
    return load_json(path) if path.is_file() else {}


def _stage_pass_scout(run_root: Path) -> bool:
    m = _manifest(run_root)
    if str(m.get("terminal_status")) == SCOUT_TERMINAL_READY:
        return True
    st3 = (m.get("stages") or {}).get("stage3_zones_plan") or {}
    if str(st3.get("status")) != "PASS":
        return False
    return (run_root / "lprod" / "lprod_target_plan.json").is_file() and (
        run_root / "scout" / "density_zones.json"
    ).is_file()


def _stage_pass_worker_plan(run_root: Path) -> bool:
    cmds = run_root / "lprod" / "worker_commands.json"
    chunk_plan = run_root / "lprod" / "worker_chunk_plan.preview.json"
    if not cmds.is_file() or not chunk_plan.is_file():
        return False
    worker_root = run_root / "worker_results"
    if not worker_root.is_dir():
        return False
    for cid in chunk_ids_from_worker_plan(run_root):
        if not (worker_root / cid / "chunk_targets.json").is_file():
            return False
    return True


def _stage_pass_checkpoint(run_root: Path, *, production_mode: bool = False) -> bool:
    m = _manifest(run_root)
    if str(m.get("terminal_status")) == CHECKPOINT_TERMINAL_READY:
        if not production_mode:
            return True
    ck = run_root / "lprod" / "checkpoint" / "checkpoint_export_manifest.json"
    if not ck.is_file():
        return False
    try:
        data = load_json(ck)
        export_ok = bool(data.get("export_pass")) or str(data.get("status")) == "PASS"
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not export_ok:
        return False
    if production_mode:
        from v2_b3_m4_production_contracts import validate_post_export_region_dof_contract  # noqa: WPS433

        core = run_root / "lprod" / "resolved_core_config.json"
        contract_errors = validate_post_export_region_dof_contract(
            run_root / "lprod" / "checkpoint",
            core_config_path=core if core.is_file() else None,
        )
        if contract_errors:
            return False
    return True


def _stage_pass_workers(run_root: Path) -> bool:
    planned = chunk_ids_from_worker_plan(run_root)
    if not planned:
        return False
    return all(chunk_worker_pass_status(run_root, cid) for cid in planned)


def _stage_pass_aggregate(run_root: Path) -> bool:
    agg_path = run_root / "aggregation" / "aggregation_result.json"
    if not agg_path.is_file():
        return False
    try:
        agg = load_json(agg_path)
        return (
            str(agg.get("status")) == AGG_STATUS_PASS
            and bool(agg.get("final_aggregation_ready"))
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _stage_pass_freeze(run_root: Path, sample_id: str, *, production_mode: bool = False) -> bool:
    if production_mode:
        from v2_b3_m4_production_freeze import production_freeze_complete  # noqa: WPS433

        return production_freeze_complete(run_root)
    return freeze_outputs_present(run_root)


def assess_stages(run_root: Path, *, production_mode: bool = False) -> Dict[str, Dict[str, Any]]:
    sample_id = _sample_id_from_run(run_root)
    checks = {
        "scout": _stage_pass_scout,
        "worker_plan": _stage_pass_worker_plan,
        "checkpoint": lambda r: _stage_pass_checkpoint(r, production_mode=production_mode),
        "workers": _stage_pass_workers,
        "aggregate": _stage_pass_aggregate,
        "freeze": lambda r: _stage_pass_freeze(r, sample_id, production_mode=production_mode),
    }
    out: Dict[str, Dict[str, Any]] = {}
    for name, fn in checks.items():
        passed = fn(run_root)
        if passed:
            reuse = "PASS_reuse"
        elif name == "scout" and run_root.is_dir() and (run_root / "scout").is_dir():
            reuse = "resume_possible"
        elif run_root.is_dir():
            reuse = "resume_possible"
        else:
            reuse = "planned_new"
        out[name] = {"pass": passed, "reuse_status": reuse}
    return out


def _policy_argv(
    *,
    workers: int,
    freq_min: float,
    freq_max: float,
    scout_spacing: float,
    scout_half_width: float,
    zone_dense: float,
    zone_medium: float,
    zone_sparse: float,
) -> List[str]:
    return [
        f"--workers={workers}",
        f"--freq-min-hz={freq_min}",
        f"--freq-max-hz={freq_max}",
        f"--scout-spacing-hz={scout_spacing}",
        f"--scout-half-width-hz={scout_half_width}",
        f"--zone-spacing-dense-hz={zone_dense}",
        f"--zone-spacing-medium-hz={zone_medium}",
        f"--zone-spacing-sparse-hz={zone_sparse}",
    ]


def _run_stage_scout(
    *,
    run_root: Path,
    policy: Sequence[str],
    force: bool,
    execute: bool,
) -> Tuple[int, str]:
    argv = ["--run-dir", str(run_root)] + list(policy)
    if execute:
        argv.append("--execute-scout")
        if force:
            argv.append("--force")
    else:
        argv.append("--dry-run")
    return scout_mod.main(argv), "v2_b3_m4_pipeline_run_scout.py"


def _run_stage_worker_plan(
    *,
    run_root: Path,
    workers: int,
    force: bool,
) -> Tuple[int, str]:
    argv = ["--run-dir", str(run_root), "--workers", str(workers), "--dry-run"]
    if force:
        argv.append("--force")
    return worker_plan_mod.main(argv), "v2_b3_m4_lprod_worker_dry_run.py"


def _run_stage_checkpoint(
    *,
    run_root: Path,
    force: bool,
    execute: bool,
) -> Tuple[int, str]:
    argv = ["--run-dir", str(run_root)]
    if execute:
        argv.append("--execute")
        if force:
            argv.append("--force")
    else:
        argv.append("--dry-run")
    return ckpt_mod.main(argv), "v2_b3_m4_lprod_checkpoint_run.py"


def _run_stage_workers(
    *,
    run_root: Path,
    workers: int,
    force: bool,
    execute: bool,
) -> Tuple[int, str]:
    argv = ["--run-dir", str(run_root), "--workers", str(workers)]
    if execute:
        argv.append("--execute")
        if force:
            argv.append("--force")
    else:
        argv.append("--dry-run")
    return workers_mod.main(argv), "v2_b3_m4_worker_run_remaining.py"


def _run_stage_aggregate(
    *,
    run_root: Path,
    force: bool,
    execute: bool,
) -> Tuple[int, str]:
    argv = ["--run-dir", str(run_root)]
    if execute:
        argv.append("--execute")
        if force:
            argv.append("--force")
    else:
        argv.append("--dry-run")
    return agg_mod.main(argv), "v2_b3_m4_aggregate_worker_results.py"


def _run_stage_freeze(
    *,
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    force: bool,
    production_mode: bool = False,
) -> Tuple[int, str]:
    if production_mode:
        from v2_b3_m4_production_freeze import (  # noqa: WPS433
            load_sample_input,
            replay_production_freeze,
        )

        rc, msg = replay_production_freeze(
            repo_root=repo_root,
            run_root=run_root,
            sample_input=load_sample_input(run_root),
            force=force,
        )
        return rc, msg

    if freeze_outputs_present(run_root) and not force:
        promote_pipeline_terminal_status(run_root, aggregation_status=AGG_STATUS_PASS)
        return 0, "freeze already present (idempotent accept)"
    errors = _validate_milestone(run_root=run_root)
    if errors:
        return 2, "freeze validation failed"
    cfg = resolve_freeze_config(sample_id)
    payload = build_freeze_payload(repo_root=repo_root, run_root=run_root, freeze_cfg=cfg)
    try:
        write_freeze_outputs(
            repo_root=repo_root,
            run_root=run_root,
            payload=payload,
            force=force,
            idempotent=True,
        )
    except FileExistsError:
        if freeze_outputs_present(run_root):
            promote_pipeline_terminal_status(run_root, aggregation_status=AGG_STATUS_PASS)
            return 0, "freeze outputs exist (accepted)"
        return 2, "freeze outputs exist"
    return 0, "v2_b3_m4_freeze_first_e2e_run.py (canonical sample_e2e)"


def _should_force_stage(name: str, *, force: bool, force_stages: Optional[Set[str]]) -> bool:
    if force:
        return True
    return bool(force_stages and name in force_stages)


def run_pipeline(
    *,
    repo_root: Path,
    run_root: Path,
    workers: int,
    force: bool,
    force_stages: Optional[Set[str]] = None,
    execute: bool,
    stop_after: Optional[str],
    m45_batch_mode: bool,
    m45_batch_spec: Path,
    production_mode: bool,
    production_samples_json: Path,
    allow_unlisted_sample: bool,
    allow_reference_mutation: bool,
    freq_min: float,
    freq_max: float,
    scout_spacing: float,
    scout_half_width: float,
    zone_dense: float,
    zone_medium: float,
    zone_sparse: float,
) -> int:
    if not run_root.is_dir():
        print(f"error: run-dir not found: {run_root}", file=sys.stderr)
        return 2

    sample_id = _sample_id_from_run(run_root)
    scope_err = validate_execution_scope(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        execute=execute,
        m45_batch_mode=m45_batch_mode,
        m45_batch_spec=m45_batch_spec,
        production_mode=production_mode,
        production_samples_json=production_samples_json,
        allow_unlisted_sample=allow_unlisted_sample,
        allow_reference_mutation=allow_reference_mutation,
    )
    if scope_err:
        print(f"error: {scope_err}", file=sys.stderr)
        return 2

    log_path = run_root / "logs" / "m4_run_one_sample.log"
    stop_rank = STOP_AFTER_RANK.get(stop_after or "", 999)

    stages = assess_stages(run_root, production_mode=production_mode)
    policy = _policy_argv(
        workers=workers,
        freq_min=freq_min,
        freq_max=freq_max,
        scout_spacing=scout_spacing,
        scout_half_width=scout_half_width,
        zone_dense=zone_dense,
        zone_medium=zone_medium,
        zone_sparse=zone_sparse,
    )

    write_json_atomic(
        run_root / "m4_run_one_sample_plan.json",
        {
            "schema": "m4_run_one_sample_plan_v1",
            "generated_utc": utc_now(),
            "will_execute": execute,
            "sample_id": sample_id,
            "run_id": run_root.name,
            "run_root": rel(run_root, repo_root=repo_root),
            "workers": workers,
            "stop_after": stop_after,
            "force": force,
            "force_stages": sorted(force_stages) if force_stages else None,
            "m45_batch_mode": m45_batch_mode,
            "production_mode": production_mode,
            "stage_assessment": stages,
        },
    )

    print(f"will_execute={str(execute).lower()}")
    print(f"sample_id={sample_id}")
    print(f"run_id={run_root.name}")
    print(f"run_dir={rel(run_root, repo_root=repo_root)}")
    if m45_batch_mode:
        print("m45_batch_mode=true")
    if production_mode:
        print("production_mode=true")
    for name in STAGE_ORDER:
        st = stages[name]
        print(f"  stage_{name}: pass={st['pass']} reuse={st['reuse_status']}")

    if not execute:
        print("planned stages (subprocess dry-run previews):")
        preview_runners = [
            ("scout", lambda: _run_stage_scout(run_root=run_root, policy=policy, force=False, execute=False)),
            ("worker_plan", lambda: _run_stage_worker_plan(run_root=run_root, workers=workers, force=False)),
            ("checkpoint", lambda: _run_stage_checkpoint(run_root=run_root, force=False, execute=False)),
            ("workers", lambda: _run_stage_workers(run_root=run_root, workers=workers, force=False, execute=False)),
            ("aggregate", lambda: _run_stage_aggregate(run_root=run_root, force=False, execute=False)),
        ]
        for name, run_fn in preview_runners:
            if stages[name]["pass"]:
                print(f"  - {name}: already PASS (would skip on --execute)")
                continue
            print(f"  - {name}: preview ...", flush=True)
            rc, script_name = run_fn()
            print(f"      script={script_name} rc={rc}")
        print("  - freeze: after AGGREGATION_PASS only")
        print("no solver executed")
        return 0

    _append_log(log_path, f"[{utc_now()}] M4.5 execute begin sample={sample_id} run={run_root.name}")

    runners = [
        (
            "scout",
            lambda sf=force_stages: _run_stage_scout(
                run_root=run_root,
                policy=policy,
                force=_should_force_stage("scout", force=force, force_stages=sf),
                execute=True,
            ),
        ),
        (
            "worker_plan",
            lambda sf=force_stages: _run_stage_worker_plan(
                run_root=run_root,
                workers=workers,
                force=_should_force_stage("worker_plan", force=force, force_stages=sf),
            ),
        ),
        (
            "checkpoint",
            lambda sf=force_stages: _run_stage_checkpoint(
                run_root=run_root,
                force=_should_force_stage("checkpoint", force=force, force_stages=sf),
                execute=True,
            ),
        ),
        (
            "workers",
            lambda sf=force_stages: _run_stage_workers(
                run_root=run_root,
                workers=workers,
                force=_should_force_stage("workers", force=force, force_stages=sf),
                execute=True,
            ),
        ),
        (
            "aggregate",
            lambda sf=force_stages: _run_stage_aggregate(
                run_root=run_root,
                force=_should_force_stage("aggregate", force=force, force_stages=sf),
                execute=True,
            ),
        ),
        (
            "freeze",
            lambda sf=force_stages: _run_stage_freeze(
                repo_root=repo_root,
                run_root=run_root,
                sample_id=sample_id,
                force=_should_force_stage("freeze", force=force, force_stages=sf),
                production_mode=production_mode,
            ),
        ),
    ]

    for idx, (name, run_fn) in enumerate(runners):
        st = stages[name]
        stage_force = _should_force_stage(name, force=force, force_stages=force_stages)
        if st["pass"] and not stage_force:
            print(f"[skip] {name}: already PASS (reuse)", flush=True)
            _append_log(log_path, f"[{utc_now()}] skip {name} PASS reuse")
            if name == "checkpoint":
                maybe_promote_checkpoint_ready_terminal(
                    run_root,
                    repair_reason="checkpoint_stage_reuse",
                )
            if STOP_AFTER_RANK.get(stop_after or "", 999) == idx:
                break
            continue

        if st["reuse_status"] == "resume_possible" and not stage_force and name not in ("worker_plan",):
            print(f"[resume] {name}: prior partial artifacts; attempting stage", flush=True)

        if production_mode and name == "workers":
            from v2_b3_m4_production_contracts import evaluate_production_region_dof_gate  # noqa: WPS433

            gate_ok, gate_errors = evaluate_production_region_dof_gate(
                run_root,
                repo_root=repo_root,
            )
            if not gate_ok:
                err_msg = ";".join(gate_errors)
                print(
                    f"error: strict production region-DOF gate blocked workers: {err_msg}",
                    file=sys.stderr,
                )
                _append_log(
                    log_path,
                    f"[{utc_now()}] workers blocked region_dof_gate={err_msg}",
                )
                return 2

        print(f"[run] {name} ...", flush=True)
        t0 = time.perf_counter()
        rc, script_name = run_fn()
        elapsed = time.perf_counter() - t0
        _append_log(
            log_path,
            f"[{utc_now()}] {name} script={script_name} rc={rc} elapsed_s={elapsed:.1f}",
        )
        print(f"  {name}: rc={rc} elapsed_s={elapsed:.1f}", flush=True)

        if name == "checkpoint" and rc == 0:
            maybe_promote_checkpoint_ready_terminal(
                run_root,
                repair_reason="checkpoint_stage_complete",
            )

        if rc != 0:
            if name in ("workers", "aggregate") and _stage_pass_checkpoint(
                run_root, production_mode=production_mode
            ):
                maybe_promote_checkpoint_ready_terminal(
                    run_root,
                    repair_reason=STALE_RUNNING_REPAIR_REASON,
                )
            if name == "freeze" and production_mode:
                mark_freeze_stage_failed(run_root, reason=script_name if isinstance(script_name, str) else f"rc={rc}")
                print(f"error: production freeze failed (rc={rc}); stopping pipeline", file=sys.stderr)
                return rc
            if name == "freeze" and _stage_pass_aggregate(run_root) and not production_mode:
                promote_pipeline_terminal_status(
                    run_root,
                    aggregation_status=AGG_STATUS_PASS,
                )
                print(
                    f"warning: freeze rc={rc} but {AGG_STATUS_PASS}; "
                    f"continuing as {AGG_PASS_FREEZE_WARNING}",
                    flush=True,
                )
                rc = 0
            else:
                print(f"error: stage {name} failed (rc={rc}); stopping pipeline", file=sys.stderr)
                return rc

        if name == "aggregate" and rc == 0 and not production_mode:
            promote_pipeline_terminal_status(run_root, aggregation_status=AGG_STATUS_PASS)

        if idx >= stop_rank:
            print(f"stop-after={stop_after} reached", flush=True)
            break

    # Refresh terminal summary
    agg_path = run_root / "aggregation" / "aggregation_result.json"
    if agg_path.is_file():
        agg = load_json(agg_path)
        print(f"aggregation_status={agg.get('status')}")
        print(f"completed_chunks={agg.get('completed_chunk_count')}/{agg.get('planned_chunk_count')}")
        print(f"deduped_modes={agg.get('deduped_mode_count')}")
        print(f"final_aggregation_ready={agg.get('final_aggregation_ready')}")
    try:
        from v2_b3_m4_runtime_provenance import collect_m4_runtime_provenance  # noqa: E402

        prov = collect_m4_runtime_provenance(run_root=run_root, workers_requested=workers)
        write_json_atomic(run_root / "m4_sample_runtime_provenance.json", prov)
        print(
            f"participation_computed_count={prov.get('participation_computed_count')} "
            f"workers_actual_parallel={prov.get('workers_actual_parallel')}",
            flush=True,
        )
    except Exception:
        pass
    if _stage_pass_aggregate(run_root):
        if production_mode:
            if not _stage_pass_freeze(run_root, sample_id, production_mode=True):
                mark_freeze_stage_failed(run_root, reason="production_freeze_incomplete")
        else:
            promote_pipeline_terminal_status(run_root, aggregation_status=AGG_STATUS_PASS)
    print(f"terminal_status={_manifest(run_root).get('terminal_status')}")
    print("pipeline execute finished")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="M4.5: run one guitar through M4 pipeline (scout→workers→aggregate→freeze)."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Assess stages and print plan only.")
    parser.add_argument("--execute", action="store_true", help="Run pipeline stages.")
    parser.add_argument("--force", action="store_true", help="Re-run PASS stages / overwrite outputs.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--m45-batch-mode",
        action="store_true",
        help="Allow execute only for sample_id/run_id listed in m4_5_small_lhs_batch_first3.json.",
    )
    parser.add_argument(
        "--m45-batch-spec",
        type=Path,
        default=DEFAULT_M45_BATCH_SPEC_REL,
        help="M4.5 small-batch spec JSON (default: m4_5_small_lhs_batch_first3.json).",
    )
    parser.add_argument(
        "--production-mode",
        action="store_true",
        help="Allow execute when sample_id/run_id are listed in --production-samples-json.",
    )
    parser.add_argument(
        "--production-samples-json",
        type=Path,
        default=DEFAULT_M45_BATCH_SPEC_REL,
        help="Production or validation batch spec JSON (with samples[]).",
    )
    parser.add_argument(
        "--allow-unlisted-sample",
        action="store_true",
        help="Override: permit execute for samples outside the M4.5 batch spec.",
    )
    parser.add_argument(
        "--allow-reference-mutation",
        action="store_true",
        help=f"Override: permit execute on frozen reference {REFERENCE_SAMPLE_ID}.",
    )
    parser.add_argument(
        "--stop-after",
        choices=tuple(STOP_AFTER_RANK.keys()),
        help="Stop after scout, checkpoint, or workers (execute mode).",
    )
    parser.add_argument("--force-checkpoint", action="store_true", help="Re-run checkpoint stage even if PASS.")
    parser.add_argument("--force-workers", action="store_true", help="Re-run worker stage even if PASS.")
    parser.add_argument("--force-aggregation", action="store_true", help="Re-run aggregation even if PASS.")
    parser.add_argument("--freq-min-hz", type=float, default=60.0)
    parser.add_argument("--freq-max-hz", type=float, default=550.0)
    parser.add_argument("--scout-spacing-hz", type=float, default=7.5)
    parser.add_argument("--scout-half-width-hz", type=float, default=3.75)
    parser.add_argument("--zone-spacing-dense-hz", type=float, default=6.0)
    parser.add_argument("--zone-spacing-medium-hz", type=float, default=9.0)
    parser.add_argument("--zone-spacing-sparse-hz", type=float, default=12.5)
    args = parser.parse_args(argv)

    if args.dry_run and args.execute:
        print("error: use --dry-run or --execute, not both", file=sys.stderr)
        return 2
    if not args.dry_run and not args.execute:
        print("error: specify --dry-run or --execute", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("error: --workers must be >= 1", file=sys.stderr)
        return 2

    repo_root = detect_repo_root(SCRIPT_DIR)
    run_root = args.run_dir if args.run_dir.is_absolute() else repo_root / args.run_dir
    run_root = run_root.resolve()

    force_stages: Optional[Set[str]] = None
    if args.force_checkpoint or args.force_workers or args.force_aggregation:
        force_stages = set()
        if args.force_checkpoint:
            force_stages.add("checkpoint")
        if args.force_workers:
            force_stages.add("workers")
        if args.force_aggregation:
            force_stages.add("aggregate")

    return run_pipeline(
        repo_root=repo_root,
        run_root=run_root,
        workers=int(args.workers),
        force=bool(args.force),
        force_stages=force_stages,
        execute=bool(args.execute),
        stop_after=args.stop_after,
        m45_batch_mode=bool(args.m45_batch_mode),
        m45_batch_spec=args.m45_batch_spec,
        production_mode=bool(args.production_mode),
        production_samples_json=args.production_samples_json,
        allow_unlisted_sample=bool(args.allow_unlisted_sample),
        allow_reference_mutation=bool(args.allow_reference_mutation),
        freq_min=float(args.freq_min_hz),
        freq_max=float(args.freq_max_hz),
        scout_spacing=float(args.scout_spacing_hz),
        scout_half_width=float(args.scout_half_width_hz),
        zone_dense=float(args.zone_spacing_dense_hz),
        zone_medium=float(args.zone_spacing_medium_hz),
        zone_sparse=float(args.zone_spacing_sparse_hz),
    )


if __name__ == "__main__":
    raise SystemExit(main())
