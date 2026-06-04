#!/usr/bin/env python3
"""M4.5.2+ — run one guitar through the full M4 pipeline (orchestrated stages)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_freeze_first_e2e_run import (  # noqa: E402
    AGG_STATUS_PASS,
    CHECKPOINT_TERMINAL_READY,
    SCOUT_TERMINAL_READY,
    TERMINAL_E2E,
    build_freeze_payload,
    resolve_freeze_config,
    write_freeze_outputs,
    _validate_milestone,
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
    """Return {sample_id: expected_run_id} for batch samples (excludes reference)."""
    spec = load_json(spec_path)
    exclude = set(spec.get("exclude_from_batch") or [])
    exclude.add(spec.get("reference_sample_id") or REFERENCE_SAMPLE_ID)
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


def _stage_pass_checkpoint(run_root: Path) -> bool:
    m = _manifest(run_root)
    if str(m.get("terminal_status")) == CHECKPOINT_TERMINAL_READY:
        return True
    ck = run_root / "lprod" / "checkpoint" / "checkpoint_export_manifest.json"
    if not ck.is_file():
        return False
    try:
        data = load_json(ck)
        return bool(data.get("export_pass")) or str(data.get("status")) == "PASS"
    except (OSError, ValueError, json.JSONDecodeError):
        return False


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


def _stage_pass_freeze(run_root: Path, sample_id: str) -> bool:
    cfg = resolve_freeze_config(sample_id)
    return (run_root / "freeze" / cfg["manifest_name"]).is_file()


def assess_stages(run_root: Path) -> Dict[str, Dict[str, Any]]:
    sample_id = _sample_id_from_run(run_root)
    checks = {
        "scout": _stage_pass_scout,
        "worker_plan": _stage_pass_worker_plan,
        "checkpoint": _stage_pass_checkpoint,
        "workers": _stage_pass_workers,
        "aggregate": _stage_pass_aggregate,
        "freeze": lambda r: _stage_pass_freeze(r, sample_id),
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
) -> Tuple[int, str]:
    errors = _validate_milestone(run_root=run_root)
    if errors:
        return 2, "freeze validation failed"
    cfg = resolve_freeze_config(sample_id)
    payload = build_freeze_payload(repo_root=repo_root, run_root=run_root, freeze_cfg=cfg)
    try:
        write_freeze_outputs(repo_root=repo_root, run_root=run_root, payload=payload, force=force)
    except FileExistsError:
        return 2, "freeze outputs exist"
    return 0, "v2_b3_m4_freeze_first_e2e_run.py (generic names)"


def run_pipeline(
    *,
    repo_root: Path,
    run_root: Path,
    workers: int,
    force: bool,
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

    stages = assess_stages(run_root)
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
            ("workers", lambda: _run_stage_workers(run_root=run_root, force=False, execute=False)),
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
        ("scout", lambda: _run_stage_scout(run_root=run_root, policy=policy, force=force, execute=True)),
        ("worker_plan", lambda: _run_stage_worker_plan(run_root=run_root, workers=workers, force=force)),
        ("checkpoint", lambda: _run_stage_checkpoint(run_root=run_root, force=force, execute=True)),
        ("workers", lambda: _run_stage_workers(run_root=run_root, force=force, execute=True)),
        ("aggregate", lambda: _run_stage_aggregate(run_root=run_root, force=force, execute=True)),
        ("freeze", lambda: _run_stage_freeze(repo_root=repo_root, run_root=run_root, sample_id=sample_id, force=force)),
    ]

    for idx, (name, run_fn) in enumerate(runners):
        st = stages[name]
        if st["pass"] and not force:
            print(f"[skip] {name}: already PASS (reuse)", flush=True)
            _append_log(log_path, f"[{utc_now()}] skip {name} PASS reuse")
            if STOP_AFTER_RANK.get(stop_after or "", 999) == idx:
                break
            continue

        if st["reuse_status"] == "resume_possible" and not force and name not in ("worker_plan",):
            print(f"[resume] {name}: prior partial artifacts; attempting stage", flush=True)

        print(f"[run] {name} ...", flush=True)
        t0 = time.perf_counter()
        rc, script_name = run_fn()
        elapsed = time.perf_counter() - t0
        _append_log(
            log_path,
            f"[{utc_now()}] {name} script={script_name} rc={rc} elapsed_s={elapsed:.1f}",
        )
        print(f"  {name}: rc={rc} elapsed_s={elapsed:.1f}", flush=True)

        if rc != 0:
            print(f"error: stage {name} failed (rc={rc}); stopping pipeline", file=sys.stderr)
            return rc

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

    return run_pipeline(
        repo_root=repo_root,
        run_root=run_root,
        workers=int(args.workers),
        force=bool(args.force),
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
