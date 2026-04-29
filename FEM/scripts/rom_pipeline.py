import argparse
import json
from pathlib import Path

from FEM.rom import ROMManager


def _parse_overrides(pairs):
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid --set value '{pair}', expected key=value")
        k, v = pair.split("=", 1)
        try:
            if "." in v or "e" in v.lower():
                out[k] = float(v)
            else:
                out[k] = int(v)
        except Exception:
            out[k] = v
    return out


def _flatten_dict(data, prefix=""):
    out = {}
    for k, v in data.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


def _print_param_preview(shape_name: str, base_cfg: dict, overrides: dict):
    base_flat = _flatten_dict(base_cfg)
    keys = sorted(set(overrides.keys()) | set(base_flat.keys()))
    rows = []
    for k in keys:
        if k in overrides:
            rows.append((k, overrides[k], "override"))
        else:
            rows.append((k, base_flat.get(k), "default"))

    # Keep terminal output focused but explicit.
    print(f"\n[online] shape={shape_name} parameter preview")
    print(f"{'parameter':<42} {'value':<24} {'source':<10}")
    print("-" * 80)
    for k, v, src in rows:
        if k.startswith("geometry.") or k.startswith("materials.") or k in overrides:
            print(f"{k:<42} {str(v):<24} {src:<10}")
    print("-" * 80)


def _read_lhs_pool_summary(manager: ROMManager, shape_name: str):
    paths = manager._shape_paths(shape_name)
    pool_path = paths.get("lhs_pool")
    if not pool_path or not pool_path.exists():
        return None
    with open(pool_path, "r", encoding="utf-8") as f:
        pool = json.load(f)
    entries = pool.get("entries", [])
    counts = {"pending": 0, "running": 0, "completed": 0, "error": 0}
    for e in entries:
        st = str(e.get("status", "pending"))
        if st in counts:
            counts[st] += 1
    return {"pool_file": str(pool_path), "total": len(entries), "counts": counts}


def main():
    parser = argparse.ArgumentParser(description="ROM workflow for 3D guitar FEM.")
    parser.add_argument("--shapes-config", type=str, default=None, help="Path to rom_shapes.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-shapes", help="List configured guitar shapes.")

    p_offline = sub.add_parser("offline", help="Run FOM parameter sweep and save snapshots.")
    p_offline.add_argument("--shape", required=True)
    p_offline.add_argument("--pool-size", type=int, default=500)
    p_offline.add_argument("--max-runs", type=int, default=50)
    p_offline.add_argument("--num-modes", type=int, default=10)
    p_offline.add_argument("--force-pool-rebuild", action="store_true", default=False)
    p_offline.add_argument("--sampling", choices=["structured", "lhs"], default="lhs")
    p_offline.add_argument("--lhs-samples", type=int, default=None)
    p_offline.add_argument("--num-samples", type=int, default=None, help="Alias for --lhs-samples")
    p_offline.add_argument("--dry-run", action="store_true")
    p_offline.add_argument("--retry-errors", action="store_true", help="Reset LHS pool error entries to pending before run.")
    p_offline.add_argument("--seed", type=int, default=123)

    p_basis = sub.add_parser("build-basis", help="Build POD basis from snapshots.")
    p_basis.add_argument("--shape", required=True)
    p_basis.add_argument("--energy", type=float, default=0.999)
    p_basis.add_argument("--max-rank", type=int, default=128)

    p_online = sub.add_parser("online", help="Run ROM projected solve.")
    p_online.add_argument("--shape", required=True)
    p_online.add_argument("--nev", type=int, default=3)
    p_online.add_argument("--set", nargs="*", default=[], help="Parameter overrides key=value")
    p_online.add_argument(
        "--params_json",
        type=str,
        default=None,
        help='Raw JSON object for overrides, e.g. \'{"geometry.thickness":0.0035}\'',
    )

    p_compare = sub.add_parser("compare", help="Run FOM vs ROM comparison.")
    p_compare.add_argument("--shape", required=True)
    p_compare.add_argument("--nev", type=int, default=3)
    p_compare.add_argument("--fom-modes", type=int, default=10)
    p_compare.add_argument("--set", nargs="*", default=[], help="Parameter overrides key=value")

    args = parser.parse_args()
    manager = ROMManager(shapes_config_path=Path(args.shapes_config) if args.shapes_config else None)

    if args.cmd == "list-shapes":
        print(json.dumps({"shapes": manager.list_shapes()}, indent=2))
        return

    if args.cmd == "offline":
        lhs_samples = args.lhs_samples if args.lhs_samples is not None else args.num_samples
        files = manager.collect_snapshots(
            args.shape,
            num_modes=args.num_modes,
            sampling=args.sampling,
            lhs_samples=lhs_samples,
            pool_size=args.pool_size,
            max_runs=args.max_runs,
            seed=args.seed,
            dry_run=args.dry_run,
            retry_errors=args.retry_errors,
            force_pool_rebuild=args.force_pool_rebuild,
        )
        batch_summary = manager.get_last_collect_summary()
        print(
            json.dumps(
                {
                    "shape": args.shape,
                    "dry_run": bool(args.dry_run),
                    "retry_errors": bool(args.retry_errors),
                    "force_pool_rebuild": bool(args.force_pool_rebuild),
                    "pool_size": int(args.pool_size),
                    "max_runs": int(args.max_runs),
                    "snapshots_written": len(files),
                    "first_snapshot": str(files[0]) if files else None,
                    "last_snapshot": str(files[-1]) if files else None,
                    "batch_summary": batch_summary,
                    "lhs_pool": _read_lhs_pool_summary(manager, args.shape),
                },
                indent=2,
            )
        )
        return

    if args.cmd == "build-basis":
        basis = manager.build_basis(args.shape, energy=args.energy, max_rank=args.max_rank)
        print(json.dumps({"shape": args.shape, "basis_file": str(basis)}, indent=2))
        return

    if args.cmd == "online":
        params = _parse_overrides(args.set)
        if args.params_json:
            raw = json.loads(args.params_json)
            if not isinstance(raw, dict):
                raise ValueError("--params_json must decode to a JSON object.")
            raw_flat = _flatten_dict(raw)
            params.update(raw_flat)

        # Missing keys keep defaults from the shape base config.
        base_cfg = manager._load_shape_base_config(args.shape)
        _print_param_preview(args.shape, base_cfg, params)
        out = manager.solve_online(args.shape, params=params, nev=args.nev)
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "compare":
        params = _parse_overrides(args.set)
        out = manager.compare(args.shape, params=params, nev=args.nev, fom_modes=args.fom_modes)
        print(json.dumps(out, indent=2))
        return


if __name__ == "__main__":
    main()

