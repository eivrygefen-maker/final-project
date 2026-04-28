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


def main():
    parser = argparse.ArgumentParser(description="ROM workflow for 3D guitar FEM.")
    parser.add_argument("--shapes-config", type=str, default=None, help="Path to rom_shapes.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-shapes", help="List configured guitar shapes.")

    p_offline = sub.add_parser("offline", help="Run FOM parameter sweep and save snapshots.")
    p_offline.add_argument("--shape", required=True)
    p_offline.add_argument("--num-modes", type=int, default=6)
    p_offline.add_argument("--sampling", choices=["structured", "lhs"], default=None)
    p_offline.add_argument("--lhs-samples", type=int, default=None)
    p_offline.add_argument("--seed", type=int, default=123)

    p_basis = sub.add_parser("build-basis", help="Build POD basis from snapshots.")
    p_basis.add_argument("--shape", required=True)
    p_basis.add_argument("--energy", type=float, default=0.999)
    p_basis.add_argument("--max-rank", type=int, default=128)

    p_online = sub.add_parser("online", help="Run ROM projected solve.")
    p_online.add_argument("--shape", required=True)
    p_online.add_argument("--nev", type=int, default=3)
    p_online.add_argument("--set", nargs="*", default=[], help="Parameter overrides key=value")

    p_compare = sub.add_parser("compare", help="Run FOM vs ROM comparison.")
    p_compare.add_argument("--shape", required=True)
    p_compare.add_argument("--nev", type=int, default=3)
    p_compare.add_argument("--fom-modes", type=int, default=6)
    p_compare.add_argument("--set", nargs="*", default=[], help="Parameter overrides key=value")

    args = parser.parse_args()
    manager = ROMManager(shapes_config_path=Path(args.shapes_config) if args.shapes_config else None)

    if args.cmd == "list-shapes":
        print(json.dumps({"shapes": manager.list_shapes()}, indent=2))
        return

    if args.cmd == "offline":
        files = manager.collect_snapshots(
            args.shape,
            num_modes=args.num_modes,
            sampling=args.sampling,
            lhs_samples=args.lhs_samples,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "shape": args.shape,
                    "snapshots_written": len(files),
                    "first_snapshot": str(files[0]) if files else None,
                    "last_snapshot": str(files[-1]) if files else None,
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

