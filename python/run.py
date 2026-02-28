# python/run.py
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def load_notes_map(notes_path: Path) -> Dict[str, float]:
    if not notes_path.exists():
        return {}
    data = json.loads(notes_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        die(f"notes.json must be a JSON object (dict). Got: {type(data)}")
    out: Dict[str, float] = {}
    for k, v in data.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            pass
    return out


def get_note_hz(stk_cfg: Dict[str, Any], notes_map: Dict[str, float]) -> float | None:
    # Prefer explicit numeric note_hz if provided
    if "note_hz" in stk_cfg:
        return float(stk_cfg["note_hz"])

    # Otherwise support note names like "A2", "F#3", ...
    note_name = stk_cfg.get("note")
    if note_name is None:
        return None

    note_name = str(note_name)
    if note_name not in notes_map:
        die(f"Unknown note '{note_name}'. Add it to python/notes.json.")
    return float(notes_map[note_name])


def resolve_run_file(arg: str, project_root: Path) -> Path:
    p = Path(arg)
    if p.suffix.lower() != ".json":
        # allow shorthand: ./run full_plate  -> runs/full_plate.json
        p = project_root / "runs" / f"{arg}.json"
    elif not p.is_absolute():
        p = project_root / p

    if not p.exists():
        die(f"Run JSON not found: {p}")
    return p


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    notes_map = load_notes_map(project_root / "python" / "notes.json")

    venv_python = project_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        die(f"Venv python missing: {venv_python}")

    if len(sys.argv) < 2:
        die(
            "Usage:\n"
            "  ./run <name>\n"
            "  ./run runs/<file>.json\n\n"
            "Examples:\n"
            "  ./run fem_only_plate\n"
            "  ./run full_plate_A2\n"
        )

    run_file = resolve_run_file(sys.argv[1], project_root)
    data: Dict[str, Any] = json.loads(run_file.read_text(encoding="utf-8"))

    steps = data.get("steps", {})
    if not isinstance(steps, dict):
        die("Invalid run JSON: 'steps' must be an object.")

    # Ensure common output dirs exist
    (project_root / "audio").mkdir(exist_ok=True)
    (project_root / "FEM" / "outputs").mkdir(parents=True, exist_ok=True)

    # ---- Step: FEM ----
    fem = steps.get("fem")
    if fem is not None:
        config = fem.get("config")
        if not config:
            die("Run JSON fem step missing: steps.fem.config")

        out = fem.get("out", "")
        solution_type = fem.get("solution_type", "")
        solution_types_file = fem.get("solution_types_file", "")
        cmd = [str(venv_python), "FEM/scripts/fem_main.py", "--config", str(config)]
        if out:
            cmd += ["--out", str(out)]
        if solution_type:
            cmd += ["--solution-type", str(solution_type)]
            if solution_types_file:
                cmd += ["--solution-types-file", str(solution_types_file)]

        run_cmd(cmd, cwd=project_root)

    # ---- Step: ANALYTIC (analytic vs numeric comparison) ----
    analytic = steps.get("analytic")
    if analytic is not None:
        config = analytic.get("config")
        result = analytic.get("result")
        if not config:
            die("Run JSON analytic step missing: steps.analytic.config")
        if not result:
            die("Run JSON analytic step missing: steps.analytic.result")

        mmax = int(analytic.get("mmax", 8))
        nmax = int(analytic.get("nmax", 8))
        top = int(analytic.get("top", 20))
        out = analytic.get("out", "")

        # Ensure output dir exists if out is provided
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(venv_python),
            "python/compare_analytic.py",
            "--config", str(config),
            "--result", str(result),
            "--mmax", str(mmax),
            "--nmax", str(nmax),
            "--top", str(top),
        ]
        if out:
            cmd += ["--out", str(out)]

        run_cmd(cmd, cwd=project_root)

    # ---- Step: STK ----
    stk = steps.get("stk")
    if stk is not None:
        binary = stk.get("binary", "./guitar_stk")
        out_wav = stk.get("out", "audio/out.wav")
        dur = float(stk.get("dur", 3.0))
        amp = float(stk.get("amp", 0.2))

        # NOTE: our updated C++ expects --note_hz (or --freq as alias)
        note_hz = get_note_hz(stk, notes_map)
        if note_hz is None:
            die("Run JSON stk step missing note. Provide 'note' (e.g. 'A2') or 'note_hz'.")

        fem_json = stk.get("fem_json", "")

        cmd = [
            str(binary),
            "--note_hz", str(float(note_hz)),
            "--dur", str(dur),
            "--amp", str(amp),
            "--out", str(out_wav),
        ]

        # If fem_json is provided, we run the "note + body coloration" mode.
        if fem_json:
            cmd += ["--fem_json", str(fem_json)]

            # Optional modal params
            if "modes" in stk:
                cmd += ["--modes", str(int(stk.get("modes")))]
            if "skip" in stk:
                cmd += ["--skip", str(int(stk.get("skip")))]

            # ---- MIX: fixed overrides range ----
            if "mix" in stk:
                cmd += ["--mix", str(float(stk.get("mix")))]
            else:
                if "mix_min" in stk and "mix_max" in stk:
                    cmd += ["--mix_min", str(float(stk.get("mix_min")))]
                    cmd += ["--mix_max", str(float(stk.get("mix_max")))]
                    if "mix_mode" in stk:
                        cmd += ["--mix_mode", str(stk.get("mix_mode"))]
                    if "mix_seed" in stk:
                        cmd += ["--mix_seed", str(int(stk.get("mix_seed")))]

            # ---- Q: fixed overrides range ----
            if "q" in stk:
                cmd += ["--q", str(float(stk.get("q")))]
            else:
                if "q_min" in stk and "q_max" in stk:
                    cmd += ["--q_min", str(float(stk.get("q_min")))]
                    cmd += ["--q_max", str(float(stk.get("q_max")))]
                    if "q_mode" in stk:
                        cmd += ["--q_mode", str(stk.get("q_mode"))]
                    if "q_seed" in stk:
                        cmd += ["--q_seed", str(int(stk.get("q_seed")))]

            # ---- rad_k ----
            if "rad_k" in stk:
                cmd += ["--rad_k", str(float(stk.get("rad_k")))]

            # ---- NEW: wet_gain ----
            if "wet_gain" in stk:
                cmd += ["--wet_gain", str(float(stk.get("wet_gain")))]

            # ---- Optional seed for STK randomness ----
            if "seed" in stk:
                cmd += ["--seed", str(int(stk.get("seed")))]

            # ---- Optional pluck_pos ----
            if "pluck_pos" in stk:
                cmd += ["--pluck_pos", str(float(stk.get("pluck_pos")))]

            # ---- Optional string controls ----
            if "string_sustain" in stk:
                cmd += ["--string_sustain", str(float(stk.get("string_sustain")))]
            if "string_detune" in stk:
                cmd += ["--string_detune", str(float(stk.get("string_detune")))]

        run_cmd(cmd, cwd=project_root)

    print("\nDone.")
    print(f"Run file: {run_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
