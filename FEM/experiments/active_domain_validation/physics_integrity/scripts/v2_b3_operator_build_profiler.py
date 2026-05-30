#!/usr/bin/env python3
"""Env-gated sub-phase timers for B3 L_prod operator build (profiling only)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PHASES: List[str] = [
    "mesh_load_b3_path",
    "coupled_replay_mesh_load_if_any",
    "function_space_creation",
    "weak_form_construction",
    "parent_matrix_assembly",
    "shell_trace_assembly",
    "raw_parent_block_capture",
    "block_compose_direct_AIJ",
    "boundary_cleanup",
    "active_reduction",
    "inactive_identification_row_norm_scans",
    "operator_contract",
    "checkpoint_export",
]

BLOCK_COMPOSE_MICRO_PHASES: List[str] = [
    "scaling_blocks",
    "pressure_restriction",
    "row_column_mapping",
    "nnz_counting",
    "preallocation",
    "value_insertion",
    "assembly_begin_end",
    "a_compose",
    "m_compose",
]


class _NullOperatorBuildProfiler:
    enabled = False

    def begin(self, phase: str) -> None:
        return None

    def end(self, phase: str) -> None:
        return None

    def begin_replay(self) -> None:
        return None

    def end_replay(self) -> None:
        return None

    def hook_mesh_load(self, action: str) -> None:
        return None

    def attach_fem3d(self) -> None:
        return None

    def detach_fem3d(self) -> None:
        return None

    def export_to_payload(self) -> None:
        return None

    def write_json(self, path: Path) -> None:
        return None

    def print_table(self, *, mesh_level: Optional[str] = None) -> None:
        return None

    def begin_block_compose_micro(self, phase: str) -> None:
        return None

    def end_block_compose_micro(self, phase: str) -> None:
        return None


class B3OperatorBuildProfiler:
    ENV_VAR = "B3_PROFILE_OPERATOR_BUILD"

    def __init__(self, payload: Optional[Dict[str, Any]] = None) -> None:
        self.enabled = os.environ.get(self.ENV_VAR, "").strip() == "1"
        self.payload = payload if payload is not None else {}
        self._intervals: Dict[str, float] = {p: 0.0 for p in PHASES}
        self._block_compose_micro: Dict[str, float] = {p: 0.0 for p in BLOCK_COMPOSE_MICRO_PHASES}
        self._open: Dict[str, float] = {}
        self._open_micro: Dict[str, float] = {}
        self._in_replay = False
        self._wall_t0 = time.perf_counter()

    @classmethod
    def maybe_from_env(cls, payload: Optional[Dict[str, Any]] = None) -> Union["B3OperatorBuildProfiler", _NullOperatorBuildProfiler]:
        if os.environ.get(cls.ENV_VAR, "").strip() != "1":
            return _NullOperatorBuildProfiler()
        return cls(payload=payload)

    def begin(self, phase: str) -> None:
        if not self.enabled or phase not in self._intervals:
            return
        if phase in self._open:
            return
        self._open[phase] = time.perf_counter()

    def end(self, phase: str) -> None:
        if not self.enabled or phase not in self._intervals:
            return
        t0 = self._open.pop(phase, None)
        if t0 is None:
            return
        self._intervals[phase] += time.perf_counter() - t0

    def begin_block_compose_micro(self, phase: str) -> None:
        if not self.enabled or phase not in self._block_compose_micro:
            return
        if phase in self._open_micro:
            return
        self._open_micro[phase] = time.perf_counter()

    def end_block_compose_micro(self, phase: str) -> None:
        if not self.enabled or phase not in self._block_compose_micro:
            return
        t0 = self._open_micro.pop(phase, None)
        if t0 is None:
            return
        self._block_compose_micro[phase] += time.perf_counter() - t0

    def block_compose_micro_rows(self) -> List[Dict[str, Any]]:
        block_s = max(float(self._intervals.get("block_compose_direct_AIJ", 0.0)), 1.0e-30)
        non_overlap = (
            "scaling_blocks",
            "pressure_restriction",
            "row_column_mapping",
            "nnz_counting",
            "preallocation",
            "value_insertion",
            "assembly_begin_end",
        )
        micro_total = sum(float(self._block_compose_micro.get(p, 0.0)) for p in non_overlap)
        micro_total = max(micro_total, 1.0e-30)
        rows: List[Dict[str, Any]] = []
        for phase in BLOCK_COMPOSE_MICRO_PHASES:
            sec = float(self._block_compose_micro.get(phase, 0.0))
            pct_micro = (
                100.0 * sec / micro_total
                if phase in non_overlap or phase in ("a_compose", "m_compose")
                else 0.0
            )
            if phase in ("a_compose", "m_compose"):
                pct_micro = 100.0 * sec / max(
                    float(self._block_compose_micro.get("value_insertion", 0.0)), 1.0e-30
                )
            rows.append(
                {
                    "phase": phase,
                    "seconds": round(sec, 3),
                    "percent_of_block_compose": round(100.0 * sec / block_s, 2),
                    "percent_of_micro_total": round(pct_micro, 2),
                }
            )
        rows.sort(key=lambda r: float(r["seconds"]), reverse=True)
        return rows

    def begin_replay(self) -> None:
        self._in_replay = True

    def end_replay(self) -> None:
        self._in_replay = False

    def hook_mesh_load(self, action: str) -> None:
        if not self.enabled or not self._in_replay:
            return
        if action == "begin":
            self.begin("coupled_replay_mesh_load_if_any")
        elif action == "end":
            self.end("coupled_replay_mesh_load_if_any")

    def attach_fem3d(self) -> None:
        if not self.enabled:
            return
        import fem_main_3d as fem3d

        fem3d._B3_operator_build_profiler = self

    def detach_fem3d(self) -> None:
        import fem_main_3d as fem3d

        if getattr(fem3d, "_B3_operator_build_profiler", None) is self:
            fem3d._B3_operator_build_profiler = None

    def phase_rows(self) -> List[Dict[str, Any]]:
        total = max(sum(self._intervals.values()), 1.0e-30)
        rows: List[Dict[str, Any]] = []
        for phase in PHASES:
            sec = float(self._intervals.get(phase, 0.0))
            rows.append(
                {
                    "phase": phase,
                    "seconds": round(sec, 3),
                    "percent": round(100.0 * sec / total, 2),
                }
            )
        rows.sort(key=lambda r: float(r["seconds"]), reverse=True)
        return rows

    def export_to_payload(self) -> None:
        if not self.enabled:
            return
        phases = {p: round(float(self._intervals.get(p, 0.0)), 3) for p in PHASES}
        phase_total = round(sum(phases.values()), 3)
        wall = round(time.perf_counter() - self._wall_t0, 3)
        self.payload["B3_PROFILE_operator_build_enabled"] = True
        self.payload["B3_PROFILE_operator_build_phases_seconds"] = phases
        self.payload["B3_PROFILE_operator_build_phases_total_seconds"] = phase_total
        self.payload["B3_PROFILE_operator_build_wall_elapsed_seconds"] = wall
        self.payload["B3_PROFILE_operator_build_phase_rows"] = self.phase_rows()
        micro = {p: round(float(self._block_compose_micro.get(p, 0.0)), 3) for p in BLOCK_COMPOSE_MICRO_PHASES}
        non_overlap = (
            "scaling_blocks",
            "pressure_restriction",
            "row_column_mapping",
            "nnz_counting",
            "preallocation",
            "value_insertion",
            "assembly_begin_end",
        )
        if any(v > 0.0 for v in micro.values()):
            self.payload["B3_PROFILE_block_compose_micro_phases_seconds"] = micro
            self.payload["B3_PROFILE_block_compose_micro_total_seconds"] = round(
                sum(float(micro.get(p, 0.0)) for p in non_overlap), 3
            )
            self.payload["B3_PROFILE_block_compose_micro_phase_rows"] = self.block_compose_micro_rows()

    def write_json(self, path: Path) -> None:
        if not self.enabled:
            return
        self.export_to_payload()
        path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "B3_PROFILE_operator_build_enabled": True,
            "B3_PROFILE_operator_build_phases_seconds": self.payload.get(
                "B3_PROFILE_operator_build_phases_seconds", {}
            ),
            "B3_PROFILE_operator_build_phases_total_seconds": self.payload.get(
                "B3_PROFILE_operator_build_phases_total_seconds"
            ),
            "B3_PROFILE_operator_build_wall_elapsed_seconds": self.payload.get(
                "B3_PROFILE_operator_build_wall_elapsed_seconds"
            ),
            "B3_PROFILE_operator_build_phase_rows": self.payload.get("B3_PROFILE_operator_build_phase_rows", []),
            "B3_PROFILE_block_compose_micro_phases_seconds": self.payload.get(
                "B3_PROFILE_block_compose_micro_phases_seconds"
            ),
            "B3_PROFILE_block_compose_micro_phase_rows": self.payload.get(
                "B3_PROFILE_block_compose_micro_phase_rows"
            ),
        }
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    def print_table(self, *, mesh_level: Optional[str] = None) -> None:
        if not self.enabled:
            return
        self.export_to_payload()
        title = "B3 operator build profile"
        if mesh_level:
            title = f"{title} ({mesh_level})"
        print(f"[{title}]", flush=True)
        print(f"{'phase':<42} {'seconds':>10} {'pct':>8}", flush=True)
        print("-" * 62, flush=True)
        for row in self.payload.get("B3_PROFILE_operator_build_phase_rows", []):
            if float(row.get("seconds", 0.0)) <= 0.0:
                continue
            print(
                f"{row['phase']:<42} {float(row['seconds']):10.3f} {float(row['percent']):7.2f}%",
                flush=True,
            )
        print("-" * 62, flush=True)
        print(
            f"{'phase_total':<42} "
            f"{float(self.payload.get('B3_PROFILE_operator_build_phases_total_seconds', 0.0)):10.3f}",
            flush=True,
        )
        print(
            f"{'wall_elapsed':<42} "
            f"{float(self.payload.get('B3_PROFILE_operator_build_wall_elapsed_seconds', 0.0)):10.3f}",
            flush=True,
        )
        micro_rows = self.payload.get("B3_PROFILE_block_compose_micro_phase_rows") or []
        if micro_rows:
            block_s = float(self._intervals.get("block_compose_direct_AIJ", 0.0))
            print(f"\n[block_compose_direct_AIJ micro-profile] parent={block_s:.3f}s", flush=True)
            print(f"{'micro_phase':<42} {'seconds':>10} {'%compose':>10} {'%micro':>8}", flush=True)
            print("-" * 72, flush=True)
            for row in micro_rows:
                if float(row.get("seconds", 0.0)) <= 0.0:
                    continue
                print(
                    f"{row['phase']:<42} {float(row['seconds']):10.3f} "
                    f"{float(row.get('percent_of_block_compose', 0.0)):9.2f}% "
                    f"{float(row.get('percent_of_micro_total', 0.0)):7.2f}%",
                    flush=True,
                )


def summarize_profile_json(path: Path) -> int:
    """Print compact timing table from a benchmark or profile JSON file."""
    if not path.is_file():
        print(f"[B3_PROFILE] missing json: {path}", flush=True)
        return 2
    data = json.loads(path.read_text(encoding="utf-8"))
    phases = data.get("B3_PROFILE_operator_build_phases_seconds")
    rows = data.get("B3_PROFILE_operator_build_phase_rows")
    if not phases and not rows:
        print(f"[B3_PROFILE] no operator build profile fields in {path}", flush=True)
        return 2
    mesh_level = data.get("B3_ST_scaling_mesh_level") or data.get("mesh_level")
    prof = B3OperatorBuildProfiler(payload={})
    prof.enabled = True
    micro = data.get("B3_PROFILE_block_compose_micro_phases_seconds") or {}
    if phases:
        for phase, sec in phases.items():
            if phase in prof._intervals:
                prof._intervals[phase] = float(sec)
    if micro:
        for phase, sec in micro.items():
            if phase in prof._block_compose_micro:
                prof._block_compose_micro[phase] = float(sec)
    print(f"[B3 operator build profile summary] source={path}", flush=True)
    if mesh_level:
        print(f"mesh_level={mesh_level}", flush=True)
    prof.print_table(mesh_level=None)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: v2_b3_operator_build_profiler.py PATH_TO_JSON", flush=True)
        return 2
    return summarize_profile_json(Path(args[0]))


if __name__ == "__main__":
    raise SystemExit(main())
