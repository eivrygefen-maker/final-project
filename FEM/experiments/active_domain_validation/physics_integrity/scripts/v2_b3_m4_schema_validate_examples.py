#!/usr/bin/env python3
"""M4.1 — validate schema example JSON (parse + keys; optional jsonschema). No execution."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PHYSICS_ROOT = SCRIPT_DIR.parent
SCHEMA_ROOT = PHYSICS_ROOT / "pipeline_runs" / "schemas" / "m4"
EXAMPLES_DIR = SCHEMA_ROOT / "examples"

EXAMPLE_TO_SCHEMA: Tuple[Tuple[str, str], ...] = (
    ("sample_input.example.json", "sample_input.schema.json"),
    ("lprod_target_plan.example.json", "lprod_target_plan.schema.json"),
    ("worker_chunk_plan.example.json", "worker_chunk_plan.schema.json"),
    ("worker_result.example.json", "worker_result.schema.json"),
    ("aggregation_result.example.json", "aggregation_result.schema.json"),
    ("pipeline_run_manifest.example.json", "pipeline_run_manifest.schema.json"),
)

REQUIRED_TOP_LEVEL: Dict[str, Sequence[str]] = {
    "sample_input.example.json": ("schema", "sample_id", "shape_name", "parameters"),
    "lprod_target_plan.example.json": (
        "schema",
        "sample_id",
        "run_id",
        "zone_policy_version",
        "target_generation_policy",
        "frequency_range_hz",
        "targets_hz",
        "target_windows_hz",
        "coverage_check",
        "estimated_runtime",
    ),
    "worker_chunk_plan.example.json": (
        "schema",
        "sample_id",
        "run_id",
        "chunk_policy_version",
        "frequency_range_hz",
        "chunks",
    ),
    "worker_result.example.json": (
        "schema",
        "sample_id",
        "run_id",
        "worker_id",
        "chunk_id",
        "status",
        "started_utc",
        "finished_utc",
        "targets_attempted",
        "accepted_modes",
        "unique_modes",
        "result_json",
        "warnings",
        "errors",
        "timing",
        "solver_metadata",
    ),
    "aggregation_result.example.json": (
        "schema",
        "sample_id",
        "run_id",
        "all_worker_results",
        "dedupe_tolerance_hz",
        "unique_modes",
        "mode_catalog_path",
        "modal_npz_path",
        "plots",
        "warnings",
        "failures",
        "status",
    ),
    "pipeline_run_manifest.example.json": (
        "schema",
        "sample_id",
        "run_id",
        "terminal_status",
        "stages",
    ),
}

CHUNK_REQUIRED = (
    "chunk_id",
    "freq_range_hz",
    "zone_ids",
    "targets_hz",
    "target_windows_hz",
    "estimated_cost",
    "status",
    "assigned_worker_id",
)

COVERAGE_REQUIRED = ("pass", "band_hz", "max_gap_hz")


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _missing_keys(data: Mapping[str, Any], required: Sequence[str]) -> List[str]:
    return [k for k in required if k not in data]


def _check_nested(data: Mapping[str, Any], example_name: str) -> List[str]:
    errors: List[str] = []
    if example_name == "worker_chunk_plan.example.json":
        chunks = data.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            errors.append("chunks must be a non-empty array")
        else:
            for i, chunk in enumerate(chunks):
                if not isinstance(chunk, dict):
                    errors.append(f"chunks[{i}] must be an object")
                    continue
                miss = _missing_keys(chunk, CHUNK_REQUIRED)
                if miss:
                    errors.append(f"chunks[{i}] missing: {', '.join(miss)}")
    if example_name == "lprod_target_plan.example.json":
        cov = data.get("coverage_check")
        if not isinstance(cov, dict):
            errors.append("coverage_check must be an object")
        else:
            miss = _missing_keys(cov, COVERAGE_REQUIRED)
            if miss:
                errors.append(f"coverage_check missing: {', '.join(miss)}")
            band = cov.get("band_hz")
            if band != [60.0, 550.0] and band != [60, 550]:
                errors.append(f"coverage_check.band_hz expected [60, 550], got {band!r}")
            if cov.get("pass") is not True:
                errors.append("coverage_check.pass must be true for gapless example")
    return errors


def _validate_with_jsonschema(
    instance: Any, schema_path: Path
) -> Tuple[bool, str]:
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return False, "jsonschema not installed"

    schema = _load_json(schema_path)
    try:
        jsonschema.validate(instance=instance, schema=schema)
    except jsonschema.ValidationError as exc:
        return False, str(exc)
    return True, "ok"


def main() -> int:
    parse_errors: List[str] = []
    key_errors: List[str] = []
    schema_notes: List[str] = []

    for example_name, schema_name in EXAMPLE_TO_SCHEMA:
        example_path = EXAMPLES_DIR / example_name
        schema_path = SCHEMA_ROOT / schema_name
        if not example_path.is_file():
            parse_errors.append(f"missing example: {example_path}")
            continue
        if not schema_path.is_file():
            parse_errors.append(f"missing schema: {schema_path}")
            continue
        try:
            data = _load_json(example_path)
        except (json.JSONDecodeError, OSError) as exc:
            parse_errors.append(f"{example_name}: {exc}")
            continue

        required = REQUIRED_TOP_LEVEL.get(example_name, ())
        miss = _missing_keys(data, required) if isinstance(data, dict) else list(required)
        if miss:
            key_errors.append(f"{example_name} missing top-level: {', '.join(miss)}")
        elif isinstance(data, dict):
            key_errors.extend(
                f"{example_name}: {e}" for e in _check_nested(data, example_name)
            )

        ok, msg = _validate_with_jsonschema(data, schema_path)
        if ok:
            schema_notes.append(f"{example_name}: jsonschema OK")
        elif msg == "jsonschema not installed":
            schema_notes.append(f"{example_name}: jsonschema skipped (not installed)")
        else:
            key_errors.append(f"{example_name} jsonschema: {msg}")

    if parse_errors:
        print("schemas/examples parse FAIL")
        for err in parse_errors:
            print(f"  {err}")
        return 1

    print("schemas/examples parse OK")

    if key_errors:
        print("required key checks FAIL")
        for err in key_errors:
            print(f"  {err}")
        return 1

    print("required key checks PASS")
    for note in schema_notes:
        print(f"  {note}")
    print("no execution performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
