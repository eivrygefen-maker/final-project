#!/usr/bin/env python3
"""Generate A/B preview WAVs for each synthesis preset (no FEM)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_response_synth import (  # noqa: E402
    DEFAULT_DURATION_S,
    DEFAULT_SAMPLE_RATE,
    load_modal_data_from_path,
    synthesize_note_with_body_response,
    synthetic_classic_body_modes,
)
from synthesis_presets import list_synthesis_preset_names  # noqa: E402

PREVIEW_NOTES = (
    ("E2", 82.41),
    ("A2", 110.0),
    ("A4", 440.0),
    ("E5", 659.25),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesis preset A/B preview WAVs")
    parser.add_argument(
        "--modal-json",
        type=Path,
        default=REPO / "FEM" / "outputs" / "rom_stk_body.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "audio" / "preset_previews",
    )
    parser.add_argument("--duration", type=float, default=2.0)
    args = parser.parse_args()

    if args.modal_json.is_file():
        modal_data = load_modal_data_from_path(args.modal_json)
    else:
        modal_data = {"predicted_modes": synthetic_classic_body_modes(), "analysis": "synthetic_fixture"}

    manifest: dict = {
        "modal_json": str(args.modal_json),
        "duration_s": args.duration,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "presets": {},
    }

    for preset in list_synthesis_preset_names():
        preset_dir = args.out_dir / preset
        preset_dir.mkdir(parents=True, exist_ok=True)
        preset_rows = []
        for note_name, hz in PREVIEW_NOTES:
            wav = preset_dir / f"{note_name}.wav"
            meta_path = preset_dir / f"{note_name}.json"
            meta = synthesize_note_with_body_response(
                frequency_hz=hz,
                note_name=note_name,
                duration_s=args.duration,
                sample_rate=DEFAULT_SAMPLE_RATE,
                modal_data=modal_data,
                output_wav=wav,
                output_metadata_json=meta_path,
                synthesis_preset=preset,
            )
            preset_rows.append(
                {
                    "note": note_name,
                    "frequency_hz": hz,
                    "wav": str(wav.relative_to(args.out_dir)),
                    "synthesis_preset": meta.get("synthesis_preset"),
                    "output_rms_dbfs": meta.get("output_rms_dbfs"),
                }
            )
        manifest["presets"][preset] = preset_rows

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "preset_preview_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote preset previews under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
