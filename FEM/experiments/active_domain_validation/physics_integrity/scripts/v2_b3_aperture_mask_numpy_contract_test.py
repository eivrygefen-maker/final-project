#!/usr/bin/env python3
"""Regression tests: aperture mask helpers must not use ndarray truth-value checks."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_aperture_pressure_mask import (  # noqa: E402
    _as_float64_coords,
    _as_int32_index_map,
    _optional_float,
    validate_aperture_mask_contract,
)


def _assert_raises_ambiguous_or(fn) -> None:
    try:
        fn()
    except ValueError as exc:
        if "ambiguous" not in str(exc):
            raise AssertionError(f"expected ambiguous truth-value error, got: {exc}") from exc
    else:
        raise AssertionError("expected ValueError for ndarray truth-value use")


def test_as_int32_index_map_none() -> None:
    arr = _as_int32_index_map(None)
    assert arr.size == 0
    assert arr.dtype == np.int32


def test_as_int32_index_map_empty_list() -> None:
    arr = _as_int32_index_map([])
    assert arr.size == 0


def test_as_int32_index_map_single_element() -> None:
    arr = _as_int32_index_map([42])
    assert arr.tolist() == [42]


def test_as_int32_index_map_multi_element_ndarray() -> None:
    src = np.arange(8, dtype=np.int32)
    arr = _as_int32_index_map(src)
    assert np.array_equal(arr, src)


def test_or_fallback_pattern_fails_on_ndarray() -> None:
    src = np.array([1, 2, 3], dtype=np.int32)

    def bad() -> None:
        np.asarray(src or [], dtype=np.int32)  # noqa: B018

    _assert_raises_ambiguous_or(bad)


def test_validate_contract_accepts_ndarray_mask_indices() -> None:
    built = {
        "p_idx": np.array([100, 101, 102], dtype=np.int32),
        "n_w": 200,
        "n_u_b3": 50,
        "active_dimension": 10,
        "active_local": np.arange(10, dtype=np.int32),
    }
    mask = {
        "p_idx_aperture": np.array([100, 101], dtype=np.int32),
        "p_active_indices": np.array([0, 1], dtype=np.int32),
        "soundhole_center_m": [0.0, 0.0, 0.09],
        "soundhole_radius_m": 0.047,
        "selected_coordinates": [[0.0, 0.0, 0.09], [0.01, 0.0, 0.09]],
    }
    validate_aperture_mask_contract(mask, built)


def test_optional_float_none_and_scalar() -> None:
    assert _optional_float(None, 1.5) == 1.5
    assert _optional_float(0.0, 1.5) == 0.0
    assert _optional_float(np.float64(2.25), 1.5) == 2.25


def test_as_float64_coords_ndarray() -> None:
    src = np.array([[0.0, 0.0, 0.1], [0.1, 0.0, 0.1]], dtype=np.float64)
    out = _as_float64_coords(src)
    assert out.shape == (2, 3)


def test_post_replay_cfg_pressure_indices_coercion() -> None:
    """Regression for _load_pressure_layout after coupled replay sets ndarray on cfg."""
    cfg = {"_coupled_air_p_air_collapsed_indices": np.arange(40463, dtype=np.int32)}

    def old_pattern() -> None:
        np.asarray(cfg.get("_coupled_air_p_air_collapsed_indices") or [], dtype=np.int32)

    _assert_raises_ambiguous_or(old_pattern)
    arr = _as_int32_index_map(cfg.get("_coupled_air_p_air_collapsed_indices"))
    assert arr.size == 40463


def main() -> int:
    tests = [
        test_as_int32_index_map_none,
        test_as_int32_index_map_empty_list,
        test_as_int32_index_map_single_element,
        test_as_int32_index_map_multi_element_ndarray,
        test_or_fallback_pattern_fails_on_ndarray,
        test_validate_contract_accepts_ndarray_mask_indices,
        test_optional_float_none_and_scalar,
        test_as_float64_coords_ndarray,
        test_post_replay_cfg_pressure_indices_coercion,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} aperture mask numpy contract tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
