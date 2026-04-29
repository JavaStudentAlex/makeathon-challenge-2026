from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from shapely.geometry import box, mapping


def _load_split_module():
    module_path = Path(__file__).parents[1] / "shinka" / "train_validation_split.py"
    spec = importlib.util.spec_from_file_location(
        "local_train_validation_split", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


split_module = _load_split_module()


def _feature_collection(rows: list[tuple[str, object]]) -> dict:
    return {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": {"name": tile_id, "origin": "SRID=4326;POINT(0 0)"},
            }
            for tile_id, geometry in rows
        ],
    }


def _write_metadata(
    data_root: Path,
    *,
    split: str,
    rows: list[tuple[str, object]],
) -> Path:
    metadata_dir = data_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"{split}_tiles.geojson"
    metadata_path.write_text(
        json.dumps(_feature_collection(rows)),
        encoding="utf-8",
    )
    return metadata_path


def test_build_train_validation_split_uses_nearest_train_tiles_and_excludes_outlier(
    tmp_path: Path,
) -> None:
    data_root = tmp_path
    _write_metadata(
        data_root,
        split="train",
        rows=[
            ("train_a", box(-75.2, 3.0, -75.0, 3.2)),
            ("train_b", box(-74.9, 2.2, -74.7, 2.4)),
            ("train_c", box(97.9, 18.9, 98.1, 19.1)),
            ("train_d", box(105.5, 12.6, 105.7, 12.8)),
        ],
    )
    _write_metadata(
        data_root,
        split="test",
        rows=[
            ("test_a", box(-75.15, 3.05, -74.95, 3.25)),
            ("test_b", box(97.95, 18.95, 98.15, 19.15)),
            ("test_far", box(30.0, -35.0, 30.2, -34.8)),
        ],
    )

    result = split_module.build_train_validation_split(
        data_root=data_root,
        max_distance_km=500.0,
    )

    assert result["validation_tile_ids"] == ["train_a", "train_c"]
    assert result["train_tile_ids"] == ["train_b", "train_d"]
    assert result["selected_test_tile_ids"] == ["test_a", "test_b"]
    assert result["excluded_test_tile_ids"] == ["test_far"]

    nearest = result["nearest_train_by_test"]
    assert [match["test_tile_id"] for match in nearest] == [
        "test_a",
        "test_b",
        "test_far",
    ]
    assert nearest[0]["train_tile_id"] == "train_a"
    assert nearest[1]["train_tile_id"] == "train_c"
    assert nearest[2]["distance_km"] > 500.0


def test_main_writes_split_json(tmp_path: Path) -> None:
    data_root = tmp_path
    _write_metadata(
        data_root,
        split="train",
        rows=[
            ("train_a", box(-75.2, 3.0, -75.0, 3.2)),
            ("train_b", box(-74.9, 2.2, -74.7, 2.4)),
        ],
    )
    _write_metadata(
        data_root,
        split="test",
        rows=[
            ("test_a", box(-75.15, 3.05, -74.95, 3.25)),
        ],
    )
    output_path = tmp_path / "split.json"

    exit_code = split_module.main(
        [
            "--data-root",
            str(data_root),
            "--output-path",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["validation_tile_ids"] == ["train_a"]
    assert payload["train_tile_ids"] == ["train_b"]
    assert payload["selected_test_tile_ids"] == ["test_a"]
