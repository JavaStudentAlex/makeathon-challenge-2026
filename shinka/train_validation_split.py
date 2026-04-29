"""Build a spatial train/validation split from challenge tile metadata.

The split selects the train tiles nearest to the test tiles by centroid
distance. A distance cutoff keeps the validation set focused on same-region
tiles and excludes distant outliers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
from pyproj import Geod, Transformer
from shapely.geometry import Point

DEFAULT_INPUT_CRS = "EPSG:4326"
DEFAULT_AREA_CRS = "EPSG:6933"
DEFAULT_MAX_DISTANCE_KM = 500.0
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data/makeathon-challenge"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "shinka/train_validation_split.json"


def build_train_validation_split(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    *,
    max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
    area_crs: str = DEFAULT_AREA_CRS,
) -> dict[str, Any]:
    """Create a train/validation split from tile metadata.

    The closest train tile is found for every test tile. Matches within
    ``max_distance_km`` are treated as validation tiles, and the remaining
    train tiles are kept for training.
    """

    data_root = Path(data_root)
    train_metadata_path = data_root / "metadata" / "train_tiles.geojson"
    test_metadata_path = data_root / "metadata" / "test_tiles.geojson"

    if not train_metadata_path.exists():
        raise FileNotFoundError(f"Train tile metadata not found: {train_metadata_path}")
    if not test_metadata_path.exists():
        raise FileNotFoundError(f"Test tile metadata not found: {test_metadata_path}")

    train_tiles = _load_tile_metadata(train_metadata_path)
    test_tiles = _load_tile_metadata(test_metadata_path)
    train_centers = _tile_centers(train_tiles, area_crs=area_crs)
    test_centers = _tile_centers(test_tiles, area_crs=area_crs)
    geod = Geod(ellps="WGS84")

    nearest_matches = [
        _nearest_train_match(test_tile_id, test_center, train_centers, geod)
        for test_tile_id, test_center in test_centers.items()
    ]
    selected_matches = [
        match for match in nearest_matches if match["distance_km"] <= max_distance_km
    ]

    if not selected_matches:
        raise ValueError(
            "No train tiles fell within the validation distance cutoff; "
            "increase max_distance_km"
        )

    validation_tile_ids = _unique_in_order(
        match["train_tile_id"] for match in selected_matches
    )
    validation_tile_id_set = set(validation_tile_ids)
    train_tile_ids = [
        tile_id
        for tile_id in train_tiles["name"].astype(str).tolist()
        if tile_id not in validation_tile_id_set
    ]

    return {
        "data_root": str(data_root),
        "train_metadata_path": str(train_metadata_path),
        "test_metadata_path": str(test_metadata_path),
        "area_crs": area_crs,
        "max_distance_km": float(max_distance_km),
        "train_tile_ids": train_tile_ids,
        "validation_tile_ids": validation_tile_ids,
        "nearest_train_by_test": nearest_matches,
        "selected_test_tile_ids": [match["test_tile_id"] for match in selected_matches],
        "excluded_test_tile_ids": [
            match["test_tile_id"]
            for match in nearest_matches
            if match["distance_km"] > max_distance_km
        ],
    }


def write_train_validation_split(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
    area_crs: str = DEFAULT_AREA_CRS,
) -> Path:
    """Write the spatial train/validation split to a JSON file."""

    output_path = Path(output_path)
    split = build_train_validation_split(
        data_root=data_root,
        max_distance_km=max_distance_km,
        area_crs=area_crs,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return output_path


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser for split generation."""

    parser = argparse.ArgumentParser(description="Build a spatial train split")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=DEFAULT_MAX_DISTANCE_KM,
        help=(
            "Maximum centroid distance for a train tile to be selected as a "
            "validation tile"
        ),
    )
    parser.add_argument(
        "--area-crs",
        default=DEFAULT_AREA_CRS,
        help=f"CRS used for centroid calculations; default {DEFAULT_AREA_CRS}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for generating the split JSON artifact."""

    parser = build_argparser()
    args = parser.parse_args(argv)
    output_path = write_train_validation_split(
        args.output_path,
        data_root=args.data_root,
        max_distance_km=args.max_distance_km,
        area_crs=args.area_crs,
    )
    print(f"wrote train/validation split to {output_path}")
    return 0


def _load_tile_metadata(metadata_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(metadata_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(DEFAULT_INPUT_CRS)
    if "name" not in gdf.columns:
        raise ValueError(f"Tile metadata missing name column: {metadata_path}")
    return gdf


def _tile_centers(
    gdf: gpd.GeoDataFrame,
    *,
    area_crs: str,
) -> dict[str, Any]:
    projected = gdf.to_crs(area_crs)
    centroids = projected.geometry.centroid
    transformer = Transformer.from_crs(area_crs, DEFAULT_INPUT_CRS, always_xy=True)
    return {
        str(tile_id): Point(*transformer.transform(center.x, center.y))
        for tile_id, center in zip(gdf["name"].astype(str), centroids, strict=False)
    }


def _nearest_train_match(
    test_tile_id: str,
    test_center: Any,
    train_centers: dict[str, Any],
    geod: Geod,
) -> dict[str, Any]:
    best_train_tile_id = ""
    best_distance_km = float("inf")

    for train_tile_id, train_center in train_centers.items():
        _, _, distance_m = geod.inv(
            float(test_center.x),
            float(test_center.y),
            float(train_center.x),
            float(train_center.y),
        )
        distance_km = float(distance_m / 1000.0)
        if distance_km < best_distance_km or (
            distance_km == best_distance_km and train_tile_id < best_train_tile_id
        ):
            best_train_tile_id = train_tile_id
            best_distance_km = distance_km

    return {
        "test_tile_id": test_tile_id,
        "train_tile_id": best_train_tile_id,
        "distance_km": best_distance_km,
    }


def _unique_in_order(values: Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        tile_id = str(value)
        if tile_id in seen:
            continue
        seen.add(tile_id)
        ordered.append(tile_id)
    return ordered


if __name__ == "__main__":
    raise SystemExit(main())
