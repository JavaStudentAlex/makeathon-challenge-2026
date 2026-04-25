from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from submission_utils import (
    is_valid_time_step,
    raster_to_geojson,
    validate_submission_geojson,
)


def _write_binary_raster(
    path: Path, data: np.ndarray, pixel_size: float = 100.0
) -> None:
    height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs="EPSG:32618",
        transform=from_origin(500000, 1000, pixel_size, pixel_size),
    ) as dst:
        dst.write(data.astype(np.uint8), 1)


def test_raster_to_geojson_filters_small_polygons_and_writes_output(
    tmp_path: Path,
) -> None:
    raster_path = tmp_path / "prediction.tif"
    output_path = tmp_path / "submission" / "prediction.geojson"

    data = np.zeros((5, 5), dtype=np.uint8)
    data[0:2, 0:2] = 1
    data[4, 4] = 1
    _write_binary_raster(raster_path, data)

    geojson = raster_to_geojson(
        raster_path=raster_path,
        output_path=output_path,
        min_area_ha=1.5,
    )

    assert output_path.exists()
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1
    assert geojson["features"][0]["properties"]["time_step"] is None


def test_raster_to_geojson_accepts_scalar_time_step(tmp_path: Path) -> None:
    raster_path = tmp_path / "prediction.tif"
    output_path = tmp_path / "submission" / "prediction.geojson"

    data = np.ones((2, 2), dtype=np.uint8)
    _write_binary_raster(raster_path, data)

    geojson = raster_to_geojson(
        raster_path=raster_path,
        output_path=output_path,
        min_area_ha=0.0,
        time_step=2204,
    )

    assert geojson["features"][0]["properties"]["time_step"] == 2204
    validate_submission_geojson(output_path)


def test_validate_submission_geojson_rejects_wrong_extension(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.json"
    submission_path.write_text(
        '{"type": "FeatureCollection", "features": []}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="extension must be .geojson"):
        validate_submission_geojson(submission_path)


def test_validate_submission_geojson_rejects_non_polygon_geometry() -> None:
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0, 0]},
                "properties": {},
            }
        ],
    }

    with pytest.raises(ValueError, match="Polygon or MultiPolygon"):
        validate_submission_geojson(geojson)


def test_validate_submission_geojson_rejects_invalid_time_step() -> None:
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0, 0], [1, 0], [1, 1], [0, 0]],
                    ],
                },
                "properties": {"time_step": 2213},
            }
        ],
    }

    with pytest.raises(ValueError, match="time_step"):
        validate_submission_geojson(geojson)


def test_is_valid_time_step_allows_null_or_yymm() -> None:
    assert is_valid_time_step(None)
    assert is_valid_time_step(2204)
    assert is_valid_time_step("2204")
    assert not is_valid_time_step(2213)
    assert not is_valid_time_step(True)


def test_raster_to_geojson_rejects_empty_binary_raster(tmp_path: Path) -> None:
    raster_path = tmp_path / "empty_prediction.tif"
    _write_binary_raster(raster_path, np.zeros((3, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="No deforestation pixels"):
        raster_to_geojson(raster_path=raster_path)


def test_raster_to_geojson_allows_empty_binary_raster(tmp_path: Path) -> None:
    raster_path = tmp_path / "empty_prediction.tif"
    output_path = tmp_path / "submission" / "empty_prediction.geojson"
    _write_binary_raster(raster_path, np.zeros((3, 3), dtype=np.uint8))

    geojson = raster_to_geojson(
        raster_path=raster_path,
        output_path=output_path,
        allow_empty=True,
    )

    assert geojson == {"type": "FeatureCollection", "features": []}
    validate_submission_geojson(output_path)


def test_raster_to_geojson_allows_empty_after_area_filter(tmp_path: Path) -> None:
    raster_path = tmp_path / "prediction.tif"
    output_path = tmp_path / "submission" / "prediction.geojson"

    data = np.zeros((2, 2), dtype=np.uint8)
    data[0, 0] = 1
    _write_binary_raster(raster_path, data)

    geojson = raster_to_geojson(
        raster_path=raster_path,
        output_path=output_path,
        min_area_ha=100.0,
        allow_empty=True,
    )

    assert geojson == {"type": "FeatureCollection", "features": []}
    validate_submission_geojson(output_path)
