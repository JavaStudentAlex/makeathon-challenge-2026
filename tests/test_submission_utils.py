from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from submission_utils import raster_to_geojson


def _write_binary_raster(path: Path, data: np.ndarray, pixel_size: float = 100.0) -> None:
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


def test_raster_to_geojson_rejects_empty_binary_raster(tmp_path: Path) -> None:
    raster_path = tmp_path / "empty_prediction.tif"
    _write_binary_raster(raster_path, np.zeros((3, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="No deforestation pixels"):
        raster_to_geojson(raster_path=raster_path)
