import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from shinka.features import ReferenceGrid
from simple_ensembles import balanced_fusion, high_recall_fusion, top_ranked_fusion
from stats_models.runner import generate_submission
from submission_utils import validate_submission_geojson

RASTER_SHAPE = (8, 8)
REFERENCE_CRS = "EPSG:32618"
POSITIVE_TIME_STEP = 2308
TILE_IDS = ("tile_a", "tile_b")
MODULES = (
    pytest.param(top_ranked_fusion, id="top_ranked_fusion"),
    pytest.param(balanced_fusion, id="balanced_fusion"),
    pytest.param(high_recall_fusion, id="high_recall_fusion"),
)


def _synthetic_reference(tile_id: str) -> ReferenceGrid:
    tile_index = TILE_IDS.index(tile_id)
    return ReferenceGrid(
        shape=RASTER_SHAPE,
        transform=from_origin(500000 + (tile_index * 1_000), 1_000, 100, 100),
        crs=REFERENCE_CRS,
    )


def _strong_positive_features() -> dict[str, np.ndarray]:
    return {
        "forest_2020": np.ones(RASTER_SHAPE, dtype=np.float32),
        "ndvi_delta": np.full(RASTER_SHAPE, -0.8, dtype=np.float32),
        "nbr_delta": np.full(RASTER_SHAPE, -0.8, dtype=np.float32),
        "ndmi_delta": np.full(RASTER_SHAPE, -0.8, dtype=np.float32),
        "evi_delta": np.full(RASTER_SHAPE, -0.8, dtype=np.float32),
        "ndre_delta": np.full(RASTER_SHAPE, -0.8, dtype=np.float32),
        "bsi_delta": np.full(RASTER_SHAPE, 0.8, dtype=np.float32),
        "vv_delta": np.full(RASTER_SHAPE, -0.2, dtype=np.float32),
        "vv_cv_delta": np.full(RASTER_SHAPE, 0.3, dtype=np.float32),
        "aef_shift": np.full(RASTER_SHAPE, 0.85, dtype=np.float32),
        "alert_consensus": np.full(RASTER_SHAPE, 0.8, dtype=np.float32),
        "first_alert_time_step": np.full(
            RASTER_SHAPE, POSITIVE_TIME_STEP, dtype=np.float32
        ),
        "strongest_anomaly_time_step": np.full(
            RASTER_SHAPE, POSITIVE_TIME_STEP, dtype=np.float32
        ),
    }


def _synthetic_feature_builder(
    data_root: Path,
    tile_id: str,
    split: str,
) -> tuple[ReferenceGrid, dict[str, np.ndarray]]:
    del data_root, split
    return _synthetic_reference(tile_id), _strong_positive_features()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_prediction_raster(path: Path, reference: ReferenceGrid) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read(1)
        assert data.shape == RASTER_SHAPE
        assert src.crs.to_string() == REFERENCE_CRS
        assert src.transform == reference.transform
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 0

    unique_values = set(np.unique(data).tolist())
    assert unique_values <= {0, 1}
    return data


def _assert_time_step_raster(path: Path, reference: ReferenceGrid) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read(1)
        assert data.shape == RASTER_SHAPE
        assert src.crs.to_string() == REFERENCE_CRS
        assert src.transform == reference.transform
        assert src.dtypes[0] == "uint16"
        assert src.nodata == 0

    return data


@pytest.mark.parametrize("module", MODULES)
def test_generate_submission_builds_valid_bundle_for_promoted_simple_ensembles(
    tmp_path: Path,
    module,
) -> None:
    data_root = tmp_path / "synthetic_data"
    data_root.mkdir()
    output_dir = tmp_path / module.__name__.rsplit(".", maxsplit=1)[-1]

    submission_path, manifest_path = generate_submission(
        module,
        data_root,
        output_dir,
        split="test",
        tiles=TILE_IDS,
        feature_builder=_synthetic_feature_builder,
        min_area_ha=0.0,
    )

    assert submission_path == output_dir / "submission.geojson"
    assert manifest_path == output_dir / "manifest.json"
    assert submission_path.exists()
    assert manifest_path.exists()

    validate_submission_geojson(submission_path)
    combined = _load_json(submission_path)
    manifest = _load_json(manifest_path)

    assert manifest["program_path"] == module.__name__
    assert manifest["split"] == "test"
    assert manifest["threshold"] == pytest.approx(0.52)
    assert manifest["min_area_ha"] == pytest.approx(0.0)
    assert manifest["submission_geojson"] == str(submission_path)
    assert [tile_summary["tile_id"] for tile_summary in manifest["tiles"]] == list(
        TILE_IDS
    )
    assert combined["features"]
    assert manifest["submission_features"] == len(combined["features"])

    total_tile_features = 0
    for tile_id, tile_summary in zip(TILE_IDS, manifest["tiles"], strict=True):
        reference = _synthetic_reference(tile_id)
        prediction_path = output_dir / "rasters" / f"pred_{tile_id}.tif"
        time_step_path = output_dir / "rasters" / f"time_step_{tile_id}.tif"
        geojson_path = output_dir / "geojson" / f"pred_{tile_id}.geojson"

        assert prediction_path.exists()
        assert time_step_path.exists()
        assert geojson_path.exists()

        validate_submission_geojson(geojson_path)
        tile_geojson = _load_json(geojson_path)
        total_tile_features += len(tile_geojson["features"])

        prediction = _assert_prediction_raster(prediction_path, reference)
        time_step = _assert_time_step_raster(time_step_path, reference)

        assert np.any(prediction == 1)
        assert tile_summary["positive_pixels"] == int(np.count_nonzero(prediction))
        assert tile_summary["features"] == len(tile_geojson["features"])
        assert 0.52 < tile_summary["mean_probability"] <= 1.0
        assert (
            tile_summary["mean_probability"]
            <= tile_summary["max_probability"] + 1e-6
        )
        assert 0.0 <= tile_summary["max_probability"] <= 1.0
        assert tile_summary["features"] > 0
        assert np.all(time_step[prediction == 1] == POSITIVE_TIME_STEP)
        assert np.all(time_step[prediction == 0] == 0)

        for feature in tile_geojson["features"]:
            assert feature["properties"].get("time_step") is None

    assert manifest["submission_features"] == total_tile_features
    for feature in combined["features"]:
        assert feature["properties"].get("time_step") is None


@pytest.mark.parametrize("module", MODULES)
def test_generate_submission_writes_empty_valid_bundle_for_strict_threshold(
    tmp_path: Path,
    module,
) -> None:
    data_root = tmp_path / "synthetic_data"
    data_root.mkdir()
    output_dir = tmp_path / f"{module.__name__.rsplit('.', maxsplit=1)[-1]}_strict"

    submission_path, manifest_path = generate_submission(
        module,
        data_root,
        output_dir,
        split="test",
        tiles=TILE_IDS,
        threshold=1.1,
        feature_builder=_synthetic_feature_builder,
        min_area_ha=0.0,
    )

    assert submission_path.exists()
    assert manifest_path.exists()

    validate_submission_geojson(submission_path)
    combined = _load_json(submission_path)
    manifest = _load_json(manifest_path)

    assert combined["features"] == []
    assert manifest["program_path"] == module.__name__
    assert manifest["threshold"] == pytest.approx(1.1)
    assert manifest["min_area_ha"] == pytest.approx(0.0)
    assert manifest["submission_features"] == 0
    assert [tile_summary["tile_id"] for tile_summary in manifest["tiles"]] == list(
        TILE_IDS
    )

    for tile_id, tile_summary in zip(TILE_IDS, manifest["tiles"], strict=True):
        reference = _synthetic_reference(tile_id)
        prediction_path = output_dir / "rasters" / f"pred_{tile_id}.tif"
        time_step_path = output_dir / "rasters" / f"time_step_{tile_id}.tif"
        geojson_path = output_dir / "geojson" / f"pred_{tile_id}.geojson"

        assert prediction_path.exists()
        assert time_step_path.exists()
        assert geojson_path.exists()

        validate_submission_geojson(geojson_path)
        tile_geojson = _load_json(geojson_path)
        prediction = _assert_prediction_raster(prediction_path, reference)
        time_step = _assert_time_step_raster(time_step_path, reference)

        assert tile_geojson["features"] == []
        assert tile_summary["positive_pixels"] == 0
        assert tile_summary["features"] == 0
        assert 0.52 < tile_summary["mean_probability"] <= 1.0
        assert (
            tile_summary["mean_probability"]
            <= tile_summary["max_probability"] + 1e-6
        )
        assert 0.0 <= tile_summary["max_probability"] <= 1.0
        assert np.count_nonzero(prediction) == 0
        assert np.all(prediction == 0)
        assert np.all(time_step == 0)
