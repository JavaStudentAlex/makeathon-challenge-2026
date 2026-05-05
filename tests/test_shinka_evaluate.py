from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping


def _load_evaluate_module():
    module_path = Path(__file__).parents[1] / "shinka" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("local_shinka_evaluate", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shinka_evaluate = _load_evaluate_module()


def _gdf(
    rows: list[tuple[object, dict]],
    *,
    crs: str = shinka_evaluate.DEFAULT_AREA_CRS,
) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [properties for _, properties in rows],
        geometry=[geometry for geometry, _ in rows],
        crs=crs,
    )


def _feature_collection(geometry, properties: dict) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": properties,
            }
        ],
    }


def _write_label_raster(path: Path, data: np.ndarray, dtype: str = "uint16") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=dtype,
        crs="EPSG:4326",
        transform=from_origin(0.0, 0.001, 0.001, 0.001),
        nodata=0,
    ) as dst:
        dst.write(data.astype(dtype), 1)


def _write_glads2_validation_truth(
    validation_data_dir: Path,
    *,
    tile_id: str = "tile_a",
    time_step: int = 2204,
) -> None:
    label_dir = validation_data_dir / "labels" / "glads2"
    alert = np.ones((1, 1), dtype=np.uint8) * 4
    year = 2000 + time_step // 100
    month = time_step % 100
    offset = (date(year, month, 1) - date(2019, 1, 1)).days
    alert_date = np.ones((1, 1), dtype=np.uint16) * offset
    _write_label_raster(label_dir / f"glads2_{tile_id}_alert.tif", alert, dtype="uint8")
    _write_label_raster(
        label_dir / f"glads2_{tile_id}_alertDate.tif",
        alert_date,
        dtype="uint16",
    )


def test_load_validation_ground_truth_decodes_supported_label_sources(
    tmp_path: Path,
) -> None:
    validation_data_dir = tmp_path / "validation"

    _write_label_raster(
        validation_data_dir / "labels" / "radd" / "radd_tile_r_labels.tif",
        np.asarray([[30055]], dtype=np.uint16),
        dtype="uint16",
    )
    _write_label_raster(
        validation_data_dir / "labels" / "gladl" / "gladl_tile_l_alert22.tif",
        np.asarray([[3]], dtype=np.uint8),
        dtype="uint8",
    )
    _write_label_raster(
        validation_data_dir / "labels" / "gladl" / "gladl_tile_l_alertDate22.tif",
        np.asarray([[91]], dtype=np.uint16),
        dtype="uint16",
    )
    _write_glads2_validation_truth(validation_data_dir, tile_id="tile_s")

    ground_truth = shinka_evaluate.load_validation_ground_truth(validation_data_dir)

    assert set(ground_truth["label_source"]) == {"radd", "gladl", "glads2"}
    assert set(ground_truth["tile_id"]) == {"tile_r", "tile_l", "tile_s"}
    assert set(ground_truth["time_step"]) == {1502, 2204}
    assert set(ground_truth["year"]) == {2015, 2022}
    assert not ground_truth.empty
    assert ground_truth.crs == "EPSG:4326"


def test_calculate_scoring_metrics_scores_perfect_spatial_and_year_match() -> None:
    geometry = box(0, 0, 10, 10)
    predictions = _gdf([(geometry, {"time_step": 2204})])
    ground_truth = _gdf([(geometry, {"time_step": 2206})])

    metrics = shinka_evaluate.calculate_scoring_metrics(predictions, ground_truth)

    assert metrics["combined_score"] == pytest.approx(1.0)
    assert metrics["union_iou"] == pytest.approx(1.0)
    assert metrics["polygon_recall"] == pytest.approx(1.0)
    assert metrics["polygon_level_fpr"] == pytest.approx(0.0)
    assert metrics["year_accuracy"] == pytest.approx(1.0)


def test_calculate_scoring_metrics_penalizes_partial_overlap_and_extra_area() -> None:
    predictions = _gdf([(box(5, 0, 15, 10), {"time_step": 2204})])
    ground_truth = _gdf([(box(0, 0, 10, 10), {"year": 2022})])

    metrics = shinka_evaluate.calculate_scoring_metrics(predictions, ground_truth)

    assert metrics["combined_score"] == pytest.approx(0.4)
    assert metrics["union_iou"] == pytest.approx(50.0 / 150.0)
    assert metrics["polygon_recall"] == pytest.approx(0.5)
    assert metrics["polygon_level_fpr"] == pytest.approx(0.5)
    assert metrics["year_accuracy"] == pytest.approx(50.0 / 150.0)


def test_calculate_scoring_metrics_penalizes_wrong_years_by_area() -> None:
    geometry = box(0, 0, 10, 10)
    predictions = _gdf([(geometry, {"time_step": 2304})])
    ground_truth = _gdf([(geometry, {"time_step": 2204})])

    metrics = shinka_evaluate.calculate_scoring_metrics(predictions, ground_truth)

    assert metrics["combined_score"] == pytest.approx(0.8)
    assert metrics["union_iou"] == pytest.approx(1.0)
    assert metrics["polygon_recall"] == pytest.approx(1.0)
    assert metrics["polygon_level_fpr"] == pytest.approx(0.0)
    assert metrics["year_accuracy"] == pytest.approx(0.0)


def test_main_scores_against_validation_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_data_dir = tmp_path / "validation"
    validation_data_dir.mkdir()
    _write_glads2_validation_truth(validation_data_dir)
    monkeypatch.setattr(shinka_evaluate, "VALIDATION_DATA_DIR", validation_data_dir)

    geometry = box(0, 0, 0.001, 0.001)
    prediction_path = tmp_path / "prediction.geojson"
    results_dir = tmp_path / "results"
    prediction_path.write_text(
        json.dumps(_feature_collection(geometry, {"time_step": 2204})),
        encoding="utf-8",
    )

    exit_code = shinka_evaluate.main(
        [
            "--prediction_path",
            str(prediction_path),
            "--results_dir",
            str(results_dir),
        ]
    )

    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    correct = json.loads((results_dir / "correct.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert correct == {"correct": True, "error": None}
    assert metrics["combined_score"] == pytest.approx(1.0)
    assert metrics["union_iou"] == pytest.approx(1.0)
    assert metrics["year_accuracy"] == pytest.approx(1.0)


def test_main_marks_candidate_timeout_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(shinka_evaluate.signal, "SIGALRM"):
        pytest.skip("SIGALRM is unavailable on this platform")

    validation_data_dir = tmp_path / "validation"
    validation_data_dir.mkdir()
    monkeypatch.setattr(shinka_evaluate, "VALIDATION_DATA_DIR", validation_data_dir)
    program_path = tmp_path / "slow_program.py"
    results_dir = tmp_path / "results"
    program_path.write_text(
        "\n".join(
            [
                "import time",
                "",
                "def run_experiment(validation_data_dir):",
                "    time.sleep(2)",
                "    return {'type': 'FeatureCollection', 'features': []}",
            ]
        ),
        encoding="utf-8",
    )
    exit_code = shinka_evaluate.main(
        [
            "--program_path",
            str(program_path),
            "--results_dir",
            str(results_dir),
            "--run_timeout_seconds",
            "1",
        ]
    )

    correct = json.loads((results_dir / "correct.json").read_text(encoding="utf-8"))

    assert exit_code == 1
    assert correct["correct"] is False
    assert "candidate execution exceeded 1 seconds" in correct["error"]


def test_calculate_scoring_metrics_penalizes_missing_prediction_time() -> None:
    geometry = box(0, 0, 10, 10)
    predictions = _gdf([(geometry, {})])
    ground_truth = _gdf([(geometry, {"time_step": "2204"})])

    metrics = shinka_evaluate.calculate_scoring_metrics(predictions, ground_truth)

    assert metrics["combined_score"] == pytest.approx(0.8)
    assert metrics["union_iou"] == pytest.approx(1.0)
    assert metrics["year_accuracy"] == pytest.approx(0.0)


def test_calculate_scoring_metrics_handles_empty_predictions() -> None:
    predictions = _gdf([])
    ground_truth = _gdf([(box(0, 0, 10, 10), {"year": 2022})])

    metrics = shinka_evaluate.calculate_scoring_metrics(predictions, ground_truth)

    assert metrics["combined_score"] == pytest.approx(0.0)
    assert metrics["union_iou"] == pytest.approx(0.0)
    assert metrics["polygon_recall"] == pytest.approx(0.0)
    assert metrics["polygon_level_fpr"] == pytest.approx(0.0)
    assert metrics["year_accuracy"] == pytest.approx(0.0)
