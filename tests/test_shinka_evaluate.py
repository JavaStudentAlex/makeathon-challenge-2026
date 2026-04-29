from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import pytest
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

    assert metrics["union_iou"] == pytest.approx(50.0 / 150.0)
    assert metrics["polygon_recall"] == pytest.approx(0.5)
    assert metrics["polygon_level_fpr"] == pytest.approx(0.5)
    assert metrics["year_accuracy"] == pytest.approx(50.0 / 150.0)


def test_calculate_scoring_metrics_penalizes_wrong_years_by_area() -> None:
    geometry = box(0, 0, 10, 10)
    predictions = _gdf([(geometry, {"time_step": 2304})])
    ground_truth = _gdf([(geometry, {"time_step": 2204})])

    metrics = shinka_evaluate.calculate_scoring_metrics(predictions, ground_truth)

    assert metrics["union_iou"] == pytest.approx(1.0)
    assert metrics["polygon_recall"] == pytest.approx(1.0)
    assert metrics["polygon_level_fpr"] == pytest.approx(0.0)
    assert metrics["year_accuracy"] == pytest.approx(0.0)


def test_main_writes_shinka_metrics_files(tmp_path: Path) -> None:
    geometry = box(0, 0, 0.001, 0.001)
    prediction_path = tmp_path / "prediction.geojson"
    ground_truth_path = tmp_path / "ground_truth.geojson"
    results_dir = tmp_path / "results"
    prediction_path.write_text(
        json.dumps(_feature_collection(geometry, {"time_step": 2204})),
        encoding="utf-8",
    )
    ground_truth_path.write_text(
        json.dumps(_feature_collection(geometry, {"year": 2022})),
        encoding="utf-8",
    )

    exit_code = shinka_evaluate.main(
        [
            "--prediction_path",
            str(prediction_path),
            "--ground_truth_path",
            str(ground_truth_path),
            "--results_dir",
            str(results_dir),
        ]
    )

    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    correct = json.loads((results_dir / "correct.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert correct == {"correct": True, "error": None}
    assert metrics["combined_score"] == pytest.approx(1.0)
    assert metrics["year_accuracy"] == pytest.approx(1.0)


def test_main_marks_candidate_timeout_as_failure(tmp_path: Path) -> None:
    if not hasattr(shinka_evaluate.signal, "SIGALRM"):
        pytest.skip("SIGALRM is unavailable on this platform")

    program_path = tmp_path / "slow_program.py"
    ground_truth_path = tmp_path / "ground_truth.geojson"
    results_dir = tmp_path / "results"
    program_path.write_text(
        "\n".join(
            [
                "import time",
                "",
                "def run_experiment():",
                "    time.sleep(2)",
                "    return {'type': 'FeatureCollection', 'features': []}",
            ]
        ),
        encoding="utf-8",
    )
    ground_truth_path.write_text(
        json.dumps(_feature_collection(box(0, 0, 0.001, 0.001), {"year": 2022})),
        encoding="utf-8",
    )

    exit_code = shinka_evaluate.main(
        [
            "--program_path",
            str(program_path),
            "--ground_truth_path",
            str(ground_truth_path),
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

    assert metrics["union_iou"] == pytest.approx(1.0)
    assert metrics["year_accuracy"] == pytest.approx(0.0)


def test_calculate_scoring_metrics_handles_empty_predictions() -> None:
    predictions = _gdf([])
    ground_truth = _gdf([(box(0, 0, 10, 10), {"year": 2022})])

    metrics = shinka_evaluate.calculate_scoring_metrics(predictions, ground_truth)

    assert metrics["union_iou"] == pytest.approx(0.0)
    assert metrics["polygon_recall"] == pytest.approx(0.0)
    assert metrics["polygon_level_fpr"] == pytest.approx(0.0)
    assert metrics["year_accuracy"] == pytest.approx(0.0)
