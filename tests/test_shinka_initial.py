from __future__ import annotations

import inspect
import importlib.util
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_aef(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 0.004, 0.001, 0.001),
        nodata=np.nan,
    ) as dst:
        dst.write(data.astype("float32"))


def _write_radd_label(path: Path, positive_year: int = 2021) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    days_since = (date(positive_year, 7, 1) - date(2014, 12, 31)).days
    label = np.zeros((4, 4), dtype=np.uint16)
    label[0, 0] = 30000 + days_since
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=from_origin(0.0, 0.004, 0.001, 0.001),
        nodata=0,
    ) as dst:
        dst.write(label, 1)


def _write_synthetic_training_tree(root: Path, tile_id: str = "tile_train") -> None:
    baseline = np.zeros((16, 4, 4), dtype=np.float32)
    changed = baseline.copy()
    changed[:, 0, 0] = 1.0
    _write_aef(root / "aef-embeddings" / f"{tile_id}_2020.tiff", baseline)
    _write_aef(root / "aef-embeddings" / f"{tile_id}_2021.tiff", changed)
    _write_radd_label(root / "labels" / "radd" / f"radd_{tile_id}_labels.tif")


def _write_synthetic_prediction_tree(root: Path, tile_id: str = "tile_pred") -> None:
    baseline = np.zeros((16, 4, 4), dtype=np.float32)
    changed = baseline.copy()
    changed[:, 0, 0] = 1.0
    _write_aef(root / "aef-embeddings" / f"{tile_id}_2020.tiff", baseline)
    _write_aef(root / "aef-embeddings" / f"{tile_id}_2021.tiff", changed)


def test_initial_seed_trains_from_training_labels_and_predicts_unlabeled_input(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).parents[1]
    initial = _load_module("local_shinka_initial", repo_root / "shinka" / "initial.py")
    initial.PREDICT_DOWNSAMPLE_FACTOR = 1
    training_data_dir = tmp_path / "training"
    prediction_data_dir = tmp_path / "validation"
    _write_synthetic_training_tree(training_data_dir)
    _write_synthetic_prediction_tree(prediction_data_dir)
    initial.TRAINING_DATA_DIR = training_data_dir

    model = initial.run_experiment()
    prediction = initial.run_inference(model, prediction_data_dir, threshold=0.05)

    assert prediction["type"] == "FeatureCollection"
    assert prediction["features"]
    assert not (prediction_data_dir / "labels").exists()
    assert prediction["features"][0]["properties"]["time_step"] == 2107
    assert inspect.signature(initial.run_experiment).parameters == {}


def test_initial_seed_program_has_30_minute_training_timeout() -> None:
    repo_root = Path(__file__).parents[1]
    initial = _load_module("local_shinka_initial", repo_root / "shinka" / "initial.py")

    assert initial.TRAINING_TIMEOUT_SECONDS == 30 * 60
