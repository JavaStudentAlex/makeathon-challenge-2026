"""Polygon scoring metrics for Shinka evolution runs.

The final score combines Union IoU, Polygon Recall, Polygon Level FPR, and Year
Accuracy. Union IoU remains the largest component because it is the primary
ranking metric. Year accuracy is computed from per-feature ``time_step`` or
``year`` properties when temporal labels are present.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import signal
import threading
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import GeometryCollection
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union
from shapely.validation import make_valid

DEFAULT_INPUT_CRS = "EPSG:4326"
DEFAULT_AREA_CRS = "EPSG:6933"
DEFAULT_RUN_TIMEOUT_SECONDS = 40 * 60


def _find_repo_root() -> Path:
    """Find the repo root even after Shinka copies this file into results/."""

    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file() and (
                candidate / "shinka"
            ).is_dir():
                return candidate
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _find_repo_root()
VALIDATION_DATA_DIR = REPO_ROOT / "data" / "makeathon-challenge" / "validation"
COMBINED_SCORE_WEIGHTS = {
    "union_iou": 0.40,
    "polygon_recall": 0.20,
    "false_positive_control": 0.20,
    "year_accuracy": 0.20,
}

GeoJSONInput = dict[str, Any] | str | Path | gpd.GeoDataFrame


def _bounded_run_timeout_seconds(value: str) -> int:
    parsed = int(value)
    if parsed <= 0 or parsed > DEFAULT_RUN_TIMEOUT_SECONDS:
        raise argparse.ArgumentTypeError(
            f"must be between 1 and {DEFAULT_RUN_TIMEOUT_SECONDS}"
        )
    return parsed


def calculate_scoring_metrics(
    predictions: GeoJSONInput,
    ground_truth: GeoJSONInput,
    *,
    area_crs: str = DEFAULT_AREA_CRS,
    default_input_crs: str = DEFAULT_INPUT_CRS,
) -> dict[str, float]:
    """Calculate challenge-style polygon metrics.

    Args:
        predictions: Predicted polygon features as a GeoDataFrame, GeoJSON dict,
            or path readable by GeoPandas.
        ground_truth: Scored ground-truth polygon features as a GeoDataFrame,
            GeoJSON dict, or path readable by GeoPandas.
        area_crs: CRS used for all area calculations. Defaults to EPSG:6933,
            a global equal-area CRS suitable for ratio metrics.
        default_input_crs: CRS assigned to inputs that do not carry CRS metadata.

    Returns:
        A metrics dictionary containing ``combined_score`` and the four public
        scoring metrics. ``combined_score`` is the weighted final score Shinka
        uses for candidate selection.
    """

    pred_gdf = _load_geodataframe(predictions, default_input_crs=default_input_crs)
    truth_gdf = _load_geodataframe(ground_truth, default_input_crs=default_input_crs)
    pred_gdf = _prepare_area_gdf(pred_gdf, area_crs=area_crs)
    truth_gdf = _prepare_area_gdf(truth_gdf, area_crs=area_crs)

    pred_union = _geometry_union(pred_gdf)
    truth_union = _geometry_union(truth_gdf)

    predicted_area = float(pred_union.area)
    ground_truth_area = float(truth_union.area)
    overlap_area = float(pred_union.intersection(truth_union).area)
    spatial_union_area = float(pred_union.union(truth_union).area)
    false_positive_area = max(predicted_area - overlap_area, 0.0)

    union_iou = _safe_divide(overlap_area, spatial_union_area)
    polygon_recall = _safe_divide(overlap_area, ground_truth_area)
    polygon_level_fpr = (
        float(false_positive_area / predicted_area) if predicted_area > 0.0 else 0.0
    )
    year_accuracy = _calculate_year_accuracy(pred_gdf, truth_gdf)
    combined_score = _calculate_combined_score(
        union_iou=union_iou,
        polygon_recall=polygon_recall,
        polygon_level_fpr=polygon_level_fpr,
        year_accuracy=year_accuracy,
        predicted_area=predicted_area,
        ground_truth_area=ground_truth_area,
    )

    return {
        "combined_score": combined_score,
        "union_iou": union_iou,
        "polygon_recall": polygon_recall,
        "polygon_level_fpr": polygon_level_fpr,
        "year_accuracy": year_accuracy,
        "predicted_area": predicted_area,
        "ground_truth_area": ground_truth_area,
        "overlap_area": overlap_area,
        "spatial_union_area": spatial_union_area,
        "false_positive_area": false_positive_area,
    }


def _calculate_combined_score(
    *,
    union_iou: float,
    polygon_recall: float,
    polygon_level_fpr: float,
    year_accuracy: float,
    predicted_area: float,
    ground_truth_area: float,
) -> float:
    recall_score = (
        polygon_recall if ground_truth_area > 0.0 else float(predicted_area == 0.0)
    )
    false_positive_control = (
        1.0 - polygon_level_fpr
        if predicted_area > 0.0
        else float(ground_truth_area == 0.0)
    )
    combined_score = (
        COMBINED_SCORE_WEIGHTS["union_iou"] * union_iou
        + COMBINED_SCORE_WEIGHTS["polygon_recall"] * recall_score
        + COMBINED_SCORE_WEIGHTS["false_positive_control"] * false_positive_control
        + COMBINED_SCORE_WEIGHTS["year_accuracy"] * year_accuracy
    )
    return float(np.clip(combined_score, 0.0, 1.0))


def score_geojson(
    prediction_geojson: GeoJSONInput,
    ground_truth_geojson: GeoJSONInput,
    *,
    area_crs: str = DEFAULT_AREA_CRS,
) -> dict[str, float]:
    """Alias for callers that want an explicit GeoJSON scoring function."""

    return calculate_scoring_metrics(
        prediction_geojson,
        ground_truth_geojson,
        area_crs=area_crs,
    )


def load_validation_ground_truth(
    validation_data_dir: str | Path = VALIDATION_DATA_DIR,
) -> gpd.GeoDataFrame:
    """Build scored validation polygons from labels in the validation split.

    The validation split is a symlink view over held-out training tiles. These
    label rasters are treated as validation truth with explicit raw decoding:
    RADD nonzero alerts, GLAD-L nonzero yearly alerts, and GLAD-S2 alerts with
    confidence >= 2.
    """

    validation_data_dir = Path(validation_data_dir)
    if not validation_data_dir.is_dir():
        raise FileNotFoundError(
            f"Validation data directory not found: {validation_data_dir}"
        )

    label_root = validation_data_dir / "labels"
    frames = [
        *_radd_ground_truth(label_root / "radd"),
        *_gladl_ground_truth(label_root / "gladl"),
        *_glads2_ground_truth(label_root / "glads2"),
    ]
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return _empty_geodataframe()

    return gpd.GeoDataFrame(
        pd.concat(non_empty, ignore_index=True),
        geometry="geometry",
        crs=DEFAULT_INPUT_CRS,
    )


def _radd_ground_truth(label_dir: Path) -> list[gpd.GeoDataFrame]:
    frames: list[gpd.GeoDataFrame] = []
    for path in sorted(label_dir.glob("radd_*_labels.tif")):
        tile_id = path.name.removeprefix("radd_").removesuffix("_labels.tif")
        with rasterio.open(path) as src:
            raw = src.read(1)
            positive = raw > 0
            time_steps, years = _radd_time_arrays(raw)
            frames.append(
                _polygonize_temporal_mask(
                    positive,
                    time_steps,
                    years,
                    transform=src.transform,
                    crs=src.crs,
                    tile_id=tile_id,
                    label_source="radd",
                )
            )
    return frames


def _gladl_ground_truth(label_dir: Path) -> list[gpd.GeoDataFrame]:
    frames: list[gpd.GeoDataFrame] = []
    for alert_path in sorted(label_dir.glob("gladl_*_alert[0-9][0-9].tif")):
        stem = alert_path.name.removeprefix("gladl_").removesuffix(".tif")
        tile_id, yy_text = stem.rsplit("_alert", maxsplit=1)
        date_path = label_dir / f"gladl_{tile_id}_alertDate{yy_text}.tif"
        if not date_path.exists():
            raise FileNotFoundError(f"GLAD-L date raster not found: {date_path}")

        year = 2000 + int(yy_text)
        with (
            rasterio.open(alert_path) as alert_src,
            rasterio.open(date_path) as date_src,
        ):
            alert = alert_src.read(1)
            alert_date = date_src.read(1)
            positive = alert > 0
            time_steps, years = _day_of_year_time_arrays(alert_date, year)
            frames.append(
                _polygonize_temporal_mask(
                    positive,
                    time_steps,
                    years,
                    transform=alert_src.transform,
                    crs=alert_src.crs,
                    tile_id=tile_id,
                    label_source="gladl",
                )
            )
    return frames


def _glads2_ground_truth(label_dir: Path) -> list[gpd.GeoDataFrame]:
    frames: list[gpd.GeoDataFrame] = []
    for alert_path in sorted(label_dir.glob("glads2_*_alert.tif")):
        tile_id = alert_path.name.removeprefix("glads2_").removesuffix("_alert.tif")
        date_path = label_dir / f"glads2_{tile_id}_alertDate.tif"
        if not date_path.exists():
            raise FileNotFoundError(f"GLAD-S2 date raster not found: {date_path}")

        with (
            rasterio.open(alert_path) as alert_src,
            rasterio.open(date_path) as date_src,
        ):
            alert = alert_src.read(1)
            alert_date = date_src.read(1)
            positive = alert >= 2
            time_steps, years = _day_offset_time_arrays(alert_date, date(2019, 1, 1))
            frames.append(
                _polygonize_temporal_mask(
                    positive,
                    time_steps,
                    years,
                    transform=alert_src.transform,
                    crs=alert_src.crs,
                    tile_id=tile_id,
                    label_source="glads2",
                )
            )
    return frames


def _polygonize_temporal_mask(
    positive: np.ndarray,
    time_steps: np.ndarray,
    years: np.ndarray,
    *,
    transform: Any,
    crs: Any,
    tile_id: str,
    label_source: str,
) -> gpd.GeoDataFrame:
    if not np.any(positive):
        return _empty_geodataframe(crs=crs)

    geometries = []
    properties = []
    values = np.where(positive, time_steps, 0).astype(np.int32)
    for geometry, value in shapes(values, mask=positive, transform=transform):
        time_step = int(value) if int(value) > 0 else None
        year_value = _year_from_time_step(time_step)
        if year_value is None:
            row, col = _first_matching_pixel(values, positive, int(value))
            year_value = int(years[row, col]) if years[row, col] > 0 else None

        geometries.append(shapely_shape(geometry))
        properties.append(
            {
                "tile_id": tile_id,
                "label_source": label_source,
                "time_step": time_step,
                "year": year_value,
            }
        )

    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs=crs)
    if gdf.crs is None:
        gdf = gdf.set_crs(DEFAULT_INPUT_CRS)
    return gdf.to_crs(DEFAULT_INPUT_CRS)


def _first_matching_pixel(
    values: np.ndarray,
    positive: np.ndarray,
    value: int,
) -> tuple[int, int]:
    if value > 0:
        locations = np.argwhere(positive & (values == value))
    else:
        locations = np.argwhere(positive)
    if locations.size == 0:
        return 0, 0
    row, col = locations[0]
    return int(row), int(col)


def _radd_time_arrays(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time_steps = np.zeros(raw.shape, dtype=np.int32)
    years = np.zeros(raw.shape, dtype=np.int32)
    for raw_value in np.unique(raw[raw > 0]):
        days_since = int(raw_value) % 10000
        observed = date(2014, 12, 31) + timedelta(days=days_since)
        time_step = (observed.year % 100) * 100 + observed.month
        mask = raw == raw_value
        time_steps[mask] = time_step
        years[mask] = observed.year
    return time_steps, years


def _day_of_year_time_arrays(
    day_of_year: np.ndarray,
    year: int,
) -> tuple[np.ndarray, np.ndarray]:
    time_steps = np.zeros(day_of_year.shape, dtype=np.int32)
    years = np.zeros(day_of_year.shape, dtype=np.int32)
    for raw_day in np.unique(day_of_year[day_of_year > 0]):
        observed = date(year, 1, 1) + timedelta(days=int(raw_day) - 1)
        time_step = (observed.year % 100) * 100 + observed.month
        mask = day_of_year == raw_day
        time_steps[mask] = time_step
        years[mask] = observed.year
    return time_steps, years


def _day_offset_time_arrays(
    day_offset: np.ndarray,
    origin: date,
) -> tuple[np.ndarray, np.ndarray]:
    time_steps = np.zeros(day_offset.shape, dtype=np.int32)
    years = np.zeros(day_offset.shape, dtype=np.int32)
    for raw_offset in np.unique(day_offset[day_offset > 0]):
        observed = origin + timedelta(days=int(raw_offset))
        time_step = (observed.year % 100) * 100 + observed.month
        mask = day_offset == raw_offset
        time_steps[mask] = time_step
        years[mask] = observed.year
    return time_steps, years


def _empty_geodataframe(crs: Any = DEFAULT_INPUT_CRS) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)


def _load_geodataframe(
    value: GeoJSONInput,
    *,
    default_input_crs: str,
) -> gpd.GeoDataFrame:
    if isinstance(value, gpd.GeoDataFrame):
        gdf = value.copy()
    elif isinstance(value, (str, Path)):
        gdf = gpd.read_file(value)
    elif isinstance(value, dict):
        if value.get("type") != "FeatureCollection":
            raise ValueError("GeoJSON input must be a FeatureCollection")
        features = value.get("features")
        if not isinstance(features, list):
            raise ValueError("FeatureCollection must contain a features list")
        if features:
            gdf = gpd.GeoDataFrame.from_features(features)
        else:
            gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry")
    else:
        raise TypeError(
            "Expected a GeoDataFrame, GeoJSON dict, or path-like input; "
            f"got {type(value).__name__}"
        )

    if "geometry" not in gdf:
        raise ValueError("Input does not contain a geometry column")
    if gdf.crs is None:
        gdf = gdf.set_crs(default_input_crs)
    return gdf


def _prepare_area_gdf(gdf: gpd.GeoDataFrame, *, area_crs: str) -> gpd.GeoDataFrame:
    prepared = gdf.copy()
    prepared = prepared[prepared.geometry.notna()].copy()
    if prepared.empty:
        return gpd.GeoDataFrame(prepared, geometry="geometry", crs=area_crs)

    prepared = prepared.to_crs(area_crs)
    prepared["geometry"] = prepared.geometry.map(_valid_geometry)
    prepared = prepared[prepared.geometry.notna()].copy()
    prepared = prepared[~prepared.geometry.is_empty].copy()
    prepared = prepared[prepared.geometry.area > 0].copy()
    return prepared


def _valid_geometry(geometry: Any) -> Any:
    if geometry is None or geometry.is_empty:
        return geometry
    if geometry.is_valid:
        return geometry
    return make_valid(geometry)


def _geometry_union(gdf: gpd.GeoDataFrame) -> Any:
    if gdf.empty:
        return GeometryCollection()
    return unary_union(list(gdf.geometry))


def _calculate_year_accuracy(
    pred_gdf: gpd.GeoDataFrame,
    truth_gdf: gpd.GeoDataFrame,
) -> float:
    temporal_truth = _filter_temporal_features(truth_gdf)
    pred_union = _geometry_union(pred_gdf)
    temporal_truth_union = _geometry_union(temporal_truth)
    denominator = float(pred_union.union(temporal_truth_union).area)
    if denominator == 0.0:
        return 1.0

    correct_intersections = []
    pred_by_year = _unions_by_year(pred_gdf)
    truth_by_year = _unions_by_year(temporal_truth)
    for year, pred_geometry in pred_by_year.items():
        truth_geometry = truth_by_year.get(year)
        if truth_geometry is None:
            continue
        intersection = pred_geometry.intersection(truth_geometry)
        if not intersection.is_empty:
            correct_intersections.append(intersection)

    correctly_dated_area = (
        float(unary_union(correct_intersections).area) if correct_intersections else 0.0
    )
    return _safe_divide(correctly_dated_area, denominator)


def _filter_temporal_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    mask = gdf.apply(lambda row: _feature_year(row) is not None, axis=1)
    return gdf[mask].copy()


def _unions_by_year(gdf: gpd.GeoDataFrame) -> dict[int, Any]:
    by_year: dict[int, list[Any]] = {}
    for _, row in gdf.iterrows():
        year = _feature_year(row)
        if year is None:
            continue
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        by_year.setdefault(year, []).append(geometry)
    return {year: unary_union(geometries) for year, geometries in by_year.items()}


def _feature_year(row: Any) -> int | None:
    if "year" in row and _coerce_year(row["year"]) is not None:
        return _coerce_year(row["year"])
    if "time_step" in row:
        return _year_from_time_step(row["time_step"])
    return None


def _coerce_year(value: Any) -> int | None:
    integer = _coerce_int(value)
    if integer is None:
        return None
    if 0 <= integer <= 99:
        return 2000 + integer
    if 1900 <= integer <= 2100:
        return integer
    return None


def _year_from_time_step(value: Any) -> int | None:
    integer = _coerce_int(value)
    if integer is None:
        return None
    if 0 <= integer <= 99:
        return 2000 + integer
    if 190001 <= integer <= 210012:
        year = integer // 100
        month = integer % 100
        return year if 1 <= month <= 12 else None
    if 100 <= integer <= 9999:
        yy = integer // 100
        month = integer % 100
        if 0 <= yy <= 99 and 1 <= month <= 12:
            return 2000 + yy
    if 1900 <= integer <= 2100:
        return integer
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or not float(value).is_integer():
            return None
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else 0.0
    return float(numerator / denominator)


@contextmanager
def _wall_time_limit(seconds: int):
    """Raise TimeoutError if candidate import or execution exceeds ``seconds``."""

    if (
        seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_remaining = signal.alarm(0)

    def _raise_timeout(_signum, _frame) -> None:
        raise TimeoutError(
            f"candidate execution exceeded {seconds} seconds; keep evaluation bounded"
        )

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_remaining > 0:
            signal.alarm(previous_remaining)


def _load_program(program_path: str | Path) -> Any:
    spec = importlib.util.spec_from_file_location("candidate_program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load candidate program: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prediction_from_program(
    program_path: str | Path,
    *,
    timeout_seconds: int = DEFAULT_RUN_TIMEOUT_SECONDS,
) -> GeoJSONInput:
    with _wall_time_limit(timeout_seconds):
        module = _load_program(program_path)
        run_experiment = getattr(module, "run_experiment", None)
        if run_experiment is None or not callable(run_experiment):
            raise AttributeError("Candidate program must define run_experiment()")
        run_inference = getattr(module, "run_inference", None)
        if run_inference is None or not callable(run_inference):
            raise AttributeError("Candidate program must define run_inference()")

        model = run_experiment()
        with _feature_only_data_dir(_validation_data_dir()) as prediction_data_dir:
            result = run_inference(model, prediction_data_dir=prediction_data_dir)
            return _coerce_prediction_result(result)


def _coerce_prediction_result(result: Any) -> GeoJSONInput:
    if isinstance(result, dict) and result.get("type") == "FeatureCollection":
        return result
    if isinstance(result, dict):
        for key in ("geojson", "prediction_geojson", "prediction_path", "path"):
            if key in result:
                return _coerce_prediction_result(result[key])
    if isinstance(result, gpd.GeoDataFrame):
        return result
    if isinstance(result, (str, Path)):
        return _load_geodataframe(result, default_input_crs=DEFAULT_INPUT_CRS)
    raise ValueError(
        "run_inference() must return a FeatureCollection, GeoDataFrame, path, "
        "or a dict containing geojson/prediction_geojson/prediction_path"
    )


@contextmanager
def _feature_only_data_dir(data_dir: Path):
    with TemporaryDirectory(prefix="shinka-inference-") as temp_name:
        feature_dir = Path(temp_name)
        for child in data_dir.iterdir():
            if child.name == "labels":
                continue
            (feature_dir / child.name).symlink_to(
                child,
                target_is_directory=child.is_dir(),
            )
        yield feature_dir


def _validation_data_dir() -> Path:
    if not VALIDATION_DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Validation data directory not found: {VALIDATION_DATA_DIR}"
        )
    return VALIDATION_DATA_DIR


def _write_shinka_results(
    results_dir: Path,
    metrics: dict[str, float],
    *,
    correct: bool,
    error: str | None = None,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )
    (results_dir / "correct.json").write_text(
        json.dumps({"correct": correct, "error": error}, indent=4),
        encoding="utf-8",
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score deforestation polygons")
    parser.add_argument("--program_path", help="Candidate program path")
    parser.add_argument("--results_dir", required=True, help="Shinka results directory")
    parser.add_argument(
        "--prediction_path",
        help="Optional existing prediction GeoJSON path; skips program execution",
    )
    parser.add_argument(
        "--area_crs",
        default=DEFAULT_AREA_CRS,
        help=f"CRS used for area calculations; default {DEFAULT_AREA_CRS}",
    )
    parser.add_argument(
        "--run_timeout_seconds",
        type=_bounded_run_timeout_seconds,
        default=DEFAULT_RUN_TIMEOUT_SECONDS,
        help=(
            "Wall-time limit for candidate import, training, and inference; "
            f"default {DEFAULT_RUN_TIMEOUT_SECONDS} seconds"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    try:
        predictions: GeoJSONInput
        if args.prediction_path:
            predictions = Path(args.prediction_path)
        elif args.program_path:
            predictions = _prediction_from_program(
                args.program_path,
                timeout_seconds=args.run_timeout_seconds,
            )
        else:
            raise ValueError("--program_path or --prediction_path is required")

        ground_truth = load_validation_ground_truth(_validation_data_dir())
        metrics = calculate_scoring_metrics(
            predictions,
            ground_truth,
            area_crs=args.area_crs,
        )
        _write_shinka_results(Path(args.results_dir), metrics, correct=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - Shinka expects result files on failure.
        metrics = {
            "combined_score": 0.0,
            "union_iou": 0.0,
            "polygon_recall": 0.0,
            "polygon_level_fpr": 1.0,
            "year_accuracy": 0.0,
        }
        _write_shinka_results(
            Path(args.results_dir),
            metrics,
            correct=False,
            error=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
