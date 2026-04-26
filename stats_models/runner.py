"""Shared submission orchestration for promoted statistical models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import numpy as np
import rasterio

from shinka.features import ReferenceGrid, build_model_features
from shinka.labels import target_from_train_labels
from submission_utils import raster_to_geojson, validate_submission_geojson

DEFAULT_DATA_ROOT = Path("data/makeathon-challenge")
REQUIRED_RESULT_KEYS = ("prediction", "probabilities", "time_step")
ALIGNMENT_THRESHOLDS = tuple(
    float(value)
    for value in np.concatenate(
        [np.arange(0.05, 0.951, 0.01), np.array([0.97, 0.98, 0.99])]
    )
)


FeatureBuilder = Callable[[Path, str, str], tuple[ReferenceGrid, dict[str, np.ndarray]]]


class SupportsRunExperiment(Protocol):
    """Protocol for promoted model modules consumed by the runner."""

    __name__: str

    def run_experiment(
        self,
        features: dict[str, np.ndarray],
        *,
        threshold: float,
    ) -> dict[str, Any]: ...


def _canonical_module_name(model_module: SupportsRunExperiment) -> str:
    """Return a stable module name for manifests and default output paths."""

    spec = getattr(model_module, "__spec__", None)
    spec_name = getattr(spec, "name", None)
    if spec_name:
        return spec_name

    module_name = getattr(model_module, "__name__", None)
    if module_name and module_name != "__main__":
        return module_name

    file_path = getattr(model_module, "__file__", None)
    if file_path:
        return Path(file_path).stem

    return model_module.__class__.__name__


def _tile_ids_from_metadata(data_root: Path, split: str) -> list[str]:
    metadata_path = data_root / "metadata" / f"{split}_tiles.geojson"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Tile metadata not found: {metadata_path}")

    with open(metadata_path, encoding="utf-8") as file:
        geojson = json.load(file)

    return [feature["properties"]["name"] for feature in geojson["features"]]


def _resolve_run_experiment(
    model_module: SupportsRunExperiment,
) -> Callable[..., dict[str, Any]]:
    module_name = getattr(model_module, "__name__", repr(model_module))
    run_experiment = getattr(model_module, "run_experiment", None)
    if run_experiment is None or not callable(run_experiment):
        raise AttributeError(
            f"{module_name} must define a callable run_experiment(features, threshold=...)"
        )
    return run_experiment


def _fit_generic_threshold_alignment(
    model_module: SupportsRunExperiment,
    run_experiment: Callable[..., dict[str, Any]],
    *,
    data_root: Path,
    split: str,
    tiles: list[str],
    initial_threshold: float,
    feature_builder: FeatureBuilder,
) -> dict[str, Any]:
    """Fit a global probability threshold from weak train labels."""

    if split != "train":
        raise ValueError(
            "generic alignment uses train-only labels; split must be train"
        )

    thresholds = np.asarray(ALIGNMENT_THRESHOLDS, dtype=np.float32)
    true_positive_pixels = np.zeros(thresholds.shape, dtype=np.int64)
    predicted_pixels = np.zeros(thresholds.shape, dtype=np.int64)
    target_pixels = 0
    total_pixels = 0
    tile_summaries: list[dict[str, Any]] = []

    for tile_id in tiles:
        reference, features = feature_builder(data_root, tile_id, split)
        result = run_experiment(features, threshold=initial_threshold)
        if not isinstance(result, dict) or "probabilities" not in result:
            module_name = getattr(model_module, "__name__", repr(model_module))
            raise ValueError(
                f"{module_name}.run_experiment(...) must return a dict containing "
                "probabilities for generic train alignment"
            )

        probabilities = np.nan_to_num(
            np.asarray(result["probabilities"], dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        target, _target_time_step = target_from_train_labels(
            data_root, tile_id, reference
        )
        target_bool = target.astype(bool)

        if probabilities.shape != target_bool.shape:
            raise ValueError(
                f"alignment target for {tile_id} has shape {target_bool.shape}, "
                f"expected {probabilities.shape}"
            )

        tile_target_pixels = int(np.count_nonzero(target_bool))
        target_pixels += tile_target_pixels
        total_pixels += int(target_bool.size)

        for index, threshold in enumerate(thresholds):
            prediction = probabilities >= threshold
            predicted_pixels[index] += int(np.count_nonzero(prediction))
            true_positive_pixels[index] += int(
                np.count_nonzero(prediction & target_bool)
            )

        tile_summaries.append(
            {
                "tile_id": tile_id,
                "target_pixels": tile_target_pixels,
                "mean_probability": float(np.mean(probabilities)),
                "max_probability": float(np.max(probabilities)),
            }
        )

    union_pixels = predicted_pixels + target_pixels - true_positive_pixels
    pixel_iou = np.divide(
        true_positive_pixels,
        union_pixels,
        out=np.zeros(thresholds.shape, dtype=np.float64),
        where=union_pixels > 0,
    )

    if target_pixels == 0:
        best_index = int(np.argmin(np.abs(thresholds - initial_threshold)))
        status = "skipped_no_positive_train_labels"
    else:
        best_iou = float(np.max(pixel_iou))
        candidate_indices = np.flatnonzero(np.isclose(pixel_iou, best_iou))
        closest_to_initial = np.argmin(
            np.abs(thresholds[candidate_indices] - initial_threshold)
        )
        best_index = int(candidate_indices[closest_to_initial])
        status = "aligned"

    return {
        "method": "generic_weak_train_label_pixel_iou_threshold_grid",
        "status": status,
        "threshold": float(thresholds[best_index]),
        "metric": "pixel_iou",
        "metric_value": float(pixel_iou[best_index]),
        "tile_count": len(tiles),
        "total_pixels": total_pixels,
        "target_pixels": target_pixels,
        "predicted_pixels": int(predicted_pixels[best_index]),
        "true_positive_pixels": int(true_positive_pixels[best_index]),
        "union_pixels": int(union_pixels[best_index]),
        "threshold_grid": {
            "min": float(np.min(thresholds)),
            "max": float(np.max(thresholds)),
            "count": int(thresholds.size),
        },
        "tile_summaries": tile_summaries,
    }


def _fit_submission_alignment(
    model_module: SupportsRunExperiment,
    run_experiment: Callable[..., dict[str, Any]],
    *,
    data_root: Path,
    split: str,
    tiles: list[str],
    initial_threshold: float,
    feature_builder: FeatureBuilder,
) -> dict[str, Any]:
    module_name = getattr(model_module, "__name__", repr(model_module))
    fit_alignment = getattr(model_module, "fit_submission_alignment", None)
    if fit_alignment is None or not callable(fit_alignment):
        alignment = _fit_generic_threshold_alignment(
            model_module,
            run_experiment,
            data_root=data_root,
            split=split,
            tiles=tiles,
            initial_threshold=initial_threshold,
            feature_builder=feature_builder,
        )
    else:
        alignment = fit_alignment(
            data_root=data_root,
            split=split,
            tiles=tiles,
            initial_threshold=initial_threshold,
            feature_builder=feature_builder,
        )

    if not isinstance(alignment, dict):
        raise ValueError(f"{module_name} alignment must return a dict")
    if "threshold" not in alignment:
        raise ValueError(f"{module_name} alignment must return a threshold")

    threshold = float(alignment["threshold"])
    if not np.isfinite(threshold):
        raise ValueError(f"{module_name} alignment returned a non-finite threshold")

    return {**alignment, "threshold": threshold}


def _write_raster(
    path: Path,
    data: np.ndarray,
    reference: ReferenceGrid,
    dtype: str,
    nodata: int | float,
) -> None:
    array = np.asarray(data)
    if array.shape != reference.shape:
        raise ValueError(
            f"Raster data for {path.name} has shape {array.shape}, expected {reference.shape}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=reference.shape[0],
        width=reference.shape[1],
        count=1,
        dtype=dtype,
        crs=reference.crs,
        transform=reference.transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(array.astype(dtype), 1)


def _write_combined_submission(
    path: Path,
    tile_feature_collections: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for geojson in tile_feature_collections:
        features.extend(geojson["features"])

    submission = {"type": "FeatureCollection", "features": features}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(submission, file)

    validate_submission_geojson(path)
    return submission


def generate_submission(
    model_module: SupportsRunExperiment,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    tiles: Iterable[str] | None = None,
    threshold: float = 0.52,
    min_area_ha: float = 0.5,
    feature_builder: FeatureBuilder = build_model_features,
    align_train: bool = False,
    alignment_split: str = "train",
    alignment_tiles: Iterable[str] | None = None,
) -> tuple[Path, Path]:
    """Run a promoted statistical model across tiles and emit a submission bundle."""

    data_root = Path(data_root)
    output_dir = Path(output_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    tile_ids = (
        _tile_ids_from_metadata(data_root, split) if tiles is None else list(tiles)
    )
    run_experiment = _resolve_run_experiment(model_module)
    output_dir.mkdir(parents=True, exist_ok=True)
    module_name = _canonical_module_name(model_module)

    alignment: dict[str, Any] | None = None
    if align_train:
        resolved_alignment_tiles = (
            _tile_ids_from_metadata(data_root, alignment_split)
            if alignment_tiles is None
            else list(alignment_tiles)
        )
        if not resolved_alignment_tiles:
            raise ValueError("train alignment requires at least one alignment tile")

        initial_threshold = threshold
        alignment = _fit_submission_alignment(
            model_module,
            run_experiment,
            data_root=data_root,
            split=alignment_split,
            tiles=resolved_alignment_tiles,
            initial_threshold=initial_threshold,
            feature_builder=feature_builder,
        )
        threshold = float(alignment["threshold"])
        alignment = {
            "enabled": True,
            "split": alignment_split,
            "tile_ids": resolved_alignment_tiles,
            **alignment,
            "initial_threshold": initial_threshold,
            "threshold": threshold,
        }
        print(
            f"aligned threshold {threshold:.4f} using "
            f"{len(resolved_alignment_tiles)} {alignment_split} tiles"
        )

    manifest: dict[str, Any] = {
        "program_path": module_name,
        "split": split,
        "threshold": threshold,
        "min_area_ha": min_area_ha,
        "tiles": [],
    }
    if alignment is not None:
        manifest["alignment"] = alignment
    tile_feature_collections: list[dict[str, Any]] = []

    raster_dir = output_dir / "rasters"
    geojson_dir = output_dir / "geojson"

    for tile_id in tile_ids:
        reference, features = feature_builder(data_root, tile_id, split)
        result = run_experiment(features, threshold=threshold)
        if not isinstance(result, dict):
            raise ValueError(
                f"run_experiment result for tile {tile_id} must be a dict with keys "
                f"{', '.join(REQUIRED_RESULT_KEYS)}"
            )

        missing_keys = [key for key in REQUIRED_RESULT_KEYS if key not in result]
        if missing_keys:
            raise ValueError(
                f"run_experiment result for tile {tile_id} missing keys: "
                f"{', '.join(missing_keys)}"
            )

        prediction = np.asarray(result["prediction"], dtype=np.uint8)
        probabilities = np.asarray(result["probabilities"], dtype=np.float32)
        time_step = np.asarray(result["time_step"], dtype=np.uint16)

        prediction_path = raster_dir / f"pred_{tile_id}.tif"
        time_step_path = raster_dir / f"time_step_{tile_id}.tif"
        geojson_path = geojson_dir / f"pred_{tile_id}.geojson"

        _write_raster(
            prediction_path,
            prediction,
            reference,
            dtype="uint8",
            nodata=0,
        )
        _write_raster(
            time_step_path,
            time_step,
            reference,
            dtype="uint16",
            nodata=0,
        )

        geojson = raster_to_geojson(
            raster_path=prediction_path,
            output_path=geojson_path,
            min_area_ha=min_area_ha,
            time_step=None,
            allow_empty=True,
        )
        tile_feature_collections.append(geojson)

        tile_summary = {
            "tile_id": tile_id,
            "positive_pixels": int(np.count_nonzero(prediction)),
            "mean_probability": float(np.mean(probabilities)),
            "max_probability": float(np.max(probabilities)),
            "features": len(geojson["features"]),
        }
        manifest["tiles"].append(tile_summary)
        print(
            f"{tile_id}: {tile_summary['positive_pixels']} positive pixels, "
            f"{tile_summary['features']} polygons -> {geojson_path}"
        )

    submission_path = output_dir / "submission.geojson"
    submission = _write_combined_submission(submission_path, tile_feature_collections)
    manifest["submission_geojson"] = str(submission_path)
    manifest["submission_features"] = len(submission["features"])

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"wrote {submission_path} with {len(submission['features'])} total features")
    return submission_path, manifest_path


def build_argparser(*, align_train_default: bool = False) -> argparse.ArgumentParser:
    """Build the CLI parser shared by promoted model entrypoints."""

    parser = argparse.ArgumentParser(
        description="Generate a validated GeoJSON submission for a promoted model."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--tiles", nargs="*")
    parser.add_argument("--threshold", type=float, default=0.52)
    parser.add_argument("--min-area-ha", type=float, default=0.5)
    parser.add_argument(
        "--align-train",
        action=argparse.BooleanOptionalAction,
        default=align_train_default,
        help=(
            "fit a supported model's submission alignment on train metadata tiles "
            "before generating the requested split"
        ),
    )
    parser.add_argument("--alignment-split", default="train")
    parser.add_argument("--alignment-tiles", nargs="*")
    return parser


def _default_output_dir(model_module: SupportsRunExperiment) -> Path:
    module_name = _canonical_module_name(model_module)
    return Path("submission") / module_name.rsplit(".", maxsplit=1)[-1]


def run_from_cli(
    model_module: SupportsRunExperiment,
    argv: list[str] | None = None,
    *,
    align_train_default: bool = False,
) -> int:
    """CLI entrypoint used by promoted model shims under ``stats_models/``."""

    parser = build_argparser(align_train_default=align_train_default)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    output_dir = args.output_dir or _default_output_dir(model_module)
    generate_submission(
        model_module,
        data_root=args.data_root,
        output_dir=output_dir,
        split=args.split,
        tiles=args.tiles,
        threshold=args.threshold,
        min_area_ha=args.min_area_ha,
        align_train=args.align_train,
        alignment_split=args.alignment_split,
        alignment_tiles=args.alignment_tiles,
    )
    return 0


__all__ = ["build_argparser", "generate_submission", "run_from_cli"]
