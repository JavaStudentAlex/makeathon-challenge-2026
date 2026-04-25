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
from submission_utils import raster_to_geojson, validate_submission_geojson

DEFAULT_DATA_ROOT = Path("data/makeathon-challenge")
REQUIRED_RESULT_KEYS = ("prediction", "probabilities", "time_step")


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
) -> tuple[Path, Path]:
    """Run a promoted statistical model across tiles and emit a submission bundle."""

    data_root = Path(data_root)
    output_dir = Path(output_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    tile_ids = _tile_ids_from_metadata(data_root, split) if tiles is None else list(tiles)
    run_experiment = _resolve_run_experiment(model_module)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "program_path": getattr(model_module, "__name__", model_module.__class__.__name__),
        "split": split,
        "threshold": threshold,
        "min_area_ha": min_area_ha,
        "tiles": [],
    }
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

    print(
        f"wrote {submission_path} with {len(submission['features'])} total features"
    )
    return submission_path, manifest_path


def build_argparser() -> argparse.ArgumentParser:
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
    return parser


def _default_output_dir(model_module: SupportsRunExperiment) -> Path:
    module_name = getattr(model_module, "__name__", "submission")
    return Path("submission") / module_name.rsplit(".", maxsplit=1)[-1]


def run_from_cli(model_module: SupportsRunExperiment, argv: list[str] | None = None) -> int:
    """CLI entrypoint used by promoted model shims under ``stats_models/``."""

    parser = build_argparser()
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
    )
    return 0


__all__ = ["build_argparser", "generate_submission", "run_from_cli"]
