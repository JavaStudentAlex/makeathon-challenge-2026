import json
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

from shinka.features import ReferenceGrid
from stats_models import (
    eligibility_and_patch_votes,
    spatial_consensus_and_time_median,
    spatial_consensus_and_timing,
)
from stats_models.runner import _default_output_dir, generate_submission
from submission_utils import validate_submission_geojson

FEATURE_KEYS = (
    "ndvi_delta",
    "nbr_delta",
    "ndmi_delta",
    "evi_delta",
    "vv_delta",
    "forest_2020",
    "bsi_delta",
    "aef_shift",
)

TIME_STEP_FEATURE_KEYS = (
    "first_alert_time_step",
    "strongest_anomaly_time_step",
    "anomaly_time_step",
    "change_time_step",
    "predicted_time_step",
)


def _write_metadata(tmp_path: Path, tile_ids: list[str], split: str = "test") -> Path:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    features = []
    for index, tile_id in enumerate(tile_ids):
        x0 = -70.0 + (index * 0.1)
        y0 = -10.0
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [x0, y0],
                            [x0 + 0.01, y0],
                            [x0 + 0.01, y0 + 0.01],
                            [x0, y0 + 0.01],
                            [x0, y0],
                        ]
                    ],
                },
                "properties": {"name": tile_id},
            }
        )

    metadata_path = metadata_dir / f"{split}_tiles.geojson"
    metadata_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )
    return metadata_path


def _synthetic_reference(shape: tuple[int, int] = (8, 8)) -> ReferenceGrid:
    return ReferenceGrid(
        shape=shape,
        transform=from_origin(500000, 1000, 100, 100),
        crs="EPSG:32618",
    )


def _make_synthetic_feature_builder(positive: bool = True):
    def feature_builder(data_root: Path, tile_id: str, split: str):
        del data_root, tile_id, split

        shape = (8, 8)
        if positive:
            features = {
                "ndvi_delta": np.full(shape, -0.7, dtype=np.float32),
                "nbr_delta": np.full(shape, -0.7, dtype=np.float32),
                "ndmi_delta": np.full(shape, -0.7, dtype=np.float32),
                "evi_delta": np.full(shape, -0.7, dtype=np.float32),
                "vv_delta": np.full(shape, -0.7, dtype=np.float32),
                "forest_2020": np.ones(shape, dtype=np.float32),
                "bsi_delta": np.full(shape, 0.7, dtype=np.float32),
                "aef_shift": np.full(shape, 0.7, dtype=np.float32),
            }
        else:
            features = {key: np.zeros(shape, dtype=np.float32) for key in FEATURE_KEYS}

        return _synthetic_reference(shape), features

    return feature_builder


def _spatial_consensus_positive_features(
    shape: tuple[int, int] = (8, 8),
    *,
    include_alert_consensus: bool = False,
) -> dict[str, np.ndarray]:
    features = {
        "ndvi_delta": np.full(shape, -0.7, dtype=np.float32),
        "nbr_delta": np.full(shape, -0.7, dtype=np.float32),
        "ndmi_delta": np.full(shape, -0.7, dtype=np.float32),
        "evi_delta": np.full(shape, -0.7, dtype=np.float32),
        "ndre_delta": np.full(shape, -0.7, dtype=np.float32),
        "vv_delta": np.full(shape, -0.7, dtype=np.float32),
        "bsi_delta": np.full(shape, 0.7, dtype=np.float32),
        "ndwi_delta": np.zeros(shape, dtype=np.float32),
        "forest_2020": np.ones(shape, dtype=np.float32),
        "aef_shift": np.full(shape, 0.7, dtype=np.float32),
    }
    if include_alert_consensus:
        features["alert_consensus"] = np.full(shape, 0.8, dtype=np.float32)
    features.update(
        {
            "first_alert_time_step": np.full(shape, 2308, dtype=np.float32),
            "strongest_anomaly_time_step": np.full(shape, 2307, dtype=np.float32),
            "anomaly_time_step": np.full(shape, 2307, dtype=np.float32),
            "change_time_step": np.full(shape, 2306, dtype=np.float32),
            "predicted_time_step": np.full(shape, 2307, dtype=np.float32),
        }
    )
    return features


def _make_spatial_consensus_feature_builder(*, include_alert_consensus: bool = False):
    def feature_builder(data_root: Path, tile_id: str, split: str):
        del data_root, tile_id, split

        shape = (8, 8)
        return _synthetic_reference(shape), _spatial_consensus_positive_features(
            shape,
            include_alert_consensus=include_alert_consensus,
        )

    return feature_builder


def test_generate_submission_combines_per_tile_geojsons(tmp_path: Path) -> None:
    tile_ids = ["tile_a", "tile_b"]
    data_root = tmp_path / "synthetic_data"
    data_root.mkdir()
    _write_metadata(data_root, tile_ids)

    output_dir = tmp_path / "output"
    submission_path, manifest_path = generate_submission(
        eligibility_and_patch_votes,
        data_root,
        output_dir,
        split="test",
        tiles=tile_ids,
        feature_builder=_make_synthetic_feature_builder(positive=True),
    )

    assert submission_path == output_dir / "submission.geojson"
    assert manifest_path == output_dir / "manifest.json"
    assert submission_path.exists()
    assert manifest_path.exists()

    validate_submission_geojson(submission_path)

    combined = json.loads(submission_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(manifest["tiles"]) == 2
    assert manifest["submission_features"] == len(combined["features"])

    for tile_id in tile_ids:
        assert (output_dir / "rasters" / f"pred_{tile_id}.tif").exists()
        assert (output_dir / "geojson" / f"pred_{tile_id}.geojson").exists()


@pytest.mark.parametrize(
    ("model_module", "expected_program_path", "include_alert_consensus"),
    [
        (
            spatial_consensus_and_time_median,
            "stats_models.spatial_consensus_and_time_median",
            False,
        ),
        (
            spatial_consensus_and_timing,
            "stats_models.spatial_consensus_and_timing",
            True,
        ),
    ],
    ids=["spatial_consensus_and_time_median", "spatial_consensus_and_timing"],
)
def test_generate_submission_builds_valid_bundle_for_spatial_consensus_models(
    tmp_path: Path,
    model_module,
    expected_program_path: str,
    include_alert_consensus: bool,
) -> None:
    tile_ids = ["tile_a", "tile_b"]
    data_root = tmp_path / "synthetic_data"
    data_root.mkdir()
    _write_metadata(data_root, tile_ids)

    output_dir = (
        tmp_path / f"{model_module.__name__.rsplit('.', maxsplit=1)[-1]}_output"
    )
    submission_path, manifest_path = generate_submission(
        model_module,
        data_root,
        output_dir,
        split="test",
        tiles=tile_ids,
        feature_builder=_make_spatial_consensus_feature_builder(
            include_alert_consensus=include_alert_consensus
        ),
    )

    assert submission_path == output_dir / "submission.geojson"
    assert manifest_path == output_dir / "manifest.json"
    assert submission_path.exists()
    assert manifest_path.exists()

    validate_submission_geojson(submission_path)

    combined = json.loads(submission_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["program_path"] == expected_program_path
    assert manifest["submission_features"] == len(combined["features"])
    assert combined["features"]
    assert len(manifest["tiles"]) == len(tile_ids)
    assert all(
        tile_summary["positive_pixels"] > 0 for tile_summary in manifest["tiles"]
    )

    for feature in combined["features"]:
        assert feature["properties"].get("time_step") is None

    for tile_id in tile_ids:
        assert (output_dir / "rasters" / f"pred_{tile_id}.tif").exists()
        assert (output_dir / "rasters" / f"time_step_{tile_id}.tif").exists()
        assert (output_dir / "geojson" / f"pred_{tile_id}.geojson").exists()


def test_generate_submission_handles_empty_predictions(tmp_path: Path) -> None:
    tile_ids = ["tile_a", "tile_b"]
    data_root = tmp_path / "synthetic_data"
    data_root.mkdir()
    _write_metadata(data_root, tile_ids)

    submission_path, _ = generate_submission(
        eligibility_and_patch_votes,
        data_root,
        tmp_path / "output",
        split="test",
        tiles=tile_ids,
        feature_builder=_make_synthetic_feature_builder(positive=False),
    )

    validate_submission_geojson(submission_path)
    combined = json.loads(submission_path.read_text(encoding="utf-8"))
    assert combined["features"] == []


def test_generate_submission_uses_metadata_tile_ids_when_tiles_none(
    tmp_path: Path,
) -> None:
    tile_ids = ["tile_from_metadata_a", "tile_from_metadata_b"]
    data_root = tmp_path / "synthetic_data"
    data_root.mkdir()
    _write_metadata(data_root, tile_ids)

    _, manifest_path = generate_submission(
        eligibility_and_patch_votes,
        data_root,
        tmp_path / "output",
        split="test",
        tiles=None,
        feature_builder=_make_synthetic_feature_builder(positive=True),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [tile["tile_id"] for tile in manifest["tiles"]] == tile_ids


def test_generate_submission_raises_for_missing_data_root(tmp_path: Path) -> None:
    missing_data_root = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError):
        generate_submission(
            eligibility_and_patch_votes,
            missing_data_root,
            tmp_path / "output",
            split="test",
            tiles=["tile_a"],
            feature_builder=_make_synthetic_feature_builder(positive=True),
        )


class _BrokenModelModule:
    __file__ = "broken_model.py"

    @staticmethod
    def run_experiment(features: dict[str, np.ndarray], threshold: float = 0.52):
        del threshold
        shape = next(iter(features.values())).shape
        return {
            "prediction": np.zeros(shape, dtype=np.uint8),
            "probabilities": np.zeros(shape, dtype=np.float32),
        }


class _MainStyleModelModule:
    __name__ = "__main__"

    class __spec__:
        name = "stats_models.fake_promoted_model"

    @staticmethod
    def run_experiment(features: dict[str, np.ndarray], threshold: float = 0.52):
        del threshold
        shape = next(iter(features.values())).shape
        return {
            "prediction": np.ones(shape, dtype=np.uint8),
            "probabilities": np.ones(shape, dtype=np.float32),
            "time_step": np.full(shape, 2307, dtype=np.uint16),
        }


def test_generate_submission_validates_run_experiment_contract(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "synthetic_data"
    data_root.mkdir()
    _write_metadata(data_root, ["tile_a"])

    with pytest.raises(ValueError, match="time_step"):
        generate_submission(
            _BrokenModelModule,
            data_root,
            tmp_path / "output",
            split="test",
            tiles=["tile_a"],
            feature_builder=_make_synthetic_feature_builder(positive=True),
        )


def test_default_output_dir_uses_canonical_module_name() -> None:
    assert _default_output_dir(_MainStyleModelModule) == Path(
        "submission/fake_promoted_model"
    )


def test_generate_submission_uses_canonical_module_name_in_manifest(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "synthetic_data"
    data_root.mkdir()
    _write_metadata(data_root, ["tile_a"])

    _, manifest_path = generate_submission(
        _MainStyleModelModule,
        data_root,
        tmp_path / "output",
        split="test",
        tiles=["tile_a"],
        feature_builder=_make_synthetic_feature_builder(positive=True),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["program_path"] == "stats_models.fake_promoted_model"


def test_eligibility_and_patch_votes_run_experiment_contract() -> None:
    shape = (4, 4)
    features = {key: np.zeros(shape, dtype=np.float32) for key in FEATURE_KEYS}

    result = eligibility_and_patch_votes.run_experiment(features, threshold=0.52)

    assert result["prediction"].dtype == np.uint8
    assert result["prediction"].shape == shape
    assert result["probabilities"].dtype == np.float32
    assert result["probabilities"].shape == shape
    assert result["time_step"].dtype == np.uint16
    assert result["time_step"].shape == shape
    assert result["year"].dtype == np.uint16
    assert result["year"].shape == shape
