from typing import Any

import numpy as np

from stats_models import spatial_consensus_and_timing as model

CORE_FEATURE_KEYS = (
    "ndvi_delta",
    "nbr_delta",
    "ndmi_delta",
    "evi_delta",
    "ndre_delta",
    "vv_delta",
    "bsi_delta",
    "ndwi_delta",
    "forest_2020",
    "aef_shift",
    "alert_consensus",
)

TIME_STEP_FEATURE_KEYS = (
    "first_alert_time_step",
    "strongest_anomaly_time_step",
    "anomaly_time_step",
    "change_time_step",
    "predicted_time_step",
)


def _positive_features(shape: tuple[int, int] = (4, 4)) -> dict[str, np.ndarray]:
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
        "alert_consensus": np.full(shape, 0.8, dtype=np.float32),
    }
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


def _zero_non_forest_features(
    shape: tuple[int, int] = (4, 4),
) -> dict[str, np.ndarray]:
    features = {
        key: np.zeros(shape, dtype=np.float32)
        for key in CORE_FEATURE_KEYS + TIME_STEP_FEATURE_KEYS
    }
    features["forest_2020"] = np.zeros(shape, dtype=np.float32)
    return features


def _assert_runner_contract(result: dict[str, Any], shape: tuple[int, int]) -> None:
    assert result["probabilities"].shape == shape
    assert result["probabilities"].dtype == np.float32
    assert result["prediction"].shape == shape
    assert result["prediction"].dtype == np.uint8
    assert result["time_step"].shape == shape
    assert result["time_step"].dtype == np.uint16
    assert result["year"].shape == shape
    assert result["year"].dtype == np.uint16


def test_run_experiment_returns_expected_contract_for_strong_positive() -> None:
    result = model.run_experiment(_positive_features())

    _assert_runner_contract(result, shape=(4, 4))
    assert np.all(result["probabilities"] > 0.9)
    assert np.all(result["prediction"] == 1)


def test_run_experiment_all_zero_non_forest_returns_no_detections() -> None:
    result = model.run_experiment(_zero_non_forest_features())

    _assert_runner_contract(result, shape=(4, 4))
    assert np.all(result["probabilities"] == 0.0)
    assert np.all(result["prediction"] == 0)
    assert np.all(result["time_step"] == 0)
    assert np.all(result["year"] == 0)


def test_run_experiment_threshold_only_changes_prediction_outputs() -> None:
    features = _positive_features()

    default_result = model.run_experiment(features, threshold=0.52)
    strict_result = model.run_experiment(features, threshold=0.99)

    assert np.array_equal(
        default_result["probabilities"], strict_result["probabilities"]
    )
    assert np.all(default_result["prediction"] == 1)
    assert np.all(strict_result["prediction"] == 0)
    assert np.all(strict_result["time_step"] == 0)
    assert np.all(strict_result["year"] == 0)


def test_run_experiment_uses_primary_time_step_consensus_and_year_projection() -> None:
    result = model.run_experiment(_positive_features())

    assert np.all(result["prediction"] == 1)
    assert np.all(result["time_step"] == 2307)
    assert np.all(result["year"] == 2023)


def test_run_experiment_falls_back_to_first_alert_when_optional_steps_missing() -> None:
    features = _positive_features()
    for key in (
        "strongest_anomaly_time_step",
        "anomaly_time_step",
        "change_time_step",
        "predicted_time_step",
    ):
        features.pop(key)

    result = model.run_experiment(features)

    assert np.all(result["prediction"] == 1)
    assert np.all(result["time_step"] == 2308)
    assert np.all(result["year"] == 2023)


def test_run_experiment_normalizes_invalid_yymm_to_default_time_step() -> None:
    features = _positive_features()
    features["first_alert_time_step"] = np.full((4, 4), 2500, dtype=np.float32)
    features["strongest_anomaly_time_step"] = np.full((4, 4), 1300, dtype=np.float32)
    features["anomaly_time_step"] = np.full((4, 4), 9913, dtype=np.float32)
    features["change_time_step"] = np.zeros((4, 4), dtype=np.float32)
    features["predicted_time_step"] = np.full((4, 4), 5000, dtype=np.float32)

    result = model.run_experiment(features)

    assert np.all(result["prediction"] == 1)
    assert np.all(result["time_step"] == 2506)
    assert np.all(result["year"] == 2025)
