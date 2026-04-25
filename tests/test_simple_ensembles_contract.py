import numpy as np
import pytest

import simple_ensembles
import simple_ensembles.balanced_fusion as balanced_fusion
import simple_ensembles.high_recall_fusion as high_recall_fusion
import simple_ensembles.top_ranked_fusion as top_ranked_fusion

RESULT_KEYS = {"probabilities", "prediction", "time_step", "year"}
MODULES = (
    pytest.param(top_ranked_fusion, id="top_ranked_fusion"),
    pytest.param(balanced_fusion, id="balanced_fusion"),
    pytest.param(high_recall_fusion, id="high_recall_fusion"),
)


def _positive_features(shape: tuple[int, int] = (4, 4)) -> dict[str, np.ndarray]:
    return {
        "forest_2020": np.ones(shape, dtype=np.float32),
        "ndvi_delta": np.full(shape, -0.8, dtype=np.float32),
        "nbr_delta": np.full(shape, -0.8, dtype=np.float32),
        "ndmi_delta": np.full(shape, -0.8, dtype=np.float32),
        "evi_delta": np.full(shape, -0.8, dtype=np.float32),
        "ndre_delta": np.full(shape, -0.8, dtype=np.float32),
        "bsi_delta": np.full(shape, 0.8, dtype=np.float32),
        "vv_delta": np.full(shape, -0.2, dtype=np.float32),
        "vv_cv_delta": np.full(shape, 0.3, dtype=np.float32),
        "aef_shift": np.full(shape, 0.85, dtype=np.float32),
        "alert_consensus": np.full(shape, 0.8, dtype=np.float32),
        "first_alert_time_step": np.full(shape, 2308, dtype=np.float32),
    }


def _zero_non_forest_features(
    shape: tuple[int, int] = (4, 4),
) -> dict[str, np.ndarray]:
    return {
        "forest_2020": np.zeros(shape, dtype=np.float32),
        "ndvi_delta": np.zeros(shape, dtype=np.float32),
        "nbr_delta": np.zeros(shape, dtype=np.float32),
        "ndmi_delta": np.zeros(shape, dtype=np.float32),
        "evi_delta": np.zeros(shape, dtype=np.float32),
        "ndre_delta": np.zeros(shape, dtype=np.float32),
        "bsi_delta": np.zeros(shape, dtype=np.float32),
        "vv_delta": np.zeros(shape, dtype=np.float32),
        "vv_cv_delta": np.zeros(shape, dtype=np.float32),
        "aef_shift": np.zeros(shape, dtype=np.float32),
        "alert_consensus": np.zeros(shape, dtype=np.float32),
        "first_alert_time_step": np.zeros(shape, dtype=np.float32),
    }


def _assert_result_contract(
    result: dict[str, np.ndarray], shape: tuple[int, int]
) -> None:
    assert set(result) == RESULT_KEYS
    assert result["probabilities"].shape == shape
    assert result["probabilities"].dtype == np.float32
    assert np.all(np.isfinite(result["probabilities"]))
    assert np.all((result["probabilities"] >= 0.0) & (result["probabilities"] <= 1.0))
    assert result["prediction"].shape == shape
    assert result["prediction"].dtype == np.uint8
    assert result["time_step"].shape == shape
    assert result["time_step"].dtype == np.uint16
    assert result["year"].shape == shape
    assert result["year"].dtype == np.uint16


def test_package_exports_all_promoted_modules() -> None:
    assert set(simple_ensembles.__all__) == {
        "balanced_fusion",
        "high_recall_fusion",
        "top_ranked_fusion",
    }


@pytest.mark.parametrize("module", MODULES)
def test_run_experiment_returns_expected_contract_for_strong_positive(module) -> None:
    result = module.run_experiment(_positive_features())

    _assert_result_contract(result, shape=(4, 4))
    assert np.all(result["probabilities"] > 0.52)
    assert np.all(result["prediction"] == 1)
    assert np.all(result["time_step"] == 2308)
    expected_year = np.where(
        result["time_step"] > 0,
        2000 + (result["time_step"] // 100),
        0,
    ).astype(np.uint16)
    np.testing.assert_array_equal(result["year"], expected_year)


@pytest.mark.parametrize("module", MODULES)
def test_run_experiment_zero_non_forest_returns_no_detections(module) -> None:
    result = module.run_experiment(_zero_non_forest_features())

    _assert_result_contract(result, shape=(4, 4))
    assert np.all(result["probabilities"] == 0.0)
    assert np.all(result["prediction"] == 0)
    assert np.all(result["time_step"] == 0)
    assert np.all(result["year"] == 0)


@pytest.mark.parametrize("module", MODULES)
def test_run_experiment_threshold_only_changes_prediction_outputs(module) -> None:
    features = _positive_features()

    default_result = module.run_experiment(features)
    strict_result = module.run_experiment(features, threshold=1.1)

    _assert_result_contract(default_result, shape=(4, 4))
    _assert_result_contract(strict_result, shape=(4, 4))
    np.testing.assert_array_equal(
        default_result["probabilities"],
        strict_result["probabilities"],
    )
    assert np.all(default_result["prediction"] == 1)
    assert np.all(strict_result["prediction"] == 0)
    assert np.all(strict_result["time_step"] == 0)
    assert np.all(strict_result["year"] == 0)


@pytest.mark.parametrize("module", MODULES)
def test_run_experiment_rejects_empty_feature_mapping(module) -> None:
    with pytest.raises(ValueError, match="features must contain at least one array"):
        module.run_experiment({})
