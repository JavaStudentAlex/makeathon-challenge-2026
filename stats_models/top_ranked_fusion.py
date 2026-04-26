"""Promoted statistical model: top_ranked_fusion (from gen_87 / best).

This module preserves the evolved public contract and formula block from
``results/simple_ensembles/gen_87/original.py``, whose artifact is byte-identical
to ``results/simple_ensembles/best/original.py`` and carries the top
``combined_score`` in the tracked metrics.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from shinka.labels import target_from_train_labels


ArrayMap = Mapping[str, np.ndarray]
FeatureBuilder = Callable[[Path, str, str], tuple[Any, dict[str, np.ndarray]]]
TRAINING_TIMEOUT_SECONDS = 30 * 60
ALIGNMENT_THRESHOLDS = tuple(
    float(value)
    for value in np.concatenate(
        [np.arange(0.05, 0.951, 0.01), np.array([0.97, 0.98, 0.99])]
    )
)


@contextmanager
def _wall_time_limit(seconds: int):
    """Fail direct local runs that spend too long in candidate model code."""

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
            f"run_experiment exceeded {seconds} seconds; keep training bounded"
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


def _feature(features: ArrayMap, name: str, default: float = 0.0) -> np.ndarray:
    try:
        reference = next(iter(features.values()))
    except StopIteration as exc:
        raise ValueError("features must contain at least one array") from exc
    value = features.get(name)
    if value is None:
        return np.full(reference.shape, default, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    return np.nan_to_num(
        array,
        nan=default,
        posinf=default,
        neginf=default,
    ).astype(np.float32, copy=False)


def _safe_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _positive_signal(values: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(values / scale, 0.0, 1.0)


def _drop_signal(values: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(-values / scale, 0.0, 1.0)


# EVOLVE-BLOCK-START
def _normalized_signals(features: ArrayMap) -> dict[str, np.ndarray]:
    """Build bounded per-pixel evidence channels for shallow ensemble members."""

    return {
        "forest": (_feature(features, "forest_2020", default=1.0) >= 0.5).astype(
            np.float32
        ),
        "ndvi": _drop_signal(_feature(features, "ndvi_delta"), 0.80),
        "nbr": _drop_signal(_feature(features, "nbr_delta"), 0.95),
        "ndmi": _drop_signal(_feature(features, "ndmi_delta"), 0.70),
        "evi": _drop_signal(_feature(features, "evi_delta"), 1.00),
        "ndre": _drop_signal(_feature(features, "ndre_delta"), 0.70),
        "bsi": _positive_signal(_feature(features, "bsi_delta"), 0.60),
        "ndwi": _positive_signal(_feature(features, "ndwi_delta"), 0.80),
        "vv": _drop_signal(_feature(features, "vv_delta"), 0.25),
        "vv_cv": _positive_signal(_feature(features, "vv_cv_delta"), 0.30),
        "aef": _positive_signal(_feature(features, "aef_shift"), 0.85),
        "alert": np.clip(_feature(features, "alert_consensus"), 0.0, 1.0),
        "seasonal": _positive_signal(_feature(features, "seasonal_drop"), 0.60),
        "ndvi_z": _drop_signal(_feature(features, "ndvi_zscore"), 3.0),
        "nbr_z": _drop_signal(_feature(features, "nbr_zscore"), 3.0),
        "ndmi_z": _drop_signal(_feature(features, "ndmi_zscore"), 3.0),
        "bsi_z": _positive_signal(_feature(features, "bsi_zscore"), 3.0),
        "vv_z": _drop_signal(_feature(features, "vv_zscore"), 3.0),
        "water": np.clip(_feature(features, "water"), 0.0, 1.0),
        "crop": np.clip(_feature(features, "crop"), 0.0, 1.0),
        "urban": np.clip(_feature(features, "urban"), 0.0, 1.0),
        "bare": np.clip(_feature(features, "bare"), 0.0, 1.0),
        "cloud": np.clip(_feature(features, "cloud"), 0.0, 1.0),
    }


def _linear_margin_model(signals: dict[str, np.ndarray]) -> np.ndarray:
    """Logistic-regression-style fixed margin over robust normalized signals."""

    margin = (
        -10.05964816
        + 2.42527754 * signals["forest"]
        + 0.48136702 * signals["ndvi"]
        - 2.14275027 * signals["nbr"]
        - 0.55701068 * signals["ndmi"]
        - 1.27410346 * signals["evi"]
        + 4.47727629 * signals["ndre"]
        + 12.13401261 * signals["bsi"]
        - 3.64696641 * signals["ndwi"]
        + 1.63595877 * signals["seasonal"]
        + 1.66100565 * signals["vv"]
        + 1.90352555 * signals["vv_cv"]
        + 0.45 * signals["aef"]
        + 0.55 * signals["alert"]
    )
    return _safe_sigmoid(margin)


def _rule_vote_model(signals: dict[str, np.ndarray]) -> np.ndarray:
    """Conservative explicit rules requiring multi-sensor or multi-index support."""

    nbr = signals["nbr"]
    ndmi = signals["ndmi"]
    ndvi = signals["ndvi"]
    ndre = signals["ndre"]
    bsi = signals["bsi"]
    ndwi = signals["ndwi"]
    vv = signals["vv"]
    seasonal = signals["seasonal"]
    aef = signals["aef"]
    alert = signals["alert"]

    votes = (
        ((nbr > 0.45) & (ndmi > 0.60) & (bsi > 0.70)).astype(np.float32)
        + ((nbr > 0.60) & (bsi > 0.70)).astype(np.float32)
        + ((ndmi > 0.63) & (bsi > 0.68)).astype(np.float32)
        + ((ndre > 0.68) & (bsi > 0.68) & (ndmi > 0.52)).astype(np.float32)
        + ((ndvi > 0.72) & (nbr > 0.55) & (bsi > 0.62)).astype(np.float32)
        + ((vv > 0.58) & (bsi > 0.58) & (ndmi > 0.50)).astype(np.float32)
        + ((nbr > 0.75) & (ndmi > 0.55) & (bsi > 0.55)).astype(np.float32)
        + ((aef > 0.75) & (bsi > 0.55)).astype(np.float32)
        + ((alert > 0.55) & (nbr > 0.45)).astype(np.float32)
        - ((ndwi > 0.95) & (bsi < 0.65)).astype(np.float32)
        - ((seasonal > 0.90) & (nbr < 0.62)).astype(np.float32)
    )
    return np.clip(votes / 5.8, 0.0, 1.0)


def _stump_ensemble_model(signals: dict[str, np.ndarray]) -> np.ndarray:
    """Tiny boosted-stump-style vote over shallow one-feature thresholds."""

    nbr = signals["nbr"]
    ndmi = signals["ndmi"]
    bsi = signals["bsi"]
    ndwi = signals["ndwi"]
    ndre = signals["ndre"]
    vv = signals["vv"]
    seasonal = signals["seasonal"]
    aef = signals["aef"]
    alert = signals["alert"]

    score = (
        0.15 * (nbr > 0.45)
        + 0.13 * (nbr > 0.60)
        + 0.16 * (ndmi > 0.55)
        + 0.14 * (ndmi > 0.67)
        + 0.17 * (bsi > 0.62)
        + 0.13 * (bsi > 0.72)
        + 0.08 * (ndre > 0.68)
        + 0.05 * (vv > 0.55)
        + 0.05 * (aef > 0.70)
        + 0.05 * (alert > 0.50)
        - 0.08 * ((ndwi > 0.92) & (bsi < 0.70))
        - 0.05 * ((seasonal > 0.88) & (nbr < 0.65))
    )
    return np.clip(score.astype(np.float32), 0.0, 1.0)


def _prototype_distance_model(signals: dict[str, np.ndarray]) -> np.ndarray:
    """Gaussian/prototype-style separation between cleared and stable pixels."""

    cleared_distance = (
        (signals["ndvi"] - 0.78) ** 2
        + (signals["nbr"] - 0.78) ** 2
        + (signals["ndmi"] - 0.80) ** 2
        + (signals["ndre"] - 0.76) ** 2
        + (signals["bsi"] - 0.82) ** 2
        + (signals["vv"] - 0.55) ** 2
    ) / 6.0
    stable_distance = (
        (signals["ndvi"] - 0.60) ** 2
        + (signals["nbr"] - 0.48) ** 2
        + (signals["ndmi"] - 0.40) ** 2
        + (signals["ndre"] - 0.55) ** 2
        + (signals["bsi"] - 0.38) ** 2
        + (signals["vv"] - 0.35) ** 2
    ) / 6.0
    return _safe_sigmoid(6.0 * (stable_distance - cleared_distance))


def _sensor_agreement_model(signals: dict[str, np.ndarray]) -> np.ndarray:
    """Require vegetation/moisture loss, structure loss, and exposure to agree."""

    optical_drop = np.mean(
        np.stack(
            [signals["ndvi"], signals["nbr"], signals["ndmi"], signals["ndre"]],
            axis=0,
        ),
        axis=0,
    )
    structural_drop = np.mean(
        np.stack([signals["vv"], signals["vv_cv"], signals["aef"]], axis=0),
        axis=0,
    )
    exposure = np.maximum(
        np.maximum(signals["bsi"], signals["bare"]),
        0.55 * signals["nbr"] + 0.45 * signals["ndmi"],
    )
    positive = np.minimum(optical_drop, exposure)
    positive = np.minimum(
        positive, 0.55 + 0.45 * np.maximum(structural_drop, signals["alert"])
    )
    penalty = np.maximum.reduce(
        [
            signals["water"],
            signals["crop"],
            signals["urban"],
            signals["cloud"],
            signals["seasonal"],
            signals["ndwi"],
        ]
    )
    return np.clip(positive - 0.45 * penalty, 0.0, 1.0)


def _temporal_anomaly_model(signals: dict[str, np.ndarray]) -> np.ndarray:
    """EWMA/z-score-style anomaly score with raw-delta fallback evidence."""

    optical_z = (
        0.31 * signals["nbr_z"]
        + 0.27 * signals["ndmi_z"]
        + 0.20 * signals["ndvi_z"]
        + 0.22 * signals["bsi_z"]
    )
    abrupt_delta = (
        0.28 * signals["nbr"]
        + 0.27 * signals["ndmi"]
        + 0.20 * signals["bsi"]
        + 0.15 * signals["ndvi"]
        + 0.10 * signals["vv"]
    )
    sar_z = 0.68 * signals["vv_z"] + 0.32 * signals["vv_cv"]
    anomaly = 0.48 * optical_z + 0.34 * abrupt_delta + 0.18 * sar_z

    negative_context = np.maximum.reduce(
        [signals["water"], signals["cloud"], signals["seasonal"], signals["ndwi"]]
    )
    return np.clip(anomaly - 0.22 * negative_context, 0.0, 1.0).astype(np.float32)


def _mean_filter3x3(values: np.ndarray) -> np.ndarray:
    """Cheap local consensus filter used as support, not as a new predictor."""

    padded = np.pad(values.astype(np.float32, copy=False), 1, mode="edge")
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0


def _conservative_fusion_model(signals: dict[str, np.ndarray]) -> np.ndarray:
    """Blend shallow model families with explicit negative-evidence suppression."""

    optical_core = np.mean(
        np.stack(
            [signals["ndvi"], signals["nbr"], signals["ndmi"], signals["ndre"]],
            axis=0,
        ),
        axis=0,
    )
    structural_core = np.mean(
        np.stack([signals["vv"], signals["vv_cv"], signals["aef"]], axis=0),
        axis=0,
    )
    context_penalty = np.maximum.reduce(
        [signals["water"], signals["crop"], signals["urban"], signals["cloud"]]
    )

    margin = (
        2.35 * _linear_margin_model(signals)
        + 0.85 * _rule_vote_model(signals)
        + 0.70 * _stump_ensemble_model(signals)
        + 0.55 * _prototype_distance_model(signals)
        + 1.10 * _sensor_agreement_model(signals)
        + 0.55 * _temporal_anomaly_model(signals)
        + 0.75 * np.minimum(1.0, 0.5 * optical_core + 0.5 * structural_core)
        + 0.35 * signals["alert"]
        - 1.85 * context_penalty
        - 0.95 * signals["seasonal"]
        - 0.75 * signals["ndwi"]
        - 2.10
    )
    return _safe_sigmoid(margin)


def predict_deforestation_probability(features: ArrayMap) -> np.ndarray:
    """Primitive ensemble of shallow models for engineered change features."""

    signals = _normalized_signals(features)
    linear = _linear_margin_model(signals)
    rules = _rule_vote_model(signals)
    stumps = _stump_ensemble_model(signals)
    prototype = _prototype_distance_model(signals)
    agreement = _sensor_agreement_model(signals)
    anomaly = _temporal_anomaly_model(signals)
    conservative = _conservative_fusion_model(signals)

    raw_probability = (
        0.42 * linear
        + 0.09 * rules
        + 0.07 * stumps
        + 0.05 * prototype
        + 0.14 * agreement
        + 0.10 * anomaly
        + 0.13 * conservative
    )

    local_support = _mean_filter3x3(raw_probability)
    strong_clearing = (
        (signals["bsi"] > 0.76)
        & (np.maximum(signals["nbr"], signals["ndmi"]) > 0.66)
        & (np.maximum(agreement, anomaly) > 0.48)
    )
    isolated_weak = (raw_probability > 0.46) & (local_support < 0.31) & ~strong_clearing

    clustered = 0.78 * raw_probability + 0.22 * local_support
    clustered = np.where(isolated_weak, 0.84 * clustered, clustered)
    clustered = np.maximum(clustered, np.where(strong_clearing, raw_probability, 0.0))

    calibrated = _safe_sigmoid(11.0 * (clustered - 0.545))
    return calibrated.astype(np.float32) * signals["forest"]


def _valid_time_step(time_step: np.ndarray) -> np.ndarray:
    yy = time_step // 100
    mm = time_step % 100
    is_valid = (yy >= 21) & (yy <= 26) & (mm >= 1) & (mm <= 12)
    return np.where(is_valid, time_step, 2506)


def predict_deforestation_time_step(
    features: ArrayMap,
    prediction: np.ndarray,
) -> np.ndarray:
    """Predict YYMM time step for pixels selected by the binary mask."""

    first_alert_time_step = np.rint(_feature(features, "first_alert_time_step")).astype(
        np.int16
    )
    anomaly_time_step = np.rint(
        _feature(features, "strongest_anomaly_time_step", default=2506.0)
    ).astype(np.int16)

    time_step = np.where(
        first_alert_time_step > 0, first_alert_time_step, anomaly_time_step
    )
    time_step = _valid_time_step(time_step)
    return np.where(prediction.astype(bool), time_step, 0).astype(np.uint16)


# EVOLVE-BLOCK-END


def run_experiment(
    features: ArrayMap,
    threshold: float = 0.52,
) -> dict[str, Any]:
    """ShinkaEvolve entrypoint used by ``evaluation.py``."""

    with _wall_time_limit(TRAINING_TIMEOUT_SECONDS):
        probabilities = predict_deforestation_probability(features)
        prediction = (probabilities >= threshold).astype(np.uint8)
        time_step = predict_deforestation_time_step(features, prediction)

    year = np.where(time_step > 0, 2000 + (time_step // 100), 0).astype(np.uint16)
    return {
        "probabilities": probabilities.astype(np.float32),
        "prediction": prediction,
        "time_step": time_step,
        "year": year,
    }


def fit_submission_alignment(
    *,
    data_root: Path,
    split: str,
    tiles: Iterable[str],
    initial_threshold: float,
    feature_builder: FeatureBuilder,
) -> dict[str, Any]:
    """Fit the final submission threshold against all requested train labels."""

    if split != "train":
        raise ValueError(
            "top-ranked alignment uses train-only labels; split must be train"
        )

    tile_ids = list(tiles)
    if not tile_ids:
        raise ValueError("top-ranked alignment requires at least one train tile")

    thresholds = np.asarray(ALIGNMENT_THRESHOLDS, dtype=np.float32)
    true_positive_pixels = np.zeros(thresholds.shape, dtype=np.int64)
    predicted_pixels = np.zeros(thresholds.shape, dtype=np.int64)
    target_pixels = 0
    total_pixels = 0
    tile_summaries: list[dict[str, Any]] = []

    for tile_id in tile_ids:
        reference, features = feature_builder(data_root, tile_id, split)
        probabilities = np.nan_to_num(
            predict_deforestation_probability(features),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)
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
        "method": "weak_train_label_pixel_iou_threshold_grid",
        "status": status,
        "threshold": float(thresholds[best_index]),
        "metric": "pixel_iou",
        "metric_value": float(pixel_iou[best_index]),
        "tile_count": len(tile_ids),
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


if __name__ == "__main__":
    import sys

    from stats_models.runner import run_from_cli

    sys.exit(run_from_cli(sys.modules[__name__], align_train_default=True))
