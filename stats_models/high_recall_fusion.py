"""Promoted statistical model: high_recall_fusion (from gen_61).

This module preserves the evolved public contract and formula block from
``results/simple_ensembles/gen_61/original.py`` while adding the maintained
empty-feature diagnostic used by promoted statistical-model modules.
"""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from typing import Any, Mapping

import numpy as np

ArrayMap = Mapping[str, np.ndarray]
TRAINING_TIMEOUT_SECONDS = 30 * 60


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
    """Require vegetation/moisture loss and bare-soil exposure to agree."""

    optical_drop = np.maximum(signals["nbr"], signals["ndmi"])
    exposure = np.maximum(
        signals["bsi"],
        0.55 * signals["nbr"] + 0.45 * signals["ndmi"],
    )
    return np.minimum(optical_drop, exposure)


def predict_deforestation_probability(features: ArrayMap) -> np.ndarray:
    """Primitive ensemble of shallow models for engineered change features."""

    signals = _normalized_signals(features)
    linear = _linear_margin_model(signals)
    rule_vote = _rule_vote_model(signals)
    stump_vote = _stump_ensemble_model(signals)
    prototype = _prototype_distance_model(signals)
    agreement = _sensor_agreement_model(signals)

    raw_probability = (
        0.78 * linear
        + 0.09 * rule_vote
        + 0.06 * stump_vote
        + 0.03 * prototype
        + 0.04 * agreement
    )
    calibrated = _safe_sigmoid(10.0 * (raw_probability - 0.725))
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


if __name__ == "__main__":
    import sys

    from stats_models.runner import run_from_cli

    sys.exit(run_from_cli(sys.modules[__name__]))
