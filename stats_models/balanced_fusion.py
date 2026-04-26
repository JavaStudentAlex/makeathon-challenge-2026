"""Promoted statistical model: balanced_fusion (from gen_73).

This module preserves the evolved public contract and formula block from
``results/simple_ensembles/gen_73/original.py`` while adding the maintained
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
class SignalBank:
    """Lazy normalized feature access with caching for shallow raster scoring."""

    __slots__ = ("features", "_cache")

    def __init__(self, features: ArrayMap):
        self.features = features
        self._cache: dict[str, np.ndarray] = {}

    def get(self, name: str) -> np.ndarray:
        cached = self._cache.get(name)
        if cached is not None:
            return cached

        if name == "forest":
            value = (_feature(self.features, "forest_2020", default=1.0) >= 0.5).astype(
                np.float32
            )
        elif name == "ndvi":
            value = _drop_signal(_feature(self.features, "ndvi_delta"), 0.80)
        elif name == "nbr":
            value = _drop_signal(_feature(self.features, "nbr_delta"), 0.95)
        elif name == "ndmi":
            value = _drop_signal(_feature(self.features, "ndmi_delta"), 0.70)
        elif name == "evi":
            value = _drop_signal(_feature(self.features, "evi_delta"), 1.00)
        elif name == "ndre":
            value = _drop_signal(_feature(self.features, "ndre_delta"), 0.70)
        elif name == "bsi":
            value = _positive_signal(_feature(self.features, "bsi_delta"), 0.60)
        elif name == "ndwi":
            value = _positive_signal(_feature(self.features, "ndwi_delta"), 0.80)
        elif name == "vv":
            value = _drop_signal(_feature(self.features, "vv_delta"), 0.25)
        elif name == "vv_cv":
            value = _positive_signal(_feature(self.features, "vv_cv_delta"), 0.30)
        elif name == "aef":
            value = _positive_signal(_feature(self.features, "aef_shift"), 0.85)
        elif name == "alert":
            value = np.clip(_feature(self.features, "alert_consensus"), 0.0, 1.0)
        elif name == "seasonal":
            value = _positive_signal(_feature(self.features, "seasonal_drop"), 0.60)
        elif name == "water":
            value = np.clip(_feature(self.features, "water"), 0.0, 1.0)
        elif name == "crop":
            value = np.clip(_feature(self.features, "crop"), 0.0, 1.0)
        elif name == "urban":
            value = np.clip(_feature(self.features, "urban"), 0.0, 1.0)
        elif name == "bare":
            value = np.clip(_feature(self.features, "bare"), 0.0, 1.0)
        elif name == "cloud":
            value = np.clip(_feature(self.features, "cloud"), 0.0, 1.0)
        elif name == "ndvi_z":
            value = _drop_signal(_feature(self.features, "ndvi_zscore"), 3.0)
        elif name == "nbr_z":
            value = _drop_signal(_feature(self.features, "nbr_zscore"), 3.0)
        elif name == "ndmi_z":
            value = _drop_signal(_feature(self.features, "ndmi_zscore"), 3.0)
        elif name == "bsi_z":
            value = _positive_signal(_feature(self.features, "bsi_zscore"), 3.0)
        elif name == "vv_z":
            value = _drop_signal(_feature(self.features, "vv_zscore"), 3.0)
        elif name == "ndvi_3m":
            value = _drop_signal(
                _feature(
                    self.features,
                    "ndvi_delta_3m",
                    default=float(np.nanmean(_feature(self.features, "ndvi_delta"))),
                ),
                0.72,
            )
        elif name == "nbr_3m":
            value = _drop_signal(
                _feature(
                    self.features,
                    "nbr_delta_3m",
                    default=float(np.nanmean(_feature(self.features, "nbr_delta"))),
                ),
                0.82,
            )
        elif name == "ndmi_3m":
            value = _drop_signal(
                _feature(
                    self.features,
                    "ndmi_delta_3m",
                    default=float(np.nanmean(_feature(self.features, "ndmi_delta"))),
                ),
                0.62,
            )
        elif name == "bsi_3m":
            value = _positive_signal(
                _feature(
                    self.features,
                    "bsi_delta_3m",
                    default=float(np.nanmean(_feature(self.features, "bsi_delta"))),
                ),
                0.55,
            )
        elif name == "ndvi_6m":
            value = _drop_signal(
                _feature(
                    self.features,
                    "ndvi_delta_6m",
                    default=float(np.nanmean(_feature(self.features, "ndvi_delta"))),
                ),
                0.66,
            )
        elif name == "nbr_6m":
            value = _drop_signal(
                _feature(
                    self.features,
                    "nbr_delta_6m",
                    default=float(np.nanmean(_feature(self.features, "nbr_delta"))),
                ),
                0.76,
            )
        elif name == "ndmi_6m":
            value = _drop_signal(
                _feature(
                    self.features,
                    "ndmi_delta_6m",
                    default=float(np.nanmean(_feature(self.features, "ndmi_delta"))),
                ),
                0.58,
            )
        elif name == "bsi_6m":
            value = _positive_signal(
                _feature(
                    self.features,
                    "bsi_delta_6m",
                    default=float(np.nanmean(_feature(self.features, "bsi_delta"))),
                ),
                0.50,
            )
        else:
            value = _feature(self.features, name)

        value = value.astype(np.float32, copy=False)
        self._cache[name] = value
        return value

    def mean(self, names: tuple[str, ...]) -> np.ndarray:
        acc = np.zeros_like(self.get(names[0]), dtype=np.float32)
        for name in names:
            acc += self.get(name)
        return acc / float(len(names))


def _mean_filter3x3(values: np.ndarray) -> np.ndarray:
    """Cheap neighborhood consensus smoother."""

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


def _max_filter3x3(values: np.ndarray) -> np.ndarray:
    """Cheap dilation-style support from the strongest neighboring pixel."""

    padded = np.pad(values.astype(np.float32, copy=False), 1, mode="edge")
    return np.maximum.reduce(
        [
            padded[:-2, :-2],
            padded[:-2, 1:-1],
            padded[:-2, 2:],
            padded[1:-1, :-2],
            padded[1:-1, 1:-1],
            padded[1:-1, 2:],
            padded[2:, :-2],
            padded[2:, 1:-1],
            padded[2:, 2:],
        ]
    )


def _linear_logistic(bank: SignalBank) -> np.ndarray:
    """Fixed logistic-regression-style score over normalized signals."""

    margin = (
        -10.05964816
        + 2.42527754 * bank.get("forest")
        + 0.48136702 * bank.get("ndvi")
        - 2.14275027 * bank.get("nbr")
        - 0.55701068 * bank.get("ndmi")
        - 1.27410346 * bank.get("evi")
        + 4.47727629 * bank.get("ndre")
        + 12.13401261 * bank.get("bsi")
        - 3.64696641 * bank.get("ndwi")
        + 1.63595877 * bank.get("seasonal")
        + 1.66100565 * bank.get("vv")
        + 1.90352555 * bank.get("vv_cv")
        + 0.45 * bank.get("aef")
        + 0.55 * bank.get("alert")
    )
    return _safe_sigmoid(margin).astype(np.float32)


def _logical_rule_score(bank: SignalBank) -> np.ndarray:
    """Decision-tree-like vote score using multi-index clearing rules."""

    ndvi = bank.get("ndvi")
    nbr = bank.get("nbr")
    ndmi = bank.get("ndmi")
    ndre = bank.get("ndre")
    bsi = bank.get("bsi")
    ndwi = bank.get("ndwi")
    vv = bank.get("vv")
    aef = bank.get("aef")
    alert = bank.get("alert")
    seasonal = bank.get("seasonal")

    votes = (
        1.00 * ((nbr > 0.45) & (ndmi > 0.60) & (bsi > 0.70))
        + 0.95 * ((nbr > 0.60) & (bsi > 0.70))
        + 0.95 * ((ndmi > 0.63) & (bsi > 0.68))
        + 0.90 * ((ndre > 0.68) & (bsi > 0.68) & (ndmi > 0.52))
        + 0.85 * ((ndvi > 0.72) & (nbr > 0.55) & (bsi > 0.62))
        + 0.75 * ((vv > 0.58) & (bsi > 0.58) & (ndmi > 0.50))
        + 0.70 * ((nbr > 0.75) & (ndmi > 0.55) & (bsi > 0.55))
        + 0.65 * ((aef > 0.75) & (bsi > 0.55))
        + 0.60 * ((alert > 0.55) & (nbr > 0.45))
        + 0.50 * ((alert > 0.45) & (bsi > 0.62) & (ndmi > 0.50))
        - 0.80 * ((ndwi > 0.95) & (bsi < 0.65))
        - 0.55 * ((seasonal > 0.90) & (nbr < 0.62))
    )
    return np.clip(votes / 6.4, 0.0, 1.0).astype(np.float32)


def _stump_additive_score(bank: SignalBank) -> np.ndarray:
    """Compact boosted-stump approximation with fixed thresholds."""

    nbr = bank.get("nbr")
    ndmi = bank.get("ndmi")
    bsi = bank.get("bsi")
    ndwi = bank.get("ndwi")
    ndre = bank.get("ndre")
    vv = bank.get("vv")
    vv_cv = bank.get("vv_cv")
    aef = bank.get("aef")
    alert = bank.get("alert")
    seasonal = bank.get("seasonal")

    score = (
        0.13 * (nbr > 0.42)
        + 0.12 * (nbr > 0.56)
        + 0.09 * (nbr > 0.72)
        + 0.14 * (ndmi > 0.52)
        + 0.12 * (ndmi > 0.65)
        + 0.16 * (bsi > 0.60)
        + 0.12 * (bsi > 0.72)
        + 0.07 * (ndre > 0.66)
        + 0.05 * (vv > 0.52)
        + 0.04 * (vv_cv > 0.58)
        + 0.04 * (aef > 0.70)
        + 0.04 * (alert > 0.50)
        - 0.07 * ((ndwi > 0.92) & (bsi < 0.70))
        - 0.05 * ((seasonal > 0.88) & (nbr < 0.65))
    )
    return np.clip(score.astype(np.float32), 0.0, 1.0)


def _prototype_margin(bank: SignalBank) -> np.ndarray:
    """Gaussian/prototype-style separation between cleared and stable pixels."""

    ndvi = bank.get("ndvi")
    nbr = bank.get("nbr")
    ndmi = bank.get("ndmi")
    ndre = bank.get("ndre")
    bsi = bank.get("bsi")
    vv = bank.get("vv")

    cleared_distance = (
        0.80 * (ndvi - 0.78) ** 2
        + 1.10 * (nbr - 0.78) ** 2
        + 1.10 * (ndmi - 0.80) ** 2
        + 0.85 * (ndre - 0.76) ** 2
        + 1.30 * (bsi - 0.82) ** 2
        + 0.65 * (vv - 0.55) ** 2
    ) / 5.80

    stable_distance = (
        0.80 * (ndvi - 0.60) ** 2
        + 1.10 * (nbr - 0.48) ** 2
        + 1.10 * (ndmi - 0.40) ** 2
        + 0.85 * (ndre - 0.55) ** 2
        + 1.30 * (bsi - 0.38) ** 2
        + 0.65 * (vv - 0.35) ** 2
    ) / 5.80

    return _safe_sigmoid(6.0 * (stable_distance - cleared_distance)).astype(np.float32)


def _sensor_agreement(bank: SignalBank) -> np.ndarray:
    """Require optical, exposure, and weak structural support to agree."""

    optical_drop = np.maximum(bank.get("nbr"), bank.get("ndmi"))
    exposure = np.maximum(
        np.maximum(bank.get("bsi"), bank.get("bare")),
        0.55 * bank.get("nbr") + 0.45 * bank.get("ndmi"),
    )
    structural = np.maximum.reduce([bank.get("vv"), bank.get("vv_cv"), bank.get("aef")])
    base = np.minimum(optical_drop, exposure)
    return np.clip(
        0.80 * base
        + 0.12 * np.minimum(base, structural)
        + 0.08 * np.minimum(base, bank.get("alert")),
        0.0,
        1.0,
    )


def _temporal_anomaly(bank: SignalBank) -> np.ndarray:
    """EWMA/z-score-style anomaly score with safe zero fallbacks."""

    optical_z = (
        0.32 * bank.get("nbr_z")
        + 0.27 * bank.get("ndmi_z")
        + 0.19 * bank.get("ndvi_z")
        + 0.22 * bank.get("bsi_z")
    )
    sar_z = 0.65 * bank.get("vv_z") + 0.35 * bank.get("vv_cv")
    abrupt_change = (
        0.35 * bank.get("nbr")
        + 0.30 * bank.get("ndmi")
        + 0.20 * bank.get("bsi")
        + 0.15 * bank.get("vv")
    )
    return np.clip(0.46 * optical_z + 0.19 * sar_z + 0.35 * abrupt_change, 0.0, 1.0)


def _persistent_loss_score(bank: SignalBank) -> np.ndarray:
    """Rolling-window permanence score, falling back to current deltas if absent."""

    abrupt_optical = np.maximum.reduce(
        [bank.get("nbr"), bank.get("ndmi"), bank.get("ndvi")]
    )
    abrupt_exposure = bank.get("bsi")
    mid_optical = np.maximum.reduce(
        [bank.get("nbr_3m"), bank.get("ndmi_3m"), bank.get("ndvi_3m")]
    )
    long_optical = np.maximum.reduce(
        [bank.get("nbr_6m"), bank.get("ndmi_6m"), bank.get("ndvi_6m")]
    )
    mid_exposure = bank.get("bsi_3m")
    long_exposure = bank.get("bsi_6m")

    persistence = np.minimum(
        0.58 * mid_optical + 0.42 * long_optical,
        0.55 * mid_exposure + 0.45 * long_exposure,
    )
    onset = np.minimum(abrupt_optical, abrupt_exposure)
    structure = np.maximum.reduce(
        [bank.get("vv"), bank.get("vv_cv"), bank.get("aef"), bank.get("alert")]
    )
    negative = np.maximum.reduce(
        [bank.get("ndwi"), bank.get("water"), bank.get("seasonal"), bank.get("cloud")]
    )

    score = (
        0.52 * persistence + 0.30 * onset + 0.18 * np.minimum(persistence, structure)
    )
    return np.clip(score - 0.24 * negative, 0.0, 1.0).astype(np.float32)


def _consensus_gate(bank: SignalBank) -> np.ndarray:
    """Independent evidence agreement gate to suppress diffuse false positives."""

    vegetation = bank.mean(("ndvi", "nbr", "ndmi", "ndre"))
    moisture_bare = np.minimum(
        np.maximum(bank.get("nbr"), bank.get("ndmi")),
        np.maximum(bank.get("bsi"), bank.get("bare")),
    )
    structure = np.maximum.reduce([bank.get("vv"), bank.get("vv_cv"), bank.get("aef")])
    alert = bank.get("alert")

    two_source = np.maximum(
        np.minimum(vegetation, moisture_bare),
        np.minimum(moisture_bare, np.maximum(structure, alert)),
    )
    strong_optical = np.minimum(
        moisture_bare, np.maximum(bank.get("nbr"), bank.get("ndmi"))
    )
    return np.clip(0.70 * two_source + 0.30 * strong_optical, 0.0, 1.0)


def _context_multiplier(bank: SignalBank) -> np.ndarray:
    """Multiplicative suppression for common false-positive contexts."""

    ndwi = bank.get("ndwi")
    seasonal = bank.get("seasonal")
    bsi = bank.get("bsi")
    vv = bank.get("vv")

    water_like = np.maximum(ndwi, bank.get("water"))
    landuse_confuser = np.maximum.reduce(
        [bank.get("crop"), bank.get("urban"), bank.get("cloud")]
    )
    bare_relief = 1.0 - 0.10 * bank.get("bare")

    water_penalty = 1.0 - 0.42 * np.clip(water_like - 0.70, 0.0, 1.0)
    seasonal_penalty = 1.0 - 0.34 * np.clip(
        seasonal * (1.0 - np.maximum(bsi, vv)),
        0.0,
        1.0,
    )
    confuser_penalty = 1.0 - 0.28 * landuse_confuser

    return np.clip(
        water_penalty * seasonal_penalty * confuser_penalty * bare_relief, 0.32, 1.0
    )


def _model_stack(bank: SignalBank) -> dict[str, np.ndarray]:
    """Run shallow model families and expose intermediate scores."""

    return {
        "linear": _linear_logistic(bank),
        "rules": _logical_rule_score(bank),
        "stumps": _stump_additive_score(bank),
        "prototype": _prototype_margin(bank),
        "agreement": _sensor_agreement(bank),
        "anomaly": _temporal_anomaly(bank),
        "persistent": _persistent_loss_score(bank),
        "gate": _consensus_gate(bank),
    }


def _fuse_scores(bank: SignalBank, scores: dict[str, np.ndarray]) -> np.ndarray:
    """Blend model outputs with conservative context gating and light smoothing."""

    raw = (
        0.45 * scores["linear"]
        + 0.10 * scores["rules"]
        + 0.07 * scores["stumps"]
        + 0.05 * scores["prototype"]
        + 0.12 * scores["agreement"]
        + 0.08 * scores["anomaly"]
        + 0.08 * scores["persistent"]
        + 0.05 * scores["gate"]
    )

    optical_core = np.maximum(bank.get("nbr"), bank.get("ndmi"))
    exposure_core = np.maximum(bank.get("bsi"), bank.get("bare"))
    structural_or_alert = np.maximum.reduce(
        [bank.get("vv"), bank.get("vv_cv"), bank.get("aef"), bank.get("alert")]
    )
    support = np.maximum(
        np.minimum(optical_core, exposure_core),
        np.minimum(exposure_core, structural_or_alert),
    )

    raw = (0.84 * raw + 0.16 * support) * _context_multiplier(bank)

    local = _mean_filter3x3(raw)
    neighbor_peak = _max_filter3x3(raw)
    clustered = 0.74 * raw + 0.23 * local + 0.03 * neighbor_peak

    strong_single_pixel = (exposure_core > 0.78) & (optical_core > 0.68) & (raw > 0.60)
    weak_isolated = (raw > 0.48) & (local < 0.36)

    # Hysteresis-style fill: expand only into adjacent pixels that already have
    # optical/exposure evidence, avoiding broad dilation into water/cloud/crop
    # confusers through the existing context multiplier.
    neighbor_core = (
        (neighbor_peak > 0.58)
        & (raw > 0.38)
        & (support > 0.50)
        & (optical_core > 0.48)
        & (exposure_core > 0.46)
    )
    neighbor_boost = np.minimum(0.90 * neighbor_peak, 0.58 + 0.30 * support)

    clustered = np.where(weak_isolated, 0.82 * clustered, clustered)
    clustered = np.where(
        neighbor_core, np.maximum(clustered, neighbor_boost), clustered
    )
    clustered = np.maximum(clustered, np.where(strong_single_pixel, 0.92 * raw, 0.0))

    return np.clip(clustered, 0.0, 1.0).astype(np.float32)


def predict_deforestation_probability(features: ArrayMap) -> np.ndarray:
    """Shallow ensemble pipeline for engineered deforestation features."""

    bank = SignalBank(features)
    scores = _model_stack(bank)
    fused = _fuse_scores(bank, scores)

    calibrated = _safe_sigmoid(10.4 * (fused - 0.615)).astype(np.float32)
    return calibrated * bank.get("forest")


def _valid_time_step(time_step: np.ndarray) -> np.ndarray:
    yy = time_step // 100
    mm = time_step % 100
    is_valid = (yy >= 21) & (yy <= 26) & (mm >= 1) & (mm <= 12)
    return np.where(is_valid, time_step, 2506)


def _time_preference(features: ArrayMap) -> np.ndarray:
    """Choose alert timing when present, otherwise anomaly timing."""

    first_alert_time_step = np.rint(_feature(features, "first_alert_time_step")).astype(
        np.int16
    )
    anomaly_time_step = np.rint(
        _feature(features, "strongest_anomaly_time_step", default=2506.0)
    ).astype(np.int16)

    time_step = np.where(
        first_alert_time_step > 0, first_alert_time_step, anomaly_time_step
    )
    return _valid_time_step(time_step)


def _min_valid_time_filter3x3(time_step: np.ndarray) -> np.ndarray:
    """Earliest valid neighboring YYMM, preserving 2506 when none is available."""

    valid = _valid_time_step(time_step.astype(np.int32, copy=False))
    padded = np.pad(valid, 1, mode="edge")
    candidates = np.stack(
        [
            padded[:-2, :-2],
            padded[:-2, 1:-1],
            padded[:-2, 2:],
            padded[1:-1, :-2],
            padded[1:-1, 1:-1],
            padded[1:-1, 2:],
            padded[2:, :-2],
            padded[2:, 1:-1],
            padded[2:, 2:],
        ],
        axis=0,
    )
    return np.min(candidates, axis=0)


def predict_deforestation_time_step(
    features: ArrayMap,
    prediction: np.ndarray,
) -> np.ndarray:
    """Predict YYMM time step for pixels selected by the binary mask."""

    time_step = _time_preference(features)
    neighbor_time = _min_valid_time_filter3x3(time_step)
    fill_from_cluster = prediction.astype(bool) & (neighbor_time < time_step)
    time_step = np.where(fill_from_cluster, neighbor_time, time_step)
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
