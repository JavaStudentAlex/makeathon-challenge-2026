"""Promoted statistical model: eligibility_and_patch_votes (from gen_178).

Duplicate of gen_190; promoted as the canonical first architecture per M001/D001.
See stats_models/runner.py for the shared submission orchestration.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


ArrayMap = Mapping[str, np.ndarray]


def _feature(features: ArrayMap, name: str, default: float = 0.0) -> np.ndarray:
    reference = next(iter(features.values()))
    value = features.get(name)
    if value is None:
        return np.full(reference.shape, default, dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _safe_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _positive_signal(values: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(values / scale, 0.0, 1.0)


def _drop_signal(values: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(-values / scale, 0.0, 1.0)


# EVOLVE-BLOCK-START
def _finite(values: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Replace NaN/Inf so primitive raster formulas remain deterministic."""
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=fill, posinf=fill, neginf=fill)


def _feature_any(features: ArrayMap, names: tuple[str, ...], default: float = 0.0) -> np.ndarray:
    """Return the first available feature among common engineered aliases."""
    for name in names:
        if name in features:
            return _finite(_feature(features, name, default=default))
    return _finite(_feature(features, names[0], default=default))


def _has_any(features: ArrayMap, names: tuple[str, ...]) -> bool:
    """Whether any alias is available; used to keep optional gates non-destructive."""
    return any(name in features for name in names)


def _scene_mask(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    v = _finite(values)
    finite = np.isfinite(v)
    if mask is None:
        return v[finite]
    eligible = finite & mask.astype(bool)
    sample = v[eligible]
    if sample.size < 32:
        sample = v[finite]
    return sample


def _percentile_score(
    values: np.ndarray,
    low_q: float,
    high_q: float,
    *,
    mask: np.ndarray | None = None,
    invert: bool = False,
) -> np.ndarray:
    """Robust scene-relative score scaled from local percentiles."""
    v = _finite(values)
    sample = _scene_mask(v, mask)
    if sample.size == 0:
        return np.zeros_like(v, dtype=np.float32)

    lo = np.nanpercentile(sample, low_q)
    hi = np.nanpercentile(sample, high_q)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1.0e-6:
        return np.zeros_like(v, dtype=np.float32)

    if invert:
        score = (hi - v) / (hi - lo)
    else:
        score = (v - lo) / (hi - lo)
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def _adaptive_drop(
    values: np.ndarray,
    floor_q: float = 2.0,
    ceil_q: float = 32.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    return _percentile_score(values, floor_q, ceil_q, mask=mask, invert=True)


def _adaptive_positive(
    values: np.ndarray,
    floor_q: float = 72.0,
    ceil_q: float = 98.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    return _percentile_score(values, floor_q, ceil_q, mask=mask, invert=False)


def _z_drop(values: np.ndarray, scale: float) -> np.ndarray:
    """Convert standardized anomalies to a 0..1 drop score."""
    return _drop_signal(_finite(values), scale)


def _z_rise(values: np.ndarray, scale: float) -> np.ndarray:
    """Convert standardized anomalies to a 0..1 positive anomaly score."""
    return _positive_signal(_finite(values), scale)


def _local_mean_3x3(values: np.ndarray) -> np.ndarray:
    """Tiny spatial-coherence smoother for patchy clearings; no training or dependencies."""
    v = _finite(values)
    padded = np.pad(v, ((1, 1), (1, 1)), mode="edge")
    total = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return (total / 9.0).astype(np.float32)


def predict_deforestation_probability(features: ArrayMap) -> np.ndarray:
    """Primitive statistical ensemble over canopy loss, structure loss, and anomaly consensus."""
    forest_raw = _feature_any(
        features,
        ("forest_2020", "forest_mask_2020", "esa_forest_2020", "treecover_2020", "hansen_treecover_2020"),
        default=1.0,
    )
    forest_threshold = 30.0 if np.nanmax(forest_raw) > 1.5 else 0.5
    forest_2020 = forest_raw >= forest_threshold

    # Optional production-safe eligibility refinements: do not score pixels already
    # cleared before 2021 or mapped as permanent/static water when such rasters exist.
    if _has_any(features, ("pre2021_loss", "loss_before_2021", "deforested_before_2021", "treecover_loss_year")):
        preloss = _feature_any(
            features,
            ("pre2021_loss", "loss_before_2021", "deforested_before_2021", "treecover_loss_year"),
            default=0.0,
        )
        forest_2020 = forest_2020 & ((preloss <= 0.0) | (preloss >= 21.0))

    if _has_any(features, ("permanent_water", "water_2020", "static_water", "water_mask")):
        static_water = _feature_any(
            features,
            ("permanent_water", "water_2020", "static_water", "water_mask"),
            default=0.0,
        )
        water_threshold = 50.0 if np.nanmax(static_water) > 1.5 else 0.5
        forest_2020 = forest_2020 & (static_water < water_threshold)

    # Core deltas / changes
    ndvi_delta = _feature_any(features, ("ndvi_delta", "delta_ndvi"))
    nbr_delta = _feature_any(features, ("nbr_delta", "delta_nbr"))
    ndmi_delta = _feature_any(features, ("ndmi_delta", "delta_ndmi"))
    evi_delta = _feature_any(features, ("evi_delta", "delta_evi"))
    ndre_delta = _feature_any(features, ("ndre_delta", "delta_ndre"))
    vv_delta = _feature_any(features, ("vv_delta", "s1_vv_delta", "sentinel1_vv_delta"))
    bsi_delta = _feature_any(features, ("bsi_delta", "delta_bsi"))
    ndwi_delta = _feature_any(features, ("ndwi_delta", "delta_ndwi"))

    # Fixed-scale primitive evidence layers.
    ndvi_drop = _drop_signal(ndvi_delta, 0.45)
    nbr_drop = _drop_signal(nbr_delta, 0.50)
    ndmi_drop = _drop_signal(ndmi_delta, 0.40)
    evi_drop = _drop_signal(evi_delta, 0.35)
    ndre_drop = _drop_signal(ndre_delta, 0.30)
    vv_drop = _drop_signal(vv_delta, 0.25)

    bsi_rise = _positive_signal(bsi_delta, 0.45)
    vv_instability = _positive_signal(_feature_any(features, ("vv_cv_delta", "s1_vv_cv_delta", "vv_instability")), 0.25)
    aef_shift = _positive_signal(_feature_any(features, ("aef_shift", "alphaearth_shift", "embedding_shift")), 0.85)

    # Seasonal / hydrologic context.
    water_rise = _positive_signal(ndwi_delta, 0.35)
    water_drop = _drop_signal(ndwi_delta, 0.40)
    seasonal_drop = _positive_signal(_feature_any(features, ("seasonal_drop", "seasonality_score")), 0.35)

    # Forest-masked adaptive anomaly layers.
    ad_ndvi = _adaptive_drop(ndvi_delta, mask=forest_2020)
    ad_nbr = _adaptive_drop(nbr_delta, mask=forest_2020)
    ad_ndmi = _adaptive_drop(ndmi_delta, mask=forest_2020)
    ad_evi = _adaptive_drop(evi_delta, mask=forest_2020)
    ad_vv = _adaptive_drop(vv_delta, mask=forest_2020)
    ad_bsi = _adaptive_positive(bsi_delta, mask=forest_2020)

    # Standardized anomalies / EWMA-style scores.
    ndvi_anom = np.maximum(
        _z_drop(_feature_any(features, ("ndvi_z", "ndvi_anomaly_z")), 3.0),
        _z_drop(_feature_any(features, ("ndvi_delta_z", "ndvi_change_z")), 2.5),
    )
    nbr_anom = np.maximum(
        _z_drop(_feature_any(features, ("nbr_z", "nbr_anomaly_z")), 3.0),
        _z_drop(_feature_any(features, ("nbr_delta_z", "nbr_change_z")), 2.5),
    )
    ndmi_anom = np.maximum(
        _z_drop(_feature_any(features, ("ndmi_z", "ndmi_anomaly_z")), 3.0),
        _z_drop(_feature_any(features, ("ndmi_delta_z", "ndmi_change_z")), 2.5),
    )
    vv_anom = np.maximum(
        _z_drop(_feature_any(features, ("vv_z", "vv_anomaly_z")), 3.0),
        _z_drop(_feature_any(features, ("vv_delta_z", "vv_change_z")), 2.5),
    )
    bsi_anom = np.maximum(
        _z_rise(_feature_any(features, ("bsi_z", "bsi_anomaly_z")), 3.0),
        _z_rise(_feature_any(features, ("bsi_delta_z", "bsi_change_z")), 2.5),
    )
    ewma_anomaly = np.clip(
        _feature_any(features, ("ewma_anomaly_score", "ewma_score", "cumulative_anomaly_score")), 0.0, 1.0
    )

    # Evidence families.
    canopy_loss = np.clip(
        0.30 * ndvi_drop
        + 0.32 * nbr_drop
        + 0.18 * ndmi_drop
        + 0.12 * evi_drop
        + 0.08 * ndre_drop,
        0.0,
        1.0,
    )
    optical_agree = np.minimum(np.maximum(ndvi_drop, evi_drop), nbr_drop)
    moisture_agree = np.minimum(nbr_drop, ndmi_drop)
    exposed_clearing = np.minimum(np.maximum(nbr_drop, ndvi_drop), bsi_rise)
    structural_agree = np.maximum(vv_drop, 0.55 * vv_drop + 0.45 * vv_instability)
    latent_structural = np.minimum(aef_shift, np.maximum(structural_agree, exposed_clearing))
    dry_context = np.minimum(water_drop, np.maximum(ndmi_drop, bsi_rise))

    temporal_anomaly = np.clip(
        0.25 * ndvi_anom
        + 0.31 * nbr_anom
        + 0.18 * ndmi_anom
        + 0.13 * vv_anom
        + 0.11 * bsi_anom
        + 0.18 * ewma_anomaly,
        0.0,
        1.0,
    )

    adaptive_canopy = np.clip(
        0.27 * ad_ndvi
        + 0.33 * ad_nbr
        + 0.17 * ad_ndmi
        + 0.10 * ad_evi
        + 0.13 * ad_bsi,
        0.0,
        1.0,
    )
    adaptive_structure = np.maximum(ad_vv, np.minimum(ad_vv + 0.20, ad_bsi))

    adaptive_loss = np.minimum(
        np.maximum(adaptive_canopy, temporal_anomaly),
        np.maximum(adaptive_structure, exposed_clearing),
    )
    persistent_loss = np.minimum(
        np.maximum(canopy_loss, np.maximum(temporal_anomaly, adaptive_canopy)),
        np.maximum(exposed_clearing, structural_agree),
    )
    optical_only_loss = np.minimum(canopy_loss, np.minimum(optical_agree, moisture_agree))

    # Conservative evidence count to reward multi-signal agreement.
    evidence_count = (
        (canopy_loss > 0.40).astype(np.float32)
        + (optical_agree > 0.36).astype(np.float32)
        + (moisture_agree > 0.34).astype(np.float32)
        + (exposed_clearing > 0.34).astype(np.float32)
        + (structural_agree > 0.34).astype(np.float32)
        + (aef_shift > 0.40).astype(np.float32)
        + (temporal_anomaly > 0.42).astype(np.float32)
        + (adaptive_loss > 0.48).astype(np.float32)
        + (persistent_loss > 0.40).astype(np.float32)
        + (optical_only_loss > 0.50).astype(np.float32)
    )
    evidence_bonus = np.clip((evidence_count - 2.0) / 3.5, 0.0, 1.0)

    # Sensor-family voting separates broad evidence sources before spatial smoothing.
    optical_family = np.maximum(canopy_loss, optical_agree)
    soil_family = exposed_clearing
    sar_family = structural_agree
    temporal_family = np.maximum(temporal_anomaly, adaptive_loss)
    latent_family = latent_structural

    family_votes = (
        (optical_family > 0.42).astype(np.float32)
        + (soil_family > 0.34).astype(np.float32)
        + (sar_family > 0.36).astype(np.float32)
        + (temporal_family > 0.40).astype(np.float32)
        + (latent_family > 0.36).astype(np.float32)
    )
    family_strength = np.clip(
        0.31 * optical_family
        + 0.22 * soil_family
        + 0.22 * sar_family
        + 0.16 * temporal_family
        + 0.09 * latent_family,
        0.0,
        1.0,
    )
    consensus_patch = _local_mean_3x3(
        (
            (family_votes >= 2.0)
            & (family_strength > 0.34)
            & (water_rise < 0.78)
            & (seasonal_drop < 0.88)
        ).astype(np.float32)
    )

    # Explicit artifact suppression for water/seasonal dynamics.
    clearing_override = np.maximum.reduce([exposed_clearing, structural_agree, adaptive_loss, family_strength])
    water_penalty = water_rise * (1.0 - 0.55 * clearing_override)
    seasonal_penalty = seasonal_drop * (1.0 - 0.30 * np.maximum(optical_agree, temporal_anomaly))

    # Weighted linear score over primitive evidence layers.
    score = (
        -3.03
        + 0.80 * canopy_loss
        + 0.58 * nbr_drop
        + 0.32 * ndmi_drop
        + 0.46 * optical_agree
        + 0.28 * moisture_agree
        + 0.76 * exposed_clearing
        + 0.80 * structural_agree
        + 0.58 * aef_shift
        + 0.40 * latent_structural
        + 0.52 * temporal_anomaly
        + 0.53 * adaptive_loss
        + 0.48 * persistent_loss
        + 0.16 * optical_only_loss
        + 0.38 * evidence_bonus
        + 0.30 * family_strength
        + 0.42 * consensus_patch
        + 0.20 * dry_context
        - 1.44 * water_penalty
        - 1.00 * seasonal_penalty
    )

    probability = _safe_sigmoid(score).astype(np.float32)

    # Rule-based boosts only when several channels agree.
    optical_clearcut = (
        (optical_agree > 0.46)
        & (exposed_clearing > 0.38)
        & (moisture_agree > 0.35)
        & (water_rise < 0.68)
    )

    optical_only_strong = (
        (canopy_loss > 0.68)
        & (optical_agree > 0.55)
        & (moisture_agree > 0.50)
        & (adaptive_canopy > 0.44)
        & (water_rise < 0.55)
        & (seasonal_drop < 0.68)
    )

    sar_supported = (
        (canopy_loss > 0.42)
        & (structural_agree > 0.46)
        & (np.maximum(aef_shift, adaptive_loss) > 0.36)
        & (water_rise < 0.72)
    )

    anomaly_supported = (
        (persistent_loss > 0.46)
        & (temporal_anomaly > 0.38)
        & (evidence_count >= 3.0)
        & (seasonal_drop < 0.78)
    )

    adaptive_hotspot = (
        (adaptive_loss > 0.58)
        & (canopy_loss > 0.34)
        & (evidence_count >= 3.0)
        & (water_rise < 0.62)
        & (seasonal_drop < 0.80)
    )

    high_confidence = (
        (optical_clearcut | optical_only_strong | sar_supported | anomaly_supported | adaptive_hotspot)
        & (seasonal_drop < 0.84)
    )

    very_consistent = (
        (evidence_count >= 4.0)
        & (persistent_loss > 0.50)
        & (np.maximum.reduce([temporal_anomaly, adaptive_loss]) > 0.36)
        & (water_rise < 0.62)
        & (seasonal_drop < 0.74)
    )

    ultra_consistent = (
        (evidence_count >= 5.0)
        & (clearing_override > 0.52)
        & (canopy_loss > 0.45)
        & (water_rise < 0.58)
    )

    probability = np.where(high_confidence, np.maximum(probability, 0.58), probability)
    probability = np.where(optical_only_strong, np.maximum(probability, 0.54), probability)
    probability = np.where(adaptive_hotspot, np.maximum(probability, 0.585), probability)
    probability = np.where(very_consistent, np.maximum(probability, 0.625), probability)
    probability = np.where(ultra_consistent, np.maximum(probability, 0.680), probability)

    local_core = (
        ((probability > 0.52) & (evidence_count >= 2.0))
        | ((persistent_loss > 0.50) & (evidence_count >= 3.0))
        | ((optical_only_loss > 0.56) & (adaptive_canopy > 0.45))
        | ((family_votes >= 2.0) & (family_strength > 0.42))
    ).astype(np.float32)
    local_support = _local_mean_3x3(local_core)

    coherent_borderline = (
        (local_support >= 0.34)
        & (consensus_patch >= 0.18)
        & (probability > 0.452)
        & (persistent_loss > 0.36)
        & (evidence_count >= 2.0)
        & (water_rise < 0.74)
        & (seasonal_drop < 0.84)
    )
    compact_high_evidence = (
        (local_support >= 0.22)
        & (consensus_patch >= 0.12)
        & (probability > 0.482)
        & (evidence_count >= 4.0)
        & (clearing_override > 0.42)
        & (water_rise < 0.66)
    )
    probable_clear_patch = (
        (local_support >= 0.44)
        & (consensus_patch >= 0.24)
        & (evidence_count >= 3.0)
        & (np.maximum.reduce([adaptive_loss, persistent_loss, exposed_clearing]) > 0.40)
        & (water_rise < 0.68)
        & (seasonal_drop < 0.86)
    )
    strong_three_sensor_patch = (
        (consensus_patch >= 0.26)
        & (family_votes >= 3.0)
        & (family_strength > 0.48)
        & (clearing_override > 0.42)
        & (water_rise < 0.66)
        & (seasonal_drop < 0.80)
    )
    two_sensor_patch = (
        (consensus_patch >= 0.34)
        & (family_votes >= 2.0)
        & (family_strength > 0.46)
        & (probability > 0.44)
        & (water_rise < 0.70)
        & (seasonal_drop < 0.84)
    )

    probability = np.where(coherent_borderline, np.maximum(probability, 0.532), probability)
    probability = np.where(compact_high_evidence, np.maximum(probability, 0.552), probability)
    probability = np.where(probable_clear_patch, np.maximum(probability, 0.586), probability)
    probability = np.where(two_sensor_patch, np.maximum(probability, 0.548), probability)
    probability = np.where(strong_three_sensor_patch, np.maximum(probability, 0.602), probability)

    likely_water_artifact = (
        (water_rise > 0.86)
        & (clearing_override < 0.45)
        & (optical_only_loss < 0.62)
        & (consensus_patch < 0.16)
        & (local_support < 0.22)
    )
    likely_seasonal_artifact = (
        (seasonal_drop > 0.92)
        & (persistent_loss < 0.50)
        & (family_votes <= 1.0)
        & (local_support < 0.18)
    )
    weak_single_channel = (
        (evidence_count <= 1.0)
        & (persistent_loss < 0.42)
        & (canopy_loss < 0.38)
    )
    isolated_weak_alarm = (
        (local_support <= 0.11)
        & (consensus_patch < 0.10)
        & (probability < 0.61)
        & (evidence_count <= 2.0)
        & (family_votes <= 1.0)
        & (persistent_loss < 0.48)
        & (optical_only_loss < 0.58)
    )

    probability = np.where(likely_water_artifact | likely_seasonal_artifact, probability * 0.45, probability)
    probability = np.where(weak_single_channel, probability * 0.78, probability)
    probability = np.where(isolated_weak_alarm, probability * 0.70, probability)

    # Eligibility gate: never predict outside 2020 forest.
    return np.clip(probability, 0.0, 1.0).astype(np.float32) * forest_2020.astype(np.float32)


def _valid_time_step(time_step: np.ndarray) -> np.ndarray:
    yy = time_step // 100
    mm = time_step % 100
    is_valid = (yy >= 21) & (yy <= 26) & (mm >= 1) & (mm <= 12)
    return np.where(is_valid, time_step, 2506)


def _is_valid_yymm(time_step: np.ndarray) -> np.ndarray:
    yy = time_step // 100
    mm = time_step % 100
    return (yy >= 21) & (yy <= 26) & (mm >= 1) & (mm <= 12)


def predict_deforestation_time_step(
    features: ArrayMap,
    prediction: np.ndarray,
) -> np.ndarray:
    """Predict YYMM using consensus among valid event-time candidates, not simply earliest."""
    first_alert = np.rint(_feature_any(features, ("first_alert_time_step", "alert_time_step"))).astype(np.int32)
    strongest_anomaly = np.rint(
        _feature_any(features, ("strongest_anomaly_time_step", "strongest_change_time_step"), default=0.0)
    ).astype(np.int32)
    anomaly_time = np.rint(_feature_any(features, ("anomaly_time_step", "anomaly_yymm"), default=0.0)).astype(np.int32)
    change_time = np.rint(_feature_any(features, ("change_time_step", "change_yymm"), default=0.0)).astype(np.int32)
    predicted_time = np.rint(
        _feature_any(features, ("predicted_time_step", "predicted_yymm"), default=0.0)
    ).astype(np.int32)

    valid_sa = np.where(_is_valid_yymm(strongest_anomaly), strongest_anomaly, 9999)
    valid_an = np.where(_is_valid_yymm(anomaly_time), anomaly_time, 9999)
    valid_ch = np.where(_is_valid_yymm(change_time), change_time, 9999)
    valid_fa = np.where(_is_valid_yymm(first_alert), first_alert, 9999)
    valid_pr = np.where(_is_valid_yymm(predicted_time), predicted_time, 9999)

    primary_stack = np.stack([valid_sa, valid_an, valid_ch, valid_pr], axis=0)
    all_stack = np.stack([valid_sa, valid_an, valid_ch, valid_fa, valid_pr], axis=0)

    primary_valid = primary_stack != 9999
    primary_count = np.sum(primary_valid, axis=0)

    primary_min = np.min(primary_stack, axis=0)
    primary_max = np.max(np.where(primary_valid, primary_stack, -9999), axis=0)
    primary_spread = np.where(primary_count > 0, primary_max - primary_min, 9999)

    primary_sorted = np.sort(primary_stack, axis=0)
    primary_median = primary_sorted[2]
    primary_second = primary_sorted[1]
    primary_consensus = np.where(primary_count >= 3, primary_median, primary_second)

    earliest_any = np.min(all_stack, axis=0)
    earliest_primary = np.min(primary_stack, axis=0)

    use_primary_consensus = (primary_count >= 2) & (primary_spread <= 103)
    use_primary_earliest = primary_count >= 1
    use_alert_fallback = (valid_fa != 9999) & (primary_count < 2)

    time_step = np.where(use_primary_consensus, primary_consensus, 9999)
    time_step = np.where((time_step == 9999) & use_alert_fallback, valid_fa, time_step)
    time_step = np.where((time_step == 9999) & use_primary_earliest, earliest_primary, time_step)
    time_step = np.where(time_step == 9999, earliest_any, time_step)
    time_step = np.where(time_step == 9999, 2506, time_step)
    time_step = _valid_time_step(time_step.astype(np.int32))
    return np.where(prediction.astype(bool), time_step, 0).astype(np.uint16)


# EVOLVE-BLOCK-END


def run_experiment(
    features: ArrayMap,
    threshold: float = 0.52,
) -> dict[str, Any]:
    """ShinkaEvolve entrypoint used by ``evaluation.py``."""

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
