"""Initial Shinka seed program for shallow deforestation ensembles.

The seed is intentionally self-contained so Shinka can evaluate it without the
downloaded challenge dataset. It trains tiny prior models on synthetic examples
that encode the report-derived feature assumptions, then applies the ensemble
to a few candidate patches and returns GeoJSON polygons.
"""

# EVOLVE-BLOCK-START
from __future__ import annotations

import signal
import threading
import warnings
from contextlib import contextmanager
from typing import Any, NamedTuple

import numpy as np
from shapely.geometry import box, mapping
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURE_NAMES: tuple[str, ...] = (
    "forest_2020",
    "ndvi_delta",
    "nbr_delta",
    "ndmi_delta",
    "evi_delta",
    "ndre_delta",
    "bsi_delta",
    "ndwi_delta",
    "vv_delta",
    "vv_cv_delta",
    "aef_shift",
    "alert_consensus",
    "seasonal_drop",
    "ndvi_zscore",
    "nbr_zscore",
    "ndmi_zscore",
    "bsi_zscore",
    "vv_zscore",
    "water",
    "crop",
    "urban",
    "cloud",
)

POSITIVE_GEOMETRY = box(0.0, 0.0, 0.01, 0.01)
POSITIVE_TIME_STEP = 2308
RANDOM_STATE = 7
TRAINING_TIMEOUT_SECONDS = 30 * 60


class EnsembleMember(NamedTuple):
    name: str
    model: Any
    weight: float


@contextmanager
def _wall_time_limit(seconds: int):
    """Bound local model fitting so evolved candidates cannot hang indefinitely."""

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
            f"model fitting exceeded {seconds} seconds; keep training bounded"
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


def _row(**overrides: float) -> dict[str, float]:
    """Create one normalized feature row with stable-forest defaults."""

    values = {
        "forest_2020": 1.0,
        "ndvi_delta": 0.0,
        "nbr_delta": 0.0,
        "ndmi_delta": 0.0,
        "evi_delta": 0.0,
        "ndre_delta": 0.0,
        "bsi_delta": 0.0,
        "ndwi_delta": 0.0,
        "vv_delta": 0.0,
        "vv_cv_delta": 0.0,
        "aef_shift": 0.0,
        "alert_consensus": 0.0,
        "seasonal_drop": 0.0,
        "ndvi_zscore": 0.0,
        "nbr_zscore": 0.0,
        "ndmi_zscore": 0.0,
        "bsi_zscore": 0.0,
        "vv_zscore": 0.0,
        "water": 0.0,
        "crop": 0.0,
        "urban": 0.0,
        "cloud": 0.0,
    }
    values.update(overrides)
    return values


def _matrix(rows: list[dict[str, float]]) -> np.ndarray:
    return np.asarray(
        [[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float
    )


def _build_training_examples() -> tuple[np.ndarray, np.ndarray]:
    """Encode report ideas as a tiny prior dataset for shallow ML models."""

    positive_rows = [
        _row(
            ndvi_delta=-0.82,
            nbr_delta=-0.88,
            ndmi_delta=-0.74,
            evi_delta=-0.63,
            ndre_delta=-0.58,
            bsi_delta=0.72,
            vv_delta=-0.23,
            vv_cv_delta=0.30,
            aef_shift=0.82,
            alert_consensus=0.90,
            ndvi_zscore=-3.2,
            nbr_zscore=-3.5,
            ndmi_zscore=-3.0,
            bsi_zscore=3.1,
            vv_zscore=-2.7,
        ),
        _row(
            ndvi_delta=-0.58,
            nbr_delta=-0.77,
            ndmi_delta=-0.70,
            bsi_delta=0.80,
            vv_delta=-0.30,
            vv_cv_delta=0.34,
            aef_shift=0.74,
            alert_consensus=0.72,
            nbr_zscore=-3.0,
            ndmi_zscore=-2.7,
            bsi_zscore=3.4,
            vv_zscore=-3.2,
        ),
        _row(
            ndvi_delta=-0.72,
            nbr_delta=-0.92,
            ndmi_delta=-0.62,
            evi_delta=-0.55,
            ndre_delta=-0.61,
            bsi_delta=0.66,
            aef_shift=0.88,
            alert_consensus=0.60,
            nbr_zscore=-3.6,
            bsi_zscore=2.8,
        ),
        _row(
            ndvi_delta=-0.49,
            nbr_delta=-0.69,
            ndmi_delta=-0.64,
            bsi_delta=0.62,
            vv_delta=-0.27,
            aef_shift=0.70,
            alert_consensus=0.82,
            ndmi_zscore=-2.8,
            vv_zscore=-2.6,
        ),
    ]
    negative_rows = [
        _row(),
        _row(ndvi_delta=-0.25, nbr_delta=-0.20, ndmi_delta=-0.18, seasonal_drop=0.95),
        _row(
            ndvi_delta=-0.45,
            nbr_delta=-0.42,
            ndmi_delta=-0.30,
            bsi_delta=0.20,
            cloud=0.90,
        ),
        _row(
            ndvi_delta=-0.52,
            nbr_delta=-0.55,
            ndmi_delta=-0.15,
            ndwi_delta=0.95,
            water=0.95,
        ),
        _row(
            forest_2020=0.0,
            ndvi_delta=-0.80,
            nbr_delta=-0.85,
            bsi_delta=0.90,
            aef_shift=0.70,
        ),
        _row(bsi_delta=0.82, crop=0.90, forest_2020=0.30),
        _row(bsi_delta=0.76, urban=0.95, forest_2020=0.20),
        _row(alert_consensus=0.72, forest_2020=1.0, nbr_delta=-0.12),
    ]

    rows = positive_rows + negative_rows
    labels = np.asarray([1] * len(positive_rows) + [0] * len(negative_rows), dtype=int)
    return _matrix(rows), labels


def _fit_ensemble(x_train: np.ndarray, y_train: np.ndarray) -> list[EnsembleMember]:
    """Fit XGBoost, LightGBM, and SVM members with safe fallbacks."""

    members: list[EnsembleMember] = []

    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=18,
            max_depth=2,
            learning_rate=0.25,
            objective="binary:logistic",
            eval_metric="logloss",
            subsample=1.0,
            colsample_bytree=0.9,
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        )
        with _wall_time_limit(TRAINING_TIMEOUT_SECONDS):
            xgb.fit(x_train, y_train)
        members.append(EnsembleMember("xgboost", xgb, 0.42))
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        lgbm = LGBMClassifier(
            n_estimators=24,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.20,
            objective="binary",
            min_child_samples=1,
            min_data_in_leaf=1,
            min_data_in_bin=1,
            random_state=RANDOM_STATE + 1,
            n_jobs=1,
            verbosity=-1,
            force_col_wise=True,
        )
        with _wall_time_limit(TRAINING_TIMEOUT_SECONDS):
            lgbm.fit(x_train, y_train)
        members.append(EnsembleMember("lightgbm", lgbm, 0.42))
    except Exception:
        pass

    try:
        svm = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE + 2,
                max_iter=1000,
            ),
        )
        with _wall_time_limit(TRAINING_TIMEOUT_SECONDS):
            svm.fit(x_train, y_train)
        members.append(EnsembleMember("svm_rbf", svm, 0.16))
    except Exception:
        pass

    return members


def _positive_signal(value: np.ndarray | float, scale: float) -> np.ndarray | float:
    return np.clip(np.asarray(value, dtype=float) / scale, 0.0, 1.0)


def _drop_signal(value: np.ndarray | float, scale: float) -> np.ndarray | float:
    return np.clip(-np.asarray(value, dtype=float) / scale, 0.0, 1.0)


def _domain_prior_probability(x_values: np.ndarray) -> np.ndarray:
    """Fixed physics-inspired prior blended with learned model probabilities."""

    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    optical_drop = np.mean(
        np.stack(
            [
                _drop_signal(x_values[:, idx["ndvi_delta"]], 0.80),
                _drop_signal(x_values[:, idx["nbr_delta"]], 0.95),
                _drop_signal(x_values[:, idx["ndmi_delta"]], 0.70),
                _drop_signal(x_values[:, idx["evi_delta"]], 1.00),
                _drop_signal(x_values[:, idx["ndre_delta"]], 0.70),
            ],
            axis=0,
        ),
        axis=0,
    )
    exposure = _positive_signal(x_values[:, idx["bsi_delta"]], 0.70)
    sar_drop = np.mean(
        np.stack(
            [
                _drop_signal(x_values[:, idx["vv_delta"]], 0.30),
                _positive_signal(x_values[:, idx["vv_cv_delta"]], 0.35),
            ],
            axis=0,
        ),
        axis=0,
    )
    semantic_shift = _positive_signal(x_values[:, idx["aef_shift"]], 0.85)
    alert = np.clip(x_values[:, idx["alert_consensus"]], 0.0, 1.0)
    forest = np.clip(x_values[:, idx["forest_2020"]], 0.0, 1.0)
    negative_context = np.maximum.reduce(
        [
            np.clip(x_values[:, idx["water"]], 0.0, 1.0),
            np.clip(x_values[:, idx["crop"]], 0.0, 1.0),
            np.clip(x_values[:, idx["urban"]], 0.0, 1.0),
            np.clip(x_values[:, idx["cloud"]], 0.0, 1.0),
            _positive_signal(x_values[:, idx["seasonal_drop"]], 0.90),
            _positive_signal(x_values[:, idx["ndwi_delta"]], 0.90),
        ]
    )

    probability = (
        0.34 * optical_drop
        + 0.18 * exposure
        + 0.16 * sar_drop
        + 0.16 * semantic_shift
        + 0.16 * alert
        - 0.40 * negative_context
    )
    return np.clip(probability, 0.0, 1.0) * (forest >= 0.5)


def _predict_ensemble_probability(
    members: list[EnsembleMember],
    x_values: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    weighted = np.zeros(x_values.shape[0], dtype=float)
    total_weight = 0.0
    names: list[str] = []

    for member in members:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            probabilities = member.model.predict_proba(x_values)[:, 1]
        weighted += float(member.weight) * probabilities
        total_weight += float(member.weight)
        names.append(member.name)

    if not members:
        return _domain_prior_probability(x_values), ["domain_prior"]

    model_probability = weighted / max(total_weight, 1e-12)
    prior_probability = _domain_prior_probability(x_values)
    return (
        np.clip(0.72 * model_probability + 0.28 * prior_probability, 0.0, 1.0),
        names,
    )


def _candidate_rows() -> list[dict[str, Any]]:
    """Candidate patches for smoke evaluation and future Shinka mutations."""

    return [
        {
            "geometry": POSITIVE_GEOMETRY,
            "time_step": POSITIVE_TIME_STEP,
            "features": _row(
                ndvi_delta=-0.86,
                nbr_delta=-0.92,
                ndmi_delta=-0.78,
                evi_delta=-0.64,
                ndre_delta=-0.61,
                bsi_delta=0.78,
                vv_delta=-0.25,
                vv_cv_delta=0.32,
                aef_shift=0.86,
                alert_consensus=0.88,
                ndvi_zscore=-3.4,
                nbr_zscore=-3.7,
                ndmi_zscore=-3.2,
                bsi_zscore=3.3,
                vv_zscore=-2.9,
            ),
        },
        {
            "geometry": box(0.02, 0.0, 0.03, 0.01),
            "time_step": 0,
            "features": _row(),
        },
        {
            "geometry": box(0.0, 0.02, 0.01, 0.03),
            "time_step": 0,
            "features": _row(
                ndvi_delta=-0.40,
                nbr_delta=-0.38,
                ndmi_delta=-0.25,
                seasonal_drop=0.95,
                cloud=0.40,
            ),
        },
        {
            "geometry": box(0.02, 0.02, 0.03, 0.03),
            "time_step": 0,
            "features": _row(
                forest_2020=0.0,
                ndvi_delta=-0.90,
                nbr_delta=-0.88,
                bsi_delta=0.88,
                aef_shift=0.80,
                urban=0.20,
            ),
        },
    ]


def run_experiment(threshold: float = 0.50) -> dict[str, Any]:
    """Return deforestation polygons predicted by a shallow ML ensemble."""

    x_train, y_train = _build_training_examples()
    members = _fit_ensemble(x_train, y_train)
    candidates = _candidate_rows()
    x_candidates = _matrix([candidate["features"] for candidate in candidates])
    probabilities, model_names = _predict_ensemble_probability(members, x_candidates)

    features = []
    for candidate, probability in zip(candidates, probabilities, strict=True):
        time_step = int(candidate["time_step"])
        if probability < threshold or time_step <= 0:
            continue
        year = 2000 + (time_step // 100)
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(candidate["geometry"]),
                "properties": {
                    "time_step": time_step,
                    "year": year,
                    "probability": float(probability),
                    "model_ensemble": "+".join(model_names),
                },
            }
        )

    return {"type": "FeatureCollection", "features": features}


if __name__ == "__main__":
    import json

    print(json.dumps(run_experiment(), indent=2))
# EVOLVE-BLOCK-END
